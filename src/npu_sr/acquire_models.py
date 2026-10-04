"""Pinned, small upstream models rebuilt into static ONNX graphs using NumPy."""

import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .acquire import SOURCE, SOURCE_SHA256, _fields, acquire, export, read_weights
from .errors import SRException
from .model import ModelSpec, model_path, model_spec, sha256
from .quicksr import export_quicksr, read_quicksr
from .weights import read_dncnn

FSRCNN_REVISION = "6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d"
LAPSRN_REVISION = "fc51c90af1b5801a357abc919160d7ff4f24b997"
SOURCES = {
    "ESPCN": (SOURCE, SOURCE_SHA256),
    "FSRCNN": (
        f"https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/{FSRCNN_REVISION}"
        "/models/FSRCNN_x2.pb",
        "366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c",
    ),
    "FSRCNN-small": (
        f"https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/{FSRCNN_REVISION}"
        "/models/FSRCNN-small_x2.pb",
        "429e4793d049c1ae16ddbbc322fd11c3c08831c0c20137390b4d098976a2b0d9",
    ),
    "LapSRN": (
        f"https://raw.githubusercontent.com/fannymonori/TF-LapSRN/{LAPSRN_REVISION}"
        "/export/LapSRN_x2.pb",
        "f59c86e6835bbca646dbd81588f07820dfe3bd09e3702976ae2d90b5fc0f0b21",
    ),
    "DnCNN": (
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth",
        "0451a70de9b672ae037270498fbb1c17a1c1c4403785df586ff65df5b858e5b0",
    ),
    "QuickSRNet-small": (
        "https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/"
        "quicksrnet_small_2x_checkpoint_float32.pth.tar",
        "d95d70f1d2366cb9c28d99f8c7aa5bb07e1ffeaf5d7e30d9c66ab0fa28c6d0f8",
    ),
    "QuickSRNet-medium": (
        "https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/"
        "quicksrnet_medium_2x_checkpoint_float32.pth.tar",
        "a0d176b40a649e45a176c3b53f45e0237015f4f2c17b157ef5c81e38c4442a0d",
    ),
}


def read_constants(data: bytes) -> dict[str, np.ndarray]:
    """Extract only float constants from a pinned TensorFlow GraphDef."""
    result = {}
    for raw in _fields(data)[1]:
        node = _fields(raw)
        if node[2][0] != b"Const":
            continue
        attributes = {_fields(a)[1][0].decode(): _fields(a)[2][0] for a in node[5]}
        tensor = _fields(_fields(attributes["value"])[8][0])
        if tensor[1][0] != 1:
            continue
        shape = [_fields(d)[1][0] for d in _fields(tensor[2][0]).get(2, [])]
        if 4 in tensor:
            values = np.frombuffer(tensor[4][0], "<f4")
        else:
            values = np.array([struct.unpack("<f", v)[0] for v in tensor[5]], np.float32)
        count = int(np.prod(shape))
        if values.size == 1 and count > 1:
            values = np.repeat(values, count)
        result[node[1][0].decode()] = values.reshape(shape).copy()
    return result


def export_model(weights: dict[str, np.ndarray], spec: ModelSpec, path: Path) -> None:
    """Export the known architecture, folding TF biases and equivalent activations."""
    nodes, initializers = [], []

    def constant(name: str, value: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(np.asarray(value, np.float32), name))
        return name

    def convolution(previous: str, key: str, bias: str | None, output: str) -> str:
        kernel = weights[key]
        if spec.architecture != "DnCNN":
            kernel = kernel.transpose(3, 2, 0, 1).copy()
        before, after = (kernel.shape[-1] - 1) // 2, kernel.shape[-1] // 2
        inputs = [previous, constant(key, kernel)]
        if bias:
            inputs.append(constant(bias, np.broadcast_to(weights[bias], (kernel.shape[0],))))
        nodes.append(
            helper.make_node(
                "Conv", inputs, [output], name=output, pads=[before, before, after, after]
            )
        )
        return output

    previous = "input"
    if spec.architecture.startswith("FSRCNN"):
        count = 5 if spec.architecture == "FSRCNN-small" else 8
        for index in range(1, count + 1):
            previous = convolution(
                previous, f"f{index}", f"b{index}" if index < count else None, f"c{index}"
            )
            if index < count:
                slope = constant(f"alpha{index}", weights[f"alpha{index}"].reshape(-1, 1, 1))
                nodes.append(helper.make_node("PRelu", [previous, slope], [f"a{index}"]))
                previous = f"a{index}"
        nodes.append(
            helper.make_node("DepthToSpace", [previous], ["shuffle"], blocksize=2, mode="DCR")
        )
        nodes.append(
            helper.make_node(
                "Add",
                ["shuffle", constant("final_bias", weights[f"b{count}"].reshape(1, 1, 1, 1))],
                ["output"],
            )
        )
    elif spec.architecture == "LapSRN":
        for index in range(10):
            key = "0_0f" if index == 0 else ("Variable" if index == 1 else f"Variable_{index - 1}")
            previous = convolution(previous, key, f"{index}_0bias", f"c{index}")
            slope = float(weights[f"mul_{2 * index + 1}/x"])
            nodes.append(
                helper.make_node(
                    "PRelu",
                    [previous, constant(f"slope{index}", np.full((64, 1, 1), slope))],
                    [f"a{index}"],
                )
            )
            previous = f"a{index}"
        convolution(previous, "reconstruction_2_deconv_f", None, "residual")
        nodes.append(
            helper.make_node("DepthToSpace", ["residual"], ["residual2x"], blocksize=2, mode="DCR")
        )
        nodes.append(
            helper.make_node(
                "PRelu",
                ["residual2x", constant("slope_res", weights["mul_21/x"].reshape(1, 1, 1))],
                ["residual_relu"],
            )
        )
        convolution("input", "reconstruction_2_deconv_f_1", None, "base")
        nodes.append(
            helper.make_node("DepthToSpace", ["base"], ["base2x"], blocksize=2, mode="DCR")
        )
        nodes.append(helper.make_node("Add", ["base2x", "residual_relu"], ["sum"]))
        nodes.append(
            helper.make_node(
                "PRelu",
                ["sum", constant("slope_out", weights["mul_23/x"].reshape(1, 1, 1))],
                ["output"],
            )
        )
    elif spec.architecture == "DnCNN":
        for index in range(17):
            previous = convolution(
                previous, f"model.{index * 2}.weight", f"model.{index * 2}.bias", f"c{index}"
            )
            if index < 16:
                nodes.append(helper.make_node("Relu", [previous], [f"r{index}"]))
                previous = f"r{index}"
        nodes.append(helper.make_node("Sub", ["input", previous], ["output"]))
    else:
        raise SRException(f"No exporter for {spec.architecture}")
    graph = helper.make_graph(
        nodes,
        spec.identifier,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, spec.input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, spec.output_shape)],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", spec.opset)], producer_name="npu-sr"
    )
    model.ir_version = 8
    helper.set_model_props(
        model,
        {
            "source": SOURCES[spec.architecture][0],
            "license": spec.license,
            "version": spec.version,
        },
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def acquire_model(identifier: str, directory: Path) -> Path:
    spec = model_spec(identifier)
    if identifier == "espcn-x2":
        return acquire(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source, digest = SOURCES[spec.architecture]
    upstream = directory / source.rsplit("/", 1)[1]
    if not upstream.is_file() or sha256(upstream) != digest:
        temporary = upstream.with_suffix(".download")
        try:
            with urllib.request.urlopen(source, timeout=60) as response:
                data = response.read(3_000_001)
            temporary.write_bytes(data)
            if sha256(temporary) != digest:
                raise SRException("Upstream model SHA256 mismatch; refusing to export.")
            temporary.replace(upstream)
        finally:
            temporary.unlink(missing_ok=True)
    data = upstream.read_bytes()
    path = model_path(directory, identifier)
    if spec.architecture == "ESPCN":
        export(read_weights(data), path, spec.input_shape)
    elif spec.architecture.startswith("QuickSRNet-"):
        weights = read_quicksr(data, spec.architecture.removeprefix("QuickSRNet-"))
        export_quicksr(weights, spec, path)
    else:
        weights = read_dncnn(data) if spec.architecture == "DnCNN" else read_constants(data)
        export_model(weights, spec, path)
    manifest = spec.info() | {
        "name": identifier,
        "sha256": sha256(path),
        "source": source,
        "source_sha256": digest,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
