"""Rebuild a small ONNX model from pinned, hash-checked upstream weights.

Only six float constants are read from TensorFlow's protobuf wire format.
No TensorFlow installation, pickle loading, or upstream Python execution is needed.
The exported graph changes layout to NCHW and folds bias additions into Conv.
"""

import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .errors import SRException
from .model import INPUT_SHAPE, MODEL_NAME, MODEL_VERSION, OUTPUT_SHAPE, model_path, sha256

REVISION = "5c628eca82028161a53e1265cc3a5b571ab8625f"
SOURCE = f"https://raw.githubusercontent.com/fannymonori/TF-ESPCN/{REVISION}/export/ESPCN_x2.pb"
SOURCE_SHA256 = "59f77351e1d7c0057bf6fe088b4a8a07e42c468c8c8aebb674a6b4ea1823221d"


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError("Invalid protobuf varint")


def _fields(data: bytes) -> dict[int, list]:
    """Read protobuf primitive fields; ignore unknown field numbers."""
    fields: dict[int, list] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, offset = _varint(data, offset)
            else:
                size = 8 if wire == 1 else 4
            value = data[offset : offset + size]
            offset += size
            if len(value) != size:
                raise ValueError("Truncated protobuf")
        else:
            raise ValueError("Unsupported protobuf wire type")
        fields.setdefault(number, []).append(value)
    return fields


def read_weights(data: bytes) -> dict[str, np.ndarray]:
    result = {}
    for encoded_node in _fields(data)[1]:
        node = _fields(encoded_node)
        name = node[1][0].decode()
        if name not in {"f1", "f2", "f3", "b1", "b2", "b3"}:
            continue
        attributes = {_fields(a)[1][0].decode(): _fields(a)[2][0] for a in node[5]}
        tensor = _fields(_fields(attributes["value"])[8][0])
        if tensor[1][0] != 1:  # TensorFlow DT_FLOAT
            raise ValueError("Expected float32 weights")
        shape = [_fields(dim)[1][0] for dim in _fields(tensor[2][0])[2]]
        if 4 in tensor:
            values = np.frombuffer(tensor[4][0], dtype="<f4")
        else:
            values = np.array([struct.unpack("<f", v)[0] for v in tensor[5]], dtype=np.float32)
        result[name] = values.reshape(shape).copy()
    expected = {
        "f1": (5, 5, 1, 64),
        "f2": (3, 3, 64, 32),
        "f3": (3, 3, 32, 4),
        "b1": (64,),
        "b2": (32,),
        "b3": (4,),
    }
    if {k: v.shape for k, v in result.items()} != expected:
        raise ValueError("Unexpected upstream weight shapes")
    return result


def export(weights: dict[str, np.ndarray], path: Path) -> None:
    nodes, initializers = [], []
    previous = "input"
    for index, pad in ((1, 2), (2, 1), (3, 1)):
        kernel = weights[f"f{index}"].transpose(3, 2, 0, 1).copy()
        initializers.extend(
            [
                numpy_helper.from_array(kernel, f"w{index}"),
                numpy_helper.from_array(weights[f"b{index}"], f"b{index}"),
            ]
        )
        conv = f"conv{index}"
        nodes.append(
            helper.make_node(
                "Conv", [previous, f"w{index}", f"b{index}"], [conv], name=conv, pads=[pad] * 4
            )
        )
        if index < 3:
            previous = f"relu{index}"
            nodes.append(helper.make_node("Relu", [conv], [previous], name=previous))
        else:
            previous = conv
    nodes.extend(
        [
            helper.make_node(
                "DepthToSpace",
                [previous],
                ["shuffle"],
                name="pixel_shuffle",
                blocksize=2,
                mode="DCR",
            ),
            helper.make_node("Tanh", ["shuffle"], ["output"], name="output_tanh"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        MODEL_NAME,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, INPUT_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, OUTPUT_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="npu-sr",
        producer_version="0.1.0",
    )
    model.ir_version = 8
    helper.set_model_props(
        model, {"source": SOURCE, "license": "Apache-2.0", "version": MODEL_VERSION}
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def acquire(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    upstream = directory / "ESPCN_x2.pb"
    if not upstream.is_file() or sha256(upstream) != SOURCE_SHA256:
        temporary = upstream.with_suffix(".download")
        try:
            with urllib.request.urlopen(SOURCE, timeout=60) as response:
                data = response.read(1_000_001)
            temporary.write_bytes(data)
            if sha256(temporary) != SOURCE_SHA256:
                raise SRException("Upstream model SHA256 mismatch; refusing to export.")
            temporary.replace(upstream)
        finally:
            temporary.unlink(missing_ok=True)
    path = model_path(directory)
    export(read_weights(upstream.read_bytes()), path)
    manifest = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "sha256": sha256(path),
        "source": SOURCE,
        "source_sha256": SOURCE_SHA256,
        "license": "Apache-2.0",
        "input_shape": INPUT_SHAPE,
        "output_shape": OUTPUT_SHAPE,
        "precision": "float32 (QNN HTP uses float16 math)",
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
