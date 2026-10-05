"""Static fusion of a pinned spatial graph and an original small temporal residual."""

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .errors import SRException
from .model import ModelSpec, sha256

SPATIAL_SHA256 = "4e54ddc579e3ec2fbcb66a14b621b88eb3d65b7d84d39672a5450a5fd78154bf"
SHAPES = {
    "conv0.weight": (8, 6, 3, 3),
    "conv0.bias": (8,),
    "conv1.weight": (8, 8, 3, 3),
    "conv1.bias": (8,),
    "conv2.weight": (4, 8, 3, 3),
    "conv2.bias": (4,),
}


def validate_weights(values: dict[str, np.ndarray]) -> None:
    if set(values) != set(SHAPES):
        raise SRException("Invalid temporal feature weight contract.")
    width = values["conv0.bias"].size
    if width not in {8, 16}:
        raise SRException("Temporal feature width must be 8 or 16.")
    inputs = values["conv0.weight"].shape[1] if values["conv0.weight"].ndim == 4 else 0
    if inputs not in {6, 10}:
        raise SRException("Temporal feature inputs must be 6 or 10.")
    shapes = {
        "conv0.weight": (width, inputs, 3, 3),
        "conv0.bias": (width,),
        "conv1.weight": (width, width, 3, 3),
        "conv1.bias": (width,),
        "conv2.weight": (4, width, 3, 3),
        "conv2.bias": (4,),
    }
    if any(
        value.shape != shapes[key] or value.dtype != np.float32 or not np.isfinite(value).all()
        for key, value in values.items()
    ):
        raise SRException("Invalid temporal feature weight contract.")
    if not np.any(values["conv2.weight"]):
        raise SRException("Temporal checkpoint has an untrained zero-output convolution.")


def residual_nodes() -> list:
    """Known six-input graph; no arbitrary checkpoint graph execution."""
    nodes, previous = [], "temporal_input"
    for index in range(3):
        name, target = f"temporal_conv{index}", f"temporal_feature{index}"
        nodes.append(
            helper.make_node(
                "Conv", [previous, name + ".weight", name + ".bias"], [target], pads=[1] * 4
            )
        )
        previous = target
        if index < 2:
            previous = f"temporal_relu{index}"
            nodes.append(helper.make_node("Relu", [target], [previous]))
    return nodes + [
        helper.make_node(
            "Clip", [previous, "temporal_minimum", "temporal_maximum"], ["temporal_bounded"]
        ),
        helper.make_node(
            "DepthToSpace", ["temporal_bounded"], ["temporal_output"], blocksize=2, mode="DCR"
        ),
    ]


def fuse_temporal(
    spatial: Path, values: dict[str, np.ndarray], spec: ModelSpec, output: Path
) -> None:
    """Preserve the useful tile core with full radius-7 context and one QNN graph."""
    validate_weights(values)
    if (
        sha256(spatial) != SPATIAL_SHA256
        or spec.input_channels not in {2, 6}
        or values["conv0.weight"].shape[1] != spec.input_channels + 4
        or spec.halo != 7
        or (spec.core, spec.height) not in {(256, 256), (320, 270)}
        or spec.scale != 2
    ):
        raise SRException("Temporal fusion requires the pinned spatial graph and halo-7 contract.")
    base = onnx.load(spatial)
    initializers = []
    for value in base.graph.initializer:
        if value.name == "cnn.0.weight":
            array = numpy_helper.to_array(value)
            expanded = np.zeros((array.shape[0], spec.input_channels, *array.shape[2:]), np.float32)
            expanded[:, 1:2] = array
            initializers.append(numpy_helper.from_array(expanded, value.name))
        else:
            initializers.append(copy.deepcopy(value))
    nodes = []
    for node in base.graph.node:
        node = copy.deepcopy(node)
        if node.op_type == "DepthToSpace":
            node.output[0] = "spatial_output"
        nodes.append(node)
    nodes.append(helper.make_node("Concat", ["input", "subpixels"], ["temporal_input"], axis=1))
    initializers.extend(
        numpy_helper.from_array(value, "temporal_" + key) for key, value in values.items()
    )
    initializers.extend(
        [
            numpy_helper.from_array(np.array(-0.05, np.float32), "temporal_minimum"),
            numpy_helper.from_array(np.array(0.05, np.float32), "temporal_maximum"),
        ]
    )
    nodes.extend(residual_nodes())
    nodes.append(helper.make_node("Add", ["spatial_output", "temporal_output"], ["output"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            spec.identifier,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, spec.input_shape)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, spec.output_shape)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    helper.set_model_props(model, {"version": spec.version, "license": spec.license})
    onnx.checker.check_model(model)
    onnx.save(model, output)
