"""Pinned QuickSRNet checkpoints and a documented neutral-RGB luminance export.

Architecture/checkpoint provenance: Qualcomm AIMET Model Zoo, BSD-3-Clause.
This implementation constructs ONNX directly; PyTorch is not a runtime dependency.
The exported function is the source RGB network evaluated on repeated Y input,
then its clipped RGB result converted to Rec.601 Y. It is not the RGB model.
"""

import io
import pickle
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .model import ModelSpec
from .weights import FloatStorage, Storage, Tensor, _tensor


class _OptimizerMetadata:
    """Inert checkpoint metadata marker; no optimizer code is loaded or executed."""


class _Parameter:
    """Identity-hashable inert parameter used by optimizer metadata dictionaries."""

    def __init__(self, tensor: Tensor):
        self.tensor = tensor


def _parameter(tensor: Tensor, *_metadata) -> _Parameter:
    return _Parameter(tensor)


class _CheckpointReader(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("collections", "defaultdict"): defaultdict,
            ("__builtin__", "dict"): dict,
            ("torch", "FloatStorage"): FloatStorage,
            ("torch._utils", "_rebuild_tensor_v2"): _tensor,
            ("torch._utils", "_rebuild_parameter"): _parameter,
            ("torch.optim.adam", "Adam"): _OptimizerMetadata,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Disallowed checkpoint global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, value):
        if not isinstance(value, tuple) or len(value) != 5:
            raise ValueError("Unsupported QuickSRNet storage record")
        kind, dtype, key, _location, size = value
        if (
            kind != "storage"
            or dtype is not FloatStorage
            or not isinstance(size, int)
            or not 0 < size <= 1_000_000
        ):
            raise ValueError("Unsupported QuickSRNet tensor storage")
        return Storage(key, size)


def expected_shapes(size: str) -> dict[str, tuple[int, ...]]:
    if size not in {"small", "medium"}:
        raise ValueError("Unsupported QuickSRNet size")
    layers = 3 if size == "small" else 6
    result = {}
    for index in range(layers):
        key = f"cnn.{index * 2}"
        result[key + ".weight"] = (32, 3 if index == 0 else 32, 3, 3)
        result[key + ".bias"] = (32,)
    return result | {"conv_last.weight": (12, 32, 3, 3), "conv_last.bias": (12,)}


def read_quicksr(data: bytes, size: str) -> dict[str, np.ndarray]:
    """Read only the known float checkpoint layout, with inert metadata globals.

    Acquisition must verify the pinned SHA256 first. ZIP members are read in
    memory, not extracted. Other checkpoint formats/globals fail closed.
    """
    if len(data) > 2_000_000:
        raise ValueError("QuickSRNet checkpoint exceeds size limit")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(info.file_size for info in archive.infolist()) > 4_000_000:
            raise ValueError("QuickSRNet decompressed checkpoint exceeds size limit")
        checkpoint = _CheckpointReader(io.BytesIO(archive.read("archive/data.pkl"))).load()
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
            raise ValueError("QuickSRNet checkpoint requires a state dictionary")
        tensors = checkpoint["state_dict"]
        shapes = expected_shapes(size)
        if set(tensors) != set(shapes):
            raise ValueError("QuickSRNet checkpoint keys do not match the architecture")
        result = {}
        for name, shape in shapes.items():
            tensor = tensors[name]
            if isinstance(tensor, _Parameter):
                tensor = tensor.tensor
            stride = tuple(int(np.prod(shape[i + 1 :])) for i in range(len(shape)))
            if (
                not isinstance(tensor, Tensor)
                or tensor.shape != shape
                or tensor.stride != stride
                or tensor.offset != 0
                or tensor.storage.size != int(np.prod(shape))
            ):
                raise ValueError("QuickSRNet tensor shape/storage layout mismatch")
            key = tensor.storage.key
            if not isinstance(key, str) or not key.isdecimal():
                raise ValueError("Invalid QuickSRNet storage key")
            values = np.frombuffer(archive.read(f"archive/data/{key}"), "<f4")
            if values.size != tensor.storage.size or not np.isfinite(values).all():
                raise ValueError("Invalid QuickSRNet float weights")
            result[name] = values.reshape(shape).copy()
        return result


def export_quicksr(weights: dict[str, np.ndarray], spec: ModelSpec, path: Path) -> None:
    """Fold repeated-Y input, preserve all Clip operations and subpixel ordering."""
    size = spec.architecture.removeprefix("QuickSRNet-")
    shapes = expected_shapes(size)
    if set(weights) != set(shapes) or any(weights[k].shape != v for k, v in shapes.items()):
        raise ValueError("QuickSRNet export weights do not match the architecture")
    initializers = [
        numpy_helper.from_array(np.array(0, np.float32), "minimum"),
        numpy_helper.from_array(np.array(1, np.float32), "maximum"),
    ]
    nodes = []
    previous = "input"
    keys = [key.removesuffix(".weight") for key in shapes if key.endswith(".weight")]
    for index, key in enumerate(keys):
        kernel = weights[key + ".weight"]
        if index == 0:
            kernel = kernel.sum(axis=1, keepdims=True)
        for name, value in ((key + ".weight", kernel), (key + ".bias", weights[key + ".bias"])):
            initializers.append(numpy_helper.from_array(np.asarray(value, np.float32), name))
        convolution, clipped = f"conv_{index}", f"clip_{index}"
        nodes.append(
            helper.make_node(
                "Conv", [previous, key + ".weight", key + ".bias"], [convolution], pads=[1, 1, 1, 1]
            )
        )
        nodes.append(helper.make_node("Clip", [convolution, "minimum", "maximum"], [clipped]))
        previous = clipped
    # Source PixelShuffle is CRD for RGB. Mix matching phases after the RGB Clip,
    # then single-channel DCR/CRD are equivalent. Mixing before Clip is incorrect.
    mix = np.zeros((4, 12, 1, 1), np.float32)
    for phase in range(4):
        mix[phase, phase::4, 0, 0] = [0.299, 0.587, 0.114]
    initializers.append(numpy_helper.from_array(mix, "rgb_to_y"))
    nodes.append(
        helper.make_node("Conv", [previous, "rgb_to_y"], ["subpixels"], kernel_shape=[1, 1])
    )
    nodes.append(
        helper.make_node("DepthToSpace", ["subpixels"], ["output"], blocksize=2, mode="DCR")
    )
    graph = helper.make_graph(
        nodes,
        spec.identifier,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, spec.input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, spec.output_shape)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    helper.set_model_props(
        model,
        {
            "license": spec.license,
            "version": spec.version,
            "adaptation": "source RGB network on repeated Y; clipped RGB output to Rec.601 Y",
        },
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)
