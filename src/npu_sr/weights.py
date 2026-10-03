"""Read the pinned KAIR legacy float state dictionary without PyTorch or code execution.

This is deliberately not a general pickle/PyTorch loader. Only OrderedDict,
FloatStorage and tensor rebuild records are allowed. Globals cannot execute code.
The acquisition layer checks the exact upstream SHA256 before calling this reader.
"""

import io
import pickle
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


class FloatStorage:
    """Marker for an allowed storage type, never instantiated by upstream code."""


@dataclass
class Storage:
    key: str
    size: int


@dataclass
class Tensor:
    storage: Storage
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _tensor(storage, offset, shape, stride, requires_grad, hooks) -> Tensor:
    return Tensor(storage, offset, shape, stride)


class _Reader(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch", "FloatStorage"): FloatStorage,
            ("torch._utils", "_rebuild_tensor_v2"): _tensor,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Disallowed weight pickle global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, value):
        kind, dtype, key, location, size, view = value
        if kind != "storage" or dtype is not FloatStorage or view is not None:
            raise ValueError("Unsupported weight storage")
        if not isinstance(size, int) or not 0 < size <= 1_000_000:
            raise ValueError("Invalid weight storage size")
        return Storage(key, size)


def read_dncnn(data: bytes) -> dict[str, np.ndarray]:
    """Decode the known legacy serialization; reject unknown globals/layouts."""
    if len(data) > 3_000_000:
        raise ValueError("Weight file exceeds size limit")
    stream = io.BytesIO(data)
    magic, protocol, system = (_Reader(stream).load() for _ in range(3))
    if magic != 0x1950A86A20F9469CFC6C or protocol != 1001 or not system["little_endian"]:
        raise ValueError("Unexpected PyTorch serialization header")
    tensors = _Reader(stream).load()
    keys = _Reader(stream).load()
    if not isinstance(tensors, OrderedDict) or len(tensors) != 34:
        raise ValueError("Unexpected DnCNN state dictionary")
    storages = {tensor.storage.key: tensor.storage for tensor in tensors.values()}
    if set(keys) != set(storages):
        raise ValueError("Weight storage keys do not match tensors")
    arrays = {}
    for key in keys:
        count = int.from_bytes(stream.read(8), "little")
        if count != storages[key].size:
            raise ValueError("Weight storage length mismatch")
        arrays[key] = np.frombuffer(stream.read(count * 4), dtype="<f4").copy()
        if arrays[key].size != count:
            raise ValueError("Truncated weight storage")
    result = {}
    for name, tensor in tensors.items():
        expected_stride = tuple(
            int(np.prod(tensor.shape[index + 1 :])) for index in range(len(tensor.shape))
        )
        if tensor.offset != 0 or tuple(tensor.stride) != expected_stride:
            raise ValueError("Only contiguous, complete tensors are supported")
        result[name] = arrays[tensor.storage.key].reshape(tensor.shape)
    if stream.read(1):
        raise ValueError("Unexpected trailing weight data")
    return result
