"""Independent upstream TF graph interpreter for the small downloaded SR graphs.

Test dimensions are reduced to keep NumPy reference convolutions inexpensive.
This validates export semantics, not merely CPU/NPU agreement on the same export.
"""

from pathlib import Path

import numpy as np
import onnx
import pytest

from npu_sr.acquire import _fields
from npu_sr.acquire_models import read_constants
from npu_sr.model import MODELS, model_directory, model_path
from npu_sr.utils import load_ort


def tf_reference(data: bytes, input_nchw: np.ndarray) -> np.ndarray:
    values = read_constants(data)
    for raw in _fields(data)[1]:
        fields = _fields(raw)
        name, op = fields[1][0].decode(), fields[2][0].decode()
        inputs = [v.decode() for v in fields.get(3, [])]
        if op == "Const":
            continue
        if op == "Placeholder":
            values[name] = input_nchw.transpose(0, 2, 3, 1)
            continue
        if op == "Transpose":
            # The sole final transpose in these pinned graphs is NHWC -> NCHW.
            values[name] = values[inputs[0]].transpose(0, 3, 1, 2)
            continue
        args = [values[v] for v in inputs]
        if op == "Conv2D":
            x, kernel = args
            height, width = x.shape[1:3]
            kh, kw, _, cout = kernel.shape
            padded = np.pad(x, ((0, 0), ((kh - 1) // 2, kh // 2), ((kw - 1) // 2, kw // 2), (0, 0)))
            out = np.zeros((1, height, width, cout), np.float32)
            for row in range(kh):
                for col in range(kw):
                    out += padded[:, row : row + height, col : col + width] @ kernel[row, col]
        elif op in {"Add", "BiasAdd"}:
            out = args[0] + args[1]
        elif op == "Sub":
            out = args[0] - args[1]
        elif op == "Mul":
            out = args[0] * args[1]
        elif op == "Relu":
            out = np.maximum(args[0], 0)
        elif op == "Abs":
            out = np.abs(args[0])
        elif op == "DepthToSpace":
            x = args[0]
            n, height, width, channels = x.shape
            out = (
                x.reshape(n, height, width, 2, 2, channels // 4)
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(n, height * 2, width * 2, channels // 4)
            )
        else:
            raise AssertionError(f"Unhandled upstream TF operator {op}")
        values[name] = out
    return values["NCHW_output"]


@pytest.mark.parametrize(
    "identifier,filename",
    [
        ("fsrcnn-small-x2", "FSRCNN-small_x2.pb"),
        ("fsrcnn-x2", "FSRCNN_x2.pb"),
        ("lapsrn-x2", "LapSRN_x2.pb"),
    ],
)
def test_export_matches_original_tf_graph(identifier: str, filename: str, tmp_path: Path):
    upstream = model_directory() / filename
    path = model_path(identifier=identifier)
    if not upstream.exists() or not path.exists():
        pytest.skip("Optional downloaded candidate unavailable")
    model = onnx.load(path)
    for value, scale in [(model.graph.input[0], 1), (model.graph.output[0], 2)]:
        value.type.tensor_type.shape.dim[2].dim_value = 9 * scale
        value.type.tensor_type.shape.dim[3].dim_value = 7 * scale
    reduced = tmp_path / "reduced.onnx"
    onnx.save(model, reduced)
    tensor = np.random.default_rng(81).random((1, 1, 9, 7), dtype=np.float32)
    expected = tf_reference(upstream.read_bytes(), tensor)
    actual = (
        load_ort()
        .InferenceSession(str(reduced), providers=["CPUExecutionProvider"])
        .run(None, {"input": tensor})[0]
    )
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("identifier", list(MODELS))
def test_downloaded_cpu_models_are_finite_and_shape_correct(identifier):
    from npu_sr.runtime import Runtime

    path = model_path(identifier=identifier)
    if not path.exists():
        pytest.skip("Optional downloaded model unavailable")
    runtime = Runtime(path, "cpu")
    result = runtime.run(np.full(runtime.input_shape, 0.5, np.float32))
    assert result.shape == tuple(runtime.output_shape)
    assert np.isfinite(result).all()
