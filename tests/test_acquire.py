import json
from pathlib import Path

import numpy as np
import onnx
import pytest

from npu_sr.acquire import SOURCE_SHA256, export, read_weights
from npu_sr.model import INPUT_SHAPE, model_path, sha256
from npu_sr.utils import load_ort


def test_export_matches_independent_convolution_reference(tmp_path: Path):
    """Verify layout, padding, nonlinearities, and DCR ordering without upstream weights."""
    rng = np.random.default_rng(12)
    weights = {}
    for i, shape in [(1, (5, 5, 1, 64)), (2, (3, 3, 64, 32)), (3, (3, 3, 32, 4))]:
        weights[f"f{i}"] = (rng.standard_normal(shape) * 0.02).astype(np.float32)
        weights[f"b{i}"] = (rng.standard_normal(shape[-1]) * 0.02).astype(np.float32)
    path = tmp_path / "test.onnx"
    export(weights, path)
    # Shrink only the input/output contract for a fast independent NumPy reference.
    model = onnx.load(path)
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 9
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 7
    model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 18
    model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = 14
    onnx.save(model, path)
    tensor = rng.random((1, 1, 9, 7), dtype=np.float32)
    expected = tensor[0].transpose(1, 2, 0)
    for i in range(1, 4):
        kernel = weights[f"f{i}"]
        pad = kernel.shape[0] // 2
        padded = np.pad(expected, ((pad, pad), (pad, pad), (0, 0)))
        result = np.broadcast_to(weights[f"b{i}"], (9, 7, kernel.shape[-1])).copy()
        for row in range(kernel.shape[0]):
            for col in range(kernel.shape[1]):
                result += padded[row : row + 9, col : col + 7] @ kernel[row, col]
        expected = np.maximum(result, 0) if i < 3 else result
    expected = np.tanh(expected.reshape(9, 7, 2, 2).transpose(0, 2, 1, 3).reshape(18, 14))
    ort = load_ort()
    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"input": tensor}
    )[0][0, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_upstream_weights_if_available():
    upstream = model_path().parent / "ESPCN_x2.pb"
    if not upstream.exists():
        pytest.skip("upstream weights unavailable")
    assert sha256(upstream) == SOURCE_SHA256
    weights = read_weights(upstream.read_bytes())
    assert weights["f1"].shape == (5, 5, 1, 64)
    assert all(np.isfinite(weight).all() for weight in weights.values())


def test_export_is_deterministic(tmp_path):
    weights = {
        "f1": np.zeros((5, 5, 1, 64), np.float32),
        "f2": np.zeros((3, 3, 64, 32), np.float32),
        "f3": np.zeros((3, 3, 32, 4), np.float32),
        "b1": np.zeros(64, np.float32),
        "b2": np.zeros(32, np.float32),
        "b3": np.zeros(4, np.float32),
    }
    first, second = tmp_path / "a.onnx", tmp_path / "b.onnx"
    export(weights, first)
    export(weights, second)
    assert sha256(first) == sha256(second)
    assert [
        dim.dim_value for dim in onnx.load(first).graph.input[0].type.tensor_type.shape.dim
    ] == INPUT_SHAPE


def test_download_hash_mismatch_does_not_install(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from npu_sr import acquire
    from npu_sr.errors import SRException

    @contextmanager
    def bad_response(*_args, **_kwargs):
        class Response:
            def read(self, _size):
                return b"bad upstream artifact"

        yield Response()

    monkeypatch.setattr(acquire.urllib.request, "urlopen", bad_response)
    with pytest.raises(SRException, match="SHA256 mismatch"):
        acquire.acquire(tmp_path)
    assert not (tmp_path / "ESPCN_x2.pb").exists()
    assert not list(tmp_path.glob("*.download"))


def test_manifest_persisted_for_downloaded_model():
    path = model_path()
    if not path.exists():
        pytest.skip("downloaded model unavailable")
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["sha256"] == sha256(path)
    assert manifest["source_sha256"] == SOURCE_SHA256
