import io
import pickle
import zipfile

import numpy as np
import onnxruntime as ort
import pytest

from npu_sr.model import ModelSpec
from npu_sr.quicksr import _CheckpointReader, expected_shapes, export_quicksr, read_quicksr


def _archive(payload):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
    return stream.getvalue()


def test_checkpoint_global_allowlist():
    with pytest.raises(ValueError, match="Disallowed"):
        read_quicksr(_archive(pickle.dumps(eval)), "small")


def test_checkpoint_schema_and_size_limits():
    with pytest.raises(ValueError, match="size limit"):
        read_quicksr(b"x" * 2_000_001, "small")
    with pytest.raises(ValueError, match="keys"):
        read_quicksr(_archive(pickle.dumps({"state_dict": {}})), "small")
    with pytest.raises(ValueError, match="size"):
        expected_shapes("large")


def test_storage_rejects_unbounded_and_wrong_types():
    reader = _CheckpointReader(io.BytesIO())
    with pytest.raises(ValueError, match="storage"):
        reader.persistent_load(("storage", int, "0", "cpu", 100))


def _conv(value, weight, bias):
    padded = np.pad(value, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
    return np.einsum("nchwkl,ockl->nohw", windows, weight) + bias[None, :, None, None]


@pytest.mark.parametrize("size", ["small", "medium"])
def test_luminance_export_matches_rgb_clip_then_pixelshuffle(size, tmp_path):
    """Independent convolution/reference catches channel folding and CRD phase errors."""
    rng = np.random.default_rng(123)
    weights = {
        name: rng.normal(0, 0.05 if name.endswith("weight") else 0.4, shape).astype(np.float32)
        for name, shape in expected_shapes(size).items()
    }
    spec = ModelSpec("test", f"QuickSRNet-{size}", core=8)
    path = tmp_path / "model.onnx"
    export_quicksr(weights, spec, path)
    y = rng.random(spec.input_shape, dtype=np.float32)
    rgb = np.repeat(y, 3, axis=1)
    for key in (key.removesuffix(".weight") for key in weights if key.endswith(".weight")):
        rgb = np.clip(_conv(rgb, weights[key + ".weight"], weights[key + ".bias"]), 0, 1)
    n, _, h, w = rgb.shape
    rgb = rgb.reshape(n, 3, 2, 2, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(n, 3, h * 2, w * 2)
    expected = (rgb * np.array([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdims=True)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": y})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_export_rejects_missing_weights(tmp_path):
    with pytest.raises(ValueError, match="architecture"):
        export_quicksr({}, ModelSpec("test", "QuickSRNet-small"), tmp_path / "out.onnx")


@pytest.mark.parametrize("size", ["small", "medium"])
def test_acquired_checkpoint_cpu_integration_when_available(size):
    from npu_sr.model import model_path
    from npu_sr.runtime import Runtime

    path = model_path(identifier=f"quicksrnet-{size}-y-x2")
    if not path.is_file():
        pytest.skip("Pinned checkpoint not acquired locally")
    runtime = Runtime(path, "cpu")
    tensor = np.random.default_rng(31).uniform(0.1, 0.9, runtime.input_shape).astype(np.float32)
    actual = runtime.run(tensor)
    assert list(actual.shape) == runtime.spec.output_shape
    assert np.isfinite(actual).all() and 0 <= actual.min() <= actual.max() <= 1.000001
