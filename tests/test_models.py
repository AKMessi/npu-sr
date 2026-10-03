import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from npu_sr.acquire_models import SOURCES, acquire_model
from npu_sr.cli import main
from npu_sr.errors import NPUUnavailable, SRException
from npu_sr.evaluate import quality_metrics, ssim_arrays
from npu_sr.image import infer_y, postprocess, preprocess
from npu_sr.model import MODELS, model_path, model_spec, resolve_model, validate_model
from npu_sr.runtime import Runtime
from npu_sr.weights import read_dncnn


def test_registry_and_legacy_paths():
    assert resolve_model(None).name == "espcn-x2.onnx"
    assert resolve_model(None, "denoise").name == "dncnn-25.onnx"
    assert resolve_model("custom.onnx") == Path("custom.onnx")
    for identifier, spec in MODELS.items():
        assert spec.input_shape[2] == spec.core + 2 * spec.halo
        assert spec.output_shape[2] == spec.input_shape[2] * spec.scale
        assert spec.architecture in SOURCES
        assert model_spec(identifier) is spec
    with pytest.raises(SRException, match="Unknown model"):
        model_spec("nonexistent")


@pytest.mark.parametrize("identifier", list(MODELS))
@pytest.mark.parametrize("size", [(1, 1), (271, 513)])
def test_shape_aware_tiling_is_exact(identifier, size):
    spec = model_spec(identifier)
    rng = np.random.default_rng(41)
    pixels = rng.integers(0, 256, (size[1], size[0], 3), np.uint8)
    prepared = preprocess(Image.fromarray(pixels), spec)

    class Identity:
        def run(self, tensor):
            return tensor.repeat(spec.scale, 2).repeat(spec.scale, 3)

    timings = {}
    y, _ = infer_y(prepared, Identity(), timings)
    expected = (pixels / 255 @ np.array([0.299, 0.587, 0.114])).astype(np.float32)
    expected = expected.repeat(spec.scale, 0).repeat(spec.scale, 1)
    np.testing.assert_allclose(y, expected, atol=1e-6)
    assert postprocess(y, prepared).size == tuple(v * spec.scale for v in size)
    assert all(v >= 0 for v in timings.values())


def test_tile_input_buffer_is_reused():
    prepared = preprocess(Image.new("RGB", (300, 130)))
    iterator = prepared.tiles()
    first, second = next(iterator)[2], next(iterator)[2]
    assert first is second
    assert second.flags.c_contiguous


def test_manifest_registry_shape_tamper(tiny_model):
    path = tiny_model.with_suffix(".json")
    manifest = json.loads(path.read_text())
    manifest["input_shape"] = [1, 1, 100, 100]
    path.write_text(json.dumps(manifest))
    with pytest.raises(SRException, match="input_shape"):
        validate_model(tiny_model)


def test_gpu_never_falls_back(tiny_model, monkeypatch):
    def unavailable(*args):
        raise NPUUnavailable("GPU execution failed")

    monkeypatch.setattr(Runtime, "_accelerated", unavailable)
    with pytest.raises(NPUUnavailable, match="GPU execution failed"):
        Runtime(tiny_model, "gpu")


def test_ssim_identity_constant_formula_symmetry():
    white, gray = np.full((25, 24), 255.0), np.full((25, 24), 128.0)
    assert ssim_arrays(white, white) == pytest.approx(1)
    expected = (2 * 255 * 128 + 2.55**2) / (255**2 + 128**2 + 2.55**2)
    assert ssim_arrays(white, gray) == pytest.approx(expected)
    assert ssim_arrays(white, gray) == pytest.approx(ssim_arrays(gray, white))
    with pytest.raises(SRException, match="11x11"):
        ssim_arrays(np.zeros((5, 5)), np.zeros((5, 5)))
    with pytest.raises(SRException, match="finite"):
        ssim_arrays(white * np.nan, white)
    metrics = quality_metrics(Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32)))
    assert metrics["psnr_y_db"] is None and metrics["ssim_y"] == pytest.approx(1)


def test_weight_reader_rejects_arbitrary_pickle_globals():
    with pytest.raises(ValueError, match="Disallowed"):
        read_dncnn(pickle.dumps(Path("dangerous")))


def test_model_cli(capsys):
    assert main(["models", "list"]) == 0
    assert "dncnn-25" in capsys.readouterr().out
    assert main(["models", "info", "fsrcnn-x2"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["source_sha256"] == SOURCES["FSRCNN"][1]
    assert info["input_shape"] == [1, 1, 268, 268]
    assert main(["models", "info", "missing"]) == 1


def test_bad_download_hash(tmp_path, monkeypatch):
    import io

    from npu_sr import acquire_models

    monkeypatch.setattr(
        acquire_models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad")
    )
    with pytest.raises(SRException, match="SHA256"):
        acquire_model("dncnn-25", tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.npu
@pytest.mark.parametrize("identifier", list(MODELS))
def test_all_models_strict_qnn_and_numerical_agreement(identifier):
    path = model_path(identifier=identifier)
    cpu, npu = Runtime(path, "cpu"), Runtime(path, "npu")
    tensor = np.random.default_rng(42).uniform(0.2, 0.8, cpu.input_shape).astype(np.float32)
    expected = cpu.run(tensor)
    for _ in range(3):
        actual = npu.run(tensor)
        # FP16 HTP accumulates differently; bound both MAE and extreme deviation.
        assert np.mean(np.abs(expected - actual)) < 0.003
        assert np.max(np.abs(expected - actual)) < 0.035
    assert npu.evidence["executed_kernel_counts"] == {"QNNExecutionProvider": 1}


@pytest.mark.npu
def test_context_reuse_is_strict_and_equivalent(tmp_path):
    path = model_path(identifier="espcn-x2-256")
    cold = Runtime(path, "npu", cache_dir=tmp_path)
    assert cold.evidence["context_cache"] == "miss"
    x = np.full(cold.input_shape, 0.4, np.float32)
    expected = cold.run(x)
    del cold
    warm = Runtime(path, "npu", cache_dir=tmp_path)
    assert warm.evidence["context_cache"] == "hit"
    np.testing.assert_allclose(warm.run(x), expected, atol=0.001)
    assert warm.evidence["executed_kernel_counts"] == {"QNNExecutionProvider": 1}


@pytest.mark.npu
def test_denoising_improves_seeded_gaussian_noise():
    from npu_sr.evaluate import psnr

    clean = np.tile(np.linspace(64, 192, 96), (64, 1))
    noise = np.random.default_rng(2026).normal(0, 25, clean.shape)
    reference = Image.fromarray(np.rint(clean).astype(np.uint8)).convert("RGB")
    noisy = Image.fromarray(np.clip(np.rint(clean + noise), 0, 255).astype(np.uint8)).convert("RGB")
    runtime = Runtime(model_path(identifier="dncnn-25"), "npu")
    prepared = preprocess(noisy, runtime.spec)
    result = postprocess(infer_y(prepared, runtime)[0], prepared)
    assert psnr(result, reference) > psnr(noisy, reference) + 5


@pytest.mark.npu
@pytest.mark.parametrize("identifier", list(MODELS))
def test_snapdragon_directml_execution(identifier):
    """Manual Snapdragon hardware group; generic CI has no proven GPU path."""
    runtime = Runtime(model_path(identifier=identifier), "gpu")
    assert runtime.evidence["executed_kernel_counts"] == {"DmlExecutionProvider": 1}
    x = np.full(runtime.input_shape, 0.5, np.float32)
    assert np.isfinite(runtime.run(x)).all()
