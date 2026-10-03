import json

from PIL import Image

from npu_sr.runtime import Runtime
from npu_sr.suite import performance_suite, quality_suite


def test_performance_suite_phases_and_checkpoint(tiny_model, tmp_path, monkeypatch):
    from npu_sr import suite

    monkeypatch.setattr(suite, "model_path", lambda **kwargs: tiny_model)
    monkeypatch.setattr(suite, "Runtime", lambda *a, **k: Runtime(tiny_model, "cpu"))
    monkeypatch.setattr(suite, "environment", lambda: {"test": True})
    source = tmp_path / "input.png"
    Image.new("RGB", (20, 16), "orange").save(source)
    checkpoint = tmp_path / "partial.json"
    report = performance_suite(
        source, ["espcn-x2"], ["cpu"], 2, 1, 2, resolutions=[(20, 16)], checkpoint=checkpoint
    )
    assert report["complete"]
    assert not json.loads(checkpoint.read_text())["complete"]
    result = report["results"][0]
    assert result["tile_count"] == 1 and result["output_resolution"] == [40, 32]
    assert len(result["trials"]) == 2
    assert result["trials"][0]["phase_statistics"]["total_ms"]["median_ms"] > 0
    json.dumps(report, allow_nan=False)


def test_quality_suite_uses_hr_ground_truth(tiny_model, tmp_path, monkeypatch):
    from npu_sr import suite

    monkeypatch.setattr(suite, "model_path", lambda **kwargs: tiny_model)
    monkeypatch.setattr(suite, "Runtime", lambda *a, **k: Runtime(tiny_model, "cpu"))
    monkeypatch.setattr(suite, "environment", lambda: {"test": True})
    Image.new("RGB", (33, 35), "orange").save(tmp_path / "reference.png")
    report = quality_suite(tmp_path, ["espcn-x2"], ["cpu"])
    result = report["results"][0]
    assert result["degradation"]["type"] == "Pillow bicubic x2 downsample"
    assert result["metrics"]["ssim_y"] > 0.99
    assert result["baseline_metrics"]["ssim_y"] > 0.99


def test_quality_rejects_hidden_alpha(tiny_model, tmp_path, monkeypatch):
    import pytest

    from npu_sr import suite
    from npu_sr.errors import SRException

    monkeypatch.setattr(suite, "model_path", lambda **kwargs: tiny_model)
    monkeypatch.setattr(suite, "Runtime", lambda *a, **k: Runtime(tiny_model, "cpu"))
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(tmp_path / "reference.png")
    with pytest.raises(SRException, match="opaque"):
        quality_suite(tmp_path, ["espcn-x2"], ["cpu"])
