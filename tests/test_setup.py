"""First-run failures, integrity repair and executed capability evidence."""

import io
from types import SimpleNamespace

import pytest

from npu_sr import capabilities, setup
from npu_sr.errors import SRException


def test_setup_stops_before_download_on_unsupported_hardware(monkeypatch):
    monkeypatch.setattr(setup, "platform_problem", lambda: "ARM64 Python required")
    monkeypatch.setattr(setup, "acquire_model", lambda *a: pytest.fail("unexpected download"))
    with pytest.raises(SRException, match="ARM64"):
        setup.prepare(["quicksrnet-small-y-x2"])


def test_setup_repairs_model_then_requires_npu_and_video_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "platform_problem", lambda: None)
    monkeypatch.setattr(setup, "load_ort", lambda: None)
    monkeypatch.setattr(setup, "model_directory", lambda: tmp_path)
    monkeypatch.setattr(setup, "model_path", lambda **kw: tmp_path / "model.onnx")
    calls = []

    def validate(path):
        if not calls:
            raise SRException("hash mismatch")
        return {"sha256": "verified fixture"}

    monkeypatch.setattr(setup, "validate_model", validate)
    monkeypatch.setattr(setup, "acquire_model", lambda *a: calls.append(a))
    monkeypatch.setattr(
        setup, "diagnose", lambda *a, **k: {"ready": True, "video": {"ready": False}}
    )
    result = setup.prepare(["quicksrnet-small-y-x2"])
    assert len(calls) == 1 and not result["setup_ready"]
    assert result["acquired_models"]["quicksrnet-small-y-x2"] == "downloaded and hash-verified"
    monkeypatch.setattr(
        setup, "diagnose", lambda *a, **k: {"ready": True, "video": {"ready": True}}
    )
    assert setup.prepare(["quicksrnet-small-y-x2"])["setup_ready"]


def test_setup_cpu_runs_real_inference_without_npu_request(tiny_model, monkeypatch):
    monkeypatch.setattr(setup, "model_path", lambda **kw: tiny_model)
    requests = []

    def diagnose(*args, **kwargs):
        requests.append(kwargs["npu"])
        return {"ready": False, "video": {"ready": False}}

    monkeypatch.setattr(setup, "diagnose", diagnose)
    result = setup.prepare(["quicksrnet-small-y-x2"], "cpu")
    assert requests == [False] and result["setup_ready"]
    assert result["cpu_proof"]["providers"] == ["CPUExecutionProvider"]


def test_setup_missing_runtime_does_not_download(monkeypatch):
    monkeypatch.setattr(setup, "platform_problem", lambda: None)
    monkeypatch.setattr(
        setup, "load_ort", lambda: (_ for _ in ()).throw(SRException("App Runtime"))
    )
    monkeypatch.setattr(setup, "acquire_model", lambda *a: pytest.fail("unexpected download"))
    with pytest.raises(SRException, match="App Runtime"):
        setup.prepare(["quicksrnet-small-y-x2"])


def test_setup_conflicting_tool_options_fail_before_download(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "acquire_model", lambda *a: pytest.fail("unexpected download"))
    with pytest.raises(SRException, match="not both"):
        setup.prepare(["quicksrnet-small-y-x2"], download_ffmpeg=True, ffmpeg=tmp_path / "ffmpeg")


def test_codec_lists_do_not_establish_hardware_capability(monkeypatch, tmp_path):
    monkeypatch.setattr(capabilities, "tool_path", lambda p: tmp_path / "ffmpeg")
    monkeypatch.setattr(
        capabilities, "run_tool", lambda *a: SimpleNamespace(stdout=b"ffmpeg fixture")
    )

    def fail(*args):
        raise SRException("no executed hardware evidence")

    monkeypatch.setattr(capabilities, "hardware_decode_probe", fail)
    monkeypatch.setattr(capabilities, "hardware_encode_probe", fail)
    report = capabilities.video_capabilities()
    assert not report["ready"]
    assert all(row["status"] == "not proven" for row in report["encode"].values())
    assert report["decode"]["h264"]["status"] == "not proven"


def test_executed_video_probes_are_kept_per_codec(monkeypatch, tmp_path):
    monkeypatch.setattr(capabilities, "tool_path", lambda p: tmp_path / "ffmpeg")
    monkeypatch.setattr(
        capabilities, "run_tool", lambda *a: SimpleNamespace(stdout=b"ffmpeg fixture")
    )
    monkeypatch.setattr(capabilities, "hardware_decode_probe", lambda *a: {"method": "D3D11VA"})
    calls = []

    def encode(tool, info, scale, codec, bitrate):
        calls.append(codec)
        if codec == "hevc":
            raise SRException("unavailable")
        return {"method": "Media Foundation hardware", "transform": "fixture"}

    monkeypatch.setattr(capabilities, "hardware_encode_probe", encode)
    report = capabilities.video_capabilities()
    assert report["ready"] and calls == ["h264", "hevc", "av1"]
    assert report["encode"]["hevc"]["status"] == "not proven"
    assert report["encode"]["av1"]["evidence"]["transform"] == "fixture"


def test_missing_ffmpeg_has_actionable_diagnostic(monkeypatch):
    monkeypatch.setattr(
        capabilities, "tool_path", lambda p: (_ for _ in ()).throw(SRException("missing"))
    )
    report = capabilities.video_capabilities()
    assert not report["ready"] and report["errors"] == ["missing"]


def test_progress_is_throttled_and_reports_submitted_frames(monkeypatch):
    from npu_sr import progress

    clock = iter([0, 0.5, 2.1, 2.2, 4.2])
    monkeypatch.setattr(progress, "perf_counter", lambda: next(clock))
    stream = io.StringIO()
    reporter = progress.VideoProgress(stream)
    reporter.initialize(
        {
            "source_frames": 120,
            "input": {
                "frame_rate": "30",
                "resolution": [960, 540],
                "duration_seconds": 4,
                "codec": "h264",
            },
            "output_resolution": [1920, 1080],
            "codec": "av1",
            "model": "test",
            "backend": "CPU",
            "codec_evidence": {"decode": {"method": "software"}},
        }
    )
    for frames in (15, 63, 66, 120):
        reporter(frames)
    result = stream.getvalue()
    assert result.count("Progress:") == 2
    assert "120 / 120 (100%)" in result and "FPS submitted" in result
    assert "QCOM" not in result
