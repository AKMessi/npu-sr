"""Tiny real FFmpeg CPU integration; manual codec/NPU tests remain separated."""

from dataclasses import replace

import numpy as np
import pytest

from npu_sr.errors import SRException
from npu_sr.ffmpeg import PipeProcess, decode_args, read_frame, tool_path, write_frame
from npu_sr.model import model_path
from npu_sr.video import VideoSettings, process_video


@pytest.fixture
def video_source(tmp_path):
    try:
        ffmpeg = tool_path()
    except SRException:
        pytest.skip("FFmpeg absent; install it to run CPU video integration")
    source = tmp_path / "source.mp4"
    # Increasing gray levels make temporal ordering observable after lossy encode.
    raw = b"".join(np.full((48, 64, 3), 40 + n * 25, np.uint8).tobytes() for n in range(6))
    child = PipeProcess(
        ffmpeg,
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "64x48",
            "-r",
            "6",
            "-i",
            "pipe:0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        input_pipe=True,
    )
    try:
        write_frame(child.process.stdin, raw)
        child.process.stdin.close()
        child.finish()
    finally:
        child.close()
    return source, ffmpeg


def cpu_settings(ffmpeg):
    if not model_path(identifier="espcn-x2-256").is_file():
        pytest.skip("Acquire espcn-x2-256 for CPU video integration")
    return VideoSettings(device="cpu", decode="software", encode="software", ffmpeg=ffmpeg)


def test_cpu_video_preserves_frames_audio_duration_and_order(video_source, tmp_path):
    source, ffmpeg = video_source
    output = tmp_path / "result.mp4"
    report = process_video(source, output, cpu_settings(ffmpeg))
    assert report["frames_processed"] == 6 and report["dropped_frames"] == 0
    assert report["output"]["resolution"] == [128, 96]
    assert report["output"]["audio"] and report["output"]["frame_rate"] == "6"
    assert abs(report["output"]["duration_seconds"] - 1) < 0.05
    decoder = PipeProcess(ffmpeg, decode_args(output, False))
    means = []
    try:
        while (raw := read_frame(decoder.process.stdout, 128 * 96 * 3)) is not None:
            means.append(np.frombuffer(raw, np.uint8).mean())
        decoder.finish()
    finally:
        decoder.close()
    assert len(means) == 6 and np.all(np.diff(means) > 10)
    assert report["end_to_end_fps"] > 0
    assert report["output_timestamps_validated"]
    from npu_sr.video_benchmark import evaluate_video

    quality = evaluate_video(output, output, ffmpeg, stride=2)
    assert quality["decoded_frames"] == 6 and quality["mean_psnr_y_db"] is None
    assert quality["mean_ssim_y"] == pytest.approx(1, abs=1e-12)
    assert quality["temporal_residual_change"] == 0


@pytest.mark.parametrize("depth", [0, 2])
def test_cancellation_closes_both_children(video_source, tmp_path, monkeypatch, depth):
    source, ffmpeg = video_source
    children = []

    def record(*args, **kwargs):
        child = PipeProcess(*args, **kwargs)
        children.append(child)
        return child

    def interrupt(frame):
        raise KeyboardInterrupt

    monkeypatch.setattr("npu_sr.video.PipeProcess", record)
    output = tmp_path / "cancelled.mp4"
    with pytest.raises(KeyboardInterrupt):
        process_video(
            source, output, replace(cpu_settings(ffmpeg), pipeline_depth=depth), interrupt
        )
    assert len(children) == 2 and all(c.process.poll() is not None for c in children)
    assert not output.exists() and not list(tmp_path.glob("*.partial.mp4"))


@pytest.mark.parametrize("depth", [0, 2])
def test_cpu_planar_video_preserves_frames_and_audio(video_source, tmp_path, depth):
    source, ffmpeg = video_source
    settings = replace(cpu_settings(ffmpeg), frame_format="nv12", pipeline_depth=depth)
    report = process_video(source, tmp_path / "planar.mp4", settings)
    assert report["frames_processed"] == 6 and report["dropped_frames"] == 0
    assert report["frame_format"] == "nv12" and report["output"]["audio"]
    assert report["output_timestamps_validated"]
    assert max(report["observed_queue_peaks"].values()) <= depth


def test_broken_encoder_is_clean_error(video_source, tmp_path, monkeypatch):
    source, ffmpeg = video_source
    monkeypatch.setattr("npu_sr.video.encode_args", lambda *args: ["-invalid-option"])
    with pytest.raises(SRException, match="pipe failed|process failed"):
        process_video(source, tmp_path / "failed.mp4", cpu_settings(ffmpeg))


def test_software_av1_video_on_installed_build(video_source, tmp_path):
    source, ffmpeg = video_source
    settings = replace(cpu_settings(ffmpeg), codec="av1")
    report = process_video(source, tmp_path / "software-av1.mp4", settings)
    assert report["frames_processed"] == 6 and report["output"]["codec"] == "av1"
    assert report["codec_evidence"]["encode"]["encoder"] in {"libsvtav1", "libaom-av1"}
    assert report["output"]["audio"]


def test_output_collision_and_corrupt_input(video_source, tmp_path):
    source, ffmpeg = video_source
    with pytest.raises(SRException, match="overwrite its input"):
        process_video(source, source, cpu_settings(ffmpeg))
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(SRException, match="Video tool failed"):
        process_video(bad, tmp_path / "bad-result.mp4", cpu_settings(ffmpeg))


def test_encoder_output_validation_failure_never_publishes(video_source, tmp_path, monkeypatch):
    source, ffmpeg = video_source
    from npu_sr.video import validate_cfr

    def fail_output(path, tool, info):
        if ".partial" in path.name:
            raise SRException("Variable or reordered frame timestamps are unsupported.")
        return validate_cfr(path, tool, info)

    monkeypatch.setattr("npu_sr.video.validate_cfr", fail_output)
    output = tmp_path / "invalid.mp4"
    with pytest.raises(SRException, match="Encoded output failed validation"):
        process_video(source, output, cpu_settings(ffmpeg))
    assert not output.exists() and not list(tmp_path.glob("*.partial.mp4"))


@pytest.mark.video_hw
@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
def test_real_snapdragon_video_hardware(video_source, tmp_path, codec):
    original, ffmpeg = video_source
    # This driver rejects the 64x48 CI fixture. Test the proven useful 360p profile.
    source = tmp_path / "hardware-source.mp4"
    from npu_sr.ffmpeg import run_tool

    run_tool(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(original),
            "-vf",
            "scale=640:360",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-c:a",
            "copy",
            str(source),
        ]
    )
    settings = replace(
        cpu_settings(ffmpeg), device="npu", decode="hardware", encode="hardware", codec=codec
    )
    report = process_video(source, tmp_path / f"{codec}.mp4", settings)
    assert report["execution_evidence"]["executed_kernel_counts"] == {"QNNExecutionProvider": 1}
    assert report["codec_evidence"]["decode"]["hardware_format_selected"] == "d3d11"
    assert "QCOM Hardware Encoder" in report["codec_evidence"]["encode"]["transform"]
    assert report["frames_processed"] == 6 and report["output"]["audio"]


@pytest.mark.video_hw
def test_realtime_planar_context_reuse_and_cancellation(video_source, tmp_path, monkeypatch):
    from npu_sr.ffmpeg import run_tool
    from npu_sr.video import settings_for_preset

    original, ffmpeg = video_source
    source = tmp_path / "planar-source.mp4"
    run_tool(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(original),
            "-vf",
            "scale=960:540",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-c:a",
            "copy",
            str(source),
        ]
    )
    settings = settings_for_preset("realtime", {"ffmpeg": ffmpeg, "cache_dir": tmp_path / "cache"})
    for index in range(2):
        report = process_video(source, tmp_path / f"reuse-{index}.mp4", settings)
        assert report["execution_evidence"]["executed_kernel_counts"] == {"QNNExecutionProvider": 1}
        assert report["frames_processed"] == 6 and report["output"]["audio"]
        assert report["neural_tile_runs"] == 72
        assert report["output_timestamps_validated"] and report["dropped_frames"] == 0
        assert report["frame_format"] == "nv12"
        assert report["npu_performance_requested"] == "burst"
        assert report["neural_strength"] == 0.5
        assert max(report["observed_queue_peaks"].values()) <= 2
        if index:
            assert report["execution_evidence"]["context_cache"] == "hit"
    decoder = PipeProcess(ffmpeg, decode_args(tmp_path / "reuse-1.mp4", False))
    means = []
    try:
        while (raw := read_frame(decoder.process.stdout, 1920 * 1080 * 3)) is not None:
            means.append(np.frombuffer(raw, np.uint8).mean())
        decoder.finish()
    finally:
        decoder.close()
    assert len(means) == 6 and np.all(np.diff(means) > 10)

    children = []

    def record(*args, **kwargs):
        child = PipeProcess(*args, **kwargs)
        children.append(child)
        return child

    def interrupt(frame):
        raise KeyboardInterrupt

    monkeypatch.setattr("npu_sr.video.PipeProcess", record)
    with pytest.raises(KeyboardInterrupt):
        process_video(source, tmp_path / "cancelled-hardware.mp4", settings, interrupt)
    assert len(children) == 2 and all(c.process.poll() is not None for c in children)
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert not (tmp_path / "cancelled-hardware.mp4").exists()
