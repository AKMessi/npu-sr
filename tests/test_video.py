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


@pytest.mark.parametrize(
    "fps,portrait,audio,extension",
    [
        ("24000/1001", False, True, "mp4"),
        ("24", False, False, "mkv"),
        ("25", False, True, "mp4"),
        ("30000/1001", False, True, "mkv"),
        ("50", False, False, "mp4"),
        ("60000/1001", False, True, "mp4"),
        ("60", False, False, "mkv"),
        ("30", True, True, "mp4"),
    ],
)
def test_fractional_cadence_portrait_and_audio_matrix(tmp_path, fps, portrait, audio, extension):
    from fractions import Fraction

    from npu_sr.ffmpeg import run_tool

    try:
        ffmpeg = tool_path()
    except SRException:
        pytest.skip("FFmpeg required for generated correctness matrix")
    width, height = (48, 64) if portrait else (64, 48)
    source, output = tmp_path / f"source.{extension}", tmp_path / f"result.{extension}"
    args = [
        str(ffmpeg),
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={fps}",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    args += [
        "-frames:v",
        "12",
        "-t",
        str(12 / float(Fraction(fps))),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    args += ["-c:a", "aac"] if audio else ["-an"]
    run_tool([*args, str(source)])
    report = process_video(source, output, cpu_settings(ffmpeg))
    assert report["frames_processed"] == report["output"]["reported_frames"] == 12
    assert report["output"]["resolution"] == [width * 2, height * 2]
    assert Fraction(report["output"]["frame_rate"]) == Fraction(fps)
    assert report["output"]["audio"] is audio
    if audio:
        assert report["output"]["audio_channels"] == 2 and report["audio_validation"]


def test_disk_write_failure_keeps_existing_output_and_cleans_children(
    video_source, tmp_path, monkeypatch
):
    import errno

    source, ffmpeg = video_source
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing user output")
    children = []

    def child(*a, **kw):
        result = PipeProcess(*a, **kw)
        children.append(result)
        return result

    def full_disk(*a):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("npu_sr.video.PipeProcess", child)
    monkeypatch.setattr("npu_sr.video.write_frame", full_disk)
    with pytest.raises(SRException, match="No space"):
        process_video(
            source, output, replace(cpu_settings(ffmpeg), overwrite=True, pipeline_depth=2)
        )
    assert output.read_bytes() == b"existing user output"
    assert all(c.process.poll() is not None for c in children)
    assert not list(tmp_path.glob("*.partial.mp4"))


@pytest.mark.parametrize("video_delay,audio_delay", [(0.5, 0), (0, 0.5), (2, 2.5)])
def test_copied_audio_preserves_offset_from_first_video_frame(tmp_path, video_delay, audio_delay):
    from npu_sr.ffmpeg import probe, run_tool

    try:
        ffmpeg = tool_path()
    except SRException:
        pytest.skip("FFmpeg required for audio timing regression")
    source, output = tmp_path / "offset.mp4", tmp_path / "enhanced.mp4"
    run_tool(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-itsoffset",
            str(video_delay),
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=30:duration=2",
            "-itsoffset",
            str(audio_delay),
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=48000:duration=2.5",
            "-c:v",
            "libx264",
            "-fps_mode",
            "passthrough",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    original = probe(source, ffmpeg)
    report = process_video(source, output, cpu_settings(ffmpeg))
    actual = probe(output, ffmpeg)
    assert report["frames_processed"] == 60
    assert actual.audio_codec == original.audio_codec == "aac"
    expected_delay = max(0, original.audio_start - original.video_start)
    assert actual.audio_start - actual.video_start == pytest.approx(expected_delay, abs=0.025)
    assert all(row["samples"] == 60 for row in report["whole_run_phase_statistics"].values())
    if video_delay > audio_delay:

        def pcm(path):
            return np.frombuffer(
                run_tool(
                    [
                        str(ffmpeg),
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        "-map",
                        "0:a:0",
                        "-f",
                        "s16le",
                        "-ac",
                        "1",
                        "-ar",
                        "48000",
                        "pipe:1",
                    ]
                ).stdout,
                np.int16,
            )

        before, after = pcm(source), pcm(output)
        start = round((video_delay - audio_delay) * 48000)
        assert np.array_equal(before[start : start + len(after)], after)


@pytest.mark.video_hw
def test_first_run_setup_proves_npu_and_hardware_codecs():
    from npu_sr.setup import prepare

    report = prepare(["quicksrnet-small-y-x2", "quicksrnet-medium-y-x2"])
    assert report["setup_ready"]
    assert report["evidence"]["executed_kernel_counts"] == {"QNNExecutionProvider": 1}
    assert report["evidence"]["cpu_fallback_disabled"]
    assert report["video"]["decode"]["h264"]["status"] == "proven"
    assert "QCOM Hardware Encoder" in report["video"]["encode"]["av1"]["evidence"]["transform"]


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


def test_full_and_packet_audits_leave_identical_delivered_pixels(video_source, tmp_path):
    from npu_sr.ffmpeg import run_tool

    source, ffmpeg = video_source
    hashes = []
    for full in (False, True):
        output = tmp_path / f"audit-{full}.mp4"
        report = process_video(source, output, replace(cpu_settings(ffmpeg), verify_full=full))
        assert report["verify_full"] is full
        for timeline in report["timeline_validation"].values():
            assert timeline["count"] == 6 and timeline["full_decode"] is full
        hashes.append(
            run_tool(
                [
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-map",
                    "0:v:0",
                    "-f",
                    "framemd5",
                    "pipe:1",
                ]
            ).stdout
        )
    assert hashes[0] == hashes[1]


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


def test_strict_container_failure_happens_before_stream_and_auto_reports_fallback(
    video_source, tmp_path, monkeypatch, caplog
):
    source, ffmpeg = video_source
    started = []

    def unavailable(tool, metadata, scale, codec, bitrate, container):
        assert container == ".mkv"
        raise SRException("Test-only hardware MKV header failure")

    def child(*args, **kwargs):
        result = PipeProcess(*args, **kwargs)
        started.append(result)
        return result

    monkeypatch.setattr("npu_sr.video.hardware_encode_probe", unavailable)
    monkeypatch.setattr("npu_sr.video.PipeProcess", child)
    output = tmp_path / "existing.mkv"
    output.write_bytes(b"existing user output")
    settings = replace(cpu_settings(ffmpeg), overwrite=True, encode="hardware")
    with pytest.raises(SRException, match="Strict hardware encode unavailable"):
        process_video(source, output, settings)
    assert not started and output.read_bytes() == b"existing user output"
    report = process_video(source, output, replace(settings, encode="auto"))
    assert report["frames_processed"] == 6
    assert report["codec_evidence"]["encode"] == {"method": "software", "auto_fallback": True}
    assert "using software" in caplog.text


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
    from npu_sr.video import inspect_timeline

    def fail_output(path, tool, info, full=False):
        if ".partial" in path.name:
            raise SRException("Variable or reordered frame timestamps are unsupported.")
        return inspect_timeline(path, tool, info, full)

    monkeypatch.setattr("npu_sr.video.inspect_timeline", fail_output)
    output = tmp_path / "invalid.mp4"
    with pytest.raises(SRException, match="Encoded output failed validation"):
        process_video(source, output, cpu_settings(ffmpeg))
    assert not output.exists() and not list(tmp_path.glob("*.partial.mp4"))


@pytest.mark.video_hw
def test_temporal_hardware_video_counts_frames_and_resets_history(
    tmp_path, temporal_hardware_model
):
    from npu_sr.ffmpeg import run_tool, tool_path

    ffmpeg = tool_path()
    source = tmp_path / "temporal-scene-cut.mp4"
    run_tool(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x404040:s=640x360:r=30:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "color=c=0xc0c0c0:s=640x360:r=30:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=500:sample_rate=48000:duration=0.4",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    settings = VideoSettings(
        model=str(temporal_hardware_model),
        device="npu",
        decode="hardware",
        encode="hardware",
        codec="av1",
        frame_format="nv12",
        pipeline_depth=2,
        npu_performance="burst",
        ffmpeg=ffmpeg,
    )
    report = process_video(source, tmp_path / "temporal-result.mp4", settings)
    assert report["frames_processed"] == 12 and report["output"]["audio"]
    assert report["neural_tile_runs"] == 72 and report["dropped_frames"] == 0
    assert report["temporal_state"]["frames_processed"] == 12
    assert report["temporal_state"]["scene_reset_counts"] == {"initial": 1, "abrupt-change": 1}
    assert report["execution_evidence"]["executed_kernel_counts"] == {"QNNExecutionProvider": 1}
    assert report["execution_evidence"]["cpu_fallback_disabled"]
    assert report["codec_evidence"]["decode"]["hardware_format_selected"] == "d3d11"
    assert "QCOM Hardware Encoder" in report["codec_evidence"]["encode"]["transform"]
    assert max(report["observed_queue_peaks"].values()) <= 2


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
def test_hardware_av1_matroska_copies_audio_and_retains_frame_cadence(video_source, tmp_path):
    from npu_sr.ffmpeg import run_tool
    from npu_sr.video import settings_for_preset

    original, ffmpeg = video_source
    source = tmp_path / "source-360p.mp4"
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
            "-c:a",
            "copy",
            str(source),
        ]
    )
    settings = settings_for_preset("realtime", {"ffmpeg": ffmpeg})
    report = process_video(source, tmp_path / "av1.mkv", settings)
    assert report["frames_processed"] == report["output"]["reported_frames"] == 6
    assert report["output"]["codec"] == "av1" and report["output"]["audio_codec"] == "aac"
    assert report["audio_validation"] and report["output_timestamps_validated"]
    assert report["execution_evidence"]["executed_kernel_counts"] == {"QNNExecutionProvider": 1}
    assert report["execution_evidence"]["cpu_fallback_disabled"]
    assert "QCOM" in report["codec_evidence"]["encode"]["transform"]


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
        report = process_video(
            source, tmp_path / f"reuse-{index}.mp4", replace(settings, verify_full=bool(index))
        )
        assert report["timeline_validation"]["output"]["full_decode"] is bool(index)
        assert report["execution_evidence"]["executed_kernel_counts"] == {"QNNExecutionProvider": 1}
        assert report["frames_processed"] == 6 and report["output"]["audio"]
        assert report["neural_tile_runs"] == 72
        assert report["output_timestamps_validated"] and report["dropped_frames"] == 0
        assert report["frame_format"] == "nv12"
        assert report["npu_performance_requested"] == "burst"
        assert report["model"] == "quicksrnet-small-y-x2"
        assert report["neural_strength"] == 1.0
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
