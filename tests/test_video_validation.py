"""Packet cadence must agree with decoded frames without masking VFR or corruption."""

from fractions import Fraction
from pathlib import Path

import pytest

from npu_sr.errors import SRException
from npu_sr.ffmpeg import VideoInfo, check_frame_times, probe, run_tool, tool_path
from npu_sr.video_validation import inspect_timeline, packet_times


def info(fps=Fraction(30)):
    return VideoInfo(64, 48, fps, 1, None, False, "h264")


def line(pts, duration="0.033333", flags="___"):
    return f"pts_time={pts}|duration_time={duration}|flags={flags}\n".encode()


def test_packet_order_is_not_presentation_order():
    timestamps = [0, 3, 1, 2, *range(4, 200)]
    ordered = packet_times((line(n / 30) for n in timestamps), info())
    assert check_frame_times(ordered, Fraction(30)) == 200


@pytest.mark.parametrize(
    "packets",
    [
        [line(0), line(0)],  # Duplicate presentation.
        [line(0), line(2 / 30)],  # Gap.
        [line(0, "0.1")],  # VFR duration despite nominal rate.
        [line(0, "nan")],
        [line("nan")],
        [line("N/A")],
        [b"duration_time=0.033333\n"],
        [line(0, flags="__C")],
        [line(0, flags="__D")],
        [b"x" * 4097],
    ],
)
def test_invalid_packets_do_not_pass_cfr(packets):
    with pytest.raises(SRException):
        check_frame_times(packet_times(packets, info()), Fraction(30))


def test_excessive_reordering_fails_instead_of_growing_memory():
    packets = [line(n / 30) for n in range(1, 130)] + [line(0)]
    with pytest.raises(SRException, match="Variable or reordered"):
        check_frame_times(packet_times(packets, info()), Fraction(30))


def test_optional_unknown_duration_still_requires_pts_cadence():
    assert (
        check_frame_times(
            packet_times((line(n / 30, "N/A") for n in range(100)), info()), Fraction(30)
        )
        == 100
    )


@pytest.mark.parametrize("extension,fps", [("mp4", "30"), ("mkv", "30000/1001")])
def test_real_b_frames_packet_and_decoded_audits_agree(tmp_path, extension, fps):
    try:
        ffmpeg = tool_path()
    except SRException:
        pytest.skip("Install FFmpeg for real packet validation")
    source = tmp_path / f"b-frames.{extension}"
    run_tool(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=64x48:rate={fps}",
            "-frames:v",
            "90",
            "-c:v",
            "libx264",
            "-bf",
            "3",
            str(source),
        ]
    )
    metadata = probe(source, ffmpeg)
    fast = inspect_timeline(source, ffmpeg, metadata)
    full = inspect_timeline(source, ffmpeg, metadata, full=True)
    assert fast["count"] == full["count"] == 90
    assert not fast["full_decode"] and full["full_decode"]
    assert "pixel integrity" in fast["limitation"] and full["limitation"] is None
    wrong = VideoInfo(64, 48, metadata.fps, metadata.duration, 89, False, "h264")
    with pytest.raises(SRException, match="disagrees with container"):
        inspect_timeline(source, ffmpeg, wrong)


def test_missing_probe_is_actionable(tmp_path):
    with pytest.raises(SRException, match="ffprobe is missing"):
        inspect_timeline(tmp_path / "input.mp4", tmp_path / "ffmpeg", info())


def test_full_mode_uses_decoded_frame_audit(monkeypatch):
    calls = []

    def full(source, ffmpeg, metadata):
        calls.append((source, ffmpeg, metadata))
        return 90

    monkeypatch.setattr("npu_sr.video_validation.validate_cfr", full)
    result = inspect_timeline(Path("input.mp4"), Path("ffmpeg"), info(), full=True)
    assert result["count"] == 90 and len(calls) == 1 and result["full_decode"]


@pytest.mark.parametrize("failure", ["timeout", "oversize", "interrupt"])
def test_full_audit_is_bounded_and_cleans_child_on_failure(monkeypatch, failure):
    import io

    from npu_sr.ffmpeg import validate_cfr

    class Child:
        stdout = io.BytesIO(b"0.0\n" if failure != "oversize" else b"1" * 150)
        code = None

        def poll(self):
            return self.code

        def kill(self):
            self.code = -9

        def wait(self, timeout):
            self.code = self.code if self.code is not None else 0
            return self.code

    class Timer:
        cancelled = False

        def __init__(self, seconds, callback):
            self.callback = callback

        def start(self):
            if failure == "timeout":
                self.callback()

        def cancel(self):
            self.cancelled = True

    child, timers = Child(), []

    def make_timer(*args):
        result = Timer(*args)
        timers.append(result)
        return result

    monkeypatch.setattr("npu_sr.ffmpeg.subprocess.Popen", lambda *a, **kw: child)
    monkeypatch.setattr("npu_sr.ffmpeg.threading.Timer", make_timer)
    if failure == "interrupt":

        def interrupt(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr("npu_sr.ffmpeg.check_frame_times", interrupt)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else SRException):
        validate_cfr(Path("source"), Path("ffmpeg"), info())
    assert child.stdout.closed and child.poll() is not None and timers[0].cancelled


@pytest.mark.parametrize("failure", ["exit", "diagnostics", "timeout", "interrupt"])
def test_inspection_failure_and_cancellation_close_child_and_pipes(tmp_path, monkeypatch, failure):
    import io

    class Child:
        stdout = io.BytesIO(b"".join(line(n / 30) for n in range(3)))
        stderr = io.BytesIO(b"damaged packet\n" if failure == "diagnostics" else b"")
        code = 1 if failure == "exit" else None

        def poll(self):
            return self.code

        def kill(self):
            self.code = -9

        def wait(self, timeout):
            if self.code is None:
                self.code = 0
            return self.code

    class Timer:
        cancelled = False

        def __init__(self, seconds, callback):
            self.callback = callback

        def start(self):
            if failure == "timeout":
                self.callback()

        def cancel(self):
            self.cancelled = True

    child = Child()
    timers = []

    def timer(*args):
        instance = Timer(*args)
        timers.append(instance)
        return instance

    monkeypatch.setattr("npu_sr.video_validation.subprocess.Popen", lambda *a, **kw: child)
    monkeypatch.setattr("npu_sr.video_validation.threading.Timer", timer)
    if failure == "interrupt":

        def interrupt(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr("npu_sr.video_validation.packet_times", interrupt)
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.with_name("ffprobe.exe" if __import__("os").name == "nt" else "ffprobe").touch()
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else SRException):
        inspect_timeline(tmp_path / "input.mp4", ffmpeg, info())
    assert child.poll() is not None and child.stdout.closed and child.stderr.closed
    assert timers[0].cancelled
