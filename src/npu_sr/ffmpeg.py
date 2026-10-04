"""FFmpeg tools, bounded stderr capture, and executed hardware codec checks."""

import json
import math
import os
import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

from .errors import SRException


def tool_path(explicit: Path | None = None) -> Path:
    """Prefer an explicit tool, then configured tool, cached native build, PATH."""
    configured = explicit or os.environ.get("NPU_SR_FFMPEG")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise SRException("FFmpeg executable missing. Supply --ffmpeg or NPU_SR_FFMPEG.")
        return path.resolve()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "npu-sr" / "tools"
        cached = sorted(root.glob("ffmpeg-*/extracted/*/bin/ffmpeg.exe"))
        if cached:
            return cached[-1]
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    raise SRException("FFmpeg unavailable. See docs/ffmpeg.md or supply --ffmpeg.")


def run_tool(
    arguments: list[str], timeout: float = 60, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            arguments, capture_output=True, timeout=timeout, check=False, cwd=cwd
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SRException(f"Video tool failed: {exc}") from exc
    if result.returncode:
        lines = result.stderr.decode("utf-8", errors="replace").splitlines()
        failures = [
            line
            for line in lines
            if re.search(r"failed|error|invalid|could not", line, re.I)
            and "0 decode errors" not in line
        ]
        raise SRException("Video tool failed: " + "\n".join(failures[-6:] + lines[-2:]))
    return result


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    duration: float
    frames: int | None
    audio: bool
    codec: str
    color_range: str = "unknown"
    color_space: str = "unknown"

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3

    def bytes_for(self, pixel_format: str) -> int:
        if pixel_format == "rgb24":
            return self.frame_bytes
        if pixel_format == "nv12":
            return self.width * self.height * 3 // 2
        raise SRException("Frame format must be rgb24 or nv12.")

    def public(self) -> dict:
        return {
            "resolution": [self.width, self.height],
            "frame_rate": str(self.fps),
            "duration_seconds": self.duration,
            "reported_frames": self.frames,
            "audio": self.audio,
            "codec": self.codec,
            "color_range": self.color_range,
            "color_space": self.color_space,
        }


def probe(path: Path, ffmpeg: Path, count_frames: bool = False) -> VideoInfo:
    ffprobe = ffmpeg.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if not ffprobe.is_file():
        raise SRException("ffprobe is missing beside FFmpeg.")
    arguments = [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_frames:
        arguments.append("-count_frames")
    result = run_tool([*arguments, str(path.resolve())], timeout=300)
    try:
        data = json.loads(result.stdout)
        streams = data["streams"]
        video = next(s for s in streams if s["codec_type"] == "video")
        fps = Fraction(video["avg_frame_rate"])
        nominal = Fraction(video["r_frame_rate"])
        duration = float(video.get("duration", data["format"].get("duration", 0)))
        width, height = int(video["width"]), int(video["height"])
    except (ValueError, KeyError, StopIteration, ZeroDivisionError) as exc:
        raise SRException("Input is not a supported timed video stream.") from exc
    if not 0 < fps <= 240 or duration <= 0 or fps != nominal:
        raise SRException("Only constant-framerate video (1–240 FPS) is supported.")
    if width <= 0 or height <= 0 or width * height > 16_000_000 or width % 2 or height % 2:
        raise SRException("Video needs positive even dimensions, at most 16 million pixels.")
    if video.get("sample_aspect_ratio", "1:1") not in {"1:1", "N/A"}:
        raise SRException("Anamorphic video is unsupported; convert to square pixels first.")
    rotation = [s.get("rotation", 0) for s in video.get("side_data_list", [])]
    if any(rotation) or video.get("tags", {}).get("rotate", "0") not in {"0", ""}:
        raise SRException("Rotated video is unsupported; normalize orientation first.")
    if video.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise SRException("HDR video is unsupported. Convert to SDR explicitly first.")
    count = video.get("nb_read_frames" if count_frames else "nb_frames")
    return VideoInfo(
        width,
        height,
        fps,
        duration,
        int(count) if count and count != "N/A" else None,
        any(s["codec_type"] == "audio" for s in streams),
        video.get("codec_name", "unknown"),
        video.get("color_range", "unknown"),
        video.get("color_space", "unknown"),
    )


def check_frame_times(times: Iterable[float], fps: Fraction) -> int:
    """Validate timestamp cadence without storing a video's timestamp list."""
    previous, count = None, 0
    interval = 1 / float(fps)
    for value in times:
        if not math.isfinite(value):
            raise SRException("Frame timestamps must be finite.")
        if previous is not None and abs(value - previous - interval) > max(0.0011, interval * 0.02):
            raise SRException("Variable or reordered frame timestamps are unsupported.")
        previous = value
        count += 1
    if not count:
        raise SRException("Input has no timed frames.")
    return count


def validate_cfr(source: Path, ffmpeg: Path, info: VideoInfo) -> int:
    """Inspect actual input timestamps; average framerate alone cannot prove CFR."""
    ffprobe = ffmpeg.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    child = subprocess.Popen(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "default=nw=1:nk=1",
            str(source.resolve()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert child.stdout is not None
        count = check_frame_times((float(line) for line in child.stdout if line.strip()), info.fps)
        if child.wait(timeout=10):
            raise SRException("Input timestamp inspection failed.")
        return count
    except ValueError as exc:
        raise SRException("Video contains missing or malformed frame timestamps.") from exc
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if child.stdout:
            child.stdout.close()


def decode_args(
    source: Path, hardware: bool, frames: int | None = None, pixel_format: str = "rgb24"
) -> list[str]:
    if pixel_format not in {"rgb24", "nv12"}:
        raise SRException("Frame format must be rgb24 or nv12.")
    args = ["-loglevel", "debug" if hardware else "error", "-noautorotate"]
    if hardware:
        args += ["-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11"]
    args += ["-i", str(source.resolve()), "-map", "0:v:0", "-an", "-sn", "-dn"]
    if hardware:
        # A software fallback cannot satisfy hwdownload's hardware-frame input contract.
        filter_text = "hwdownload,format=nv12"
        if pixel_format == "rgb24":
            filter_text += ",format=rgb24"
        args += ["-vf", filter_text]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    return args + ["-fps_mode", "passthrough", "-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"]


def codec_args(codec: str, hardware: bool, bitrate: str, quality: int) -> list[str]:
    if codec not in {"h264", "hevc", "av1"}:
        raise SRException(f"Unsupported output codec: {codec}")
    if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?[kKmM]?", bitrate):
        raise SRException("Bitrate must be a positive FFmpeg rate such as 8M.")
    if not 0 <= quality <= 51:
        raise SRException("Quality must be between 0 and 51 (lower is better, software only).")
    if hardware:
        # Default vendor rate control dropped frames at scene cuts in local tests.
        # FFmpeg documents camera_record as CFR; output is still verified afterward.
        return [
            "-c:v",
            codec + "_mf",
            "-hw_encoding",
            "1",
            "-rate_control",
            "u_vbr",
            "-scenario",
            "camera_record",
            "-b:v",
            bitrate,
            "-pix_fmt",
            "nv12",
        ]
    encoder = {"h264": "libx264", "hevc": "libx265", "av1": "libaom-av1"}[codec]
    extra = ["-preset", "veryfast"] if codec != "av1" else ["-cpu-used", "8", "-b:v", "0"]
    return ["-c:v", encoder, "-crf", str(quality), *extra, "-pix_fmt", "yuv420p"]


def encode_args(
    source: Path,
    output: Path,
    info: VideoInfo,
    scale: int,
    codec: str,
    hardware: bool,
    audio: str,
    bitrate: str,
    quality: int,
    pixel_format: str = "rgb24",
) -> list[str]:
    args = [
        "-loglevel",
        "verbose",
        "-thread_queue_size",
        "2",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "-video_size",
        f"{info.width * scale}x{info.height * scale}",
        "-framerate",
        str(info.fps),
        "-i",
        "pipe:0",
    ]
    if audio == "copy" and info.audio:
        args += ["-i", str(source.resolve()), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        args += ["-map", "0:v:0", "-an"]
    colors = []
    if pixel_format == "nv12":
        colors += ["-color_range", "tv"]
        if info.color_space != "unknown":
            colors += ["-colorspace", info.color_space]
    return (
        args
        + codec_args(codec, hardware, bitrate, quality)
        + colors
        + [
            "-fps_mode",
            "passthrough",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            str(output.resolve()),
        ]
    )


def hardware_decode_probe(ffmpeg: Path, source: Path) -> dict:
    # Capture only a few frames during initialization, never a whole video in RAM.
    result = run_tool([str(ffmpeg), "-hide_banner", "-nostdin", *decode_args(source, True, 3)])
    log = result.stderr.decode("utf-8", errors="replace")
    return decode_evidence(log, bool(result.stdout))


def decode_evidence(log: str, produced_frames: bool) -> dict:
    selected = "Format d3d11 chosen by get_format()" in log
    if not selected or not produced_frames:
        raise SRException("Strict hardware decode has no executed D3D11 frame evidence.")
    return {
        "method": "D3D11VA hardware",
        "hardware_format_selected": "d3d11",
        "explicit_hwdownload": True,
    }


def encode_evidence(log: str) -> dict:
    names = re.findall(r"MFT name: '([^']+)'", log)
    if not names:
        raise SRException(
            "Strict hardware encode did not identify an activated Media Foundation transform."
        )
    return {
        "method": "Media Foundation hardware",
        "forced_hardware_enumeration": True,
        "transform": names[-1],
    }


def hardware_encode_probe(
    ffmpeg: Path, info: VideoInfo, scale: int, codec: str, bitrate: str
) -> dict:
    result = run_tool(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "verbose",
            "-f",
            "lavfi",
            "-i",
            f"color=size={info.width * scale}x{info.height * scale}:rate={info.fps}",
            "-frames:v",
            "3",
            *codec_args(codec, True, bitrate, 20),
            "-f",
            "null",
            "-",
        ]
    )
    return encode_evidence(result.stderr.decode("utf-8", errors="replace"))


class PipeProcess:
    """Drain diagnostics continuously, retaining bounded text and selected evidence."""

    def __init__(self, ffmpeg: Path, arguments: list[str], *, input_pipe: bool = False):
        try:
            self.process = subprocess.Popen(
                [str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-y", *arguments],
                stdin=subprocess.PIPE if input_pipe else subprocess.DEVNULL,
                stdout=subprocess.DEVNULL if input_pipe else subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise SRException(f"Cannot start FFmpeg: {exc}") from exc
        self.tail: deque[str] = deque(maxlen=16)
        self.selected: deque[str] = deque(maxlen=16)
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        assert self.process.stderr is not None
        for raw in self.process.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            self.tail.append(line[:1000])
            if "MFT name:" in line or "Format d3d11 chosen by get_format()" in line:
                self.selected.append(line[:1000])

    @property
    def evidence_log(self) -> str:
        return "\n".join(self.selected)

    def finish(self) -> None:
        try:
            code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            self.close()
            raise SRException("FFmpeg did not finish within 60 seconds.") from exc
        self.thread.join(timeout=5)
        if code:
            raise SRException("FFmpeg process failed: " + "\n".join(self.tail))

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.thread.join(timeout=5)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream:
                stream.close()


def read_frame(stream: BinaryIO, size: int) -> bytes | None:
    """Read exactly one frame even when a pipe returns partial chunks."""
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            if offset:
                raise SRException("FFmpeg returned a truncated raw frame.")
            return None
        offset += count
    return bytes(buffer)


def write_frame(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = stream.write(view)
        if not count:
            raise SRException("FFmpeg encoder pipe stopped accepting frames.")
        view = view[count:]
