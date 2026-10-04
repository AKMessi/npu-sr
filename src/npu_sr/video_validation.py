"""Bounded packet timeline checks, with an explicit decoded-frame forensic audit."""

import heapq
import math
import os
import subprocess
import threading
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from time import perf_counter

from .errors import SRException
from .ffmpeg import VideoInfo, check_frame_times, validate_cfr


def packet_times(lines: Iterable[bytes], info: VideoInfo) -> Iterator[float]:
    """Reorder at most 64 presentation timestamps from coded packet order.

    This checks timed access units, not decoded pictures. The caller must compare
    the count with actual frames processed. Unsupported packet layouts fail with
    a full-audit suggestion rather than being accepted as frame evidence.
    """
    pending: list[float] = []
    interval = 1 / float(info.fps)
    for raw in lines:
        if len(raw) > 4096:
            raise SRException("Video packet metadata exceeded its bounded line size.")
        try:
            fields = dict(part.split("=", 1) for part in raw.decode("ascii").strip().split("|"))
            pts = float(fields["pts_time"])
            duration = fields.get("duration_time", "N/A")
            if duration != "N/A":
                value = float(duration)
                if not math.isfinite(value) or abs(value - interval) > max(0.0011, interval * 0.02):
                    raise SRException("Variable packet duration is unsupported.")
            if any(flag in fields.get("flags", "") for flag in ("C", "D")):
                raise SRException("Corrupt or discard-marked video packets require a full audit.")
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            raise SRException("Missing or malformed video packet timestamps.") from exc
        if not math.isfinite(pts):
            raise SRException("Video packet timestamps must be finite.")
        heapq.heappush(pending, pts)
        if len(pending) > 64:
            yield heapq.heappop(pending)
    while pending:
        yield heapq.heappop(pending)


def inspect_timeline(source: Path, ffmpeg: Path, info: VideoInfo, full: bool = False) -> dict:
    """Inspect every packet by default; full=True decodes every frame instead.

    Packet counts are only valid when they agree with both container metadata
    and the enhancement pipeline's actual decoded/encoded frame accounting.
    They cannot prove pixel integrity; full audit remains available for that.
    """
    started = perf_counter()
    if full:
        count = validate_cfr(source, ffmpeg, info)
        method = "decoded frame timestamps (full audit)"
    else:
        ffprobe = ffmpeg.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if not ffprobe.is_file():
            raise SRException("ffprobe is missing beside FFmpeg.")
        try:
            child = subprocess.Popen(
                [
                    str(ffprobe),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "packet=pts_time,duration_time,flags",
                    "-of",
                    "compact=p=0:nk=0",
                    str(source.resolve()),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise SRException("Cannot start video timeline inspection.") from exc
        errors: deque[str] = deque(maxlen=8)
        expired = threading.Event()

        def drain() -> None:
            assert child.stderr is not None
            for line in iter(lambda: child.stderr.readline(4097), b""):
                errors.append(line.decode("utf-8", "replace")[:1000])

        def timeout() -> None:
            expired.set()
            if child.poll() is None:
                try:
                    child.kill()
                except OSError:
                    pass  # Child may exit between poll and kill.

        reader = threading.Thread(target=drain, daemon=True)
        timer = threading.Timer(300, timeout)
        reader.start()
        try:
            timer.start()
            assert child.stdout is not None
            # readline's limit prevents an untrusted metadata line allocating arbitrarily.
            lines = iter(lambda: child.stdout.readline(4097), b"")
            count = check_frame_times(packet_times(lines, info), info.fps)
            code = child.wait(timeout=10)
            reader.join(timeout=5)
            if expired.is_set() or code or errors:
                raise SRException("Video packet inspection failed or timed out.")
        except (SRException, subprocess.TimeoutExpired) as exc:
            raise SRException(
                "Packet timeline validation failed. Try --verify-full for a decoded-frame "
                "audit; VFR or corrupt streams still cannot be processed. " + str(exc)
            ) from exc
        finally:
            timer.cancel()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            reader.join(timeout=5)
            child.stdout.close()
            child.stderr.close()
        method = "all packet presentation timestamps; actual pipeline frame count required"
    if info.frames is not None and count != info.frames:
        raise SRException("Timeline count disagrees with container frames. Try --verify-full.")
    return {
        "method": method,
        "count": count,
        "seconds": perf_counter() - started,
        "full_decode": full,
        "limitation": None if full else "Packet inspection does not prove decoded pixel integrity.",
    }
