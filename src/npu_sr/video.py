"""Ordered raw-frame streaming with one enhancement session for the whole video."""

import logging
import os
import threading
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from PIL import Image

from .errors import SRException
from .ffmpeg import (
    PipeProcess,
    VideoInfo,
    decode_args,
    decode_evidence,
    encode_args,
    encode_evidence,
    hardware_decode_probe,
    hardware_encode_probe,
    probe,
    read_frame,
    run_tool,
    tool_path,
    validate_cfr,
    write_frame,
)
from .image import infer_y, postprocess, preprocess
from .model import resolve_model, sha256
from .monitoring import memory_usage
from .runtime import Runtime

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoSettings:
    model: str = "espcn-x2-256"
    device: str = "auto"
    decode: str = "auto"
    encode: str = "auto"
    codec: str = "h264"
    audio: str = "copy"
    bitrate: str = "8M"
    quality: int = 20
    ffmpeg: Path | None = None
    cache_dir: Path | None = None
    overwrite: bool = False
    verbose: bool = False


def select_codecs(
    ffmpeg: Path, source: Path, info: VideoInfo, settings: VideoSettings, scale: int
) -> tuple[bool, bool, dict]:
    evidence = {}
    selected = []
    for kind in ("decode", "encode"):
        mode = getattr(settings, kind)
        if mode not in {"auto", "hardware", "software"}:
            raise SRException(f"Invalid {kind} mode: {mode}")
        hardware = False
        if mode != "software":
            try:
                if kind == "decode":
                    result = hardware_decode_probe(ffmpeg, source)
                else:
                    result = hardware_encode_probe(
                        ffmpeg, info, scale, settings.codec, settings.bitrate
                    )
                evidence[kind] = result
                hardware = True
            except SRException as exc:
                if mode == "hardware":
                    raise SRException(f"Strict hardware {kind} unavailable: {exc}") from exc
                log.warning("Hardware %s unavailable — using software: %s", kind, exc)
                evidence[kind] = {"method": "software", "auto_fallback": True}
        else:
            evidence[kind] = {"method": "software", "auto_fallback": False}
        selected.append(hardware)
    return selected[0], selected[1], evidence


def frames(process: PipeProcess, info: VideoInfo) -> Iterator[tuple[int, bytes, float]]:
    """A sequential source today; no seeking or known frame count is required."""
    assert process.process.stdout is not None
    index = 0
    while True:
        started = perf_counter()
        frame = read_frame(process.process.stdout, info.frame_bytes)
        if frame is None:
            return
        yield index, frame, (perf_counter() - started) * 1000
        index += 1


def _statistics(samples: deque[float]) -> dict:
    values = np.asarray(samples, np.float64)
    return {
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def process_video(
    source: Path,
    output: Path,
    settings: VideoSettings,
    progress: Callable[[int], None] | None = None,
) -> dict:
    if not source.is_file():
        raise SRException("Input video is missing.")
    if output.resolve() == source.resolve():
        raise SRException("Video output would overwrite its input.")
    if output.exists() and not settings.overwrite:
        raise SRException("Video output already exists. Use --overwrite deliberately.")
    if output.suffix.lower() not in {".mp4", ".mkv"}:
        raise SRException("Video output must be MP4 or MKV.")
    if settings.audio not in {"copy", "none"}:
        raise SRException("Audio mode must be copy or none.")
    initialization = perf_counter()
    ffmpeg = tool_path(settings.ffmpeg)
    info = probe(source, ffmpeg)
    source_frames = validate_cfr(source, ffmpeg, info)
    runtime = Runtime(
        resolve_model(settings.model), settings.device, settings.verbose, settings.cache_dir
    )
    if runtime.spec.task != "upscale" or runtime.spec.scale != 2:
        raise SRException("Video v0.3 requires a 2x super-resolution model.")
    hardware_decode, hardware_encode, codec_evidence = select_codecs(
        ffmpeg, source, info, settings, 2
    )
    version = (
        run_tool([str(ffmpeg), "-version"]).stdout.decode("utf-8", errors="replace").splitlines()[0]
    )
    for _ in range(3):
        runtime.run(np.full(runtime.input_shape, 0.5, np.float32))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.partial{output.suffix}")
    startup_ms = (perf_counter() - initialization) * 1000
    phases: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=8192))
    completed, decoder, encoder = 0, None, None
    done = threading.Event()
    activity = [perf_counter()]
    timed_out = threading.Event()

    def watchdog() -> None:
        # Killing a stalled child unblocks an IO wait; never leave a broken pipe hanging forever.
        while not done.wait(0.5):
            if perf_counter() - activity[0] > 60:
                timed_out.set()
                for child in (decoder, encoder):
                    if child and child.process.poll() is None:
                        child.process.kill()
                return

    watcher = threading.Thread(target=watchdog, daemon=True)
    started, cpu_started = perf_counter(), process_time()
    try:
        decoder = PipeProcess(ffmpeg, decode_args(source, hardware_decode))
        encoder = PipeProcess(
            ffmpeg,
            encode_args(
                source,
                temporary,
                info,
                2,
                settings.codec,
                hardware_encode,
                settings.audio,
                settings.bitrate,
                settings.quality,
            ),
            input_pipe=True,
        )
        watcher.start()
        assert encoder.process.stdin is not None
        for index, raw, read_ms in frames(decoder, info):
            frame_started = perf_counter()
            image = Image.frombytes("RGB", (info.width, info.height), raw)
            before = perf_counter()
            prepared = preprocess(image, runtime.spec)
            timings = {"preprocessing_ms": (perf_counter() - before) * 1000}
            y, _ = infer_y(prepared, runtime, timings)
            before = perf_counter()
            enhanced = postprocess(y, prepared)
            data = enhanced.tobytes()
            timings["postprocessing_ms"] = (perf_counter() - before) * 1000
            before = perf_counter()
            write_frame(encoder.process.stdin, data)
            timings["encoder_pipe_wait_ms"] = (perf_counter() - before) * 1000
            timings["decoder_pipe_wait_ms"] = read_ms
            timings["frame_total_ms"] = (perf_counter() - frame_started) * 1000 + read_ms
            for key, value in timings.items():
                phases[key].append(value)
            completed = index + 1
            activity[0] = perf_counter()
            if progress:
                progress(completed)
        encoder.process.stdin.close()
        decoder.finish()
        encoder.finish()
        processing_seconds = perf_counter() - started
        cpu_seconds = process_time() - cpu_started
        if timed_out.is_set():
            raise SRException("Video pipeline stalled for more than 60 seconds.")
        if not completed:
            raise SRException("Input video produced no frames.")
        if completed != source_frames or (info.frames is not None and completed != info.frames):
            raise SRException("Decoded frame count does not match input metadata.")
        if abs(completed / float(info.fps) - info.duration) > max(0.1, 2 / float(info.fps)):
            raise SRException("Decoded frames do not match input duration; VFR or damaged input.")
        if hardware_decode:
            codec_evidence["decode"] = decode_evidence(decoder.evidence_log, True)
        if hardware_encode:
            codec_evidence["encode"] = encode_evidence(encoder.evidence_log)
        # Count the final encoded frames: passthrough must not skip or duplicate frames.
        try:
            actual = probe(temporary, ffmpeg, count_frames=True)
            encoded_frames = validate_cfr(temporary, ffmpeg, actual)
        except SRException as exc:
            raise SRException(
                "Encoded output failed validation; no output published. "
                "Try another codec or explicit software encode. " + str(exc)
            ) from exc
        expected_size = (info.width * 2, info.height * 2)
        if (
            actual.frames != completed
            or encoded_frames != completed
            or actual.fps != info.fps
            or (actual.width, actual.height) != expected_size
        ):
            raise SRException(
                "Encoded video failed frame-count, framerate or resolution validation."
            )
        if abs(actual.duration - completed / float(info.fps)) > max(0.1, 2 / float(info.fps)):
            raise SRException("Encoded video duration does not match processed frames.")
        if settings.audio == "copy" and info.audio and not actual.audio:
            raise SRException("Audio stream did not survive encoding.")
        os.replace(temporary, output)
        from .suite import environment

        inference_seconds = sum(phases["inference_ms"]) / 1000
        return {
            "schema_version": 3,
            "environment": environment(),
            "ffmpeg_version": version,
            "ffmpeg_sha256": sha256(ffmpeg),
            "input_sha256": sha256(source),
            "model": runtime.manifest["name"],
            "model_sha256": runtime.manifest["sha256"],
            "backend": runtime.backend,
            "execution_evidence": runtime.evidence,
            "codec_evidence": codec_evidence,
            "codec": settings.codec,
            "bitrate": settings.bitrate if hardware_encode else None,
            "software_quality_crf": settings.quality if not hardware_encode else None,
            "audio": settings.audio,
            "input": info.public(),
            "output": actual.public(),
            "frames_processed": completed,
            "input_timestamps_validated": True,
            "output_timestamps_validated": True,
            "hardware_rate_control": "u_vbr / camera_record" if hardware_encode else None,
            "dropped_frames": 0,
            "startup_ms": startup_ms,
            "neural_startup_ms": runtime.startup_ms,
            "warmup_tensor_runs": 3,
            "processing_seconds": processing_seconds,
            "end_to_end_fps": completed / processing_seconds,
            "real_time_factor": processing_seconds / (completed / float(info.fps)),
            "process_cpu_seconds": cpu_seconds,
            "process_memory": memory_usage(),
            "phase_statistics": {key: _statistics(values) for key, values in phases.items()},
            "phase_sample_scope": f"last {min(completed, 8192)} frames; bounded storage",
            "inference_only_theoretical_fps": completed / inference_seconds
            if completed <= 8192
            else None,
            "timing_scope": (
                "decode process start through encoder flush; excludes initialization, "
                "output validation and hashing"
            ),
            "pipe_timings": "read/write blocking time, not isolated hardware codec execution time",
        }
    except (BrokenPipeError, OSError) as exc:
        tail = "\n".join(encoder.tail) if encoder else ""
        raise SRException(f"Video pipe failed: {exc}\n{tail}") from exc
    finally:
        done.set()
        for child in (decoder, encoder):
            if child:
                child.close()
        if watcher.is_alive():
            watcher.join(timeout=2)
        temporary.unlink(missing_ok=True)
