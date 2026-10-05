"""Ordered raw-frame streaming with one enhancement session for the whole video."""

import logging
import os
import threading
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
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
    software_av1_encoder,
    tool_path,
    write_frame,
)
from .image import infer_y, postprocess, preprocess
from .model import resolve_model, sha256
from .monitoring import memory_usage, power_state
from .runtime import Runtime
from .stream import Frame, process_stream
from .video_stats import LatencyHistogram, SustainedStatistics, resource_summary
from .video_validation import inspect_timeline

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
    frame_format: str = "rgb24"
    npu_performance: str = "default"
    pipeline_depth: int = 0
    preset: str | None = None
    neural_strength: float = 1.0
    verify_full: bool = False


PRESETS = {
    "quality": {
        "model": "quicksrnet-medium-y-x2",
        "frame_format": "nv12",
        "npu_performance": "sustained_high_performance",
        "pipeline_depth": 2,
    },
    "balanced": {
        "model": "quicksrnet-small-y-x2",
        "frame_format": "nv12",
        "npu_performance": "sustained_high_performance",
        "pipeline_depth": 2,
    },
    "realtime": {
        "model": "quicksrnet-small-y-x2",
        "frame_format": "nv12",
        "npu_performance": "burst",
        "neural_strength": 1.0,
        "pipeline_depth": 2,
        "device": "npu",
        "decode": "hardware",
        "encode": "hardware",
        "codec": "av1",
    },
}


def settings_for_preset(preset: str | None, overrides: dict) -> VideoSettings:
    """Materialize a transparent CLI preset, then apply explicit user options."""
    if preset is not None and preset not in PRESETS:
        raise SRException(f"Unknown video preset: {preset}")
    settings = replace(VideoSettings(), **PRESETS.get(preset, {}), preset=preset)
    return replace(
        settings, **{key: value for key, value in overrides.items() if value is not None}
    )


def select_codecs(
    ffmpeg: Path,
    source: Path,
    info: VideoInfo,
    settings: VideoSettings,
    scale: int,
    container: str = ".mp4",
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
                        ffmpeg, info, scale, settings.codec, settings.bitrate, container
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


def frames(process: PipeProcess, info: VideoInfo, pixel_format: str = "rgb24") -> Iterator[Frame]:
    """A sequential source today; no seeking or known frame count is required."""
    assert process.process.stdout is not None
    index = 0
    while True:
        started = perf_counter()
        frame = read_frame(process.process.stdout, info.bytes_for(pixel_format))
        if frame is None:
            return
        yield Frame(index, frame, (perf_counter() - started) * 1000, started)
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
    initialized: Callable[[dict], None] | None = None,
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
    environment_start = {
        "timestamp": datetime.now(UTC).isoformat(),
        "power_state": power_state(),
        "scope": "power state at command start; full software/hardware environment recorded at end",
    }
    ffmpeg = tool_path(settings.ffmpeg)
    info = probe(source, ffmpeg)
    input_validation = inspect_timeline(source, ffmpeg, info, settings.verify_full)
    source_frames = input_validation["count"]
    runtime = Runtime(
        resolve_model(settings.model),
        settings.device,
        settings.verbose,
        settings.cache_dir,
        settings.npu_performance,
    )
    if runtime.spec.task != "upscale" or runtime.spec.scale != 2:
        raise SRException("Video requires a 2x super-resolution model.")
    planar = None
    if settings.frame_format == "nv12":
        from .planar import NV12Enhancer

        if runtime.spec.input_channels in {2, 6}:
            from .temporal import TemporalNV12Enhancer

            planar = TemporalNV12Enhancer(info, runtime, settings.neural_strength)
        else:
            planar = NV12Enhancer(info, runtime, settings.neural_strength)
    elif settings.frame_format != "rgb24":
        raise SRException("Frame format must be rgb24 or nv12.")
    elif runtime.spec.input_channels != 1:
        raise SRException("Temporal models require --frame-format nv12.")
    elif settings.neural_strength != 1:
        raise SRException("Neural blending requires --frame-format nv12.")
    hardware_decode, hardware_encode, codec_evidence = select_codecs(
        ffmpeg, source, info, settings, 2, output.suffix.lower()
    )
    software_av1 = "libaom-av1"
    if not hardware_encode and settings.codec == "av1":
        software_av1 = software_av1_encoder(ffmpeg)
        codec_evidence["encode"]["encoder"] = software_av1
    version = (
        run_tool([str(ffmpeg), "-version"]).stdout.decode("utf-8", errors="replace").splitlines()[0]
    )
    for _ in range(3):
        runtime.run(np.full(runtime.input_shape, 0.5, np.float32))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.partial{output.suffix}")
    startup_ms = (perf_counter() - initialization) * 1000
    if initialized:
        initialized(
            {
                "input": info.public(),
                "source_frames": source_frames,
                "model": runtime.spec.identifier,
                "backend": runtime.label,
                "codec_evidence": codec_evidence,
                "codec": settings.codec,
                "output_resolution": [info.width * 2, info.height * 2],
            }
        )
    phases: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=8192))
    whole_phases: defaultdict[str, LatencyHistogram] = defaultdict(LatencyHistogram)
    completed, decoder, encoder = 0, None, None
    done = threading.Event()
    activity = [perf_counter()]
    timed_out = threading.Event()
    sustained = SustainedStatistics()
    resources: deque[dict] = deque(maxlen=7200)
    previous_completion = [0.0]

    def stop_children() -> None:
        for child in (decoder, encoder):
            if child and child.process.poll() is None:
                child.process.kill()

    def enhance(raw: bytes) -> tuple[bytes, dict[str, float]]:
        if planar:
            return planar.process(raw)
        image = Image.frombytes("RGB", (info.width, info.height), raw)
        before = perf_counter()
        prepared = preprocess(image, runtime.spec)
        timings = {"preprocessing_ms": (perf_counter() - before) * 1000}
        y, _ = infer_y(prepared, runtime, timings)
        before = perf_counter()
        data = postprocess(y, prepared).tobytes()
        timings["postprocessing_ms"] = (perf_counter() - before) * 1000
        return data, timings

    def record(count: int, timings: dict[str, float]) -> None:
        nonlocal completed
        timings["frame_total_ms"] = (
            timings["enhancement_ms"]
            + timings["decoder_pipe_wait_ms"]
            + timings["encoder_pipe_wait_ms"]
        )
        for key, value in timings.items():
            phases[key].append(value)
            whole_phases[key].record(value)
        completed = count
        elapsed = perf_counter() - started
        timings["completion_interval_ms"] = (elapsed - previous_completion[0]) * 1000
        phases["completion_interval_ms"].append(timings["completion_interval_ms"])
        whole_phases["completion_interval_ms"].record(timings["completion_interval_ms"])
        previous_completion[0] = elapsed
        sustained.record(elapsed)
        activity[0] = perf_counter()
        if progress:
            progress(completed)

    def watchdog() -> None:
        # Killing a stalled child unblocks an IO wait; never leave a broken pipe hanging forever.
        while not done.wait(1):
            readings = {"python": memory_usage()}
            for label, child in (("decoder", decoder), ("encoder", encoder)):
                if child:
                    readings[label] = memory_usage(child.process.pid)
            resources.append({"elapsed_seconds": perf_counter() - started, "processes": readings})
            if perf_counter() - activity[0] > 60:
                timed_out.set()
                stop_children()
                return

    watcher = threading.Thread(target=watchdog, daemon=True)
    started, cpu_started = perf_counter(), process_time()
    try:
        decoder = PipeProcess(
            ffmpeg, decode_args(source, hardware_decode, pixel_format=settings.frame_format)
        )
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
                settings.frame_format,
                software_av1,
            ),
            input_pipe=True,
        )
        watcher.start()
        assert encoder.process.stdin is not None
        queue_peaks = process_stream(
            frames(decoder, info, settings.frame_format),
            enhance,
            lambda data: write_frame(encoder.process.stdin, data),
            record,
            settings.pipeline_depth,
            stop_children,
        )
        encoder.process.stdin.close()
        decoder.finish()
        encoder.finish()
        processing_seconds = perf_counter() - started
        cpu_seconds = process_time() - cpu_started
        done.set()
        watcher.join(timeout=5)
        if timed_out.is_set():
            raise SRException("Video pipeline stalled for more than 60 seconds.")
        if not completed:
            raise SRException("Input video produced no frames.")
        tile_count = (
            (info.width + runtime.spec.core - 1)
            // runtime.spec.core
            * ((info.height + runtime.spec.height - 1) // runtime.spec.height)
        )
        neural_runs = runtime.run_calls - 3  # Successful calls, excluding explicit warmups.
        temporal_state = (
            planar.evidence() if runtime.spec.input_channels in {2, 6} and planar else None
        )
        if temporal_state is not None and temporal_state["frames_processed"] != completed:
            raise SRException("Temporal state/frame accounting mismatch.")
        if neural_runs != completed * tile_count:
            raise SRException("Neural tile-call count does not match every decoded frame.")
        if completed != source_frames or (info.frames is not None and completed != info.frames):
            raise SRException("Decoded frame count does not match input metadata.")
        if abs(completed / float(info.fps) - info.duration) > max(0.1, 2 / float(info.fps)):
            raise SRException("Decoded frames do not match input duration; VFR or damaged input.")
        if hardware_decode:
            codec_evidence["decode"] = decode_evidence(decoder.evidence_log, True)
        if hardware_encode:
            codec_evidence["encode"] = encode_evidence(encoder.evidence_log) | {
                "initial_probe": codec_evidence["encode"]
            }
        # Count the final encoded frames: passthrough must not skip or duplicate frames.
        validation_started = perf_counter()
        try:
            actual = probe(temporary, ffmpeg)
            output_validation = inspect_timeline(temporary, ffmpeg, actual, settings.verify_full)
            encoded_frames = output_validation["count"]
        except SRException as exc:
            raise SRException(
                "Encoded output failed validation; no output published. "
                "Try another codec or explicit software encode. " + str(exc)
            ) from exc
        expected_size = (info.width * 2, info.height * 2)
        if (
            (actual.frames is not None and actual.frames != completed)
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
        audio_validation = None
        if settings.audio == "copy" and info.audio:
            if (
                actual.audio_codec != info.audio_codec
                or actual.audio_channels != info.audio_channels
            ):
                raise SRException("Copied audio codec or channel count changed.")
            expected_offset = max(0, info.audio_start - info.video_start)
            actual_offset = actual.audio_start - actual.video_start
            # One AAC priming packet/Matroska millisecond rounding is allowed.
            if abs(actual_offset - expected_offset) > 0.03:
                raise SRException("Copied audio offset changed; no output published.")
            audio_validation = {
                "expected_offset_seconds": expected_offset,
                "actual_offset_seconds": actual_offset,
                "tolerance_seconds": 0.03,
                "scope": (
                    "first audio stream codec/channels/start relative to first video frame; "
                    "leading audio trimmed"
                ),
            }
        actual = replace(actual, frames=encoded_frames)
        os.replace(temporary, output)
        validation_seconds = perf_counter() - validation_started
        from .suite import environment

        report_started = perf_counter()
        measured_environment = environment()
        ffmpeg_hash, input_hash = sha256(ffmpeg), sha256(source)
        report_seconds = perf_counter() - report_started
        inference_seconds = whole_phases["inference_ms"].total / 1000
        total_seconds = perf_counter() - initialization
        return {
            "schema_version": 5,
            "environment": measured_environment,
            "environment_start": environment_start,
            "ffmpeg_version": version,
            "ffmpeg_sha256": ffmpeg_hash,
            "input_sha256": input_hash,
            "model": runtime.manifest["name"],
            "model_sha256": runtime.manifest["sha256"],
            "model_precision": runtime.spec.precision,
            "tile_core": [runtime.spec.core, runtime.spec.height],
            "tile_halo": runtime.spec.halo,
            "tile_count_per_frame": tile_count,
            "neural_tile_runs": neural_runs,
            "backend": runtime.backend,
            "execution_evidence": runtime.evidence,
            "temporal_state": temporal_state,
            "codec_evidence": codec_evidence,
            "codec": settings.codec,
            "bitrate": settings.bitrate if hardware_encode else None,
            "frame_format": settings.frame_format,
            "neural_strength": settings.neural_strength,
            "preset": settings.preset,
            "pipeline_depth": settings.pipeline_depth,
            "observed_queue_peaks": queue_peaks,
            "frame_latency_scope": (
                "raw read start through encoder pipe submission; excludes encoder completion"
            ),
            "npu_performance_requested": settings.npu_performance,
            "software_quality_crf": settings.quality if not hardware_encode else None,
            "audio": settings.audio,
            "audio_validation": audio_validation,
            "input": info.public(),
            "output": actual.public(),
            "frames_processed": completed,
            "input_timestamps_validated": True,
            "timeline_validation": {"input": input_validation, "output": output_validation},
            "verify_full": settings.verify_full,
            "output_timestamps_validated": True,
            "hardware_rate_control": "u_vbr / camera_record" if hardware_encode else None,
            "dropped_frames": 0,
            "startup_ms": startup_ms,
            "neural_startup_ms": runtime.startup_ms,
            "warmup_tensor_runs": 3,
            "processing_seconds": processing_seconds,
            "output_validation_seconds": validation_seconds,
            "report_preparation_seconds": report_seconds,
            "total_seconds": total_seconds,
            "total_seconds_scope": "process_video function; excludes CLI import, printing and JSON",
            "whole_command_fps": completed / total_seconds,
            "whole_command_real_time_factor": total_seconds / (completed / float(info.fps)),
            "end_to_end_fps": completed / processing_seconds,
            "real_time_factor": processing_seconds / (completed / float(info.fps)),
            "process_cpu_seconds": cpu_seconds,
            "process_memory": memory_usage(),
            "resource_samples": list(resources),
            "resources": resource_summary(list(resources)),
            "resource_sample_scope": (
                "1s Windows process CPU/working set; last 7200 samples; no process IDs"
            ),
            "sustained": sustained.report(processing_seconds),
            "phase_statistics": {key: _statistics(values) for key, values in phases.items()},
            "whole_run_phase_statistics": {
                key: values.report() for key, values in whole_phases.items()
            },
            "phase_sample_scope": f"last {min(completed, 8192)} frames; bounded storage",
            "inference_only_theoretical_fps": completed / inference_seconds,
            "timing_scope": (
                "decode process start through encoder flush; excludes initialization, "
                "output validation and hashing"
            ),
            "pipe_timings": "read/write blocking time, not isolated hardware codec execution time",
        }
    except (BrokenPipeError, OSError) as exc:
        if encoder:
            encoder.thread.join(timeout=1)
        tail = encoder.failure_log if encoder else ""
        raise SRException(f"Video pipe failed: {exc}\n{tail}") from exc
    finally:
        done.set()
        for child in (decoder, encoder):
            if child:
                child.close()
        if watcher.is_alive():
            watcher.join(timeout=2)
        temporary.unlink(missing_ok=True)
