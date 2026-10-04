"""Measured video trials and aligned decoded-frame quality comparisons."""

import json
import tempfile
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .benchmark import save_report
from .errors import SRException
from .evaluate import luminance, quality_metrics
from .ffmpeg import PipeProcess, decode_args, probe, read_frame, tool_path
from .model import sha256
from .video import VideoSettings, process_video


def benchmark_video(
    source: Path,
    directory: Path,
    settings: VideoSettings,
    trials: int,
    checkpoint: Path | None = None,
) -> dict:
    if not 1 <= trials <= 20:
        raise SRException("Video benchmark trials must be 1–20.")
    directory.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 3, "complete": False, "trials": []}
    for trial in range(trials):
        output = directory / f"trial-{trial + 1}.mp4"
        result = process_video(source, output, settings)
        result["trial"] = trial + 1
        report["trials"].append(result)
        if checkpoint:
            save_report(report, checkpoint)
        print(
            f"Trial {trial + 1}: {result['end_to_end_fps']:.1f} FPS, "
            f"RTF {result['real_time_factor']:.2f}"
        )
    report.update(
        complete=True,
        median_trial_end_to_end_fps=float(
            np.median([r["end_to_end_fps"] for r in report["trials"]])
        ),
        median_trial_real_time_factor=float(
            np.median([r["real_time_factor"] for r in report["trials"]])
        ),
        aggregation="median across complete measured trials; startup separate",
    )
    return report


def evaluate_video(
    output: Path, reference: Path, ffmpeg: Path | None = None, stride: int = 12, vmaf: bool = False
) -> dict:
    """Compare delivered video against matching HR frames, with bounded memory.

    Temporal error change is an unregistered diagnostic, not a standard perceptual
    score: mean absolute change of reconstruction residual at 256x144, without
    motion compensation. It may reward smoothing and includes source motion.
    """
    if stride < 1:
        raise SRException("Quality sampling stride must be positive.")
    ffmpeg = tool_path(ffmpeg)
    actual, expected = probe(output, ffmpeg), probe(reference, ffmpeg)
    if (actual.width, actual.height, actual.fps) != (expected.width, expected.height, expected.fps):
        raise SRException("Video quality requires matching HR dimensions and framerates.")
    if abs(actual.duration - expected.duration) > 2 / float(expected.fps):
        raise SRException("Video quality requires aligned durations.")
    result_process, reference_process = None, None
    samples: deque[dict] = deque(maxlen=8192)
    temporal_sum, temporal_count, previous = 0.0, 0, None
    count = 0
    try:
        result_process = PipeProcess(ffmpeg, decode_args(output, False))
        reference_process = PipeProcess(ffmpeg, decode_args(reference, False))
        while True:
            result_bytes = read_frame(result_process.process.stdout, actual.frame_bytes)
            reference_bytes = read_frame(reference_process.process.stdout, expected.frame_bytes)
            if result_bytes is None or reference_bytes is None:
                if result_bytes != reference_bytes:
                    raise SRException("Video quality frame counts differ.")
                break
            image = Image.frombytes("RGB", (actual.width, actual.height), result_bytes)
            truth = Image.frombytes("RGB", (expected.width, expected.height), reference_bytes)
            if count % stride == 0:
                samples.append({"frame": count, **quality_metrics(image, truth)})
            small = luminance(image.resize((256, 144), Image.Resampling.BILINEAR))
            residual = small - luminance(truth.resize((256, 144), Image.Resampling.BILINEAR))
            if previous is not None:
                temporal_sum += float(np.abs(residual - previous).mean()) / 255
                temporal_count += 1
            previous = residual
            count += 1
        result_process.finish()
        reference_process.finish()
    finally:
        for child in (result_process, reference_process):
            if child:
                child.close()
    if not samples:
        raise SRException("No aligned quality frames were produced.")
    psnr = [row["psnr_y_db"] for row in samples if row["psnr_y_db"] is not None]
    return {
        "schema_version": 3,
        "output_sha256": sha256(output),
        "reference_sha256": sha256(reference),
        "properties": expected.public(),
        "decoded_frames": count,
        "sample_stride": stride,
        "samples": list(samples),
        "sample_scope": "up to last 8192 sampled frames",
        "mean_psnr_y_db": float(np.mean(psnr)) if len(psnr) == len(samples) else None,
        "mean_ssim_y": float(np.mean([row["ssim_y"] for row in samples])),
        "temporal_residual_change": temporal_sum / temporal_count if temporal_count else None,
        "temporal_metric": (
            "mean absolute consecutive reconstruction-residual change /255 at 256x144; "
            "unregistered, no motion compensation"
        ),
        "vmaf": measure_vmaf(output, reference, ffmpeg, stride) if vmaf else "not measured",
    }


def measure_vmaf(output: Path, reference: Path, ffmpeg: Path, stride: int = 12) -> dict:
    """Use FFmpeg's native libvmaf with an explicit built-in 1080p model."""
    from .ffmpeg import run_tool

    with tempfile.TemporaryDirectory(prefix="npu-sr-vmaf-") as directory:
        run_tool(
            [
                str(ffmpeg.resolve()),
                "-v",
                "error",
                "-nostats",
                "-i",
                str(output.resolve()),
                "-i",
                str(reference.resolve()),
                "-filter_complex",
                "[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];"
                f"[d][r]libvmaf=model=version=vmaf_v0.6.1:n_threads=2:n_subsample={stride}:"
                "log_fmt=json:log_path=metrics.json",
                "-an",
                "-f",
                "null",
                "-",
            ],
            timeout=300,
            cwd=Path(directory),
        )
        data = json.loads((Path(directory) / "metrics.json").read_text())
        return {
            "mean": data["pooled_metrics"]["vmaf"]["mean"],
            "model": "vmaf_v0.6.1",
            "libvmaf_version": data["version"],
            "sample_stride": stride,
            "samples": len(data["frames"]),
        }


def decode_throughput(source: Path, ffmpeg: Path, hardware: bool) -> dict:
    """Standalone decoder including hardware download and raw pipe transfer."""
    info = probe(source, ffmpeg)
    started, count = perf_counter(), 0
    child = PipeProcess(ffmpeg, decode_args(source, hardware))
    try:
        while read_frame(child.process.stdout, info.frame_bytes) is not None:
            count += 1
        child.finish()
        seconds = perf_counter() - started
        from .ffmpeg import decode_evidence

        evidence = (
            decode_evidence(child.evidence_log, bool(count)) if hardware else {"method": "software"}
        )
        return {
            "frames": count,
            "seconds": seconds,
            "fps": count / seconds,
            "evidence": evidence,
            "scope": "standalone decode, CPU download and raw pipe read, including startup",
        }
    finally:
        child.close()


def encode_throughput(
    reference: Path, ffmpeg: Path, codec: str = "h264", bitrate: str = "8M"
) -> dict:
    """Standalone hardware encode from a bounded predecoded 16-frame cycle."""
    from .ffmpeg import codec_args, encode_evidence, write_frame

    info = probe(reference, ffmpeg)
    decoder = PipeProcess(ffmpeg, decode_args(reference, False, 16))
    buffers = []
    try:
        while (raw := read_frame(decoder.process.stdout, info.frame_bytes)) is not None:
            buffers.append(raw)
        decoder.finish()
    finally:
        decoder.close()
    if not buffers:
        raise SRException("Encoder benchmark reference has no frames.")
    count = info.frames or round(info.duration * float(info.fps))
    started = perf_counter()
    child = PipeProcess(
        ffmpeg,
        [
            "-loglevel",
            "verbose",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{info.width}x{info.height}",
            "-r",
            str(info.fps),
            "-i",
            "pipe:0",
            *codec_args(codec, True, bitrate, 20),
            "-f",
            "null",
            "-",
        ],
        input_pipe=True,
    )
    try:
        for index in range(count):
            write_frame(child.process.stdin, buffers[index % len(buffers)])
        child.process.stdin.close()
        child.finish()
        seconds = perf_counter() - started
        return {
            "frames": count,
            "seconds": seconds,
            "fps": count / seconds,
            "evidence": encode_evidence(child.evidence_log),
            "buffered_frames": len(buffers),
            "scope": (
                "standalone hardware encode with CPU RGB/NV12 conversion, pipe transfer "
                "and flush; cyclic 16-frame input, startup included"
            ),
        }
    finally:
        child.close()
