"""Paired native-Y quality evaluation with fixed film ROI and explicit temporal scope."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import SRException
from .evaluate import ssim_arrays
from .ffmpeg import PipeProcess, decode_args, probe, read_frame
from .model import sha256
from .video_benchmark import measure_vmaf


def validate_roi(roi: tuple[int, int, int, int], width: int, height: int) -> None:
    x, y, w, h = roi
    if min(x, y) < 0 or min(w, h) < 16 or x + w > width or y + h > height:
        raise SRException("Quality ROI is invalid or exceeds the decoded frame.")


def full_temporal_difference(current: np.ndarray, previous: np.ndarray) -> float:
    """Native-Y reconstruction residual change /255, a project diagnostic, not flow.

    Inputs are signed coded-Y errors in [-255,255]. Promote before subtraction
    so opposite extrema differ by 510, rather than wrapping through uint8.
    Constant reconstruction bias has zero temporal error; spatial metrics are
    still required. This does not establish perceptual temporal quality alone.
    """
    if current.shape != previous.shape or current.ndim != 2 or not current.size:
        raise SRException("Temporal residual planes must have equal nonempty geometry.")
    if any(
        plane.dtype != np.int16 or plane.min() < -255 or plane.max() > 255
        for plane in (current, previous)
    ):
        raise SRException("Temporal residuals must be signed coded-Y errors in [-255,255].")
    return float(np.abs(current - previous).mean()) / 255


def evaluate_native_video(
    output: Path,
    reference: Path,
    ffmpeg: Path,
    roi: tuple[int, int, int, int] = (0, 0, 1920, 1080),
    stride: int = 3,
) -> dict:
    """Use decoded coded-Y peak 255, not an implicit RGB/Rec.601 conversion.

    PSNR is computed from mean sampled MSE per clip so a perfect black frame
    cannot introduce infinity into an average. SSIM uses the existing independent
    Gaussian implementation. VMAF uses every frame. This is not the v0.4 protocol.
    """
    if stride < 1:
        raise SRException("Quality stride must be positive.")
    actual, truth = probe(output, ffmpeg), probe(reference, ffmpeg)
    if (actual.width, actual.height, actual.fps) != (truth.width, truth.height, truth.fps):
        raise SRException("Native video quality requires aligned dimensions and framerates.")
    if abs(actual.duration - truth.duration) > 2 / float(actual.fps):
        raise SRException("Native video quality requires aligned durations.")
    validate_roi(roi, actual.width, actual.height)
    x, y, width, height = roi
    children = []
    samples = []
    previous = None
    previous_full = None
    full_temporal_sum = 0.0
    temporal_sum, count = 0.0, 0
    try:
        for path in (output, reference):
            children.append(PipeProcess(ffmpeg, decode_args(path, False, pixel_format="nv12")))
        while True:
            raw = [read_frame(child.process.stdout, actual.bytes_for("nv12")) for child in children]
            if raw[0] is None or raw[1] is None:
                if (raw[0] is None) != (raw[1] is None):
                    raise SRException("Quality comparison has unequal decoded frame counts.")
                break
            planes = [
                np.frombuffer(data, np.uint8, actual.width * actual.height).reshape(
                    actual.height, actual.width
                )[y : y + height, x : x + width]
                for data in raw
            ]
            if count % stride == 0:
                a, b = [plane[2:-2, 2:-2].astype(np.float64) for plane in planes]
                mse = float(np.square(a - b).mean())
                samples.append({"frame": count, "mse_y": mse, "ssim_y": ssim_arrays(a, b)})
            small = [
                np.asarray(
                    Image.fromarray(plane).resize((256, 144), Image.Resampling.BILINEAR), np.float32
                )
                for plane in planes
            ]
            residual = small[0] - small[1]
            if previous is not None:
                temporal_sum += float(np.abs(residual - previous).mean()) / 255
            previous = residual
            full_error = planes[0][2:-2, 2:-2].astype(np.int16) - planes[1][2:-2, 2:-2].astype(
                np.int16
            )
            if previous_full is not None:
                full_temporal_sum += full_temporal_difference(full_error, previous_full)
            previous_full = full_error
            count += 1
        for child in children:
            child.finish()
    finally:
        for child in children:
            child.close()
    if not samples:
        raise SRException("Quality comparison produced no frames.")
    mean_mse = float(np.mean([row["mse_y"] for row in samples]))
    return {
        "output_sha256": sha256(output),
        "reference_sha256": sha256(reference),
        "decoded_frames": count,
        "metric_roi": list(roi),
        "psnr_y_db": None if mean_mse == 0 else 10 * math.log10(255**2 / mean_mse),
        "ssim_y": float(np.mean([row["ssim_y"] for row in samples])),
        "mean_mse_y": mean_mse,
        "sample_stride": stride,
        "samples": samples,
        "metric_definition": "native decoded 8-bit coded Y, peak255, shave2; "
        "clip PSNR from mean MSE",
        "temporal_difference_error": temporal_sum / (count - 1) if count > 1 else None,
        "temporal_difference_error_full_resolution": (
            full_temporal_sum / (count - 1) if count > 1 else None
        ),
        "full_temporal_definition": (
            "mean absolute consecutive native-Y reconstruction residual change /255; "
            "every frame, declared ROI with 2px border; no optical flow; project diagnostic, "
            "not a standard perceptual metric; retain spatial and coarse temporal metrics"
        ),
        "temporal_definition": "project diagnostic: absolute consecutive residual change /255; "
        "ROI resized to 256x144, no motion compensation, every frame; may reward smoothing",
        "vmaf": measure_vmaf(output, reference, ffmpeg, 1, count, roi=roi),
        "vmaf_neg": measure_vmaf(
            output, reference, ffmpeg, 1, count, roi=roi, model="vmaf_v0.6.1neg"
        ),
    }
