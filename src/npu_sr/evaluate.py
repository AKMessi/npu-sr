"""Quality measurements require a matching high-resolution reference."""

import math

import numpy as np
from PIL import Image

from .errors import SRException


def psnr(result: Image.Image, reference: Image.Image) -> float:
    """RGB PSNR over all pixels; images must have exactly matching dimensions."""
    if result.size != reference.size:
        raise SRException(
            "Ground truth must be exactly 2x the input dimensions (after EXIF rotation)."
        )
    if "A" in result.getbands() or "A" in reference.getbands():
        raise SRException("PSNR evaluation requires opaque images; flatten alpha first.")
    difference = np.asarray(result.convert("RGB"), dtype=np.float64) - np.asarray(
        reference.convert("RGB"), dtype=np.float64
    )
    mse = float(np.mean(difference**2))
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def luminance(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    return pixels @ np.array([0.299, 0.587, 0.114])


def _gaussian_valid(array: np.ndarray) -> np.ndarray:
    """Separable 11x11 Gaussian (sigma 1.5), valid windows only."""
    kernel = np.exp(-(np.arange(-5, 6, dtype=np.float64) ** 2) / (2 * 1.5**2))
    kernel /= kernel.sum()
    horizontal = sum(kernel[i] * array[:, i : array.shape[1] - 10 + i] for i in range(11))
    return sum(kernel[i] * horizontal[i : array.shape[0] - 10 + i] for i in range(11))


def ssim_arrays(result: np.ndarray, reference: np.ndarray, data_range: float = 255) -> float:
    """Wang et al. SSIM: Gaussian weights, population covariance, valid windows."""
    x, y = np.asarray(result, np.float64), np.asarray(reference, np.float64)
    if x.shape != y.shape or x.ndim != 2 or min(x.shape) < 11:
        raise SRException("SSIM requires matching 2D arrays at least 11x11.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or data_range <= 0:
        raise SRException("SSIM requires finite values and a positive data range.")
    mx, my = _gaussian_valid(x), _gaussian_valid(y)
    vx = np.maximum(_gaussian_valid(x * x) - mx * mx, 0)
    vy = np.maximum(_gaussian_valid(y * y) - my * my, 0)
    covariance = _gaussian_valid(x * y) - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mx * my + c1) * (2 * covariance + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2)
    )
    return float(score.mean())


def quality_metrics(result: Image.Image, reference: Image.Image, border: int = 2) -> dict:
    """Full-range Rec.601 luminance metrics with an explicit HR boundary shave."""
    if result.size != reference.size or border < 0:
        raise SRException("Quality evaluation requires matching dimensions and a valid border.")
    if "A" in result.getbands() or "A" in reference.getbands():
        raise SRException("Quality evaluation requires opaque images.")
    x, y = luminance(result), luminance(reference)
    if border:
        x, y = x[border:-border, border:-border], y[border:-border, border:-border]
    if min(x.shape, default=0) < 11:
        raise SRException("Image too small after quality evaluation border crop.")
    mse = float(np.mean((x - y) ** 2))
    # Null represents a perfect match in machine-readable reports (JSON has no infinity).
    return {
        "psnr_y_db": None if mse == 0 else 10 * math.log10(255**2 / mse),
        "ssim_y": ssim_arrays(x, y),
        "border_hr_pixels": border,
    }
