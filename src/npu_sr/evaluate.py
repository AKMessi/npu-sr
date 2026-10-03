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
