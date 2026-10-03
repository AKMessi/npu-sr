"""Deterministic luminance SR, bicubic chroma/alpha, and fixed-shape tile stitching."""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import SRException
from .model import CORE, HALO, TILE

if TYPE_CHECKING:
    from .runtime import Runtime

MAX_PIXELS = 16_000_000
FORMATS = {"PNG", "JPEG", "WEBP"}


@dataclass
class PreparedImage:
    rgb: Image.Image
    alpha: Image.Image | None
    padded_y: np.ndarray
    cb: np.ndarray
    cr: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.size

    @property
    def tile_count(self) -> int:
        width, height = self.size
        return ((width + CORE - 1) // CORE) * ((height + CORE - 1) // CORE)

    def tiles(self) -> Iterator[tuple[int, int, np.ndarray]]:
        width, height = self.size
        for top in range(0, height, CORE):
            for left in range(0, width, CORE):
                tensor = self.padded_y[top : top + TILE, left : left + TILE][None, None]
                yield left, top, np.ascontiguousarray(tensor, dtype=np.float32)


def load_image(path: Path) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in FORMATS:
                    raise SRException("Unsupported input format. Use PNG, JPEG, or WebP.")
                if image.width * image.height > MAX_PIXELS:
                    raise SRException(f"Image exceeds the v0.1 limit of {MAX_PIXELS:,} pixels.")
                if getattr(image, "n_frames", 1) > 1:
                    raise SRException("Animated images are unsupported. Supply one still image.")
                oriented = ImageOps.exif_transpose(image)
                has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
                return oriented.convert("RGBA" if has_alpha else "RGB")
    except SRException:
        raise
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise SRException(f"Cannot load image '{path}': {exc}") from exc


def preprocess(image: Image.Image) -> PreparedImage:
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    red, green, blue = pixels.transpose(2, 0, 1)
    y = red * 0.299 + green * 0.587 + blue * 0.114
    cb, cr = (blue - y) * 0.564 + 0.5, (red - y) * 0.713 + 0.5
    height, width = y.shape
    # Repeat edge pixels only at the image boundary, not at internal tile boundaries.
    bottom = HALO + (-height % CORE)
    right = HALO + (-width % CORE)
    padded = np.pad(y, ((HALO, bottom), (HALO, right)), mode="edge")
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    return PreparedImage(rgb, alpha, padded, cb, cr)


def resize_plane(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(plane.astype(np.float32)).resize(size, Image.Resampling.BICUBIC),
        dtype=np.float32,
    )


def postprocess(y: np.ndarray, image: PreparedImage) -> Image.Image:
    width, height = image.size
    size = (width * 2, height * 2)
    if y.shape != (size[1], size[0]) or not np.isfinite(y).all():
        raise SRException("Invalid luminance output shape or values.")
    cb, cr = resize_plane(image.cb, size) - 0.5, resize_plane(image.cr, size) - 0.5
    red = y + cr / 0.713
    blue = y + cb / 0.564
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    pixels = np.rint(np.clip(np.stack([red, green, blue], axis=-1), 0, 1) * 255).astype(np.uint8)
    result = Image.fromarray(pixels)
    if image.alpha is not None:
        result.putalpha(image.alpha.resize(size, Image.Resampling.BICUBIC))
    return result


def infer_y(image: PreparedImage, runtime: "Runtime") -> tuple[np.ndarray, float]:
    width, height = image.size
    output = np.empty((height * 2, width * 2), dtype=np.float32)
    inference_seconds = 0.0
    for left, top, tensor in image.tiles():
        started = perf_counter()
        tile = runtime.run(tensor)[0, 0]
        inference_seconds += perf_counter() - started
        kept_w, kept_h = min(CORE, width - left) * 2, min(CORE, height - top) * 2
        border = HALO * 2
        output[top * 2 : top * 2 + kept_h, left * 2 : left * 2 + kept_w] = tile[
            border : border + kept_h, border : border + kept_w
        ]
    return output, inference_seconds * 1000


def save_image(image: Image.Image, path: Path) -> None:
    extension = path.suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise SRException("Unsupported output extension. Use .png, .jpg, .jpeg, or .webp.")
    if extension in {".jpg", ".jpeg"} and "A" in image.getbands():
        # JPEG has no alpha: flatten onto white rather than discard hidden transparency.
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        image = background
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, **({"quality": 95} if extension != ".png" else {}))
    except (OSError, ValueError) as exc:
        raise SRException(f"Cannot save output '{path}': {exc}") from exc
