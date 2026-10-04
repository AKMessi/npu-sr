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
from .model import MODELS, ModelSpec

if TYPE_CHECKING:
    from .runtime import Runtime

MAX_PIXELS = 16_000_000
FORMATS = {"PNG", "JPEG", "WEBP"}


@dataclass
class LuminanceInput:
    size: tuple[int, int]
    padded_y: np.ndarray
    spec: ModelSpec = MODELS["espcn-x2"]
    buffer: np.ndarray | None = None

    @property
    def tile_count(self) -> int:
        width, height = self.size
        core = self.spec.core
        return ((width + core - 1) // core) * ((height + self.spec.height - 1) // self.spec.height)

    def tiles(self) -> Iterator[tuple[int, int, np.ndarray]]:
        width, height = self.size
        core = self.spec.core
        tile_h, tile_w = self.spec.input_shape[2:]
        # One contiguous input buffer per image, reused for each synchronous ORT call.
        buffer = (
            self.buffer if self.buffer is not None else np.empty(self.spec.input_shape, np.float32)
        )
        for top in range(0, height, self.spec.height):
            for left in range(0, width, core):
                buffer[0, 0] = self.padded_y[top : top + tile_h, left : left + tile_w]
                yield left, top, buffer


@dataclass
class PreparedImage:
    rgb: Image.Image
    alpha: Image.Image | None
    padded_y: np.ndarray
    cb: np.ndarray
    cr: np.ndarray
    spec: ModelSpec = MODELS["espcn-x2"]

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.size

    @property
    def tile_count(self) -> int:
        return LuminanceInput(self.size, self.padded_y, self.spec).tile_count

    def tiles(self) -> Iterator[tuple[int, int, np.ndarray]]:
        return LuminanceInput(self.size, self.padded_y, self.spec).tiles()


def load_image(path: Path) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in FORMATS:
                    raise SRException("Unsupported input format. Use PNG, JPEG, or WebP.")
                if image.width * image.height > MAX_PIXELS:
                    raise SRException(f"Image exceeds the limit of {MAX_PIXELS:,} pixels.")
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


def preprocess(image: Image.Image, spec: ModelSpec = MODELS["espcn-x2"]) -> PreparedImage:
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    red, green, blue = pixels.transpose(2, 0, 1)
    y = red * 0.299 + green * 0.587 + blue * 0.114
    cb, cr = (blue - y) * 0.564 + 0.5, (red - y) * 0.713 + 0.5
    height, width = y.shape
    # Repeat edge pixels only at the image boundary, not at internal tile boundaries.
    bottom = spec.halo + (-height % spec.height)
    right = spec.halo + (-width % spec.core)
    padded = np.pad(y, ((spec.halo, bottom), (spec.halo, right)), mode="edge")
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    return PreparedImage(rgb, alpha, padded, cb, cr, spec)


def resize_plane(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(plane.astype(np.float32)).resize(size, Image.Resampling.BICUBIC),
        dtype=np.float32,
    )


def postprocess(y: np.ndarray, image: PreparedImage) -> Image.Image:
    width, height = image.size
    size = (width * image.spec.scale, height * image.spec.scale)
    if y.shape != (size[1], size[0]) or not np.isfinite(y).all():
        raise SRException("Invalid luminance output shape or values.")
    cb = (resize_plane(image.cb, size) if image.spec.scale != 1 else image.cb) - 0.5
    cr = (resize_plane(image.cr, size) if image.spec.scale != 1 else image.cr) - 0.5
    red = y + cr / 0.713
    blue = y + cb / 0.564
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    # Quantize each plane in place; avoid several full-size float RGB temporaries.
    pixels = np.empty((size[1], size[0], 3), dtype=np.uint8)
    for index, plane in enumerate((red, green, blue)):
        np.clip(plane, 0, 1, out=plane)
        plane *= 255
        np.rint(plane, out=plane)
        pixels[:, :, index] = plane
    result = Image.fromarray(pixels)
    if image.alpha is not None:
        result.putalpha(image.alpha.resize(size, Image.Resampling.BICUBIC))
    return result


def infer_y(
    image: PreparedImage | LuminanceInput,
    runtime: "Runtime",
    timings: dict[str, float] | None = None,
    output: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    width, height = image.size
    core, scale = image.spec.core, image.spec.scale
    if hasattr(runtime, "spec") and runtime.spec != image.spec:
        raise SRException("Image tile contract does not match the selected model.")
    if output is None:
        output = np.empty((height * scale, width * scale), dtype=np.float32)
    elif output.shape != (height * scale, width * scale) or output.dtype != np.float32:
        raise SRException("Invalid reusable luminance output buffer.")
    inference_seconds = 0.0
    extraction_seconds, stitching_seconds = 0.0, 0.0
    iterator = iter(image.tiles())
    while True:
        started = perf_counter()
        try:
            left, top, tensor = next(iterator)
        except StopIteration:
            break
        extraction_seconds += perf_counter() - started
        started = perf_counter()
        tile = runtime.run(tensor)[0, 0]
        inference_seconds += perf_counter() - started
        started = perf_counter()
        kept_w = min(core, width - left) * scale
        kept_h = min(image.spec.height, height - top) * scale
        border = image.spec.halo * scale
        output[top * scale : top * scale + kept_h, left * scale : left * scale + kept_w] = tile[
            border : border + kept_h, border : border + kept_w
        ]
        stitching_seconds += perf_counter() - started
    if timings is not None:
        timings.update(
            tile_extraction_ms=extraction_seconds * 1000,
            inference_ms=inference_seconds * 1000,
            stitching_ms=stitching_seconds * 1000,
        )
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
