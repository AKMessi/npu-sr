"""Reusable limited-range NV12 buffers: enhance native Y and resize native chroma."""

from time import perf_counter

import numpy as np
from PIL import Image

from .errors import SRException
from .ffmpeg import VideoInfo
from .image import LuminanceInput, infer_y
from .runtime import Runtime


class NV12Enhancer:
    """One video's buffers. Returned bytes own their data, including in a queue.

    The network receives native video luminance, normalized from 16–235 to 0–1.
    Unlike the RGB path, this does not recompute Rec.601 Y from RGB. Quality must
    be measured for this delivered path. Unknown YUV range is assumed limited;
    explicitly full-range sources require the RGB path.
    """

    def __init__(self, info: VideoInfo, runtime: Runtime):
        if info.color_range not in {"unknown", "tv"}:
            raise SRException(
                "NV12 enhancement requires limited-range SDR; use --frame-format rgb24."
            )
        self.info, self.runtime = info, runtime
        spec = runtime.spec
        height = info.height + (-info.height % spec.height) + 2 * spec.halo
        width = info.width + (-info.width % spec.core) + 2 * spec.halo
        self.input = LuminanceInput(
            (info.width, info.height),
            np.empty((height, width), np.float32),
            spec,
            np.empty(spec.input_shape, np.float32),
        )
        self.y = np.empty((info.height * 2, info.width * 2), np.float32)
        self.output = np.empty(info.width * info.height * 6, np.uint8)

    def process(self, raw: bytes) -> tuple[bytes, dict[str, float]]:
        info, spec = self.info, self.runtime.spec
        if len(raw) != info.bytes_for("nv12"):
            raise SRException("Invalid NV12 frame byte count.")
        started = perf_counter()
        pixels = np.frombuffer(raw, np.uint8)
        count = info.width * info.height
        source_y = pixels[:count].reshape(info.height, info.width)
        uv = pixels[count:].reshape(info.height // 2, info.width // 2, 2)
        padded, halo = self.input.padded_y, spec.halo
        core = padded[halo : halo + info.height, halo : halo + info.width]
        np.subtract(source_y, 16, out=core, dtype=np.float32)
        core *= 1 / 219
        np.clip(core, 0, 1, out=core)
        rows = padded[halo : halo + info.height]
        rows[:, :halo] = core[:, :1]
        rows[:, halo + info.width :] = core[:, -1:]
        padded[:halo] = padded[halo]
        padded[halo + info.height :] = padded[halo + info.height - 1]
        timings = {"preprocessing_ms": (perf_counter() - started) * 1000}
        infer_y(self.input, self.runtime, timings, self.y)
        started = perf_counter()
        if not np.isfinite(self.y).all():
            raise SRException("Invalid planar luminance output.")
        self.y *= 219
        self.y += 16
        np.clip(self.y, 16, 235, out=self.y)
        np.rint(self.y, out=self.y)
        self.output[: count * 4] = self.y.ravel()
        target_uv = self.output[count * 4 :].reshape(info.height, info.width, 2)
        for channel in range(2):
            plane = Image.fromarray(uv[:, :, channel]).resize(
                (info.width, info.height), Image.Resampling.BICUBIC
            )
            target_uv[:, :, channel] = np.asarray(plane)
        result = self.output.tobytes()
        timings["postprocessing_ms"] = (perf_counter() - started) * 1000
        return result, timings
