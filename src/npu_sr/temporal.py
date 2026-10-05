"""Bounded previous-frame state guards for experimental temporal enhancement."""

from time import perf_counter

import numpy as np

from .errors import SRException
from .ffmpeg import VideoInfo
from .planar import NV12Enhancer
from .runtime import Runtime


class SceneDetector:
    """Conservative coarse-Y reset heuristic, not an optical-flow or shot classifier.

    Reset on initial input, black transitions, large histogram/brightness changes,
    or a large decorrelated change. Fast motion can produce a false reset; that
    drops history rather than contaminating the new frame. Fades retain adjacent
    history until black, where it is reset. State is at most 32x32 float values.
    """

    def __init__(self) -> None:
        self.previous: np.ndarray | None = None
        self.resets = 0

    def observe(self, luminance: np.ndarray) -> str | None:
        if luminance.ndim != 2 or min(luminance.shape) < 1:
            raise SRException("Scene detection requires a nonempty luminance plane.")
        y = np.linspace(0, luminance.shape[0] - 1, min(32, luminance.shape[0]), dtype=int)
        x = np.linspace(0, luminance.shape[1] - 1, min(32, luminance.shape[1]), dtype=int)
        sample = luminance[np.ix_(y, x)].astype(np.float32)
        if not np.isfinite(sample).all() or sample.min() < 0 or sample.max() > 1:
            raise SRException("Scene detection expects finite normalized luminance.")
        previous = self.previous
        reason = None
        if previous is None or previous.shape != sample.shape:
            reason = "initial" if previous is None else "geometry-change"
        else:
            black, was_black = float(sample.mean()) < 0.015, float(previous.mean()) < 0.015
            if black != was_black:
                reason = "black-transition"
            else:
                difference = float(np.abs(sample - previous).mean())
                a, b = sample - sample.mean(), previous - previous.mean()
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                correlation = float(np.sum(a * b)) / denominator if denominator > 1e-8 else 1
                histograms = [np.histogram(v, bins=16, range=(0, 1))[0] for v in (sample, previous)]
                histogram_change = float(np.abs(histograms[0] - histograms[1]).sum()) / (
                    2 * sample.size
                )
                if (
                    difference > 0.45
                    or (difference > 0.18 and histogram_change > 0.25)
                    or (difference > 0.12 and correlation < 0.1)
                ):
                    reason = "abrupt-change"
        self.previous = sample
        if reason is not None:
            self.resets += 1
        return reason


class TemporalNV12Enhancer(NV12Enhancer):
    """Experimental two-frame graph; one owned LR history, every frame still inferred.

    First frames and scene changes use current/current rather than stale history.
    The strict Runtime receives both planes directly through the shared tile
    iterator. No second session, CPU neural correction or per-frame model load.
    """

    def __init__(self, info: VideoInfo, runtime: Runtime, strength: float = 1.0) -> None:
        if runtime.spec.input_channels not in {2, 6} or runtime.spec.halo < 7 or strength != 1:
            raise SRException(
                "Temporal NV12 requires a two/six-plane graph, halo >=7 and strength 1."
            )
        super().__init__(info, runtime, strength)
        self.previous = np.empty_like(self.input.padded_y)
        self.input.previous_y = self.previous
        self.previous_features = (
            np.empty((4, *self.previous.shape), np.float32)
            if runtime.spec.input_channels == 6
            else None
        )
        self.input.previous_features = self.previous_features
        self.scene = SceneDetector()
        self.reset_counts: dict[str, int] = {}
        self.frames = 0

    def prepare_history(self) -> None:
        halo = self.runtime.spec.halo
        core = self.input.padded_y[halo : halo + self.info.height, halo : halo + self.info.width]
        if reason := self.scene.observe(core):
            self.previous[:] = self.input.padded_y
            if self.previous_features is not None:
                self.previous_features[:] = self.input.padded_y
            self.reset_counts[reason] = self.reset_counts.get(reason, 0) + 1

    def process(self, raw: bytes) -> tuple[bytes, dict[str, float]]:
        result, timings = super().process(raw)
        started = perf_counter()
        self.previous[:] = self.input.padded_y
        if self.previous_features is not None:
            halo = self.runtime.spec.halo
            height, width = self.info.height, self.info.width
            for phase, plane in enumerate(self.previous_features):
                core = plane[halo : halo + height, halo : halo + width]
                np.subtract(self.y[phase // 2 :: 2, phase % 2 :: 2], 16, out=core)
                core *= 1 / 219
                rows = plane[halo : halo + height]
                rows[:, :halo] = core[:, :1]
                rows[:, halo + width :] = core[:, -1:]
                plane[:halo] = plane[halo]
                plane[halo + height :] = plane[halo + height - 1]
        elapsed = (perf_counter() - started) * 1000
        timings["temporal_state_update_ms"] = elapsed
        timings["postprocessing_ms"] += elapsed
        self.frames += 1
        return result, timings

    def evidence(self) -> dict:
        return {
            "mode": "experimental recurrent subpixel features, not the default"
            if self.previous_features is not None
            else "experimental two-frame neural residual, not the default",
            "history_frames": 1,
            "history_bytes": self.previous.nbytes,
            "feature_state_bytes": self.previous_features.nbytes
            if self.previous_features is not None
            else 0,
            "scene_state_bytes": self.scene.previous.nbytes
            if self.scene.previous is not None
            else 0,
            "scene_reset_counts": dict(self.reset_counts),
            "frames_processed": self.frames,
            "scene_reset_method": "coarse luminance heuristic; fast motion may reset history",
        }
