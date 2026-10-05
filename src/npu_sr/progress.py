"""Bounded console progress with measured submission rate, never model-only FPS."""

import sys
from time import perf_counter
from typing import TextIO


class VideoProgress:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr
        self.started: float | None = None
        self.last_print: float | None = None
        self.total: int | None = None
        self.fps: float | None = None

    def initialize(self, report: dict) -> None:
        from fractions import Fraction

        self.total = report["source_frames"]
        self.fps = float(Fraction(report["input"]["frame_rate"]))
        self.started = perf_counter()
        self.last_print = self.started
        properties = report["input"]
        w, h = properties["resolution"]
        ow, oh = report["output_resolution"]
        print(
            f"Input: {w} × {h} @ {properties['frame_rate']} FPS, "
            f"{properties['duration_seconds']:.1f} s, {properties['codec']}",
            file=self.stream,
        )
        print(f"Output: {ow} × {oh}, {report['codec'].upper()}", file=self.stream)
        print(f"Enhancement: {report['model']} / {report['backend']}", file=self.stream)
        for kind, evidence in report["codec_evidence"].items():
            print(f"{kind.title()}: {evidence['method']}", file=self.stream)

    def __call__(self, frames: int) -> None:
        now = perf_counter()
        if self.started is None:
            self.started = now
        if self.last_print is not None and now - self.last_print < 2 and frames != self.total:
            return
        elapsed = now - self.started
        if elapsed <= 0:
            return
        self.last_print = now
        amount = (
            f"{frames} / {self.total} ({frames / self.total:.0%})" if self.total else str(frames)
        )
        rate = frames / elapsed
        factor = f", {self.fps / rate:.2f}× processing/source time" if self.fps and rate else ""
        print(f"Progress: {amount} frames, {rate:.1f} FPS submitted{factor}", file=self.stream)
