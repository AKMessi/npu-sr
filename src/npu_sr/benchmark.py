"""Warm full-image inference measurements, separate from model startup and image IO."""

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import NPUUnavailable, SRException
from .image import PreparedImage, infer_y
from .utils import software_versions

if TYPE_CHECKING:
    from .runtime import Runtime


def statistics(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms or any(not np.isfinite(v) or v <= 0 for v in samples_ms):
        raise SRException("Latency samples must be finite, positive, and nonempty.")
    median = float(np.median(samples_ms))
    return {
        "median_ms": median,
        "mean_ms": float(np.mean(samples_ms)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "fps_equivalent": 1000 / median,
    }


def measure(image: PreparedImage, runtime: "Runtime", runs: int, warmups: int) -> dict[str, Any]:
    if runs < 1 or warmups < 1:
        raise SRException("Runs and warmups must be at least 1.")
    for _ in range(warmups):
        infer_y(image, runtime)
    samples = [infer_y(image, runtime)[1] for _ in range(runs)]
    return {
        "backend": runtime.backend,
        "label": runtime.label,
        "iterations": runs,
        "warmups": warmups,
        "startup_ms": runtime.startup_ms,
        "latency": statistics(samples),
        "samples_ms": samples,
        "evidence": runtime.evidence,
    }


def benchmark(
    image: PreparedImage, path: Path, runs: int, warmups: int, verbose: bool = False
) -> dict[str, Any]:
    from .runtime import Runtime

    results = []
    cpu = Runtime(path, "cpu", verbose)
    manifest = cpu.manifest
    results.append(measure(image, cpu, runs, warmups))
    del cpu
    unavailable = None
    try:
        npu = Runtime(path, "npu", verbose)
        results.append(measure(image, npu, runs, warmups))
        del npu
    except NPUUnavailable as exc:
        unavailable = str(exc)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": {k: manifest[k] for k in ("name", "version", "sha256", "precision")},
        "device": {"os": platform.system(), "architecture": platform.machine()},
        "input_dimensions": {"width": image.size[0], "height": image.size[1]},
        "tiles_per_image": image.tile_count,
        "timing_scope": "sum of ORT run calls per image",
        "software_versions": {"python": platform.python_version(), **software_versions()},
        "results": results,
        "npu_unavailable": unavailable,
    }
    if sys.platform == "win32":
        report["device"]["windows_build"] = sys.getwindowsversion().build
    if len(results) == 2:
        report["npu_speedup"] = (
            results[0]["latency"]["median_ms"] / results[1]["latency"]["median_ms"]
        )
    return report


def save_report(report: dict[str, Any], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise SRException(f"Cannot save JSON report: {exc}") from exc
