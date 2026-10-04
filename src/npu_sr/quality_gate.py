"""Complete paired-corpus quality aggregation; no selection of favorable clips."""

import math
from collections import defaultdict

import numpy as np

from .errors import SRException

METRICS = ("psnr_y_db", "ssim_y", "vmaf", "vmaf_neg", "temporal_difference_error")


def _values(row: dict) -> dict[str, float]:
    metrics = row["metrics"]
    values = {
        key: metrics[key]["mean"] if key.startswith("vmaf") else metrics[key] for key in METRICS
    }
    if any(value is None or not math.isfinite(value) for value in values.values()):
        raise SRException("Quality aggregation requires finite metrics for every clip.")
    if (
        values["psnr_y_db"] < 0
        or not -1.000001 <= values["ssim_y"] <= 1.000001
        or any(not 0 <= values[key] <= 100 for key in ("vmaf", "vmaf_neg"))
        or values["temporal_difference_error"] < 0
    ):
        raise SRException("Quality metric exceeds its valid range.")
    return values


def paired_quality_summary(results: list[dict], clips: list[dict], candidate: str) -> dict:
    """Validate exact pairs then report equal-clip means, per-source/category/split.

    Temporal error and VMAF NEG are diagnostics, not replacements for manual
    temporal inspection. Full release acceptance requires that additional review.
    """
    indexed = {}
    expected = {row["identifier"] for row in clips}
    if len(expected) != len(clips) or not clips:
        raise SRException("Quality declaration contains duplicate clips or is empty.")
    for row in results:
        if row["model"] not in {"bicubic", candidate}:
            continue
        key = row["clip"], row["model"]
        if key in indexed or row["clip"] not in expected:
            raise SRException("Duplicate or undeclared quality pair.")
        indexed[key] = row
    if set(indexed) != {(clip, model) for clip in expected for model in ("bicubic", candidate)}:
        raise SRException("Quality aggregation requires every declared clip for both methods.")
    groups = defaultdict(list)
    pairs, failures = [], []
    for clip in clips:
        baseline, neural = (indexed[clip["identifier"], model] for model in ("bicubic", candidate))
        for key in ("reference_sha256", "decoded_frames", "metric_roi", "sample_stride"):
            if baseline["metrics"][key] != neural["metrics"][key]:
                raise SRException(f"Quality pair differs in {key}.")
        if neural["metrics"]["decoded_frames"] != clip["frames"]:
            raise SRException("Quality clip frame count differs from its declaration.")
        if neural["metrics"]["metric_roi"] != clip["metric_roi"]:
            raise SRException("Quality metric ROI differs from its declaration.")
        execution = neural["execution"]
        evidence = execution.get("execution_evidence", {})
        kernels = evidence.get("executed_kernel_counts", {})
        if (
            execution.get("backend") != "npu"
            or not evidence.get("cpu_fallback_disabled")
            or set(kernels) != {"QNNExecutionProvider"}
            or not kernels.get("QNNExecutionProvider")
            or execution.get("tile_count_per_frame", 0) <= 0
            or execution.get("neural_tile_runs")
            != clip["frames"] * execution.get("tile_count_per_frame", 0)
        ):
            failures.append(f"{clip['identifier']}: strict QNN/every-frame neural proof missing")
        bicubic, actual = _values(baseline), _values(neural)
        pair = {
            "clip": clip["identifier"],
            "bicubic": bicubic,
            "candidate": actual,
            "gain": {key: actual[key] - bicubic[key] for key in METRICS},
        }
        pairs.append(pair)
        for group in ("all", f"split:{clip['split']}", f"source:{clip['source']}"):
            groups[group].append(pair)
        for category in clip["categories"]:
            groups[f"category:{category}"].append(pair)
    summary = {
        name: {
            "clips": len(rows),
            **{
                method: {
                    metric: float(np.mean([row[method][metric] for row in rows]))
                    for metric in METRICS
                }
                for method in ("bicubic", "candidate", "gain")
            },
        }
        for name, rows in groups.items()
    }
    for group in ("all", "split:holdout"):
        if group not in summary:
            continue
        for metric in ("psnr_y_db", "ssim_y", "vmaf"):
            if summary[group]["gain"][metric] <= 0:
                failures.append(f"{group}: average {metric} does not beat bicubic")
    return {
        "candidate": candidate,
        "aggregation": "equal per-clip arithmetic means; all declared clips",
        "numeric_gate_passed": not failures,
        "failures": failures,
        "release_gate": "manual temporal/artifact review and performance validation also required",
        "groups": summary,
        "pairs": pairs,
    }
