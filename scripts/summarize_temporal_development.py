"""Compare complete paired development runs; retain every metric and clip."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(baseline: dict, experiments: list[dict]) -> dict:
    if not baseline.get("complete") or not experiments:
        raise ValueError("Need complete baseline and at least one experiment")
    rows = baseline["results"]
    reference = {row["clip"]: row for row in rows}
    if len(reference) != len(rows):
        raise ValueError("Duplicate baseline clips")

    def scores(metrics: dict, full: float) -> dict:
        values = {
            "psnr_y_db": metrics["psnr_y_db"],
            "ssim_y": metrics["ssim_y"],
            "vmaf": metrics["vmaf"]["mean"],
            "vmaf_neg": metrics["vmaf_neg"]["mean"],
            "temporal_coarse": metrics["temporal_difference_error"],
            "temporal_full": full,
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite quality score")
        return values

    result = {
        "scope": "development experiments only; not holdout or sustained realtime evidence",
        "temporal_definition": (
            "project diagnostic: mean absolute consecutive native coded-Y reconstruction "
            "residual change /255; ROI shave2; no optical flow; may reward smoothing"
        ),
        "aggregate": "equal clip mean for every score; temporal ratio is ratio of clip means",
        "baseline": [
            {
                "clip": row["clip"],
                "output_sha256": row["metrics"]["output_sha256"],
                "scores": scores(
                    row["metrics"], row["full_resolution_temporal_difference"]["mean"]
                ),
            }
            for row in rows
        ],
        "experiments": [],
    }
    base = {row["clip"]: row["scores"] for row in result["baseline"]}
    for experiment in experiments:
        clips = [row["clip"] for row in experiment["results"]]
        if (
            not experiment.get("complete")
            or len(set(clips)) != len(clips)
            or set(clips) != set(reference)
        ):
            raise ValueError("Incomplete, duplicate or unpaired experiment clips")
        measured = []
        for row in experiment["results"]:
            original, metrics, execution = reference[row["clip"]], row["metrics"], row["execution"]
            if (
                metrics["reference_sha256"] != original["metrics"]["reference_sha256"]
                or metrics["decoded_frames"] != original["metrics"]["decoded_frames"]
            ):
                raise ValueError("Reference or frame count changed")
            evidence = execution["execution_evidence"]
            if not evidence["cpu_fallback_disabled"] or set(evidence["executed_kernel_counts"]) != {
                "QNNExecutionProvider"
            }:
                raise ValueError("Strict NPU evidence missing")
            current = scores(metrics, metrics["temporal_difference_error_full_resolution"])
            if not all(np.isfinite(value) for value in current.values()):
                raise ValueError("Nonfinite quality score")
            measured.append(
                {
                    "clip": row["clip"],
                    "scores": current,
                    "deltas": {
                        key: value - base[row["clip"]][key] for key, value in current.items()
                    },
                    "temporal_full_ratio": current["temporal_full"]
                    / base[row["clip"]]["temporal_full"]
                    if base[row["clip"]]["temporal_full"] > 0
                    else None,
                    "output_sha256": metrics["output_sha256"],
                    "reference_sha256": metrics["reference_sha256"],
                    "frames": metrics["decoded_frames"],
                    "execution_evidence": evidence,
                    "neural_tile_runs": execution["neural_tile_runs"],
                    "codec_evidence": execution["codec_evidence"],
                }
            )
        means = {key: float(np.mean([r["scores"][key] for r in measured])) for key in current}
        baseline_means = {key: float(np.mean([r[key] for r in base.values()])) for key in current}
        result["experiments"].append(
            {
                "model_sha256": experiment["model_sha256"],
                "training": experiment["manifest"]["training"],
                "environment_at_start": experiment["environment_at_start"],
                "scores": means,
                "deltas": {key: value - baseline_means[key] for key, value in means.items()},
                "temporal_full_ratio": means["temporal_full"] / baseline_means["temporal_full"]
                if baseline_means["temporal_full"] > 0
                else None,
                "clips": measured,
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        json.loads(args.baseline.read_text()), [json.loads(p.read_text()) for p in args.experiments]
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
