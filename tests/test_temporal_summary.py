"""Pairing and strict-execution guards on temporal research reports."""

import copy
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "temporal_summary", Path(__file__).parents[1] / "scripts/summarize_temporal_development.py"
)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def paired():
    metrics = {
        "psnr_y_db": 35.0,
        "ssim_y": 0.95,
        "vmaf": {"mean": 90.0},
        "vmaf_neg": {"mean": 89.0},
        "temporal_difference_error": 0.01,
        "temporal_difference_error_full_resolution": 0.009,
        "output_sha256": "output",
        "reference_sha256": "reference",
        "decoded_frames": 60,
    }
    baseline = {
        "complete": True,
        "results": [
            {
                "clip": "face",
                "metrics": metrics,
                "full_resolution_temporal_difference": {"mean": 0.01},
            }
        ],
    }
    experiment = {
        "complete": True,
        "model_sha256": "model",
        "manifest": {"training": {}},
        "environment_at_start": {},
        "results": [
            {
                "clip": "face",
                "metrics": copy.deepcopy(metrics),
                "execution": {
                    "execution_evidence": {
                        "cpu_fallback_disabled": True,
                        "executed_kernel_counts": {"QNNExecutionProvider": 1},
                    },
                    "neural_tile_runs": 720,
                    "codec_evidence": {},
                },
            }
        ],
    }
    return baseline, experiment


def test_equal_clip_summary_preserves_unfavorable_spatial_score():
    baseline, experiment = paired()
    experiment["results"][0]["metrics"]["psnr_y_db"] = 34.9
    result = module.summarize(baseline, [experiment])["experiments"][0]
    assert result["deltas"]["psnr_y_db"] == pytest.approx(-0.1)
    assert result["temporal_full_ratio"] == pytest.approx(0.9)


@pytest.mark.parametrize("kind", ["incomplete", "duplicate", "missing", "reference", "frames"])
def test_invalid_pairs_do_not_create_quality_claim(kind):
    baseline, experiment = paired()
    if kind == "incomplete":
        experiment["complete"] = False
    elif kind == "duplicate":
        experiment["results"] *= 2
    elif kind == "missing":
        experiment["results"] = []
    elif kind == "reference":
        experiment["results"][0]["metrics"]["reference_sha256"] = "different"
    else:
        experiment["results"][0]["metrics"]["decoded_frames"] = 59
    with pytest.raises(ValueError):
        module.summarize(baseline, [experiment])


def test_cpu_kernel_cannot_be_reported_as_strict_npu():
    baseline, experiment = paired()
    experiment["results"][0]["execution"]["execution_evidence"]["executed_kernel_counts"][
        "CPUExecutionProvider"
    ] = 1
    with pytest.raises(ValueError, match="Strict NPU"):
        module.summarize(baseline, [experiment])


def test_nonfinite_baseline_is_rejected():
    baseline, experiment = paired()
    baseline["results"][0]["metrics"]["ssim_y"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        module.summarize(baseline, [experiment])


def test_zero_baseline_error_has_no_relative_improvement_claim():
    baseline, experiment = paired()
    baseline["results"][0]["full_resolution_temporal_difference"]["mean"] = 0.0
    result = module.summarize(baseline, [experiment])["experiments"][0]
    assert result["temporal_full_ratio"] is None
    assert result["clips"][0]["temporal_full_ratio"] is None
