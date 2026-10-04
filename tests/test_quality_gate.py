from copy import deepcopy

import pytest

from npu_sr.errors import SRException
from npu_sr.quality_gate import paired_quality_summary


def fixture():
    clip = {
        "identifier": "face",
        "split": "holdout",
        "source": "film",
        "categories": ["faces"],
        "frames": 60,
        "metric_roi": [0, 0, 1920, 1080],
    }
    metrics = {
        "reference_sha256": "a" * 64,
        "decoded_frames": 60,
        "metric_roi": [0, 0, 1920, 1080],
        "sample_stride": 3,
        "psnr_y_db": 35,
        "ssim_y": 0.95,
        "vmaf": {"mean": 90},
        "vmaf_neg": {"mean": 89},
        "temporal_difference_error": 0.001,
    }
    baseline = {"clip": "face", "model": "bicubic", "metrics": metrics}
    candidate = deepcopy(baseline)
    candidate.update(
        model="neural",
        execution={
            "backend": "npu",
            "neural_tile_runs": 720,
            "tile_count_per_frame": 12,
            "execution_evidence": {
                "cpu_fallback_disabled": True,
                "executed_kernel_counts": {"QNNExecutionProvider": 1},
            },
        },
    )
    candidate["metrics"].update(psnr_y_db=36, ssim_y=0.96, vmaf={"mean": 92})
    return [baseline, candidate], [clip]


def test_paired_gains_and_diagnostic_scope():
    rows, clips = fixture()
    report = paired_quality_summary(rows, clips, "neural")
    assert report["numeric_gate_passed"]
    assert report["groups"]["all"]["gain"]["vmaf"] == 2
    assert report["groups"]["category:faces"]["clips"] == 1
    assert "manual" in report["release_gate"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reference", "frames", "nonfinite"])
def test_rejects_invalid_or_selective_pairs(kind):
    rows, clips = fixture()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows.append(rows[0])
    elif kind == "reference":
        rows[1]["metrics"]["reference_sha256"] = "b" * 64
    elif kind == "frames":
        for row in rows:
            row["metrics"]["decoded_frames"] = 59
    else:
        rows[1]["metrics"]["vmaf"]["mean"] = float("nan")
    with pytest.raises(SRException):
        paired_quality_summary(rows, clips, "neural")


@pytest.mark.parametrize("failure", ["vmaf", "cpu", "skipped"])
def test_numeric_gate_rejects_quality_loss_or_false_execution(failure):
    rows, clips = fixture()
    if failure == "vmaf":
        rows[1]["metrics"]["vmaf"]["mean"] = 89
    elif failure == "cpu":
        rows[1]["execution"]["execution_evidence"]["executed_kernel_counts"][
            "CPUExecutionProvider"
        ] = 1
    else:
        rows[1]["execution"]["neural_tile_runs"] -= 12
    report = paired_quality_summary(rows, clips, "neural")
    assert not report["numeric_gate_passed"]
