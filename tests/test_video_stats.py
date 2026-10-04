"""Sustained gate and timer aggregation; synthetic samples are logic fixtures."""

from copy import deepcopy

from npu_sr.video import settings_for_preset
from npu_sr.video_benchmark import image_quality_acceptance, realtime_acceptance
from npu_sr.video_stats import SustainedStatistics


def test_sustained_windows_detect_slow_tail():
    stats = SustainedStatistics()
    for index in range(400):
        stats.record(index / 40)
    for index in range(200):
        stats.record(10 + index / 20)
    report = stats.report(20)
    assert report["first_10s_fps"] == 40
    assert report["final_10s_fps"] == 20
    assert report["minimum_rolling_10s_fps"] == 20


def test_benchmark_gate_requires_duration_throughput_and_proofs():
    trial = {
        "input": {"frame_rate": "30", "resolution": [960, 540]},
        "output": {"resolution": [1920, 1080], "reported_frames": 3600},
        "backend": "npu",
        "frames_processed": 3600,
        "tile_count_per_frame": 12,
        "neural_tile_runs": 43200,
        "dropped_frames": 0,
        "input_timestamps_validated": True,
        "output_timestamps_validated": True,
        "processing_seconds": 90,
        "end_to_end_fps": 40,
        "sustained": {"minimum_rolling_10s_fps": 38},
        "execution_evidence": {
            "cpu_fallback_disabled": True,
            "executed_kernel_counts": {"QNNExecutionProvider": 1},
        },
        "codec_evidence": {"encode": {"method": "Media Foundation hardware"}},
    }
    report = {"complete": True, "trials": [deepcopy(trial) for _ in range(3)]}
    assert realtime_acceptance(report)["passed"]
    report["trials"][0]["processing_seconds"] = 59
    assert not realtime_acceptance(report)["passed"]
    report["trials"][0] = deepcopy(trial)
    report["trials"][0]["sustained"]["minimum_rolling_10s_fps"] = 29.9
    assert not realtime_acceptance(report)["passed"]
    report["trials"][0] = deepcopy(trial)
    report["trials"][0]["execution_evidence"]["executed_kernel_counts"]["CPUExecutionProvider"] = 1
    assert not realtime_acceptance(report)["passed"]
    report["trials"][0] = deepcopy(trial)
    report["trials"][0]["neural_tile_runs"] -= 1
    assert not realtime_acceptance(report)["passed"]


def test_presets_materialize_and_explicit_options_override():
    normal = settings_for_preset(None, {})
    assert normal.codec == "h264" and normal.frame_format == "rgb24" and normal.device == "auto"
    realtime = settings_for_preset("realtime", {})
    assert realtime.device == "npu" and realtime.encode == "hardware" and realtime.codec == "av1"
    assert realtime.frame_format == "nv12" and realtime.pipeline_depth == 2
    assert realtime.neural_strength == 1.0 and realtime.npu_performance == "burst"
    assert realtime.model == "quicksrnet-small-y-x2"
    overridden = settings_for_preset("realtime", {"codec": "hevc", "frame_format": "rgb24"})
    assert overridden.codec == "hevc" and overridden.frame_format == "rgb24"


def test_image_quality_gate_rejects_quality_collapse_and_cpu_fallback():
    rows = [
        {
            "model": "espcn-x2-256",
            "backend": "npu",
            "image": str(index),
            "metrics": {"psnr_y_db": 31, "ssim_y": 0.91},
            "baseline_metrics": {"psnr_y_db": 30, "ssim_y": 0.9},
            "evidence": {
                "executed_kernel_counts": {"QNNExecutionProvider": 1},
                "cpu_fallback_disabled": True,
            },
        }
        for index in range(5)
    ]
    assert image_quality_acceptance({"results": rows})["passed"]
    rows[0]["metrics"]["psnr_y_db"] = 20
    assert not image_quality_acceptance({"results": rows})["passed"]
    rows[0]["metrics"]["psnr_y_db"] = 31
    rows[0]["evidence"]["executed_kernel_counts"]["CPUExecutionProvider"] = 1
    assert not image_quality_acceptance({"results": rows})["passed"]
