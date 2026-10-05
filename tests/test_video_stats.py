"""Sustained gate and timer aggregation; synthetic samples are logic fixtures."""

from copy import deepcopy

from npu_sr.video import settings_for_preset
from npu_sr.video_benchmark import image_quality_acceptance, realtime_acceptance
from npu_sr.video_stats import SustainedStatistics


def test_whole_run_histogram_retains_early_stalls_with_fixed_memory():
    import numpy as np
    import pytest

    from npu_sr.video_stats import LatencyHistogram

    stats = LatencyHistogram()
    values = [800.01] * 1000 + [10.01] * 9000
    for value in values:
        stats.record(value)
    report = stats.report()
    assert report["samples"] == 10000
    assert report["mean_ms"] == pytest.approx(np.mean(values))
    assert report["p95_ms"] == pytest.approx(np.percentile(values, 95), abs=0.025)
    assert report["p99_ms"] > 800
    assert len(stats.bins) == 20001
    stats.record(2000)
    assert stats.report()["maximum_ms"] == 2000
    overflow = LatencyHistogram()
    overflow.record(2000)
    assert overflow.report()["median_ms"] is None
    for invalid in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            stats.record(invalid)


def test_custom_power_scheme_does_not_leak_user_name_or_identifier():
    from npu_sr.monitoring import public_power_scheme

    assert (
        public_power_scheme("GUID: 381b4222-f694-41f0-9685-ff5bb260df2e (localized)") == "Balanced"
    )
    assert (
        public_power_scheme("GUID: 12345678-1234-1234-1234-123456789abc (Private owner)")
        == "custom"
    )
    assert public_power_scheme("Private owner") == "unknown"


def test_production_gate_requires_actual_ten_minutes_not_source_duration():
    trial = {
        "input": {
            "frame_rate": "30",
            "resolution": [960, 540],
            "duration_seconds": 1200,
            "audio": True,
        },
        "output": {"resolution": [1920, 1080], "reported_frames": 36000},
        "frames_processed": 36000,
        "neural_tile_runs": 432000,
        "tile_count_per_frame": 12,
        "backend": "npu",
        "dropped_frames": 0,
        "input_timestamps_validated": True,
        "output_timestamps_validated": True,
        "processing_seconds": 650,
        "end_to_end_fps": 36000 / 650,
        "sustained": {"minimum_rolling_10s_fps": 40},
        "execution_evidence": {
            "cpu_fallback_disabled": True,
            "executed_kernel_counts": {"QNNExecutionProvider": 1},
        },
        "codec_evidence": {"encode": {"method": "Media Foundation hardware"}},
        "pipeline_depth": 2,
        "observed_queue_peaks": {"decode_queue": 2, "encode_queue": 2},
        "whole_run_phase_statistics": {
            key: {"samples": 36000}
            for key in ["frame_latency_ms", "inference_ms", "postprocessing_ms"]
        },
        "resources": {
            "first_window_mean_working_set_bytes": 600_000_000,
            "final_window_mean_working_set_bytes": 600_000_000,
            "peak_sampled_aggregate_working_set_bytes": 700_000_000,
        },
        "audio_validation": {"actual_offset_seconds": 0},
    }
    report = {"complete": True, "trials": [deepcopy(trial) for _ in range(3)]}
    assert realtime_acceptance(report, production=True)["passed"]
    for field in ("processing_seconds", "end_to_end_fps"):
        report["trials"][0] = deepcopy(trial)
        report["trials"][0][field] = float("nan")
        assert not realtime_acceptance(report, production=True)["passed"]
    report["trials"][0] = deepcopy(trial)
    report["trials"][0]["processing_seconds"] = 599
    assert not realtime_acceptance(report, production=True)["passed"]
    report["trials"][0] = deepcopy(trial)
    report["trials"][0]["resources"]["final_window_mean_working_set_bytes"] = 1_500_000_000
    assert not realtime_acceptance(report, production=True)["passed"]
    report["trials"][0] = deepcopy(trial)
    report["trials"][0]["whole_run_phase_statistics"]["inference_ms"]["samples"] -= 1
    assert not realtime_acceptance(report, production=True)["passed"]


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
    quality = settings_for_preset("quality", {})
    assert quality.model == "quicksrnet-medium-y-x2" and quality.codec == "av1"
    assert settings_for_preset("quality", {"codec": "h264"}).codec == "h264"


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
