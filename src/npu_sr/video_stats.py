"""Bounded sustained-throughput and resource observations, without power estimates."""

import math
import os
from array import array
from collections import deque


class LatencyHistogram:
    """Whole-run latency quantiles with fixed storage and 0.05 ms bins.

    Quantiles interpolate bin midpoints, with at most 0.025 ms binning error.
    Samples >=1000 ms use an overflow bin: affected quantiles return None,
    while count, actual mean and extrema still include them.
    """

    def __init__(self) -> None:
        self.bins = array("Q", [0]) * 20001
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = 0.0

    def record(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Latency must be finite and nonnegative.")
        self.bins[min(int(value * 20), 20000)] += 1
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def quantile(self, fraction: float) -> float | None:
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Quantile must be between zero and one.")
        if not self.count:
            return None
        rank = (self.count - 1) * fraction
        lower, upper = math.floor(rank), math.ceil(rank)
        accumulated, values = 0, []
        for index, count in enumerate(self.bins):
            accumulated += count
            if accumulated > lower and not values:
                values.append(None if index == 20000 else (index + 0.5) / 20)
            if accumulated > upper:
                values.append(None if index == 20000 else (index + 0.5) / 20)
                break
        if any(value is None for value in values):
            return None
        return values[0] + (values[1] - values[0]) * (rank - lower)

    def report(self) -> dict:
        if not self.count:
            return {"samples": 0}
        return {
            "samples": self.count,
            "mean_ms": self.total / self.count,
            "minimum_ms": self.minimum,
            "maximum_ms": self.maximum,
            "median_ms": self.quantile(0.5),
            "p95_ms": self.quantile(0.95),
            "p99_ms": self.quantile(0.99),
            "overflow_samples": self.bins[-1],
            "quantile_bin_width_ms": 0.05,
            "scope": "all frames; approximate midpoint histogram quantiles; actual mean/extrema",
        }


def resource_summary(samples: list[dict]) -> dict:
    """Process CPU deltas and sampled concurrent RSS, including codec children."""
    if len(samples) < 2 or not all(samples[0]["processes"].values()):
        return {"status": "not measured"}
    seconds = samples[-1]["elapsed_seconds"] - samples[0]["elapsed_seconds"]
    if seconds <= 0:
        return {"status": "not measured"}
    cpu = {}
    for label, first in samples[0]["processes"].items():
        last = samples[-1]["processes"].get(label, {})
        if "cpu_seconds" in first and "cpu_seconds" in last:
            cpu[label] = max(0, last["cpu_seconds"] - first["cpu_seconds"])
    sets = [
        sum(row["working_set_bytes"] for row in sample["processes"].values())
        for sample in samples
        if all("working_set_bytes" in row for row in sample["processes"].values())
    ]
    window = min(60, len(sets) // 2)
    first_window = sum(sets[:window]) / window if window else None
    final_window = sum(sets[-window:]) / window if window else None
    return {
        "sample_window_seconds": seconds,
        "process_cpu_seconds": cpu,
        "process_cpu_percent_of_machine": sum(cpu.values()) / seconds / (os.cpu_count() or 1) * 100,
        "logical_cpu_count": os.cpu_count(),
        "peak_sampled_aggregate_working_set_bytes": max(sets) if sets else None,
        "first_sample_working_set_bytes": sets[0] if sets else None,
        "last_sample_working_set_bytes": sets[-1] if sets else None,
        "first_window_mean_working_set_bytes": first_window,
        "final_window_mean_working_set_bytes": final_window,
        "memory_comparison_samples_per_window": window,
        "scope": (
            "Python + decoder + encoder; 1s sampled working set, CPU deltas between samples; "
            "excludes other apps"
        ),
    }


class SustainedStatistics:
    """Ten-second throughput windows evaluated on one-second boundaries."""

    def __init__(self) -> None:
        self.second, self.count, self.total, self.first_ten = 0, 0, 0, 0
        self.buckets: deque[tuple[int, int]] = deque(maxlen=600)
        self.last_times: deque[float] = deque(maxlen=8192)
        self.minimum: float | None = None

    def advance(self, elapsed: float) -> None:
        while self.second < int(elapsed):
            self.buckets.append((self.second, self.count))
            if len(self.buckets) >= 10:
                rate = sum(row[1] for row in list(self.buckets)[-10:]) / 10
                self.minimum = rate if self.minimum is None else min(self.minimum, rate)
            self.second += 1
            self.count = 0

    def record(self, elapsed: float) -> None:
        self.advance(elapsed)
        self.count += 1
        self.total += 1
        self.first_ten += elapsed < 10
        self.last_times.append(elapsed)

    def report(self, elapsed: float) -> dict:
        self.advance(elapsed)
        final_count = sum(value >= elapsed - 10 for value in self.last_times)
        complete_final = len(self.last_times) < 8192 or self.last_times[0] <= elapsed - 10
        return {
            "first_10s_fps": self.first_ten / 10 if elapsed >= 10 else None,
            "final_10s_fps": final_count / 10 if elapsed >= 10 and complete_final else None,
            "minimum_rolling_10s_fps": self.minimum,
            "scope": (
                "encoder pipe submissions, windows on 1s boundaries; first window includes "
                "codec startup, final includes flush; encoder completion timing not available"
            ),
            "throughput_seconds": [
                {"second": second, "frames": count} for second, count in self.buckets
            ],
            "throughput_history_scope": "last 600 complete seconds; minimum retained for whole run",
        }
