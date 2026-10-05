"""Prove standalone or fused temporal experiments through the strict runtime."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from export_temporal_residual import IDENTIFIER
from temporal_experiment_contract import register_fused

from npu_sr.model import MODELS, ModelSpec, sha256
from npu_sr.runtime import Runtime
from npu_sr.suite import environment


class TemporalSpec(ModelSpec):
    @property
    def input_shape(self) -> list[int]:
        return [1, self.input_channels, 262, 262]


def register_experiment(channels: int = 2) -> None:
    if channels not in {2, 6, 10}:
        raise ValueError("Unsupported temporal channels")
    MODELS[IDENTIFIER] = TemporalSpec(
        IDENTIFIER,
        "FeatureTemporalResidual" if channels == 6 else "TinyTemporalResidual",
        task="temporal",
        halo=3,
        license="MIT",
        input_channels=channels,
    )


def probe(args: argparse.Namespace) -> None:
    identifier = register_fused(args.geometry, args.state_features) if args.fused else None
    channels = json.loads(args.model.with_suffix(".json").read_text())["input_shape"][1]
    if not args.fused:
        register_experiment(channels)
    cpu = Runtime(args.model, "cpu")
    if args.fused and cpu.spec.identifier != identifier:
        raise ValueError("Fused proof requires the registered temporal graph contract")
    tensor = np.random.default_rng(20261004).random(tuple(cpu.input_shape), dtype=np.float32)
    expected = cpu.run(tensor)
    npu = Runtime(args.model, "npu", performance_mode="burst")
    actual = npu.run(tensor)
    changed = tensor.copy()
    changed[:, 0] = tensor[:, 1]
    if args.state_features:
        changed[:, 2:] = tensor[:, 1:2]
    dependence = float(np.abs(cpu.run(changed) - expected).mean())
    maximum = float(np.abs(actual - expected).max())
    if maximum > (0.005 if args.fused else 0.002) or dependence < 1e-6:
        raise ValueError("Temporal numerical agreement/input dependence failed")
    for _ in range(3):
        npu.run(tensor)
    calls = 6 if args.geometry == "320x270" else 12
    trials = []
    for _ in range(3):
        samples = []
        for _ in range(10):
            start = perf_counter()
            for _ in range(calls):
                npu.run(tensor)
            samples.append((perf_counter() - start) * 1000)
        trials.append({"median_calls_ms": float(np.median(samples)), "samples_ms": samples})
    np.savez(args.json.with_suffix(".npz"), input=tensor, output_cpu=expected, output_npu=actual)
    report = {
        "environment": environment(),
        "model_sha256": sha256(args.model),
        "execution_evidence": npu.evidence,
        "startup_ms": npu.startup_ms,
        "max_cpu_npu_absolute_error": maximum,
        "mean_previous_input_effect": dependence,
        "trials": trials,
        "calls_per_sample": calls,
        "scope": "random numerical proof and model-only latency; not video FPS or quality",
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Strict QNN proof", npu.evidence["executed_kernel_counts"], flush=True)
    print("CPU/NPU max error", maximum, "previous-frame mean effect", dependence, flush=True)
    print("Calls", calls, "medians", [t["median_calls_ms"] for t in trials], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--geometry", choices=["256", "320x270"], default="256")
    parser.add_argument("--state-features", action="store_true")
    probe(parser.parse_args())
