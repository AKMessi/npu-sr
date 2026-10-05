"""Independent sustained hardware trials; require ten minutes of processing, not source length."""

import argparse
import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from time import perf_counter

import numpy as np

from npu_sr import __version__
from npu_sr.benchmark import save_report
from npu_sr.model import sha256
from npu_sr.video import process_video, settings_for_preset
from npu_sr.video_benchmark import realtime_acceptance


def run(args: argparse.Namespace) -> None:
    if importlib.metadata.version("npu-sr") != __version__:
        raise ValueError("Installed metadata differs from source; reinstall before benchmarking")
    if not 1 <= args.trials <= 5:
        raise ValueError("Trials must be 1–5; production acceptance requires at least three")
    source = args.input.resolve()
    directory = args.directory.resolve()
    report_path = args.json.resolve()
    if report_path == source:
        raise ValueError("JSON would overwrite the input")
    directory.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "npu-sr/contexts"
    settings = settings_for_preset("realtime", {"cache_dir": cache})
    report = {
        "complete": False,
        "trials": [],
        "methodology": {
            "aggregation": "median of independent trial throughputs; all trials retained",
            "required_processing_seconds": 600,
            "output_policy": "retain"
            if args.retain_videos
            else "hash and delete owned output after validation",
            "power_temperature_accelerator_utilization": "not measured",
            "background_processes": "uncontrolled; no concurrent project tests/training",
        },
    }
    if args.provenance:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        if sha256(source) != provenance["input_sha256"]:
            raise ValueError("Prepared long-run input hash changed")
        report["input_provenance"] = provenance
    for index in range(args.trials):
        output = directory / f"longrun-{index + 1}-{uuid.uuid4().hex}.mp4"
        last_update = [perf_counter()]

        def progress(frames: int, last_update=last_update, index=index) -> None:
            now = perf_counter()
            if now - last_update[0] >= 30:
                print(f"Trial {index + 1}: {frames} frames enhanced", flush=True)
                last_update[0] = now

        result = process_video(source, output, settings, progress)
        result.update(
            trial=index + 1, output_sha256=sha256(output), output_bytes=output.stat().st_size
        )
        report["trials"].append(result)
        save_report(report, report_path)
        if not args.retain_videos:
            # Only this newly generated, verified, unique file is eligible.
            if not output.resolve().is_relative_to(directory) or output.resolve() == source:
                raise ValueError("Output cleanup target escaped the owned trial directory")
            output.unlink()
            result["output_retained"] = False
        else:
            result["output_retained"] = True
        save_report(report, report_path)
        print(
            f"Trial {index + 1}: {result['end_to_end_fps']:.2f} FPS; "
            f"{result['processing_seconds']:.1f} seconds actual processing",
            flush=True,
        )
    report.update(
        complete=True,
        median_trial_end_to_end_fps=float(
            np.median([trial["end_to_end_fps"] for trial in report["trials"]])
        ),
    )
    report["production_realtime_acceptance"] = realtime_acceptance(report, production=True)
    save_report(report, report_path)
    if not report["production_realtime_acceptance"]["passed"]:
        raise ValueError(str(report["production_realtime_acceptance"]["failures"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--retain-videos", action="store_true")
    run(parser.parse_args())
