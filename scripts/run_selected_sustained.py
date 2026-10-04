"""Reproduce sustained trials of a locked candidate, without changing preset defaults."""

import argparse
import json
from pathlib import Path

from npu_sr.benchmark import save_report
from npu_sr.model import resolve_model, sha256
from npu_sr.video import VideoSettings
from npu_sr.video_benchmark import benchmark_video, realtime_acceptance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path, help="Prepared sustained fixture")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--selection-lock", type=Path, default=Path("benchmarks/v0.5/selection-lock.json")
    )
    args = parser.parse_args()
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    provenance = json.loads(
        (args.directory / "sustained-provenance.json").read_text(encoding="utf-8")
    )
    source = args.directory / provenance["input_file"]
    if sha256(source) != provenance["input_sha256"]:
        raise ValueError("Sustained input bytes changed")
    if sha256(resolve_model(lock["model"])) != lock["model_sha256"]:
        raise ValueError("Selected model differs from the locked candidate")
    config = lock["settings"]
    settings = VideoSettings(
        model=lock["model"],
        device="npu",
        decode=config["decode"],
        encode=config["encode"],
        codec=config["codec"],
        bitrate=config["bitrate"],
        frame_format=config["frame_format"],
        neural_strength=config["neural_strength"],
        npu_performance=config["npu_performance"],
        pipeline_depth=config["pipeline_depth"],
        cache_dir=args.output_directory / "context-cache",
        overwrite=True,
    )
    report = benchmark_video(source, args.output_directory, settings, args.trials, args.json)
    if any(row["model_sha256"] != lock["model_sha256"] for row in report["trials"]):
        raise ValueError("Measured model differs from the selection lock")
    report["provenance"] = provenance
    report["selection_lock"] = lock
    report["realtime_acceptance"] = realtime_acceptance(report)
    save_report(report, args.json)
    print("Sustained numeric gate:", report["realtime_acceptance"])
    if not report["realtime_acceptance"]["passed"]:
        raise ValueError("Sustained acceptance failed; no release allowed")
