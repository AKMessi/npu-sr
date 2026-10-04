"""Recheck locked v0.5 delivered quality after pipeline changes; retain every result."""

import argparse
import json
import math
import re
from pathlib import Path

from npu_sr.benchmark import save_report
from npu_sr.ffmpeg import tool_path
from npu_sr.model import resolve_model, sha256
from npu_sr.video import VideoSettings, process_video
from npu_sr.video_quality import evaluate_native_video


def run(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    comparison = baseline["comparisons"]["selected-development"]
    lock = baseline["selection_lock"]
    if sha256(resolve_model(lock["model"])) != lock["model_sha256"]:
        raise ValueError("Regression model differs from locked weights")
    settings = VideoSettings(
        **{k: v for k, v in lock["settings"].items() if k in VideoSettings.__dataclass_fields__},
        model=lock["model"],
        device="npu",
        audio="none",
        overwrite=True,
        verify_full=args.verify_full,
        cache_dir=args.directory / "context-cache",
    )
    report = {"complete": False, "baseline_sha256": sha256(args.baseline), "results": []}
    failures = []
    if len(args.clips) != len(set(args.clips)):
        raise ValueError("Regression clips must be distinct")
    for name in args.clips:
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name):
            raise ValueError("Regression clip must be a corpus identifier, not a path")
        clip = next(c for c in comparison["preparation"]["clips"] if c["identifier"] == name)
        source, reference = (
            args.directory / clip["input_file"],
            args.directory / clip["reference_file"],
        )
        if not all(
            p.resolve().is_relative_to(args.directory.resolve()) for p in (source, reference)
        ):
            raise ValueError("Regression media must stay within the prepared directory")
        if sha256(source) != clip["input_sha256"] or sha256(reference) != clip["reference_sha256"]:
            raise ValueError("Regression input/reference changed")
        old = next(
            r["metrics"]
            for r in comparison["results"]
            if r["clip"] == name and r["model"] == lock["model"]
        )
        output = args.directory / "regression-results" / f"{name}.mp4"
        execution = process_video(source, output, settings)
        metrics = evaluate_native_video(output, reference, tool_path(), tuple(clip["metric_roi"]))
        gain = {
            "psnr_y_db": metrics["psnr_y_db"] - old["psnr_y_db"],
            "ssim_y": metrics["ssim_y"] - old["ssim_y"],
            "vmaf": metrics["vmaf"]["mean"] - old["vmaf"]["mean"],
        }
        if (
            not all(math.isfinite(value) for value in gain.values())
            or gain["psnr_y_db"] < -0.005
            or gain["ssim_y"] < -0.0001
            or gain["vmaf"] < -0.05
        ):
            failures.append(name)
        report["results"].append(
            {
                "clip": name,
                "baseline": old,
                "metrics": metrics,
                "gain": gain,
                "execution": execution,
            }
        )
        save_report(report, args.json)
        print(name, gain, flush=True)
    report.update(
        complete=True,
        failures=failures,
        tolerance="0.005 dB, 0.0001 SSIM, 0.05 VMAF (encode variation); report all differences",
    )
    save_report(report, args.json)
    if failures:
        raise ValueError("Delivered quality regression: " + ", ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/v0.5/quality.json"))
    parser.add_argument(
        "--clips", nargs="+", default=["conversation", "animated-leaves", "aerial-landscape"]
    )
    parser.add_argument("--verify-full", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    run(parser.parse_args())
