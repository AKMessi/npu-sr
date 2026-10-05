"""Score a known temporal graph through the production pipeline, without changing defaults."""

import argparse
import json
from pathlib import Path

from temporal_experiment_contract import register_fused

from npu_sr.corpus import validate_definition
from npu_sr.ffmpeg import tool_path
from npu_sr.model import sha256, validate_model
from npu_sr.suite import environment
from npu_sr.video import VideoSettings, process_video
from npu_sr.video_quality import evaluate_native_video


def run(args: argparse.Namespace) -> None:
    register_fused(args.geometry, args.state_features)
    definition = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_definition(definition)
    provenance = json.loads(
        (args.directory / f"provenance-{args.split}.json").read_text(encoding="utf-8")
    )
    if not provenance["complete"] or provenance["definition"] != definition:
        raise ValueError("Prepared and published definitions differ in parsed settings")
    model = validate_model(args.model)
    digest = sha256(args.model)
    if args.split == "holdout":
        if args.selection_lock is None:
            raise ValueError("Holdout requires a predeclared selection lock")
        lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
        if lock["model_sha256"] != digest or (
            lock.get("published_validation_definition_sha256") != sha256(args.manifest)
        ):
            raise ValueError("Model or definition differs from frozen holdout selection")
    selected = [c for c in provenance["clips"] if not args.clips or c["identifier"] in args.clips]
    if not selected or (args.clips and {c["identifier"] for c in selected} != set(args.clips)):
        raise ValueError("Requested clips are missing from prepared split")
    destination = args.directory / "delivered-temporal-experiment" / digest[:16] / args.split
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "complete": False,
        "environment_at_start": environment(),
        "script_sha256": sha256(Path(__file__)),
        "model_sha256": digest,
        "manifest": model,
        "published_definition_sha256": sha256(args.manifest),
        "definition_sha256_at_preparation": provenance["definition_sha256"],
        "output_subdirectory": destination.relative_to(args.directory).as_posix(),
        "results": [],
    }
    for clip in selected:
        source, reference = (
            args.directory / clip["input_file"],
            args.directory / clip["reference_file"],
        )
        if sha256(source) != clip["input_sha256"] or sha256(reference) != clip["reference_sha256"]:
            raise ValueError("Prepared source/reference bytes changed")
        output = destination / f"{clip['identifier']}-temporal.mp4"
        execution = process_video(
            source,
            output,
            VideoSettings(
                model=str(args.model),
                device="npu",
                decode="hardware",
                encode="hardware",
                codec="av1",
                audio="none",
                frame_format="nv12",
                npu_performance="burst",
                pipeline_depth=2,
                overwrite=True,
                cache_dir=args.directory / "temporal-context-cache",
            ),
        )
        metrics = evaluate_native_video(output, reference, tool_path(), tuple(clip["metric_roi"]))
        if metrics["decoded_frames"] != clip["frames"]:
            raise ValueError("Output lost frames")
        report["results"].append(
            {"clip": clip["identifier"], "execution": execution, "metrics": metrics}
        )
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            clip["identifier"],
            "PSNR",
            metrics["psnr_y_db"],
            "SSIM",
            metrics["ssim_y"],
            "VMAF",
            metrics["vmaf"]["mean"],
            "full TDE",
            metrics["temporal_difference_error_full_resolution"],
            flush=True,
        )
    report["complete"] = True
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/corpus-v1.json"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--clips", nargs="*")
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--geometry", choices=["256", "320x270"], default="256")
    parser.add_argument("--state-features", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    run(parser.parse_args())
