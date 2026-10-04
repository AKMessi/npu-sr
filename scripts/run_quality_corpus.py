"""Run paired delivered-video quality trials on a hash-checked predeclared corpus."""

import argparse
import json
import os
from pathlib import Path

from npu_sr.benchmark import save_report
from npu_sr.corpus import validate_definition
from npu_sr.ffmpeg import codec_args, encode_evidence, run_tool, tool_path
from npu_sr.model import resolve_model, sha256
from npu_sr.suite import environment
from npu_sr.video import VideoSettings, process_video
from npu_sr.video_quality import evaluate_native_video


def run(args: argparse.Namespace) -> None:
    provenance_path = args.directory / f"provenance-{args.split}.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    validate_definition(provenance["definition"])
    if not provenance["complete"] or provenance["definition_sha256"] != sha256(args.manifest):
        raise ValueError("Corpus preparation is incomplete or its definition changed")
    selection = None
    if args.split in {"holdout", "all"}:
        selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
        if (
            args.models != ["bicubic", selection["model"]]
            or sha256(args.manifest) != selection["corpus_definition_sha256"]
            or args.strength != selection["settings"]["neural_strength"]
            or args.frame_format != selection["settings"]["frame_format"]
            or sha256(resolve_model(selection["model"])) != selection["model_sha256"]
        ):
            raise ValueError("Holdout settings/weights differ from the development selection lock")
    ffmpeg = tool_path(args.ffmpeg)
    report = {
        "schema_version": 1,
        "complete": False,
        "environment": environment(),
        "ffmpeg_sha256": sha256(ffmpeg),
        "corpus_definition_sha256": sha256(args.manifest),
        "preparation": provenance,
        "selection_lock": selection,
        "configuration": {
            "models": args.models,
            "split": args.split,
            "codec": "av1",
            "bitrate": "8M",
            "neural_strength": args.strength,
            "frame_format": args.frame_format,
            "metric_protocol": "native-Y-v1; clip PSNR from mean MSE; every-frame VMAF",
        },
        "results": [],
    }
    destination = args.directory / "delivered-tagged" / args.split
    destination.mkdir(parents=True, exist_ok=True)
    for clip in provenance["clips"]:
        source, reference = (
            args.directory / clip["input_file"],
            args.directory / clip["reference_file"],
        )
        if sha256(source) != clip["input_sha256"] or sha256(reference) != clip["reference_sha256"]:
            raise ValueError("Paired source/reference bytes changed")
        for model in args.models:
            output = destination / (
                f"{clip['identifier']}-{model}-{args.strength:g}-{args.frame_format}.mp4"
            )
            if model == "bicubic":
                completed = run_tool(
                    [
                        str(ffmpeg),
                        "-v",
                        "verbose",
                        "-y",
                        "-i",
                        str(source),
                        "-vf",
                        "scale=1920:1080:flags=bicubic",
                        "-an",
                        *codec_args("av1", True, "8M", 20),
                        "-color_range",
                        "tv",
                        "-colorspace",
                        "bt709",
                        str(output),
                    ]
                )
                execution = {
                    "method": "FFmpeg bicubic",
                    "codec_evidence": {
                        "encode": encode_evidence(completed.stderr.decode("utf-8", "replace"))
                    },
                }
            else:
                execution = process_video(
                    source,
                    output,
                    VideoSettings(
                        model=model,
                        device="npu",
                        decode="hardware",
                        encode="hardware",
                        codec="av1",
                        audio="none",
                        frame_format=args.frame_format,
                        neural_strength=args.strength,
                        npu_performance="burst",
                        pipeline_depth=2,
                        cache_dir=args.directory / "context-cache",
                        overwrite=True,
                        ffmpeg=ffmpeg,
                    ),
                )
            metrics = evaluate_native_video(output, reference, ffmpeg, tuple(clip["metric_roi"]))
            if metrics["decoded_frames"] != clip["frames"]:
                raise ValueError("Quality output lost frames")
            report["results"].append(
                {
                    "clip": clip["identifier"],
                    "split": clip["split"],
                    "source": clip["source"],
                    "categories": clip["categories"],
                    "model": model,
                    "execution": execution,
                    "metrics": metrics,
                }
            )
            save_report(report, args.json)
            print(
                clip["identifier"],
                model,
                f"PSNR {metrics['psnr_y_db']:.3f}",
                f"SSIM {metrics['ssim_y']:.6f}",
                f"VMAF {metrics['vmaf']['mean']:.3f}",
                flush=True,
            )
    report["complete"] = True
    save_report(report, args.json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/corpus-v1.json"))
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
        / "npu-sr/benchmarks/v1-quality",
    )
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--models", nargs="+", default=["bicubic", "quicksrnet-small-y-x2"])
    parser.add_argument("--strength", type=float, default=1)
    parser.add_argument("--frame-format", choices=["nv12", "rgb24"], default="nv12")
    parser.add_argument(
        "--selection-lock", type=Path, default=Path("benchmarks/v0.5/selection-lock.json")
    )
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--json", type=Path, default=Path("outputs/v1-quality-development.json"))
    run(parser.parse_args())
