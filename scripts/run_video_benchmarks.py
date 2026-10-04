"""Run the small v0.3 video matrix and delivered-output quality comparisons."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from npu_sr.benchmark import save_report
from npu_sr.ffmpeg import codec_args, encode_evidence, run_tool, tool_path
from npu_sr.model import sha256
from npu_sr.suite import environment
from npu_sr.video import VideoSettings, process_video
from npu_sr.video_benchmark import (
    benchmark_video,
    decode_throughput,
    encode_throughput,
    evaluate_video,
)


def run(directory: Path, reports: Path, trials: int, quality: bool) -> None:
    ffmpeg = tool_path()
    provenance = json.loads((directory / "provenance.json").read_text())
    for clip in provenance["clips"]:
        if sha256(directory / f"{clip['name']}-reference.mkv") != clip["reference_sha256"]:
            raise ValueError("Benchmark reference hash changed; regenerate the clips")
        for item in clip["inputs"].values():
            if sha256(directory / item["file"]) != item["sha256"]:
                raise ValueError("Benchmark input hash changed; regenerate the clips")
    reports.mkdir(parents=True, exist_ok=True)
    settings = VideoSettings(
        device="npu",
        decode="hardware",
        encode="hardware",
        codec="h264",
        ffmpeg=ffmpeg,
        cache_dir=directory / "qnn-cache",
        overwrite=True,
    )
    if quality:
        settings = replace(settings, codec="av1")
        report = {
            "schema_version": 3,
            "environment": environment(),
            "provenance": provenance,
            "complete": False,
            "results": [],
        }
        for clip in provenance["clips"]:
            name = clip["name"]
            source = directory / f"{name}-960x540.mp4"
            reference = directory / f"{name}-reference.mkv"
            for model in ("bicubic", "espcn-x2-256", "fsrcnn-x2"):
                output = directory / "quality-output" / f"{name}-{model}.mp4"
                output.parent.mkdir(exist_ok=True)
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
                            *codec_args("av1", True, "8M", 20),
                            "-c:a",
                            "copy",
                            str(output),
                        ]
                    )
                    execution = {
                        "method": "FFmpeg bicubic",
                        "encode": encode_evidence(completed.stderr.decode("utf-8", "replace")),
                    }
                else:
                    result = process_video(source, output, replace(settings, model=model))
                    execution = {
                        "neural": result["execution_evidence"],
                        "codecs": result["codec_evidence"],
                        "model_sha256": result["model_sha256"],
                        "end_to_end_fps": result["end_to_end_fps"],
                    }
                metrics = evaluate_video(output, reference, ffmpeg, stride=12, vmaf=True)
                report["results"].append(
                    {"clip": name, "model": model, "execution": execution, **metrics}
                )
                save_report(report, reports / "video-quality.json")
                print(name, model, metrics["mean_psnr_y_db"], metrics["mean_ssim_y"], flush=True)
        report.update(
            complete=True,
            methodology=(
                "delivered AV1 hardware output at 8M vs aligned FFV1 HR; "
                "full-range Rec601 Y, shave2, every12th frame; temporal residual everyframe"
            ),
        )
        save_report(report, reports / "video-quality.json")
    else:
        report = {
            "schema_version": 3,
            "environment": environment(),
            "provenance": provenance,
            "complete": False,
            "results": [],
            "standalone_codecs": [],
        }
        for width, height in ((640, 360), (960, 540), (1280, 720)):
            source = directory / f"faces-{width}x{height}.mp4"
            for backend in ("cpu", "npu"):
                destination = directory / "performance-output" / f"{width}x{height}-{backend}"
                measured = benchmark_video(
                    source, destination, replace(settings, device=backend), trials
                )
                report["results"].append(
                    {"resolution": [width, height], "backend": backend, **measured}
                )
                save_report(report, reports / "video-performance.json")
            reference = directory / f"codec-{width * 2}x{height * 2}.mkv"
            run_tool(
                [
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(directory / "faces-reference.mkv"),
                    "-vf",
                    f"scale={width * 2}:{height * 2}:flags=bicubic",
                    "-an",
                    "-c:v",
                    "ffv1",
                    str(reference),
                ],
                timeout=300,
            )
            for trial in range(trials):
                report["standalone_codecs"].append(
                    {
                        "resolution": [width, height],
                        "trial": trial + 1,
                        "decode": decode_throughput(source, ffmpeg, True),
                        "encode": encode_throughput(reference, ffmpeg),
                    }
                )
        report["complete"] = True
        save_report(report, reports / "video-performance.json")
    save_report(
        environment()
        | {
            "ffmpeg_version": run_tool([str(ffmpeg), "-version"]).stdout.decode().splitlines()[0],
            "ffmpeg_sha256": sha256(ffmpeg),
        },
        reports / "environment.json",
    )
    capabilities = []
    for argument in ("-version", "-hwaccels", "-encoders"):
        completed = run_tool([str(ffmpeg), "-hide_banner", argument])
        capabilities.append(completed.stdout.decode("utf-8", "replace"))
    (reports / "ffmpeg-capabilities.txt").write_text("\n".join(capabilities), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
        / "npu-sr"
        / "benchmarks"
        / "video"
    )
    parser.add_argument("--directory", type=Path, default=default)
    parser.add_argument("--reports", type=Path, default=Path("outputs/v03-validation"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    run(args.directory, args.reports, args.trials, args.quality)
