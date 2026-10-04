"""Reproduce the v0.4 sustained gate or aligned preset quality/speed comparisons."""

import argparse
import json
import os
from pathlib import Path

from npu_sr.benchmark import save_report
from npu_sr.ffmpeg import codec_args, encode_evidence, run_tool, tool_path
from npu_sr.model import sha256
from npu_sr.suite import environment, quality_suite
from npu_sr.video import process_video, settings_for_preset
from npu_sr.video_benchmark import benchmark_video, evaluate_video, realtime_acceptance


def run(directory: Path, reports: Path, trials: int, quality: bool, images: Path) -> None:
    ffmpeg = tool_path()
    reports.mkdir(parents=True, exist_ok=True)
    overrides = {
        "device": "npu",
        "decode": "hardware",
        "encode": "hardware",
        "codec": "av1",
        "ffmpeg": ffmpeg,
        "cache_dir": directory / "qnn-cache",
        "overwrite": True,
    }
    if not quality:
        provenance = json.loads((directory / "sustained-provenance.json").read_text())
        source = directory / provenance["input_file"]
        if sha256(source) != provenance["input_sha256"]:
            raise ValueError("Sustained benchmark input hash changed")
        report = benchmark_video(
            source,
            directory / "realtime-output",
            settings_for_preset("realtime", overrides),
            trials,
            reports / "realtime-performance.json",
        )
        report["provenance"] = provenance
        report["realtime_acceptance"] = realtime_acceptance(report)
        save_report(report, reports / "realtime-performance.json")
        if not report["realtime_acceptance"]["passed"]:
            raise ValueError(f"Real-time gate failed: {report['realtime_acceptance']}")
        save_report(
            {
                "temperature": "not measured",
                "accelerator_utilization": "not measured",
                "power": "not measured",
                "trials": [
                    {
                        key: trial[key]
                        for key in (
                            "trial",
                            "processing_seconds",
                            "sustained",
                            "resources",
                            "resource_samples",
                            "process_memory",
                            "observed_queue_peaks",
                            "phase_statistics",
                        )
                    }
                    for trial in report["trials"]
                ],
            },
            reports / "thermal-or-sustained.json",
        )
    else:
        provenance = json.loads((directory / "provenance.json").read_text())
        report = {
            "environment": environment(),
            "provenance": provenance,
            "complete": False,
            "video": [],
            "preset_performance": [],
            "methodology": (
                "AV1 hardware 8M u_vbr/camera_record; delivered output vs aligned FFV1 HR; "
                "Rec601 Y PSNR/SSIM shave2 every12th frame; ordinal-clock VMAF v0.6.1; "
                "temporal residual all frames; different declared RGB/native-Y pipelines"
            ),
        }
        for clip in provenance["clips"]:
            name = clip["name"]
            source, reference = (
                directory / f"{name}-960x540.mp4",
                directory / f"{name}-reference.mkv",
            )
            if (
                sha256(source) != clip["inputs"]["960x540"]["sha256"]
                or sha256(reference) != clip["reference_sha256"]
            ):
                raise ValueError("Quality input/reference hash changed")
            for preset in ("bicubic", "realtime", "balanced", "quality"):
                output = directory / "v04-quality" / f"{name}-{preset}.mp4"
                output.parent.mkdir(exist_ok=True)
                if preset == "bicubic":
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
                    result = process_video(source, output, settings_for_preset(preset, overrides))
                    execution = {
                        key: result[key]
                        for key in (
                            "model",
                            "model_sha256",
                            "execution_evidence",
                            "codec_evidence",
                            "frame_format",
                            "npu_performance_requested",
                            "end_to_end_fps",
                        )
                    }
                metrics = evaluate_video(output, reference, ffmpeg, stride=12, vmaf=True)
                report["video"].append(
                    {"clip": name, "preset": preset, "execution": execution, **metrics}
                )
                save_report(report, reports / "quality.json")
                print(name, preset, metrics["mean_psnr_y_db"], metrics["mean_ssim_y"], flush=True)
        for preset in ("realtime", "balanced", "quality"):
            result = benchmark_video(
                directory / "faces-960x540.mp4",
                directory / "v04-pareto" / preset,
                settings_for_preset(preset, overrides),
                trials,
            )
            report["preset_performance"].append({"preset": preset, **result})
            save_report(report, reports / "quality.json")
        report["image"] = quality_suite(
            images, ["espcn-x2-256", "fsrcnn-x2"], ["cpu", "npu", "gpu"]
        )
        report["complete"] = True
        save_report(report, reports / "quality.json")
    save_report(
        environment()
        | {
            "ffmpeg_version": run_tool([str(ffmpeg), "-version"]).stdout.decode().splitlines()[0],
            "ffmpeg_sha256": sha256(ffmpeg),
        },
        reports / "environment.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=(
            Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "npu-sr/benchmarks/video"
        ),
    )
    parser.add_argument("--reports", type=Path, default=Path("outputs/v04-validation"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--images", type=Path, default=Path("datasets/bsds300-five"))
    args = parser.parse_args()
    run(args.directory, args.reports, args.trials, args.quality, args.images)
