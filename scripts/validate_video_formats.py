"""Small executed codec/cadence/orientation/audio matrix, not performance headline data."""

import argparse
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from npu_sr.benchmark import save_report
from npu_sr.errors import SRException
from npu_sr.ffmpeg import codec_args, probe, run_tool, software_av1_encoder, tool_path
from npu_sr.model import sha256
from npu_sr.video import process_video, settings_for_preset

# Deliberate coverage rather than a claim that every cross-product was tested.
CASES = [
    ("film-23976", "h264", 640, 360, "24000/1001", "aac", "stereo", "mp4"),
    ("hevc-24", "hevc", 854, 480, "24", None, None, "mkv"),
    ("av1-25", "av1", 960, 540, "25", "aac", "stereo", "mp4"),
    ("ntsc-2997", "h264", 960, 540, "30000/1001", "aac", "stereo", "mp4"),
    ("primary-30", "h264", 960, 540, "30", "aac", "stereo", "mp4"),
    ("hevc-50", "hevc", 1280, 720, "50", None, None, "mp4"),
    ("ntsc-5994", "h264", 854, 480, "60000/1001", "aac", "stereo", "mp4"),
    ("av1-60", "av1", 640, 360, "60", None, None, "mp4"),
    ("portrait", "h264", 360, 640, "30", "aac", "stereo", "mp4"),
    ("surround", "h264", 640, 360, "30", "aac", "5.1", "mp4"),
    ("opus-mkv", "h264", 640, 360, "30", "libopus", "stereo", "mkv"),
    ("1080p", "h264", 1920, 1080, "30", None, None, "mp4"),
]


def run(args: argparse.Namespace) -> None:
    ffmpeg = tool_path(args.ffmpeg)
    args.directory.mkdir(parents=True, exist_ok=True)
    settings = settings_for_preset(
        "realtime" if args.device == "npu" else "balanced",
        {
            "device": args.device,
            "decode": "hardware" if args.device == "npu" else "software",
            "encode": "hardware" if args.device == "npu" else "software",
            "codec": "av1" if args.device == "npu" else "h264",
            "ffmpeg": ffmpeg,
            "overwrite": True,
        },
    )
    if args.output_codec:
        settings = replace(settings, codec=args.output_codec)
    report = {
        "complete": False,
        "scope": "twelve-frame generated SDR correctness cases, not sustained performance",
        "results": [],
    }
    for identifier, codec, width, height, fps, audio, layout, container in CASES:
        if args.cases and identifier not in args.cases:
            continue
        source = args.directory / f"{identifier}-source.{container}"
        output = args.directory / f"{identifier}-enhanced.{container}"
        row = {
            "case": identifier,
            "requested": {
                "source_codec": codec,
                "resolution": [width, height],
                "fps": fps,
                "audio_codec": audio,
                "channel_layout": layout,
                "container": container,
            },
        }
        try:
            options = [
                str(ffmpeg),
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={width}x{height}:rate={fps}",
            ]
            if audio:
                options += [
                    "-f",
                    "lavfi",
                    "-i",
                    f"anullsrc=channel_layout={layout}:sample_rate=48000",
                ]
            options += ["-frames:v", "12", "-t", str(12 / float(Fraction(fps)))]
            options += codec_args(
                codec,
                False,
                "8M",
                20,
                software_av1_encoder(ffmpeg) if codec == "av1" else "libaom-av1",
            )
            if codec == "hevc":
                options += ["-x265-params", "pools=2:frame-threads=1"]
            options += ["-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709"]
            options += ["-c:a", audio] if audio else ["-an"]
            run_tool([*options, str(source)], timeout=180)
            metadata = probe(source, ffmpeg)
            execution = process_video(source, output, replace(settings))
            if execution["frames_processed"] != 12 or execution["output"]["audio"] != bool(audio):
                raise SRException("Matrix frame/audio preservation failed")
            if metadata.audio_channels != execution["output"]["audio_channels"]:
                raise SRException("Copied channel count changed")
            row.update(
                status="passed",
                source_sha256=sha256(source),
                output_sha256=sha256(output),
                execution=execution,
            )
        except (SRException, OSError) as exc:
            # Keep every requested case, including genuine unsupported paths.
            row.update(status="failed", reason=str(exc))
        report["results"].append(row)
        save_report(report, args.json)
        print(identifier, row["status"], row.get("reason", ""), flush=True)
    report.update(
        complete=True,
        all_requested_paths_proven=all(row["status"] == "passed" for row in report["results"]),
    )
    save_report(report, args.json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--output-codec", choices=["h264", "hevc", "av1"])
    parser.add_argument("--cases", nargs="+", choices=[row[0] for row in CASES])
    run(parser.parse_args())
