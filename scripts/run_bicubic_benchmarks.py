"""Measure a conventional 2x bicubic pipeline with the same proven hardware codecs."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from npu_sr.benchmark import save_report
from npu_sr.errors import SRException
from npu_sr.ffmpeg import (
    codec_args,
    decode_evidence,
    encode_evidence,
    probe,
    run_tool,
    tool_path,
    validate_cfr,
)
from npu_sr.model import sha256
from npu_sr.suite import environment


def run(source: Path, directory: Path, reports: Path, trials: int) -> dict:
    ffmpeg = tool_path()
    info = probe(source, ffmpeg)
    count = validate_cfr(source, ffmpeg, info)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for trial in range(1, trials + 1):
        output = directory / f"trial-{trial}.mp4"
        started = perf_counter()
        completed = run_tool(
            [
                str(ffmpeg),
                "-y",
                "-v",
                "debug",
                "-hwaccel",
                "d3d11va",
                "-hwaccel_output_format",
                "d3d11",
                "-i",
                str(source),
                "-vf",
                f"hwdownload,format=nv12,scale={info.width * 2}:{info.height * 2}:flags=bicubic",
                *codec_args("av1", True, "8M", 20),
                "-c:a",
                "copy",
                "-fps_mode",
                "passthrough",
                str(output),
            ],
            timeout=300,
        )
        seconds = perf_counter() - started
        log = completed.stderr.decode("utf-8", "replace")
        actual = probe(output, ffmpeg, count_frames=True)
        if (
            validate_cfr(output, ffmpeg, actual) != count
            or actual.frames != count
            or actual.fps != info.fps
            or [actual.width, actual.height] != [info.width * 2, info.height * 2]
            or (info.audio and not actual.audio)
            or abs(actual.duration - count / float(info.fps)) > max(0.1, 2 / float(info.fps))
        ):
            raise SRException("Bicubic output failed count/cadence/size/audio validation")
        result = dict(
            trial=trial,
            frames=count,
            processing_seconds=seconds,
            end_to_end_fps=count / seconds,
            mean_processing_budget_ms=seconds / count * 1000,
            codec_evidence=dict(decode=decode_evidence(log, True), encode=encode_evidence(log)),
            input=info.public(),
            output=actual.public(),
            output_sha256=sha256(output),
        )
        results.append(result)
        print(f"Trial {trial}: {count / seconds:.1f} FPS", flush=True)
    report = dict(
        environment=environment(),
        input_sha256=sha256(source),
        complete=True,
        ffmpeg_sha256=sha256(ffmpeg),
        ffmpeg_version=run_tool([str(ffmpeg), "-version"]).stdout.decode().splitlines()[0],
        backend="FFmpeg CPU bicubic",
        codec="av1",
        bitrate="8M",
        trials=results,
        median_trial_end_to_end_fps=float(np.median([r["end_to_end_fps"] for r in results])),
        timing_scope=(
            "whole FFmpeg process: hardware decode/download, CPU scale, hardware encode/flush; "
            "excludes output safety audit"
        ),
        aggregation="median across complete short trials; not a sustained benchmark",
    )
    save_report(report, reports)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "npu-sr/benchmarks/video"
    parser.add_argument("--input", type=Path, default=root / "faces-960x540.mp4")
    parser.add_argument("--directory", type=Path, default=root / "v04-bicubic-speed")
    parser.add_argument(
        "--json", type=Path, default=Path("outputs/v04-monthly-final/bicubic-performance.json")
    )
    parser.add_argument("--trials", type=int, choices=range(3, 21), default=3)
    args = parser.parse_args()
    run(args.input, args.directory, args.json, args.trials)
