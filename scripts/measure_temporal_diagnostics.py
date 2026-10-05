"""Supplement immutable paired quality results with native-resolution temporal errors."""

import argparse
import json
from pathlib import Path

import numpy as np

from npu_sr.ffmpeg import PipeProcess, decode_args, probe, read_frame, tool_path
from npu_sr.model import sha256
from npu_sr.suite import environment
from npu_sr.video_quality import full_temporal_difference, validate_roi


def measure(actual: Path, reference: Path, roi: list[int], ffmpeg: Path, frames: int) -> dict:
    properties = [probe(path, ffmpeg) for path in (actual, reference)]
    if any(
        (p.width, p.height, p.fps) != (properties[0].width, properties[0].height, properties[0].fps)
        for p in properties
    ):
        raise ValueError("Temporal diagnostic requires aligned dimensions and cadence")
    info = properties[0]
    validate_roi(tuple(roi), info.width, info.height)
    x, y, width, height = roi
    children = []
    previous, samples = None, []
    try:
        for path in (actual, reference):
            children.append(PipeProcess(ffmpeg, decode_args(path, False, pixel_format="nv12")))
        count = 0
        while True:
            raw = [read_frame(child.process.stdout, info.bytes_for("nv12")) for child in children]
            if any(value is None for value in raw):
                if not all(value is None for value in raw):
                    raise ValueError("Unpaired temporal frames")
                break
            planes = [
                np.frombuffer(value, np.uint8, info.width * info.height)
                .reshape(info.height, info.width)[y + 2 : y + height - 2, x + 2 : x + width - 2]
                .astype(np.int16)
                for value in raw
            ]
            error = planes[0] - planes[1]
            if previous is not None:
                samples.append(full_temporal_difference(error, previous))
            previous = error
            count += 1
        for child in children:
            child.finish()
    finally:
        for child in children:
            child.close()
    if count != frames or len(samples) != frames - 1:
        raise ValueError("Temporal diagnostic frame count differs from frozen clip")
    return {
        "mean": float(np.mean(samples)),
        "p95_interval": float(np.percentile(samples, 95)),
        "intervals": samples,
        "decoded_frames": count,
    }


def run(args):
    provenance = json.loads((args.directory / "provenance-holdout.json").read_text())
    if (
        json.loads(args.manifest.read_text()) != provenance["definition"]
        or not provenance["complete"]
    ):
        raise ValueError("Prepared definition differs from published parsed settings")
    clips = {row["identifier"]: row for row in provenance["clips"]}
    baseline = json.loads(args.baseline_json.read_text())
    temporal = json.loads(args.temporal_json.read_text())
    if not baseline["complete"] or not temporal["complete"]:
        raise ValueError("Quality comparison is incomplete")
    ffmpeg = tool_path()
    report = {
        "complete": False,
        "environment": environment(),
        "script_sha256": sha256(Path(__file__)),
        "published_definition_sha256": sha256(args.manifest),
        "definition_sha256_at_preparation": provenance["definition_sha256"],
        "protocol": (
            "every-frame native coded-Y residual change /255; declared ROI with 2px shave; "
            "retain coarse diagnostic and standard metrics; no flow or academic metric claim"
        ),
        "results": [],
    }
    for row in baseline["results"] + temporal["results"]:
        clip = clips[row["clip"]]
        reference = args.directory / clip["reference_file"]
        if "model" in row:
            model = row["model"]
            actual = (
                args.directory / "delivered-tagged/holdout" / f"{row['clip']}-{model}-1-nv12.mp4"
            )
        else:
            model = "quicksrnet-temporal-feature-experiment"
            actual = args.directory / "delivered-fused-temporal-probe" / f"{row['clip']}-fused.mp4"
        if (
            sha256(actual) != row["metrics"]["output_sha256"]
            or sha256(reference) != row["metrics"]["reference_sha256"]
        ):
            raise ValueError("Scored output or reference bytes changed")
        metrics = measure(actual, reference, clip["metric_roi"], ffmpeg, clip["frames"])
        report["results"].append(
            {
                "clip": row["clip"],
                "model": model,
                "output_sha256": sha256(actual),
                "reference_sha256": sha256(reference),
                "metric_roi": clip["metric_roi"],
                "full_resolution_temporal_difference": metrics,
                "coarse_temporal_difference": row["metrics"]["temporal_difference_error"],
            }
        )
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(row["clip"], model, metrics["mean"], flush=True)
    report["complete"] = True
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/temporal-validation-v2.json")
    )
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--temporal-json", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    run(parser.parse_args())
