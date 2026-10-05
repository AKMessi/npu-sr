"""Prepare licensed adjacent-frame patches for a tiny temporal residual experiment."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from npu_sr.corpus import acquire_source, prepare_clip, validate_definition
from npu_sr.ffmpeg import PipeProcess, decode_args, read_frame, tool_path
from npu_sr.model import resolve_model, sha256
from npu_sr.runtime import Runtime
from npu_sr.suite import environment


def decoded_y(path: Path, width: int, height: int, ffmpeg: Path):
    child = PipeProcess(ffmpeg, decode_args(path, False, pixel_format="nv12"))
    try:
        while (raw := read_frame(child.process.stdout, width * height * 3 // 2)) is not None:
            yield np.frombuffer(raw, np.uint8, width * height).reshape(height, width).copy()
        child.finish()
    finally:
        child.close()


def prepare(args: argparse.Namespace) -> None:
    definition = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_definition(definition)
    args.directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = tool_path()
    clips = [c for c in definition["clips"] if c["split"] == args.split]
    prepared = args.prepared_directory or args.directory
    if args.prepared_directory:
        if prepared.resolve() == args.directory.resolve():
            raise ValueError("Reused prepared clips need a separate sampling output directory")
        provenance = json.loads((prepared / f"provenance-{args.split}.json").read_text())
        if (
            not provenance["complete"]
            or provenance["definition"] != definition
            or {c["identifier"] for c in provenance["clips"]} != {c["identifier"] for c in clips}
            or len(provenance["clips"]) != len(clips)
        ):
            raise ValueError("Reused preparation differs from declared split")
        for clip in provenance["clips"]:
            for key in ("input", "reference"):
                if sha256(prepared / clip[f"{key}_file"]) != clip[f"{key}_sha256"]:
                    raise ValueError("Reused clip bytes changed")
        provenance["sampling_environment"] = environment()
        provenance["published_definition_sha256"] = sha256(args.manifest)
    else:
        sources = {
            key: acquire_source(value, args.source_directory, key)
            for key, value in definition["sources"].items()
        }
        provenance = {
            "definition": definition,
            "definition_sha256": sha256(args.manifest),
            "environment": environment(),
            "ffmpeg_sha256": sha256(ffmpeg),
            "complete": False,
            "clips": [],
        }
        for clip in clips:
            row = prepare_clip(clip, sources[clip["source"]], prepared, ffmpeg)
            provenance["clips"].append(row)
            (args.directory / f"provenance-{args.split}.json").write_text(
                json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
            )
            print("Prepared", clip["identifier"], flush=True)
        provenance["complete"] = True
    destination = args.directory / f"provenance-{args.split}.json"
    destination.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if args.split == "holdout":
        return
    config = definition["training"]
    baseline = resolve_model(config["baseline"])
    if sha256(baseline) != config["baseline_sha256"]:
        raise ValueError("Training teacher differs from frozen model")
    runtime = Runtime(baseline, args.teacher_device, performance_mode="burst")
    rng = np.random.default_rng(config["seed"])
    patch, halo = config["patch_lr"], config["halo"]
    fields = {key: [] for key in ("input0", "input1", "baseline0", "baseline1", "truth0", "truth1")}
    if args.recurrent_state:
        fields["baseline_previous"] = []
    identifiers = []
    for clip_index, clip in enumerate(provenance["clips"]):
        low = decoded_y(prepared / clip["input_file"], 960, 540, ffmpeg)
        high = decoded_y(prepared / clip["reference_file"], 1920, 1080, ffmpeg)
        history = deque(maxlen=3)
        count = 0
        try:
            for frame, truth in zip(low, high, strict=True):
                normalized = np.clip((frame.astype(np.float32) - 16) / 219, 0, 1)
                history.append((normalized, truth))
                if len(history) < 3 or count % 2:
                    count += 1
                    continue
                current = history[1]
                for _ in range(config["samples_per_triple"]):
                    following = history[2]
                    x = int(rng.integers(0, 960 - patch + 1))
                    top = clip["metric_roi"][1] // 2
                    bottom = (clip["metric_roi"][1] + clip["metric_roi"][3]) // 2
                    y = int(rng.integers(top, bottom - patch + 1))
                    sx, sy = min(max(x - 80, 0), 704), min(max(y - 80, 0), 284)
                    patches = []
                    for normalized, _truth in history:
                        padded = np.pad(normalized, halo, mode="edge")
                        patches.append(padded[y : y + patch + 2 * halo, x : x + patch + 2 * halo])
                    # Ten percent identical-frame triples train reset/initialization behavior.
                    reset = rng.random() < 0.1
                    if reset:
                        patches = [patches[1]] * 3
                        following = current
                    fields["input0"].append(np.stack(patches[:2]).astype(np.float16))
                    fields["input1"].append(np.stack(patches[1:]).astype(np.float16))
                    if args.recurrent_state:
                        if reset:
                            # Production resets fill each phase with the current LR sample.
                            previous = np.repeat(
                                np.repeat(current[0][y : y + patch, x : x + patch], 2, axis=0),
                                2,
                                axis=1,
                            )
                        else:
                            padded_previous = np.pad(history[0][0], 4, mode="edge")
                            tensor = padded_previous[sy : sy + 264, sx : sx + 264][
                                None, None
                            ].copy()
                            previous = runtime.run(tensor)[0, 0]
                            oy, ox = 8 + 2 * (y - sy), 8 + 2 * (x - sx)
                            previous = previous[oy : oy + patch * 2, ox : ox + patch * 2]
                        fields["baseline_previous"].append(previous[None].astype(np.float16))
                    for index, (normalized, truth) in enumerate((current, following)):
                        padded = np.pad(normalized, 4, mode="edge")
                        tensor = padded[sy : sy + 264, sx : sx + 264][None, None].copy()
                        output = runtime.run(tensor)[0, 0]
                        oy, ox = 8 + 2 * (y - sy), 8 + 2 * (x - sx)
                        fields[f"baseline{index}"].append(
                            output[oy : oy + patch * 2, ox : ox + patch * 2][None].astype(
                                np.float16
                            )
                        )
                        target = np.clip(
                            (
                                truth[y * 2 : (y + patch) * 2, x * 2 : (x + patch) * 2].astype(
                                    np.float32
                                )
                                - 16
                            )
                            / 219,
                            0,
                            1,
                        )
                        fields[f"truth{index}"].append(target[None].astype(np.float16))
                    identifiers.append(clip_index)
                count += 1
            if count != clip["frames"]:
                raise ValueError("Training sequence frame count differs from frozen clip")
        finally:
            low.close()
            high.close()
        print("Sampled", clip["identifier"], len(identifiers), "patch triples", flush=True)
    dataset = args.directory / "training-patches.npz"
    np.savez_compressed(
        dataset,
        **{key: np.stack(value) for key, value in fields.items()},
        clip=np.array(identifiers),
    )
    (args.directory / "training-data.json").write_text(
        json.dumps(
            {
                "dataset_sha256": sha256(dataset),
                "samples": len(identifiers),
                "dtype": "float16 storage; float32 training",
                "teacher": runtime.label,
                "teacher_device": runtime.backend,
                "teacher_model_sha256": sha256(baseline),
                "teacher_execution_evidence": runtime.evidence,
                "teacher_tensor_runs": runtime.run_calls,
                "recurrent_state": args.recurrent_state,
                "provenance_sha256": sha256(destination),
                "configuration": config,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Training dataset", len(identifiers), "samples", sha256(dataset), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/temporal-training-v1.json")
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--teacher-device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--prepared-directory", type=Path)
    parser.add_argument("--recurrent-state", action="store_true")
    prepare(parser.parse_args())
