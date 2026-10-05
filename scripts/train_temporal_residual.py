"""Train a tiny residual on real frame triples; training-only Torch."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class TinyResidual(nn.Module):
    """Previous/current Y to bounded subpixel correction; no recurrent growth."""

    def __init__(self, input_channels=2, hidden_channels=8):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, 4, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, value):
        value = torch.relu(self.conv0(value))
        value = torch.relu(self.conv1(value))
        return torch.pixel_shuffle(torch.clamp(self.conv2(value), -0.05, 0.05), 2)


def delivered_luminance(value):
    """Straight-through limited-range 8-bit rounding; not network quantization."""
    rounded = (torch.clamp(torch.round(value * 219 + 16), 16, 235) - 16) / 219
    return value + (rounded - value).detach()


def train(args: argparse.Namespace) -> None:
    script_digest = digest(Path(__file__))
    metadata = json.loads((args.directory / "training-data.json").read_text(encoding="utf-8"))
    dataset = args.directory / "training-patches.npz"
    if digest(dataset) != metadata["dataset_sha256"]:
        raise ValueError("Temporal training dataset SHA256 mismatch")
    config = metadata["configuration"]
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config["seed"])
    with np.load(dataset, allow_pickle=False) as archive:
        keys = ["input0", "input1", "baseline0", "baseline1", "truth0", "truth1"]
        if args.recurrent_state:
            if not metadata.get("recurrent_state"):
                raise ValueError("Recurrent training needs prepared previous-frame features")
            keys.append("baseline_previous")
        fields = {key: archive[key] for key in keys}
    count = len(fields["input0"])
    if not count or any(
        len(value) != count or not np.isfinite(value).all() for value in fields.values()
    ):
        raise ValueError("Invalid temporal training arrays")
    if args.recurrent_state and (not args.teacher_features or not args.output_rounding):
        raise ValueError(
            "Recurrent training requires teacher features and delivered output rounding"
        )
    channels = 10 if args.recurrent_state else (6 if args.teacher_features else 2)
    model = TinyResidual(channels, args.hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    started = time.perf_counter()
    samples = []
    border = config["halo"] * 2
    architecture = (
        f"Conv({model.conv0.in_channels},{args.hidden_channels},3)-Relu-"
        f"Conv({args.hidden_channels},{args.hidden_channels},3)-Relu-"
        f"Conv({args.hidden_channels},4,3)-Clip(-0.05,0.05)-DepthToSpace(2)"
    )

    def charbonnier(value):
        return torch.sqrt(value.square() + 1e-6).mean()

    for step in range(config["steps"]):
        selected = rng.integers(count, size=config["batch_size"])
        batch = {key: torch.from_numpy(value[selected]).float() for key, value in fields.items()}
        inputs = [batch[f"input{i}"] for i in range(2)]
        output = []
        previous = delivered_luminance(batch["baseline_previous"]) if args.recurrent_state else None
        for index in range(2):
            features = [inputs[index]]
            if args.recurrent_state:
                phases = torch.pixel_unshuffle(previous, 2)
                features.append(torch.nn.functional.pad(phases, (3, 3, 3, 3), mode="replicate"))
            if args.teacher_features:
                phases = torch.pixel_unshuffle(batch[f"baseline{index}"], 2)
                features.append(torch.nn.functional.pad(phases, (3, 3, 3, 3), mode="replicate"))
            correction = model(torch.cat(features, dim=1))[..., border:-border, border:-border]
            current = batch[f"baseline{index}"] + correction
            if args.output_rounding:
                current = delivered_luminance(current)
            output.append(current)
            previous = current  # Keep the gradient through both consecutive predictions.
        errors = [output[i] - batch[f"truth{i}"] for i in range(2)]
        if args.teacher_features:
            # Exclude the receptive radius of the padded teacher feature boundary.
            margin = 12 if args.recurrent_state else 6
            errors = [error[..., margin:-margin, margin:-margin] for error in errors]
        spatial = (charbonnier(errors[0]) + charbonnier(errors[1])) * 0.5
        temporal = charbonnier(errors[1] - errors[0])
        loss = spatial + args.temporal_weight * temporal
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == config["steps"] - 1:
            row = {
                "step": step + 1,
                "spatial_loss": float(spatial.detach()),
                "temporal_loss": float(temporal.detach()),
                "loss": float(loss.detach()),
            }
            samples.append(row)
            print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output, **{key: value.detach().numpy() for key, value in model.state_dict().items()}
    )
    report = {
        "complete": True,
        "trainer_sha256": script_digest,
        "seed": config["seed"],
        "configuration": config,
        "temporal_weight": args.temporal_weight,
        "input_channels": channels,
        "hidden_channels": args.hidden_channels,
        "architecture": architecture,
        "teacher_features": args.teacher_features,
        "recurrent_state": args.recurrent_state,
        "recurrent_unroll_frames": 2 if args.recurrent_state else None,
        "output_rounding": args.output_rounding,
        "output_rounding_definition": (
            "straight-through round/clamp to limited-range 8-bit Y; "
            "inference graph remains FP32, QNN HTP uses FP16 math"
        )
        if args.output_rounding
        else None,
        "teacher_feature_boundary": (
            "replicate padded; recurrent loss excludes 12 HR pixels"
            if args.recurrent_state
            else "replicate padded; loss excludes 6 HR pixels"
        )
        if args.teacher_features
        else None,
        "samples": count,
        "dataset_sha256": digest(dataset),
        "training_teacher": metadata.get("teacher", "native ORT CPU FP32"),
        "teacher_execution_evidence": metadata.get("teacher_execution_evidence"),
        "provenance_sha256": metadata["provenance_sha256"],
        "weights_sha256": digest(args.output),
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "CPU",
        "process_architecture": platform.machine(),
        "os": platform.system(),
        "python": platform.python_version(),
        "threads": 4,
        "training_seconds": time.perf_counter() - started,
        "loss_log": samples,
        "checkpoint_license": config["checkpoint_license"],
        "status": "trained experiment only; no quality or NPU success claimed",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Weights", report["weights_sha256"], "seconds", report["training_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-weight", type=float, choices=[0, 0.25, 1, 4], default=0.25)
    parser.add_argument("--teacher-features", action="store_true")
    parser.add_argument("--output-rounding", action="store_true")
    parser.add_argument("--hidden-channels", type=int, choices=[8, 16], default=8)
    parser.add_argument("--recurrent-state", action="store_true")
    train(parser.parse_args())
