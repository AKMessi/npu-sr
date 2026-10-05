"""Fuse an original six-feature correction with the separately acquired spatial model."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from export_temporal_residual import export
from temporal_experiment_contract import register_fused

from npu_sr.model import model_spec, sha256
from npu_sr.temporal_model import fuse_temporal


def run(weights: Path, spatial: Path, output: Path, geometry: str = "256") -> dict:
    # Apply the same fixed-shape, size and provenance checks as standalone export.
    with tempfile.TemporaryDirectory(prefix="npu-sr-temporal-export-") as directory:
        export(weights, Path(directory) / "residual.onnx")
    training = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
    if training.get("input_channels") not in {6, 10}:
        raise ValueError("Fused experiment requires six or ten teacher-feature input channels")
    with np.load(weights, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if training.get("recurrent_state"):
        if training["input_channels"] != 10:
            raise ValueError("Recurrent correction requires ten inputs")
    spec = model_spec(register_fused(geometry, bool(training.get("recurrent_state"))))
    output.parent.mkdir(parents=True, exist_ok=True)
    fuse_temporal(spatial, values, spec, output)
    manifest = spec.info() | {
        "name": spec.architecture,
        "sha256": sha256(output),
        "source": "original local temporal training; experimental, not a quality guarantee",
        "weights_sha256": sha256(weights),
        "spatial_sha256": sha256(spatial),
        "training": training,
        "exporter_sha256": sha256(Path(__file__)),
        "experimental": True,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("spatial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--geometry", choices=["256", "320x270"], default="256")
    arguments = parser.parse_args()
    print(run(arguments.weights, arguments.spatial, arguments.output, arguments.geometry)["sha256"])
