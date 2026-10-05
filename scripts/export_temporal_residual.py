"""Export verified original experiment weights; no Torch required for export."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from npu_sr.model import sha256

IDENTIFIER = "temporal-residual-v1-experiment"
SHAPES = {
    "conv0.weight": (8, 2, 3, 3),
    "conv0.bias": (8,),
    "conv1.weight": (8, 8, 3, 3),
    "conv1.bias": (8,),
    "conv2.weight": (4, 8, 3, 3),
    "conv2.bias": (4,),
}


def export(weights: Path, output: Path) -> dict:
    report = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
    channels = report.get("input_channels", 2)
    width = report.get("hidden_channels", 8)
    if channels not in {2, 6, 10}:
        raise ValueError("Unsupported temporal input channel count")
    if width not in {8, 16}:
        raise ValueError("Unsupported temporal hidden channel count")
    shapes = {
        "conv0.weight": (width, channels, 3, 3),
        "conv0.bias": (width,),
        "conv1.weight": (width, width, 3, 3),
        "conv1.bias": (width,),
        "conv2.weight": (4, width, 3, 3),
        "conv2.bias": (4,),
    }
    if (
        not report["complete"]
        or sha256(weights) != report["weights_sha256"]
        or weights.stat().st_size > 100_000
    ):
        raise ValueError("Temporal weights changed or training is incomplete")
    with zipfile.ZipFile(weights) as archive:
        members = archive.infolist()
        if (
            len(members) != len(SHAPES)
            or {member.filename for member in members} != {key + ".npy" for key in SHAPES}
            or any(member.file_size > 10_000 for member in members)
            or sum(member.file_size for member in members) > 20_000
        ):
            raise ValueError("Temporal weight archive exceeds its fixed array contract")
    with np.load(weights, allow_pickle=False) as archive:
        if set(archive.files) != set(SHAPES):
            raise ValueError("Unexpected temporal weight keys")
        values = {key: archive[key] for key in shapes}
    if any(
        v.shape != shapes[k] or v.dtype != np.float32 or not np.isfinite(v).all()
        for k, v in values.items()
    ):
        raise ValueError("Invalid temporal weight arrays")
    if not np.any(values["conv2.weight"]):
        raise ValueError("Untrained zero-output temporal graph is not a valid candidate")
    initializers = [numpy_helper.from_array(value, key) for key, value in values.items()]
    initializers += [
        numpy_helper.from_array(np.array(-0.05, np.float32), "minimum"),
        numpy_helper.from_array(np.array(0.05, np.float32), "maximum"),
    ]
    nodes, previous = [], "input"
    for index in range(3):
        name, target = f"conv{index}", f"feature{index}"
        nodes.append(
            helper.make_node(
                "Conv", [previous, name + ".weight", name + ".bias"], [target], pads=[1, 1, 1, 1]
            )
        )
        previous = target
        if index < 2:
            previous = f"relu{index}"
            nodes.append(helper.make_node("Relu", [target], [previous]))
    nodes += [
        helper.make_node("Clip", [previous, "minimum", "maximum"], ["bounded"]),
        helper.make_node("DepthToSpace", ["bounded"], ["output"], blocksize=2, mode="DCR"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            IDENTIFIER,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, 262, 262])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 524, 524])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    helper.set_model_props(
        model,
        {
            "license": "MIT",
            "version": IDENTIFIER + "-export1",
            "task": "bounded temporal luminance residual",
            "weights_sha256": sha256(weights),
        },
    )
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    manifest = {
        "identifier": IDENTIFIER,
        "name": IDENTIFIER,
        "version": IDENTIFIER + "-export1",
        "sha256": sha256(output),
        "source_sha256": sha256(weights),
        "license": "MIT",
        "precision": "float32 (QNN HTP uses float16 math)",
        "input_shape": [1, channels, 262, 262],
        "output_shape": [1, 1, 524, 524],
        "task": "temporal residual experiment",
        "training": report,
        "exporter_sha256": sha256(Path(__file__)),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(export(args.weights, args.output)["sha256"])
