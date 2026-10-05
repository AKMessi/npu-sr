import json
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from npu_sr.model import INPUT_SHAPE, MODEL_VERSION, OUTPUT_SHAPE, model_path, sha256


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    """A real ORT-executable x2 nearest-neighbor model, no download or NPU required."""
    import onnx
    from onnx import numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "Resize",
                ["input", "", "scales"],
                ["output"],
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            )
        ],
        "test-only",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, INPUT_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, OUTPUT_SHAPE)],
        [numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = model_path(tmp_path)
    onnx.save(model, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": MODEL_VERSION,
                "sha256": sha256(path),
                "name": "test-only",
                "precision": "float32",
            }
        )
    )
    return path


@pytest.fixture
def temporal_hardware_model(tmp_path, monkeypatch):
    """Seeded test correction: verifies execution/state, never a quality model."""
    from npu_sr.model import MODELS, ModelSpec
    from npu_sr.temporal_model import SHAPES, fuse_temporal

    spec = ModelSpec("test-temporal-hardware", "test-temporal", halo=7, input_channels=2)
    monkeypatch.setitem(MODELS, spec.identifier, spec)
    values = {
        key: np.random.default_rng(index).normal(0, 0.001, shape).astype(np.float32)
        for index, (key, shape) in enumerate(SHAPES.items())
    }
    path = tmp_path / "test-temporal.onnx"
    fuse_temporal(model_path(identifier="quicksrnet-small-y-x2"), values, spec, path)
    path.with_suffix(".json").write_text(
        json.dumps(spec.info() | {"name": spec.identifier, "sha256": sha256(path)})
    )
    return path
