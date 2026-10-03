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
