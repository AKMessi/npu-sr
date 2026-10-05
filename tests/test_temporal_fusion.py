"""Independent full-frame versus tiled inference catches effective halo mistakes."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from npu_sr.errors import SRException
from npu_sr.model import ModelSpec, sha256
from npu_sr.temporal_model import SHAPES, fuse_temporal, validate_weights


def spatial_fixture(path):
    nodes, values, previous = [], [], "input"
    for index in range(4):
        name = "cnn.0.weight" if index == 0 else f"kernel{index}"
        values.append(numpy_helper.from_array(np.full((1, 1, 3, 3), 1 / 9, np.float32), name))
        target = f"stage{index}"
        nodes.append(helper.make_node("Conv", [previous, name], [target], pads=[1] * 4))
        previous = target
    values.append(numpy_helper.from_array(np.ones((4, 1, 1, 1), np.float32), "phases"))
    nodes += [
        helper.make_node("Conv", [previous, "phases"], ["subpixels"]),
        helper.make_node("DepthToSpace", ["subpixels"], ["output"], blocksize=2, mode="DCR"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "independent-radius-four-fixture",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 264, 264])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 528, 528])],
            values,
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.save(model, path)


@pytest.mark.parametrize("width", [8, 16])
@pytest.mark.parametrize("geometry", [(256, 256), (320, 270)])
@pytest.mark.parametrize("channels", [2, 6])
def test_fused_tiled_output_matches_independent_full_context(
    tmp_path, monkeypatch, width, geometry, channels
):
    from npu_sr import temporal_model

    spatial = tmp_path / "spatial.onnx"
    spatial_fixture(spatial)
    monkeypatch.setattr(temporal_model, "SPATIAL_SHA256", sha256(spatial))
    rng = np.random.default_rng(88)
    shapes = SHAPES | {
        "conv0.weight": (width, channels + 4, 3, 3),
        "conv0.bias": (width,),
        "conv1.weight": (width, width, 3, 3),
        "conv1.bias": (width,),
        "conv2.weight": (4, width, 3, 3),
    }
    weights = {key: rng.normal(0, 0.02, shape).astype(np.float32) for key, shape in shapes.items()}
    core_width, core_height = geometry
    spec = ModelSpec(
        "test-fused",
        "test",
        core=core_width,
        core_height=core_height,
        halo=7,
        input_channels=channels,
    )
    output = tmp_path / "fused.onnx"
    fuse_temporal(spatial, weights, spec, output)
    tiled = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    full_model = onnx.load(output)
    full_width = core_width * 2 + 14
    full_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = full_width
    full_model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = full_width * 2
    full = ort.InferenceSession(full_model.SerializeToString(), providers=["CPUExecutionProvider"])
    tile_height, tile_width = spec.input_shape[2:]
    tensor = rng.random((1, channels, tile_height, full_width), dtype=np.float32)
    expected = full.run(None, {"input": tensor})[0][:, :, 14:-14, 14:-14]
    pieces = [
        tiled.run(None, {"input": tensor[..., offset : offset + tile_width].copy()})[0][
            :, :, 14:-14, 14:-14
        ]
        for offset in (0, core_width)
    ]
    np.testing.assert_allclose(np.concatenate(pieces, axis=3), expected, rtol=1e-5, atol=1e-7)
    # The radius-four spatial halo is insufficient for radius-three feature correction.
    wrong_model = copy.deepcopy(full_model)
    wrong_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = core_height + 8
    wrong_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = core_width + 8
    wrong_model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = (core_height + 8) * 2
    wrong_model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = (core_width + 8) * 2
    wrong = ort.InferenceSession(
        wrong_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    wrong_pieces = [
        wrong.run(None, {"input": tensor[:, :, 3:-3, offset + 3 : offset + tile_width - 3].copy()})[
            0
        ][:, :, 8:-8, 8:-8]
        for offset in (0, core_width)
    ]
    assert np.max(np.abs(np.concatenate(wrong_pieces, axis=3) - expected)) > 1e-6


def test_fusion_rejects_wrong_context_or_unverified_spatial_source(tmp_path):
    spatial = tmp_path / "spatial.onnx"
    spatial_fixture(spatial)
    weights = {key: np.ones(shape, np.float32) for key, shape in SHAPES.items()}
    with pytest.raises(SRException, match="pinned spatial"):
        fuse_temporal(
            spatial, weights, ModelSpec("test", "test", input_channels=2), tmp_path / "x.onnx"
        )
    assert not (tmp_path / "x.onnx").exists()


@pytest.mark.parametrize("kind", ["nonfinite", "wrong-keys", "zero-output", "mixed-width"])
def test_fusion_weight_contract(kind):
    weights = {key: np.ones(shape, np.float32) for key, shape in SHAPES.items()}
    if kind == "nonfinite":
        weights["conv0.bias"][0] = np.inf
    elif kind == "wrong-keys":
        weights.pop("conv0.bias")
    elif kind == "zero-output":
        weights["conv2.weight"][:] = 0
    else:
        weights["conv1.weight"] = np.ones((16, 16, 3, 3), np.float32)
    with pytest.raises(SRException):
        validate_weights(weights)


def test_public_fused_export_has_a_runtime_valid_manifest(tmp_path, monkeypatch):
    from npu_sr import temporal_model
    from npu_sr.model import validate_model

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    loader = importlib.util.spec_from_file_location(
        "fused_export", scripts / "export_fused_temporal.py"
    )
    exporter = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(exporter)
    spatial = tmp_path / "spatial.onnx"
    spatial_fixture(spatial)
    monkeypatch.setattr(temporal_model, "SPATIAL_SHA256", sha256(spatial))
    values = {key: np.ones(shape, np.float32) * 0.001 for key, shape in SHAPES.items()}
    weights = tmp_path / "weights.npz"
    np.savez(weights, **values)
    weights.with_suffix(".json").write_text(
        json.dumps(
            {
                "complete": True,
                "weights_sha256": sha256(weights),
                "input_channels": 6,
            }
        )
    )
    output = tmp_path / "fused.onnx"
    report = exporter.run(weights, spatial, output)
    assert validate_model(output) == report
    assert report["name"] == "QuickSRNet+TemporalResidual"
    assert report["sha256"] == sha256(output)
    assert report["experimental"] is True
