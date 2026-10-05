"""Original experiment export: inert arrays, bounded correction and channel phases."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from npu_sr.model import sha256

_spec = importlib.util.spec_from_file_location(
    "temporal_export", Path(__file__).parents[1] / "scripts/export_temporal_residual.py"
)
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


def checkpoint(tmp_path, values=None, channels=2, width=8):
    values = values or {key: np.zeros(shape, np.float32) for key, shape in exporter.SHAPES.items()}
    weights = tmp_path / "weights.npz"
    np.savez(weights, **values)
    weights.with_suffix(".json").write_text(
        json.dumps(
            {
                "complete": True,
                "weights_sha256": sha256(weights),
                "input_channels": channels,
                "hidden_channels": width,
            }
        )
    )
    return weights


def test_export_rejects_untrained_output(tmp_path):
    weights = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Untrained"):
        exporter.export(weights, tmp_path / "result.onnx")


def test_export_requires_verified_weight_bytes(tmp_path):
    weights = checkpoint(tmp_path)
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        exporter.export(weights, tmp_path / "result.onnx")


@pytest.mark.parametrize("kind", ["nonfinite", "dtype", "shape", "archive"])
def test_export_rejects_invalid_arrays(tmp_path, kind):
    values = {key: np.ones(shape, np.float32) for key, shape in exporter.SHAPES.items()}
    if kind == "nonfinite":
        values["conv0.bias"][0] = np.nan
    elif kind == "dtype":
        values["conv0.bias"] = values["conv0.bias"].astype(np.float64)
    elif kind == "shape":
        values["conv0.bias"] = values["conv0.bias"][:1]
    else:
        values["extra"] = np.ones(50_000, np.uint8)
    with pytest.raises(ValueError, match="archive|Invalid"):
        exporter.export(checkpoint(tmp_path, values), tmp_path / "result.onnx")


def test_export_matches_independent_two_channel_phase_reference(tmp_path):
    values = {key: np.zeros(shape, np.float32) for key, shape in exporter.SHAPES.items()}
    values["conv0.weight"][0, 0, 1, 1] = 1
    values["conv0.weight"][1, 1, 1, 1] = 1
    values["conv1.weight"][0, 0, 1, 1] = 1
    values["conv1.weight"][1, 1, 1, 1] = 1
    for phase, coefficient in enumerate([0.01, 0.02, 0.03, 0.04]):
        values["conv2.weight"][phase, 0, 1, 1] = coefficient
        values["conv2.weight"][phase, 1, 1, 1] = -coefficient
    output = tmp_path / "result.onnx"
    manifest = exporter.export(checkpoint(tmp_path, values), output)
    tensor = np.random.default_rng(52).uniform(-1, 1, (1, 2, 262, 262)).astype(np.float32)
    expected = np.empty((1, 1, 524, 524), np.float32)
    difference = np.maximum(tensor[:, :1], 0) - np.maximum(tensor[:, 1:], 0)
    for phase, coefficient in enumerate([0.01, 0.02, 0.03, 0.04]):
        expected[..., phase // 2 :: 2, phase % 2 :: 2] = np.clip(
            coefficient * difference, -0.05, 0.05
        )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": tensor})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    assert manifest["sha256"] == sha256(output)


@pytest.mark.parametrize("channels", [6, 10])
def test_feature_variant_preserves_channel_contract(tmp_path, channels):
    values = {key: np.ones(shape, np.float32) for key, shape in exporter.SHAPES.items()}
    values["conv0.weight"] = np.ones((8, channels, 3, 3), np.float32)
    output = tmp_path / "features.onnx"
    manifest = exporter.export(checkpoint(tmp_path, values, channels), output)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, channels, 262, 262]
    assert manifest["input_shape"] == [1, channels, 262, 262]


def test_unknown_channel_variant_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="channel"):
        exporter.export(checkpoint(tmp_path, channels=3), tmp_path / "invalid.onnx")


def test_wide_variant_exports_actual_weight_dimensions(tmp_path):
    shapes = {
        "conv0.weight": (16, 6, 3, 3),
        "conv0.bias": (16,),
        "conv1.weight": (16, 16, 3, 3),
        "conv1.bias": (16,),
        "conv2.weight": (4, 16, 3, 3),
        "conv2.bias": (4,),
    }
    values = {key: np.ones(shape, np.float32) * 0.01 for key, shape in shapes.items()}
    output = tmp_path / "wide.onnx"
    exporter.export(checkpoint(tmp_path, values, channels=6, width=16), output)
    tensor = np.ones((1, 6, 262, 262), np.float32)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": tensor})[0]
    assert actual.shape == (1, 1, 524, 524)
    assert np.isfinite(actual).all() and np.max(actual) <= 0.05


def test_unknown_width_is_rejected_before_loading_arrays(tmp_path):
    with pytest.raises(ValueError, match="hidden channel"):
        exporter.export(checkpoint(tmp_path, width=32), tmp_path / "invalid.onnx")
