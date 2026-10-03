from pathlib import Path

import numpy as np
import pytest

from npu_sr.errors import NPUUnavailable, SRException
from npu_sr.model import INPUT_SHAPE, model_path
from npu_sr.runtime import Runtime, profile_providers


def test_cpu_inference(tiny_model):
    runtime = Runtime(tiny_model, "cpu")
    assert runtime.backend == "cpu"
    assert runtime.session.get_providers() == ["CPUExecutionProvider"]
    tensor = np.arange(np.prod(INPUT_SHAPE), dtype=np.float32).reshape(INPUT_SHAPE)
    np.testing.assert_array_equal(runtime.run(tensor), tensor.repeat(2, 2).repeat(2, 3))


def test_npu_never_falls_back(tiny_model, monkeypatch):
    def unavailable(*_):
        raise NPUUnavailable("no NPU")

    monkeypatch.setattr(Runtime, "_npu", unavailable)
    with pytest.raises(NPUUnavailable, match="no NPU"):
        Runtime(tiny_model, "npu")
    automatic = Runtime(tiny_model, "auto")
    assert automatic.backend == "cpu"
    assert automatic.fallback_reason == "no NPU"


def test_model_integrity_and_bad_tensor(tiny_model):
    runtime = Runtime(tiny_model, "cpu")
    with pytest.raises(SRException, match="Expected float32"):
        runtime.run(np.zeros((2, 3), np.float64))
    tiny_model.write_bytes(b"tampered")
    with pytest.raises(SRException, match="SHA256"):
        Runtime(tiny_model, "cpu")
    with pytest.raises(SRException, match="Model missing"):
        Runtime(Path("absent-model.onnx"), "npu")


def test_malformed_manifest_is_actionable(tiny_model):
    tiny_model.with_suffix(".json").write_text("[]")
    with pytest.raises(SRException, match="expected a JSON object"):
        Runtime(tiny_model, "cpu")


def test_profile_ignores_fences():
    events = [
        {"cat": "Node", "args": {"provider": "QNNExecutionProvider"}},
        {"cat": "Node", "args": {}},
        {"cat": "Node", "args": {"provider": "CPUExecutionProvider"}},
    ]
    assert profile_providers(events) == {"QNNExecutionProvider": 1, "CPUExecutionProvider": 1}


@pytest.mark.parametrize(
    "providers,accepted",
    [
        ([], False),
        (["CPUExecutionProvider"], False),
        (["QNNExecutionProvider", "CPUExecutionProvider"], False),
        (["QNNExecutionProvider"], True),
    ],
)
def test_actual_session_profile_is_required(tiny_model, monkeypatch, providers, accepted):
    import json
    from types import SimpleNamespace

    from npu_sr import runtime as module
    from npu_sr.model import OUTPUT_SHAPE

    options_created = []

    class Options:
        def __init__(self):
            self.config = {}
            options_created.append(self)

        def add_session_config_entry(self, name, value):
            self.config[name] = value

        def add_provider_for_devices(self, devices, config):
            self.devices, self.provider_config = devices, config

    class Session:
        def __init__(self, path, sess_options, providers):
            self.options = sess_options
            self.fallback_disabled = False

        def disable_fallback(self):
            self.fallback_disabled = True

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=INPUT_SHAPE, type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=OUTPUT_SHAPE)]

        def get_providers(self):
            return ["QNNExecutionProvider", "CPUExecutionProvider"]

        def run(self, names, feed):
            assert self.fallback_disabled
            assert self.options.config["session.disable_cpu_ep_fallback"] == "1"
            assert self.options.provider_config["backend_type"] == "htp"
            return [np.zeros(OUTPUT_SHAPE, np.float32)]

        def end_profiling(self):
            path = Path(self.options.profile_file_prefix + ".json")
            path.write_text(
                json.dumps(
                    [{"cat": "Node", "args": {"provider": provider}} for provider in providers]
                )
            )
            return str(path)

    monkeypatch.setattr(
        module, "ort", SimpleNamespace(SessionOptions=Options, InferenceSession=Session)
    )
    monkeypatch.setattr(module, "ensure_qnn", lambda: "NPU-device")
    monkeypatch.setattr(module, "package_version", lambda: "test")
    if accepted:
        result = Runtime(tiny_model, "npu")
        assert result.backend == "npu"
    else:
        with pytest.raises(NPUUnavailable):
            Runtime(tiny_model, "npu")
    assert not Path(options_created[0].profile_file_prefix).parent.exists()


def test_real_downloaded_cpu_model_if_available():
    path = model_path()
    if not path.exists():
        pytest.skip("downloaded pretrained model not available")
    runtime = Runtime(path, "cpu")
    output = runtime.run(np.full(INPUT_SHAPE, 0.5, np.float32))
    assert np.isfinite(output).all()
    assert abs(float(output[0, 0, 16:-16, 16:-16].mean()) - 0.5) < 0.02


@pytest.mark.npu
def test_real_qnn_matches_cpu():
    path = model_path()
    if not path.exists():
        pytest.fail("Run python scripts/download_model.py before NPU hardware tests")
    cpu, npu = Runtime(path, "cpu"), Runtime(path, "npu", verbose=True)
    tensor = np.random.default_rng(11).random(INPUT_SHAPE, dtype=np.float32)
    expected, actual = cpu.run(tensor), npu.run(tensor)
    np.testing.assert_allclose(actual, expected, atol=0.015, rtol=0.02)
    assert npu.evidence["executed_kernel_counts"] == {"QNNExecutionProvider": 1}
