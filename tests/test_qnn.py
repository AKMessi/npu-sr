from types import SimpleNamespace

import pytest

from npu_sr import qnn
from npu_sr.errors import NPUUnavailable


@pytest.fixture
def catalog_mocks(monkeypatch):
    events = []
    result = SimpleNamespace(status=1, diagnostic_text="download failed", extended_error=-1)
    version = SimpleNamespace(major=2, minor=3, build=4, revision=5)
    provider = SimpleNamespace(
        name="QNNExecutionProvider",
        ready_state=SimpleNamespace(name="READY"),
        library_path="catalog-owned.dll",
        package_id=SimpleNamespace(version=version),
        ensure_ready_async=lambda: events.append("ensure") or SimpleNamespace(get=lambda: result),
    )
    catalog = SimpleNamespace(find_all_providers=lambda: [provider])
    winml = SimpleNamespace(
        ExecutionProviderCatalog=SimpleNamespace(get_default=lambda: catalog),
        ExecutionProviderReadyResultState=SimpleNamespace(SUCCESS=1),
    )
    npu = SimpleNamespace(ep_name="QNNExecutionProvider", device=SimpleNamespace(type=2))
    gpu = SimpleNamespace(ep_name="QNNExecutionProvider", device=SimpleNamespace(type=1))
    devices = [gpu, npu]
    ort = SimpleNamespace(
        register_execution_provider_library=lambda name, path: events.append(
            ("register", name, path)
        ),
        get_ep_devices=lambda: devices,
        OrtHardwareDeviceType=SimpleNamespace(NPU=2),
    )
    monkeypatch.setattr(qnn, "platform_problem", lambda: None)
    monkeypatch.setattr(qnn, "bootstrap_windows", lambda: events.append("bootstrap"))
    monkeypatch.setattr(qnn.importlib, "import_module", lambda _: winml)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", ort)
    monkeypatch.setattr(qnn, "_catalog", None)
    monkeypatch.setattr(qnn, "_provider", None)
    qnn.ensure_qnn.cache_clear()
    yield events, result, devices, npu
    qnn.ensure_qnn.cache_clear()


def test_catalog_flow_and_npu_selection(catalog_mocks):
    events, _, _, npu = catalog_mocks
    assert qnn.ensure_qnn() is npu
    assert events == [
        "bootstrap",
        "ensure",
        ("register", "QNNExecutionProvider", "catalog-owned.dll"),
    ]
    assert qnn.package_version() == "2.3.4.5"
    qnn.ensure_qnn()
    assert len(events) == 3  # Provider lifetime/registration is reused in one process.


def test_ready_failure_prevents_registration(catalog_mocks):
    events, result, _, _ = catalog_mocks
    result.status = 2
    with pytest.raises(NPUUnavailable, match="download failed"):
        qnn.ensure_qnn()
    assert events == ["bootstrap", "ensure"]


def test_qnn_gpu_cannot_satisfy_npu(catalog_mocks):
    _, _, devices, _ = catalog_mocks
    devices.pop()
    with pytest.raises(NPUUnavailable, match="no NPU device"):
        qnn.ensure_qnn()
