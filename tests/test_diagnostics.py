from npu_sr import diagnostics, qnn
from npu_sr.errors import NPUUnavailable


def test_unsupported_os(monkeypatch):
    monkeypatch.setattr(qnn.sys, "platform", "linux")
    assert "Windows 11" in qnn.platform_problem()


def test_wrong_python_architecture(monkeypatch):
    monkeypatch.setattr(qnn.sys, "platform", "win32")
    monkeypatch.setattr(qnn.platform, "machine", lambda: "AMD64")
    assert "ARM64 Python" in qnn.platform_problem()


def test_doctor_does_not_claim_readiness(tiny_model, monkeypatch):
    monkeypatch.setattr(diagnostics, "hardware_info", lambda: {})
    monkeypatch.setattr(diagnostics, "platform_problem", lambda: None)

    def unavailable():
        raise NPUUnavailable("QNN unavailable")

    monkeypatch.setattr(diagnostics, "ensure_qnn", unavailable)
    report = diagnostics.diagnose(tiny_model)
    assert not report["ready"]
    assert report["cpu_backend"] == "available"
    assert "QNN unavailable" in report["errors"]
