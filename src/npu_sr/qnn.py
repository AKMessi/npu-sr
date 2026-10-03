"""Windows ML bootstrap and explicit registration into Python's ORT environment."""

import atexit
import importlib
import logging
import platform
import sys
from functools import lru_cache
from typing import Any

from .errors import NPUUnavailable

log = logging.getLogger(__name__)
# Keep the runtime dependency graph and catalog alive for the lifetime of ORT sessions.
_bootstrap: Any = None
_catalog: Any = None
_provider: Any = None


def bootstrap_windows() -> None:
    """Load the installed App Runtime before importing the Windows ML ORT wheel."""
    global _bootstrap
    if sys.platform == "win32" and _bootstrap is None:
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import initialize

        _bootstrap = initialize()
        atexit.register(_bootstrap)


def platform_problem() -> str | None:
    if sys.platform != "win32":
        return "NPU inference requires Windows 11 24H2 or newer. CPU mode is available."
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        return "NPU inference requires ARM64 Python on Windows ARM64. Use py -3.12-arm64."
    if sys.getwindowsversion().build < 26100:
        return "Windows build too old: QNN catalog requires build 26100 (24H2) or newer."
    return None


@lru_cache(maxsize=1)
def ensure_qnn() -> Any:
    """Ensure/download QNN, register its library, and select a real NPU device.

    Native catalog registration alone does not register with Python's ORT Env.
    Do not replace this flow with EnsureAndRegisterCertifiedAsync.
    """
    global _bootstrap, _catalog, _provider
    if problem := platform_problem():
        raise NPUUnavailable(problem)
    try:
        bootstrap_windows()
        import onnxruntime as ort

        winml = importlib.import_module("winui3.microsoft.windows.ai.machinelearning")

        _catalog = winml.ExecutionProviderCatalog.get_default()
        _provider = next(
            (p for p in _catalog.find_all_providers() if p.name == "QNNExecutionProvider"), None
        )
        if _provider is None:
            raise NPUUnavailable("QNN is absent from the Windows ML catalog for this hardware.")
        log.info("Windows ML QNN state before ensure: %s", _provider.ready_state.name)
        result = _provider.ensure_ready_async().get()
        if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
            raise NPUUnavailable(
                f"QNN installation failed: {result.diagnostic_text} "
                f"(HRESULT {result.extended_error})"
            )
        log.info("Registering catalog library: %s", _provider.library_path)
        ort.register_execution_provider_library(_provider.name, _provider.library_path)
        devices = [
            d
            for d in ort.get_ep_devices()
            if d.ep_name == "QNNExecutionProvider"
            and d.device.type == ort.OrtHardwareDeviceType.NPU
        ]
        if not devices:
            raise NPUUnavailable(
                "QNN registered, but no NPU device is available. "
                "Update the Qualcomm NPU driver through Windows Update/OEM support."
            )
        log.info("Selected %s, hardware type NPU", devices[0].ep_name)
        return devices[0]
    except NPUUnavailable:
        raise
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise NPUUnavailable(
            "Windows ML/QNN initialization failed. Install the matching Windows "
            f"App Runtime 2.3 and VC++ ARM64 redistributable. Details: {exc}"
        ) from exc


def package_version() -> str:
    """Version of the catalog-acquired QNN package, not a guessed SDK version."""
    if _provider is None:
        return "unavailable"
    version = _provider.package_id.version
    return f"{version.major}.{version.minor}.{version.build}.{version.revision}"
