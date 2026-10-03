"""Small runtime helpers; no machine identifiers are collected."""

import importlib.metadata
from functools import lru_cache
from typing import Any

from .errors import SRException


@lru_cache(maxsize=1)
def load_ort() -> Any:
    from .qnn import bootstrap_windows

    try:
        bootstrap_windows()
        import onnxruntime as ort

        ort.disable_telemetry_events()
        return ort
    except (ImportError, OSError, RuntimeError) as exc:
        raise SRException(
            "ONNX Runtime/Windows ML unavailable. Install the package dependencies "
            f"and Windows App Runtime 2.3. Details: {exc}"
        ) from exc


def software_versions() -> dict[str, str]:
    names = [
        "npu-sr",
        "numpy",
        "Pillow",
        "onnx",
        "onnxruntime",
        "onnxruntime-windowsml",
        "wasdk-Microsoft.Windows.AI.MachineLearning",
        "winrt-runtime",
    ]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result
