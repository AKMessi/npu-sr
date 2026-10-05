"""Report real system facts and run the same strict proof used by NPU inference."""

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .errors import SRException
from .model import MODEL_NAME, validate_model
from .qnn import ensure_qnn, platform_problem
from .utils import load_ort, software_versions


def hardware_info() -> dict[str, Any]:
    result: dict[str, Any] = {"processor": "unknown", "npu_driver": "unknown"}
    if sys.platform != "win32":
        return result
    command = (
        "$cpu=(Get-CimInstance Win32_Processor | Select-Object -First 1).Name; "
        "$npu=Get-CimInstance Win32_PnPSignedDriver | "
        "Where-Object DeviceName -Match 'Hexagon.*NPU' | Select-Object -First 1; "
        "@{processor=$cpu; npu_driver=$npu.DriverVersion} | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        result.update(json.loads(completed.stdout))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return result


def diagnose(
    path: Path,
    verbose: bool = False,
    *,
    video: bool = False,
    ffmpeg: Path | None = None,
    gpu: bool = False,
    npu: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "os": platform.system(),
        "architecture": platform.machine(),
        "windows_build": sys.getwindowsversion().build if sys.platform == "win32" else None,
        **hardware_info(),
        "software_versions": software_versions(),
        "model": MODEL_NAME,
        "model_status": "missing",
        "windows_ml": "unavailable",
        "qnn_status": "unavailable",
        "cpu_backend": "unavailable",
        "errors": [],
        "ready": False,
    }
    try:
        ort = load_ort()
        report["ort_version"] = ort.__version__
        report["ort_providers"] = ort.get_available_providers()
        report["cpu_backend"] = (
            "available" if "CPUExecutionProvider" in report["ort_providers"] else "unavailable"
        )
    except SRException as exc:
        report["errors"].append(str(exc))
    try:
        manifest = validate_model(path)
        report["model"] = manifest["name"]
        report["model_status"] = "ready"
    except SRException as exc:
        report["errors"].append(str(exc))
    if not npu:
        report["qnn_status"] = "not requested"
    elif problem := platform_problem():
        report["errors"].append(problem)
    else:
        try:
            ensure_qnn()
            report["windows_ml"] = "available"
            report["qnn_status"] = "registered (inference not yet verified)"
            if report["model_status"] == "ready":
                from .runtime import Runtime

                runtime = Runtime(path, "npu", verbose)
                report["qnn_status"] = "ready (strict inference and ORT profile verified)"
                report["evidence"] = runtime.evidence
                report["startup_ms"] = runtime.startup_ms
                report["ready"] = True
        except SRException as exc:
            report["errors"].append(str(exc))
    from .capabilities import gpu_capability, model_inventory, video_capabilities

    report["models"] = model_inventory()
    if video:
        report["video"] = video_capabilities(ffmpeg)
    if gpu:
        report["gpu"] = (
            gpu_capability(path, verbose)
            if report["model_status"] == "ready"
            else {"status": "not proven", "reason": "model missing or invalid"}
        )
    return report
