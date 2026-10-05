"""Prepare models and tools, then run the normal strict proof path."""

import os
import platform
import sys
from pathlib import Path

import numpy as np

from .acquire_models import acquire_model
from .diagnostics import diagnose
from .errors import SRException
from .model import model_directory, model_path, model_spec, validate_model
from .qnn import platform_problem
from .utils import load_ort


def prepare(
    identifiers: list[str],
    device: str = "npu",
    download_ffmpeg: bool = False,
    ffmpeg: Path | None = None,
    verbose: bool = False,
) -> dict:
    if device not in {"npu", "cpu"}:
        raise SRException("Setup device must be npu or cpu.")
    if download_ffmpeg and ffmpeg is not None:
        raise SRException("Choose --ffmpeg or --download-ffmpeg, not both.")
    if download_ffmpeg and sys.platform != "win32":
        raise SRException("Install FFmpeg with your OS package manager on non-Windows systems.")
    if not identifiers or any(model_spec(name).input_channels != 1 for name in identifiers):
        raise SRException("Setup requires at least one supported spatial model.")
    if device == "npu" and (problem := platform_problem()):
        raise SRException(problem)
    # Detect a missing App Runtime before acquiring large dependencies.
    load_ort()
    acquired = {}
    for identifier in identifiers:
        path = model_path(identifier=identifier)
        try:
            validate_model(path)
            acquired[identifier] = "verified existing model"
        except SRException:
            acquire_model(identifier, model_directory())
            validate_model(path)
            acquired[identifier] = "downloaded and hash-verified"
    if download_ffmpeg:
        from .install_ffmpeg import download

        architecture = "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64"
        directory = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "npu-sr" / "tools"
        ffmpeg = download(architecture, directory / f"ffmpeg-{architecture}-monthly-202608")
    path = model_path(identifier=identifiers[0])
    report = diagnose(path, verbose, video=True, ffmpeg=ffmpeg, gpu=False, npu=device == "npu")
    report["acquired_models"] = acquired
    if device == "cpu":
        from .runtime import Runtime

        runtime = Runtime(path, "cpu", verbose)
        runtime.run(np.full(runtime.input_shape, 0.5, np.float32))
        report["cpu_proof"] = runtime.evidence
        report["setup_ready"] = True
    else:
        report["setup_ready"] = report["ready"] and report["video"]["ready"]
    report["setup_device"] = device
    return report
