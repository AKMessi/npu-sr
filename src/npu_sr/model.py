"""Pinned model contract and artifact integrity checks."""

import hashlib
import json
import os
from pathlib import Path

from .errors import SRException

MODEL_NAME = "ESPCN-x2"
MODEL_VERSION = "tf-espcn-5c628ec-export1"
TILE = 136
HALO = 4  # Receptive radius: 5x5 + 3x3 + 3x3 convolutions.
CORE = TILE - HALO * 2
INPUT_SHAPE = [1, 1, TILE, TILE]
OUTPUT_SHAPE = [1, 1, TILE * 2, TILE * 2]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_directory() -> Path:
    """Use an explicit override, clone-local models, then the user cache."""
    if directory := os.environ.get("NPU_SR_MODEL_DIR"):
        return Path(directory).expanduser().resolve()
    if (Path.cwd() / "pyproject.toml").is_file() and (Path.cwd() / "models").is_dir():
        return Path.cwd() / "models"
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return root / "npu-sr" / "models"


def model_path(directory: Path | None = None) -> Path:
    return (directory or model_directory()) / "espcn-x2.onnx"


def validate_model(path: Path) -> dict:
    if not path.is_file():
        raise SRException("Model missing. Run: python scripts/download_model.py")
    try:
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not all(
            isinstance(manifest.get(key), str) for key in ("name", "version", "sha256", "precision")
        ):
            raise ValueError("expected a JSON object with name, version, sha256, and precision")
        if manifest["version"] != MODEL_VERSION or manifest["sha256"] != sha256(path):
            raise ValueError("artifact version or SHA256 mismatch")
        return manifest
    except (OSError, ValueError, KeyError) as exc:
        raise SRException(f"Invalid model manifest: {exc}. Download the model again.") from exc
