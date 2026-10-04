"""Pinned model contract and artifact integrity checks."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import SRException

MODEL_NAME = "ESPCN-x2"
MODEL_VERSION = "tf-espcn-5c628ec-export1"
TILE = 136
HALO = 4  # Receptive radius: 5x5 + 3x3 + 3x3 convolutions.
CORE = TILE - HALO * 2
INPUT_SHAPE = [1, 1, TILE, TILE]
OUTPUT_SHAPE = [1, 1, TILE * 2, TILE * 2]


@dataclass(frozen=True)
class ModelSpec:
    """The spatial/color contract shared by export, runtime and image tiling."""

    identifier: str
    architecture: str
    task: str = "upscale"
    scale: int = 2
    core: int = 256
    core_height: int | None = None
    halo: int = 4
    license: str = "Apache-2.0"
    color_space: str = "luminance"
    opset: int = 13
    precision: str = "float32 (QNN HTP uses float16 math)"

    @property
    def input_shape(self) -> list[int]:
        return [1, 1, self.height + self.halo * 2, self.core + self.halo * 2]

    @property
    def height(self) -> int:
        return self.core if self.core_height is None else self.core_height

    @property
    def output_shape(self) -> list[int]:
        return [1, 1, self.input_shape[2] * self.scale, self.input_shape[3] * self.scale]

    @property
    def version(self) -> str:
        return MODEL_VERSION if self.identifier == "espcn-x2" else f"{self.identifier}-export1"

    def info(self) -> dict:
        return asdict(self) | {
            "version": self.version,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "preferred_backend": "npu",
        }


MODELS = {
    spec.identifier: spec
    for spec in (
        ModelSpec("espcn-x2", "ESPCN", core=128),
        ModelSpec("espcn-x2-256", "ESPCN"),
        ModelSpec("fsrcnn-small-x2", "FSRCNN-small"),
        ModelSpec("fsrcnn-x2", "FSRCNN", halo=6),
        ModelSpec("lapsrn-x2", "LapSRN", halo=12),
        ModelSpec("dncnn-25", "DnCNN", task="denoise", scale=1, halo=17, license="MIT"),
        ModelSpec("quicksrnet-small-y-x2", "QuickSRNet-small", license="BSD-3-Clause"),
        ModelSpec("quicksrnet-medium-y-x2", "QuickSRNet-medium", halo=7, license="BSD-3-Clause"),
    )
}


def model_spec(identifier: str) -> ModelSpec:
    try:
        return MODELS[identifier]
    except KeyError as exc:
        raise SRException(f"Unknown model: {identifier}. Run: npu-sr models list") from exc


def resolve_model(value: str | Path | None, task: str = "upscale") -> Path:
    """Identifiers and existing v0.1 --model path syntax are both accepted."""
    identifier = (
        str(value) if value is not None else ("dncnn-25" if task == "denoise" else "espcn-x2")
    )
    return model_path(identifier=identifier) if identifier in MODELS else Path(identifier)


def manifest_spec(manifest: dict) -> ModelSpec:
    # v0.1 manifests have no identifier; retain their exact fixed contract.
    return model_spec(manifest.get("identifier", "espcn-x2"))


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


def model_path(directory: Path | None = None, identifier: str = "espcn-x2") -> Path:
    model_spec(identifier)
    return (directory or model_directory()) / f"{identifier}.onnx"


def validate_model(path: Path) -> dict:
    if not path.is_file():
        if path.stem in MODELS:
            raise SRException(f"Model missing. Run: npu-sr models download {path.stem}")
        raise SRException("Model missing. Supply an existing ONNX path or run: npu-sr models list")
    try:
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not all(
            isinstance(manifest.get(key), str) for key in ("name", "version", "sha256", "precision")
        ):
            raise ValueError("expected a JSON object with name, version, sha256, and precision")
        spec = manifest_spec(manifest)
        if manifest["version"] != spec.version or manifest["sha256"] != sha256(path):
            raise ValueError("artifact version or SHA256 mismatch")
        for key in ("input_shape", "output_shape"):
            if key in manifest and manifest[key] != getattr(spec, key):
                raise ValueError(f"{key} does not match registry")
        return manifest
    except (OSError, ValueError, KeyError) as exc:
        raise SRException(f"Invalid model manifest: {exc}. Download the model again.") from exc
