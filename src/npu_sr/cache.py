"""Local, version/device-specific QNN contexts. Never distribute these artifacts."""

import hashlib
import json
import platform
import uuid
from functools import lru_cache
from pathlib import Path

from .diagnostics import hardware_info
from .model import sha256


@lru_cache(maxsize=1)
def device_signature() -> dict:
    return {"architecture": platform.machine(), **hardware_info()}


class ContextCache:
    """Embedded context plus integrity metadata; publish only after strict proof."""

    def __init__(
        self,
        root: Path,
        model_hash: str,
        ort_version: str,
        qnn_version: str,
        performance_mode: str = "default",
    ):
        self.signature = {
            "model_sha256": model_hash,
            "ort": ort_version,
            "qnn": qnn_version,
            "hardware": device_signature(),
            "backend": "htp",
            "precision": "fp16",
            "cache_schema": 1,
        }
        if performance_mode != "default":
            self.signature["htp_performance_mode"] = performance_mode
        key = hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{key}.onnx"
        self.metadata = root / f"{key}.json"
        self.pending = root / f"{key}.{uuid.uuid4().hex}.pending.onnx"

    def valid(self) -> bool:
        try:
            metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
            return metadata["signature"] == self.signature and metadata["sha256"] == sha256(
                self.path
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def publish(self) -> None:
        if not self.pending.is_file():
            raise ValueError("QNN did not produce the requested embedded context")
        digest = sha256(self.pending)
        self.pending.replace(self.path)
        temporary = self.metadata.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "signature": self.signature,
                        "sha256": digest,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.metadata)
        finally:
            temporary.unlink(missing_ok=True)

    def invalidate(self) -> None:
        self.metadata.unlink(missing_ok=True)

    def cleanup(self) -> None:
        self.pending.unlink(missing_ok=True)
