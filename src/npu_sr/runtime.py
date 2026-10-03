"""Strict backend selection and per-session execution evidence."""

import json
import logging
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .errors import NPUUnavailable, SRException
from .model import INPUT_SHAPE, OUTPUT_SHAPE, validate_model
from .qnn import ensure_qnn, package_version
from .utils import load_ort

ort = load_ort()

log = logging.getLogger(__name__)


def profile_providers(events: list[dict]) -> dict[str, int]:
    """Count executed kernels by provider, ignoring synchronization fence events."""
    counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") == "Node" and event.get("args", {}).get("provider"):
            provider = event["args"]["provider"]
            counts[provider] = counts.get(provider, 0) + 1
    return counts


class Runtime:
    """One fixed-shape SR session with a verified backend and separate startup timing."""

    def __init__(self, path: Path, device: str = "auto", verbose: bool = False) -> None:
        self.manifest = validate_model(path)
        self.path = path
        self.backend = "cpu"
        self.evidence: dict[str, Any] = {}
        self.fallback_reason: str | None = None
        started = perf_counter()
        if device not in {"cpu", "npu", "auto"}:
            raise SRException(f"Unknown device: {device}")
        if device != "cpu":
            try:
                self.session = self._npu(path, verbose)
                self.backend = "npu"
            except NPUUnavailable as exc:
                if device == "npu":
                    raise
                self.fallback_reason = str(exc)
                log.warning("NPU unavailable — using CPU: %s", exc)
        if self.backend == "cpu":
            options = ort.SessionOptions()
            options.log_severity_level = 0 if verbose else 3
            try:
                self.session = ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                )
                self.session.disable_fallback()
            except Exception as exc:
                raise SRException(f"CPU model initialization failed: {exc}") from exc
            self.evidence = {"providers": self.session.get_providers()}
        self._validate_contract()
        self.startup_ms = (perf_counter() - started) * 1000

    @property
    def label(self) -> str:
        return "Qualcomm QNN / Hexagon NPU" if self.backend == "npu" else "ONNX Runtime CPU"

    def _npu(self, path: Path, verbose: bool) -> ort.InferenceSession:
        device = ensure_qnn()
        options = ort.SessionOptions()
        options.log_severity_level = 0 if verbose else 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(
            [device], {"backend_type": "htp", "enable_htp_fp16_precision": "1"}
        )
        # Profile the actual session once; stop profiling before timed measurements.
        with tempfile.TemporaryDirectory(prefix="npu-sr-proof-") as directory:
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / "ort")
            session = None
            profile = None
            try:
                session = ort.InferenceSession(str(path), sess_options=options, providers=[])
                session.disable_fallback()
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                if (
                    len(inputs) != 1
                    or inputs[0].shape != INPUT_SHAPE
                    or inputs[0].type != "tensor(float)"
                    or len(outputs) != 1
                    or outputs[0].shape != OUTPUT_SHAPE
                ):
                    raise NPUUnavailable("The model does not match the fixed ESPCN-x2 contract.")
                output = session.run(None, {inputs[0].name: np.full(INPUT_SHAPE, 0.5, np.float32)})[
                    0
                ]
                if output.shape != tuple(OUTPUT_SHAPE) or not np.isfinite(output).all():
                    raise NPUUnavailable("QNN proof inference returned an invalid output.")
                profile = Path(session.end_profiling())
                providers = profile_providers(json.loads(profile.read_text(encoding="utf-8")))
                if not providers.get("QNNExecutionProvider"):
                    raise NPUUnavailable("ORT profiling contains no executed QNN kernel.")
                other = {p: n for p, n in providers.items() if p != "QNNExecutionProvider"}
                if other:
                    raise NPUUnavailable(f"Strict NPU mode rejected partial assignment: {other}")
                self.evidence = {
                    "device_type": "NPU",
                    "qnn_backend": "htp",
                    "cpu_fallback_disabled": True,
                    "executed_kernel_counts": providers,
                    "providers": session.get_providers(),
                    "qnn_package_version": package_version(),
                }
                log.info("Execution evidence: %s", self.evidence)
                return session
            except NPUUnavailable:
                raise
            except Exception as exc:
                raise NPUUnavailable(
                    f"Strict QNN inference failed: {exc}. See docs/troubleshooting.md."
                ) from exc
            finally:
                if session is not None and profile is None:
                    session.end_profiling()

    def _validate_contract(self) -> None:
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].shape != INPUT_SHAPE
            or inputs[0].type != "tensor(float)"
            or len(outputs) != 1
            or outputs[0].shape != OUTPUT_SHAPE
        ):
            raise SRException(
                "Incompatible ONNX model: expected fixed ESPCN-x2 input/output shapes."
            )
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def run(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.shape != tuple(INPUT_SHAPE) or tensor.dtype != np.float32:
            raise SRException(f"Expected float32 input with shape {INPUT_SHAPE}.")
        try:
            output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except Exception as exc:
            raise SRException(f"{self.label} inference failed: {exc}") from exc
        if output.shape != tuple(OUTPUT_SHAPE) or not np.isfinite(output).all():
            raise SRException("Model returned an invalid tensor.")
        return output
