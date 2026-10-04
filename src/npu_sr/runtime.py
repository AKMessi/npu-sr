"""Strict backend selection and per-session execution evidence."""

import json
import logging
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .errors import NPUUnavailable, SRException
from .model import manifest_spec, validate_model
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
    """One reusable static enhancement session with verified accelerator execution."""

    def __init__(
        self,
        path: Path,
        device: str = "auto",
        verbose: bool = False,
        cache_dir: Path | None = None,
        performance_mode: str = "default",
    ) -> None:
        started = perf_counter()
        self.manifest = validate_model(path)
        self.startup_phases = {"artifact_validation_ms": (perf_counter() - started) * 1000}
        self.cache_dir = cache_dir
        if performance_mode not in {"default", "burst", "sustained_high_performance"}:
            raise SRException(f"Unsupported QNN performance mode: {performance_mode}")
        self.performance_mode = performance_mode
        self._loaded_context = False
        self.spec = manifest_spec(self.manifest)
        self.input_shape = self.spec.input_shape
        self.output_shape = self.spec.output_shape
        self.path = path
        self.backend = "cpu"
        self.evidence: dict[str, Any] = {}
        self.fallback_reason: str | None = None
        if device not in {"cpu", "npu", "gpu", "auto"}:
            raise SRException(f"Unknown device: {device}")
        if device == "gpu":
            self.session = self._accelerated(path, verbose, "gpu")
            self.backend = "gpu"
        elif device != "cpu":
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
                session_started = perf_counter()
                self.session = ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                )
                self.session.disable_fallback()
                self.startup_phases["session_creation_ms"] = (
                    perf_counter() - session_started
                ) * 1000
            except Exception as exc:
                raise SRException(f"CPU model initialization failed: {exc}") from exc
            self.evidence = {"providers": self.session.get_providers()}
        self._validate_contract()
        self.startup_ms = (perf_counter() - started) * 1000

    @property
    def label(self) -> str:
        return {
            "npu": "Qualcomm QNN / Hexagon NPU",
            "gpu": "ONNX Runtime DirectML / GPU",
            "cpu": "ONNX Runtime CPU",
        }[self.backend]

    def _npu(self, path: Path, verbose: bool) -> ort.InferenceSession:
        try:
            return self._accelerated(path, verbose, "npu")
        except NPUUnavailable:
            if not self._loaded_context:
                raise
            log.warning("Cached QNN context failed validation; recompiling the source model.")
            return self._accelerated(path, verbose, "npu")

    def _accelerated(self, path: Path, verbose: bool, backend: str) -> ort.InferenceSession:
        """Require assignment and a successful proof run on the selected accelerator."""
        options = ort.SessionOptions()
        options.log_severity_level = 0 if verbose else 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers: list[Any] = []
        expected_provider = "QNNExecutionProvider" if backend == "npu" else "DmlExecutionProvider"
        cache = None
        cached = False
        session_path = path
        if backend == "npu":
            catalog_started = perf_counter()
            device = ensure_qnn()
            self.startup_phases["catalog_ms"] = (perf_counter() - catalog_started) * 1000
            options.add_provider_for_devices(
                [device],
                {
                    "backend_type": "htp",
                    "enable_htp_fp16_precision": "1",
                    "htp_performance_mode": self.performance_mode,
                },
            )
            if self.cache_dir is not None:
                from .cache import ContextCache

                cache = ContextCache(
                    self.cache_dir,
                    self.manifest["sha256"],
                    ort.__version__,
                    package_version(),
                    self.performance_mode,
                )
                cached = cache.valid()
                self._loaded_context = cached
                if cached:
                    session_path = cache.path
                else:
                    options.add_session_config_entry("ep.context_enable", "1")
                    options.add_session_config_entry("ep.context_embed_mode", "1")
                    options.add_session_config_entry(
                        "ep.context_file_path", str(cache.pending.resolve())
                    )
        else:
            if "DmlExecutionProvider" not in ort.get_available_providers():
                raise NPUUnavailable(
                    "DirectML GPU provider unavailable; GPU mode cannot fall back."
                )
            devices = [
                ep
                for ep in ort.get_ep_devices()
                if ep.ep_name == expected_provider
                and ep.device.type == ort.OrtHardwareDeviceType.GPU
            ]
            devices = [
                ep
                for ep in devices
                if not any(
                    term in ep.device.metadata.get("Description", "").lower()
                    for term in ("warp", "basic render")
                )
            ]
            if not devices:
                raise NPUUnavailable("No hardware DirectML GPU device is enumerated.")
            gpu = devices[0].device
            adapter = gpu.metadata.get("DxgiAdapterNumber")
            if adapter is None:
                raise NPUUnavailable("DirectML device has no explicit DXGI adapter index.")
            self.gpu_description = gpu.metadata.get("Description", "unknown GPU")
            # DirectML requires sequential execution and no memory pattern optimization.
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [(expected_provider, {"device_id": adapter})]
        # Profile the actual session once; stop profiling before timed measurements.
        with tempfile.TemporaryDirectory(prefix="npu-sr-proof-") as directory:
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / "ort")
            session = None
            profile = None
            proved = False
            try:
                session_started = perf_counter()
                session = ort.InferenceSession(
                    str(session_path), sess_options=options, providers=providers
                )
                self.startup_phases["session_creation_ms"] = (
                    perf_counter() - session_started
                ) * 1000
                session.disable_fallback()
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                if (
                    len(inputs) != 1
                    or inputs[0].shape != self.input_shape
                    or inputs[0].type != "tensor(float)"
                    or len(outputs) != 1
                    or outputs[0].shape != self.output_shape
                ):
                    raise NPUUnavailable("The model does not match its fixed registry contract.")
                proof_started = perf_counter()
                output = session.run(
                    None, {inputs[0].name: np.full(self.input_shape, 0.5, np.float32)}
                )[0]
                self.startup_phases["proof_inference_ms"] = (perf_counter() - proof_started) * 1000
                if output.shape != tuple(self.output_shape) or not np.isfinite(output).all():
                    raise NPUUnavailable("QNN proof inference returned an invalid output.")
                verification_started = perf_counter()
                profile = Path(session.end_profiling())
                providers = profile_providers(json.loads(profile.read_text(encoding="utf-8")))
                if not providers.get(expected_provider):
                    raise NPUUnavailable(
                        f"ORT profiling contains no executed {expected_provider} kernel."
                    )
                other = {p: n for p, n in providers.items() if p != expected_provider}
                if other:
                    raise NPUUnavailable(
                        f"Strict {backend.upper()} mode rejected partial assignment: {other}"
                    )
                self.evidence = {
                    "device_type": backend.upper(),
                    "cpu_fallback_disabled": True,
                    "executed_kernel_counts": providers,
                    "providers": session.get_providers(),
                }
                if backend == "npu":
                    self.evidence.update(qnn_backend="htp", qnn_package_version=package_version())
                    self.evidence["htp_performance_mode_requested"] = self.performance_mode
                    self.evidence["context_cache"] = (
                        "hit" if cached else "miss" if cache else "disabled"
                    )
                else:
                    # Do not serialize LUID/device identifiers from adapter metadata.
                    self.evidence["gpu_description"] = self.gpu_description
                self.startup_phases["profile_verification_ms"] = (
                    perf_counter() - verification_started
                ) * 1000
                if cache and not cached:
                    cache.publish()
                proved = True
                log.info("Execution evidence: %s", self.evidence)
                return session
            except NPUUnavailable:
                raise
            except Exception as exc:
                raise NPUUnavailable(
                    f"Strict {backend.upper()} inference failed: {exc}. "
                    "See docs/troubleshooting.md."
                ) from exc
            finally:
                if session is not None and profile is None:
                    session.end_profiling()
                if cache:
                    if cached and not proved:
                        cache.invalidate()
                    cache.cleanup()

    def _validate_contract(self) -> None:
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].shape != self.input_shape
            or inputs[0].type != "tensor(float)"
            or len(outputs) != 1
            or outputs[0].shape != self.output_shape
        ):
            raise SRException(
                "Incompatible ONNX model: expected fixed registry input/output shapes."
            )
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def run(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.shape != tuple(self.input_shape) or tensor.dtype != np.float32:
            raise SRException(f"Expected float32 input with shape {self.input_shape}.")
        try:
            output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except Exception as exc:
            raise SRException(f"{self.label} inference failed: {exc}") from exc
        if output.shape != tuple(self.output_shape) or not np.isfinite(output).all():
            raise SRException("Model returned an invalid tensor.")
        return output
