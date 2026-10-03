"""Reproducible full-image quality and performance comparisons, no invented samples."""

import hashlib
import io
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from PIL import Image

from . import __version__
from .benchmark import save_report, statistics
from .diagnostics import hardware_info
from .errors import SRException
from .evaluate import quality_metrics
from .image import infer_y, load_image, postprocess, preprocess
from .model import model_path, sha256
from .monitoring import memory_usage, power_state
from .qnn import package_version
from .runtime import Runtime
from .utils import software_versions

log = logging.getLogger(__name__)
RESOLUTIONS = [(256, 160), (640, 360), (960, 540), (1280, 720), (1920, 1080)]


def environment() -> dict:
    """Only software/hardware classes and versions; no hostname, serial or user paths."""
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            ).stdout.strip()
            or None
        )
        dirty = (
            bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                ).stdout.strip()
            )
            if commit
            else None
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    result = {
        "npu_sr_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "python_source_sha256": digest.hexdigest(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "software_versions": software_versions(),
        "power": "not measured",
        "background_processes": "uncontrolled",
        "power_state": power_state(),
        "qnn_catalog_package": package_version(),
    }
    if sys.platform == "win32":
        result.update(windows_build=sys.getwindowsversion().build, **hardware_info())
    return result


def image_run(image: Image.Image, runtime: Runtime, save: bool = False) -> dict[str, float]:
    """Separate orchestration, ORT invocation, stitching, reconstruction and PNG encode."""
    phases = {}
    total, cpu = perf_counter(), process_time()
    started = perf_counter()
    prepared = preprocess(image, runtime.spec)
    phases["preprocessing_ms"] = (perf_counter() - started) * 1000
    y, _ = infer_y(prepared, runtime, phases)
    started = perf_counter()
    output = postprocess(y, prepared)
    phases["postprocessing_ms"] = (perf_counter() - started) * 1000
    if save:
        started = perf_counter()
        output.save(io.BytesIO(), format="PNG")
        phases["png_encode_ms"] = (perf_counter() - started) * 1000
    phases["total_ms"] = (perf_counter() - total) * 1000
    phases["process_cpu_ms"] = (process_time() - cpu) * 1000
    return phases


def performance_suite(
    source: Path,
    models: list[str],
    devices: list[str],
    runs: int,
    warmups: int,
    trials: int,
    resolutions: list[tuple[int, int]] = RESOLUTIONS,
    cache_dir: Path | None = None,
    checkpoint: Path | None = None,
) -> dict:
    if min(runs, warmups, trials) < 1:
        raise SRException("Runs, warmups and trials must be positive")
    original = load_image(source)
    results = []
    report = {
        "schema_version": 2,
        "environment": environment(),
        "results": results,
        "input_source_sha256": sha256(source),
        "complete": False,
        "timing_scope": "warm image processing excluding startup, disk IO and PNG encoding",
        "aggregation": "median of trial medians; all trials retained",
    }
    for identifier in models:
        path = model_path(identifier=identifier)
        for backend in devices:
            runtime = Runtime(path, backend, cache_dir=cache_dir)
            cold_startup, cold_phases = runtime.startup_ms, runtime.startup_phases.copy()
            del runtime
            runtime = Runtime(path, backend, cache_dir=cache_dir)
            for size in resolutions:
                # Deterministic workload dimensions; not a quality measurement.
                image = original.resize(size, Image.Resampling.BICUBIC)
                trial_results = []
                for trial in range(trials):
                    for _ in range(warmups):
                        image_run(image, runtime)
                    cpu_started = process_time()
                    samples = [image_run(image, runtime) for _ in range(runs)]
                    cpu_ms = (process_time() - cpu_started) * 1000
                    phases = {
                        key: {
                            metric: value
                            for metric, value in statistics(
                                [sample[key] for sample in samples]
                            ).items()
                            if metric != "fps_equivalent"
                        }
                        for key in samples[0]
                        if key != "process_cpu_ms"
                    }
                    trial_results.append(
                        {
                            "trial": trial + 1,
                            "phase_statistics": phases,
                            "process_cpu_ms": cpu_ms,
                        }
                    )
                median = float(
                    np.median(
                        [
                            trial["phase_statistics"]["total_ms"]["median_ms"]
                            for trial in trial_results
                        ]
                    )
                )
                prepared = preprocess(image, runtime.spec)
                result = {
                    "model": identifier,
                    "model_sha256": sha256(path),
                    "backend": backend,
                    "evidence": runtime.evidence,
                    "input_resolution": list(size),
                    "output_resolution": [v * runtime.spec.scale for v in size],
                    "tile_shape": runtime.input_shape,
                    "tile_core": runtime.spec.core,
                    "halo": runtime.spec.halo,
                    "tile_count": prepared.tile_count,
                    "iterations_per_trial": runs,
                    "warmups_per_trial": warmups,
                    "cold_session_startup_ms": cold_startup,
                    "cold_startup_phases": cold_phases,
                    "warm_session_startup_ms": runtime.startup_ms,
                    "warm_startup_phases": runtime.startup_phases,
                    "median_of_trial_medians_ms": median,
                    "theoretical_images_per_second": 1000 / median,
                    "trials": trial_results,
                    "png_encode_probe_ms": image_run(image, runtime, save=True)["png_encode_ms"],
                }
                result["process_memory"] = memory_usage()
                results.append(result)
                if checkpoint:
                    save_report(report, checkpoint)
                print(f"{identifier} {backend} {size}: total median {median:.1f} ms", flush=True)
            del runtime
    report["complete"] = True
    report["environment_end"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "power_state": power_state(),
        "qnn_catalog_package": package_version(),
    }
    return report


def quality_suite(directory: Path, models: list[str], devices: list[str]) -> dict:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".bmp"})
    if not paths:
        raise SRException("Quality input directory contains no reference images")
    provenance_path = directory / "provenance.json"
    if provenance_path.exists():
        import json

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for path in paths:
            if provenance["files"].get(path.name) != sha256(path):
                raise SRException(f"Dataset image integrity mismatch: {path.name}")
    results = []
    for identifier in models:
        for backend in devices:
            runtime = Runtime(model_path(identifier=identifier), backend)
            for path in paths:
                reference = load_image(path)
                if "A" in reference.getbands():
                    raise SRException(
                        "Quality references must be opaque; flatten alpha explicitly."
                    )
                reference = reference.convert("RGB")
                if runtime.spec.scale == 2:
                    width, height = reference.size
                    reference = reference.crop((0, 0, width - width % 2, height - height % 2))
                    source = reference.resize(
                        (reference.width // 2, reference.height // 2), Image.Resampling.BICUBIC
                    )
                    baseline = source.resize(reference.size, Image.Resampling.BICUBIC)
                    degradation = {"type": "Pillow bicubic x2 downsample", "border": 2}
                else:
                    seed, sigma = 2026, 25.0
                    noise = np.random.default_rng(seed).normal(
                        0, sigma, (reference.height, reference.width)
                    )
                    # Equal channel noise; grayscale avoids leaving chroma noise untreated.
                    reference = reference.convert("L").convert("RGB")
                    noisy = np.asarray(reference, np.float64) + noise[:, :, None]
                    source = Image.fromarray(np.clip(np.rint(noisy), 0, 255).astype(np.uint8))
                    baseline = source
                    degradation = {
                        "type": "Gaussian luminance noise, clipped uint8",
                        "sigma": sigma,
                        "seed": seed,
                        "border": 0,
                    }
                prepared = preprocess(source, runtime.spec)
                result = postprocess(infer_y(prepared, runtime)[0], prepared)
                border = degradation["border"]
                metrics = quality_metrics(result, reference, border)
                results.append(
                    {
                        "model": identifier,
                        "model_sha256": runtime.manifest["sha256"],
                        "backend": backend,
                        "evidence": runtime.evidence,
                        "image": path.name,
                        "reference_sha256": sha256(path),
                        "degradation": degradation,
                        "metrics": metrics,
                        "baseline_metrics": quality_metrics(baseline, reference, border),
                    }
                )
                print(f"{identifier} {backend} {path.name}: {metrics}", flush=True)
            del runtime
    return {
        "schema_version": 2,
        "environment": environment(),
        "results": results,
        "metric_definition": "full-range Rec.601 Y; PSNR; Gaussian 11x11 sigma=1.5 population SSIM",
        "dataset": (
            "user supplied; image names and SHA256 recorded; "
            "first-five BSDS300 test subset for release"
        ),
        "methodology": "Pillow bicubic LR generation, not canonical MATLAB BSD100 scores",
    }
