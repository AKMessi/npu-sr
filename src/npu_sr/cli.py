"""Student-readable command line interface; backend failures are always visible."""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

from . import __version__
from .errors import SRException
from .model import model_path


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="npu-sr", description="2x image SR on Snapdragon X NPUs")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check system and run strict NPU proof inference")
    upscale = commands.add_parser("upscale", help="Create a 2x image")
    bench = commands.add_parser("benchmark", help="Measure CPU vs NPU warm inference")
    evaluate = commands.add_parser(
        "evaluate", help="Compare against a real high-resolution reference"
    )
    for sub in (doctor, upscale, bench, evaluate):
        sub.add_argument(
            "--model", type=Path, default=None, help="Model path with adjacent JSON manifest"
        )
        sub.add_argument(
            "--verbose", action="store_true", help="ORT logging, evidence, and tracebacks"
        )
    for sub in (upscale, bench, evaluate):
        sub.add_argument("input", type=Path)
    for sub in (upscale, evaluate):
        sub.add_argument("--device", choices=["auto", "npu", "cpu"], default="auto")
    upscale.add_argument("-o", "--output", type=Path, default=Path("output.png"))
    upscale.add_argument("--comparison-dir", type=Path, help="Save original, bicubic, and SR PNGs")
    bench.add_argument("--runs", type=positive, default=30)
    bench.add_argument("--warmups", type=positive, default=5)
    bench.add_argument("--json", type=Path)
    evaluate.add_argument("reference", type=Path)
    return root


def _doctor(args: argparse.Namespace) -> int:
    from .diagnostics import diagnose

    report = diagnose(args.model or model_path(), args.verbose)
    print("NPU-SR diagnostics\n")
    fields = [
        ("OS", report["os"]),
        ("Architecture", report["architecture"]),
        ("Windows build", report["windows_build"]),
        ("Processor", report["processor"]),
        ("NPU driver", report["npu_driver"]),
        ("Windows ML", report["windows_ml"]),
        ("QNN status", report["qnn_status"]),
        ("ONNX Runtime", report.get("ort_version", "missing")),
        ("CPU backend", report["cpu_backend"]),
        ("Model", report["model"]),
        ("Model status", report["model_status"]),
    ]
    for label, value in fields:
        print(f"{label}: {value}")
    if report["ready"]:
        print("\nResult: system ready for NPU inference (strict QNN proof passed)")
    else:
        print("\nResult: system not ready for NPU inference")
    for error in report["errors"]:
        print(f"  Error: {error}")
    if not report["ready"]:
        print("See: docs/troubleshooting.md")
    return 0 if report["ready"] else 1


def _upscale(args: argparse.Namespace) -> int:
    from .image import infer_y, load_image, postprocess, preprocess, save_image
    from .runtime import Runtime

    if args.input.resolve() == args.output.resolve():
        raise SRException(
            "Output must differ from input; the source image will not be overwritten."
        )
    started = perf_counter()
    prepared = preprocess(load_image(args.input))
    runtime = Runtime(args.model or model_path(), args.device, args.verbose)
    y, inference_ms = infer_y(prepared, runtime)
    output = postprocess(y, prepared)
    save_image(output, args.output)
    total_ms = (perf_counter() - started) * 1000
    if args.comparison_dir:
        from PIL import Image

        original = prepared.rgb.copy()
        if prepared.alpha is not None:
            original.putalpha(prepared.alpha)
        paths = [
            args.comparison_dir / name
            for name in ("original.png", "bicubic_2x.png", "npu_sr_2x.png")
        ]
        if any(path.resolve() == args.input.resolve() for path in paths):
            raise SRException("Comparison directory would overwrite the source image.")
        save_image(original, paths[0])
        save_image(original.resize(output.size, Image.Resampling.BICUBIC), paths[1])
        save_image(output, paths[2])
    print(f"Input:      {prepared.size[0]} × {prepared.size[1]}")
    print(f"Output:     {output.width} × {output.height}")
    print("Scale:      2×")
    print(f"Backend:    {runtime.label}")
    print(f"Inference:  {inference_ms:.1f} ms ({prepared.tile_count} tiles)")
    print(f"Startup:    {runtime.startup_ms:.0f} ms (includes NPU proof when selected)")
    print(f"Total:      {total_ms:.0f} ms (load through save, includes startup)")
    print(f"Saved:      {args.output}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .benchmark import benchmark, save_report
    from .image import load_image, preprocess

    if args.json and args.json.resolve() == args.input.resolve():
        raise SRException("JSON report must differ from the input image.")
    prepared = preprocess(load_image(args.input))
    report = benchmark(prepared, args.model or model_path(), args.runs, args.warmups, args.verbose)
    print("Backend       Median      Mean       p95       FPS equivalent")
    print("------------------------------------------------------------")
    for result in report["results"]:
        latency = result["latency"]
        print(
            f"{result['backend'].upper():<10} {latency['median_ms']:8.1f} ms "
            f"{latency['mean_ms']:7.1f} ms {latency['p95_ms']:7.1f} ms "
            f"{latency['fps_equivalent']:9.1f}"
        )
        print(f"  Backend: {result['label']}; startup: {result['startup_ms']:.0f} ms")
    if report["npu_unavailable"]:
        print(f"\nNPU unavailable — CPU measurements only: {report['npu_unavailable']}")
    else:
        print(f"\nNPU speedup: {report['npu_speedup']:.2f}×")
    print(f"{args.runs} measured images; {args.warmups} warmups; {prepared.tile_count} tiles/image")
    print("Inference timings exclude startup, image IO, preprocessing, and tile stitching.")
    if args.json:
        save_report(report, args.json)
        print(f"Saved: {args.json}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from PIL import Image

    from .evaluate import psnr
    from .image import infer_y, load_image, postprocess, preprocess
    from .runtime import Runtime

    prepared = preprocess(load_image(args.input))
    reference = load_image(args.reference)
    expected = (prepared.size[0] * 2, prepared.size[1] * 2)
    if reference.size != expected:
        raise SRException(
            "Ground truth must be exactly 2x the input dimensions (after EXIF rotation)."
        )
    runtime = Runtime(args.model or model_path(), args.device, args.verbose)
    output = postprocess(infer_y(prepared, runtime)[0], prepared)
    bicubic = prepared.rgb.resize(output.size, Image.Resampling.BICUBIC)
    print(f"Backend: {runtime.label}")
    print(f"SR RGB PSNR:      {psnr(output, reference):.2f} dB")
    print(f"Bicubic RGB PSNR: {psnr(bicubic, reference):.2f} dB")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    try:
        return {
            "doctor": _doctor,
            "upscale": _upscale,
            "benchmark": _benchmark,
            "evaluate": _evaluate,
        }[args.command](args)
    except (SRException, OSError, ValueError, ImportError) as exc:
        print(f"Error: {exc}\n\nRun: npu-sr doctor\nSee: docs/troubleshooting.md", file=sys.stderr)
        if args.verbose:
            logging.exception("Detailed failure")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
