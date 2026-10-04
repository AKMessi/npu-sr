"""Student-readable command line interface; backend failures are always visible."""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

from . import __version__
from .errors import SRException
from .model import MODELS, model_directory, model_spec, resolve_model


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
    denoise = commands.add_parser("denoise", help="Clean luminance Gaussian noise (sigma 25/255)")
    models = commands.add_parser("models", help="List, inspect or download a model")
    models.add_argument("action", choices=["list", "info", "download"])
    models.add_argument("identifier", nargs="?")
    models.add_argument("--verbose", action="store_true")
    bench = commands.add_parser("benchmark", help="Measure CPU vs NPU warm inference")
    evaluate = commands.add_parser(
        "evaluate", help="Compare against a real high-resolution reference"
    )
    for sub in (doctor, upscale, denoise, bench, evaluate):
        sub.add_argument(
            "--model", default=None, help="Registry identifier or ONNX path with JSON manifest"
        )
        sub.add_argument(
            "--verbose", action="store_true", help="ORT logging, evidence, and tracebacks"
        )
    for sub in (upscale, denoise, bench, evaluate):
        sub.add_argument("input", type=Path)
    for sub in (upscale, denoise, evaluate):
        sub.add_argument("--device", choices=["auto", "npu", "cpu", "gpu"], default="auto")
        sub.add_argument("--cache-dir", type=Path, help="Local QNN compiled context cache")
    for sub in (upscale, denoise):
        sub.add_argument("-o", "--output", type=Path, default=Path("output.png"))
        sub.add_argument("--comparison-dir", type=Path, help="Save original and enhanced PNGs")
    bench.add_argument("--runs", type=positive, default=30)
    bench.add_argument("--warmups", type=positive, default=5)
    bench.add_argument("--json", type=Path)
    bench.add_argument("--gpu", action="store_true", help="Also require a strict GPU measurement")
    suite = commands.add_parser(
        "benchmark-suite", help="Reproducible model performance/quality matrix"
    )
    suite.add_argument(
        "input", type=Path, help="Image directory for quality, image file for performance"
    )
    suite.add_argument("--quality", action="store_true")
    suite.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    suite.add_argument(
        "--devices", nargs="+", choices=["cpu", "npu", "gpu"], default=["cpu", "npu", "gpu"]
    )
    suite.add_argument("--runs", type=positive, default=10)
    suite.add_argument("--warmups", type=positive, default=3)
    suite.add_argument("--trials", type=positive, default=3)
    suite.add_argument("--json", type=Path, required=True)
    suite.add_argument("--verbose", action="store_true")
    suite.add_argument("--cache-dir", type=Path, help="Local QNN compiled context cache")
    evaluate.add_argument("reference", type=Path)
    video = commands.add_parser(
        "video", help="Stream 2x video through one persistent model session"
    )
    video.add_argument("input", type=Path)
    video.add_argument("-o", "--output", type=Path, required=True)
    video.add_argument("--model", default="espcn-x2-256")
    video.add_argument("--device", choices=["auto", "npu", "gpu", "cpu"], default="auto")
    video.add_argument("--decode", choices=["auto", "hardware", "software"], default="auto")
    video.add_argument("--encode", choices=["auto", "hardware", "software"], default="auto")
    video.add_argument("--codec", choices=["h264", "hevc", "av1"], default="h264")
    video.add_argument("--audio", choices=["copy", "none"], default="copy")
    video.add_argument("--bitrate", default="8M", help="Hardware encoder target rate")
    video.add_argument(
        "--quality", type=int, default=20, help="Software CRF, 0–51 (lower is better)"
    )
    video.add_argument("--ffmpeg", type=Path)
    video.add_argument("--cache-dir", type=Path)
    video.add_argument("--json", type=Path)
    video.add_argument("--overwrite", action="store_true")
    video.add_argument("--verbose", action="store_true")
    video_bench = commands.add_parser(
        "benchmark-video",
        parents=[video],
        add_help=False,
        help="Run multiple complete video processing trials",
    )
    video_bench.add_argument("--trials", type=positive, default=3)
    video_quality = commands.add_parser(
        "evaluate-video", help="Compare with an aligned HR reference video"
    )
    video_quality.add_argument("input", type=Path)
    video_quality.add_argument("reference", type=Path)
    video_quality.add_argument("--ffmpeg", type=Path)
    video_quality.add_argument("--stride", type=positive, default=12)
    video_quality.add_argument("--json", type=Path, required=True)
    video_quality.add_argument("--verbose", action="store_true")
    video_quality.add_argument(
        "--vmaf", action="store_true", help="Require native libvmaf measurement"
    )
    return root


def _doctor(args: argparse.Namespace) -> int:
    from .diagnostics import diagnose

    report = diagnose(resolve_model(args.model), args.verbose)
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
    runtime = Runtime(
        resolve_model(args.model, args.command), args.device, args.verbose, args.cache_dir
    )
    if runtime.spec.task != args.command:
        raise SRException(
            f"Model {runtime.spec.identifier} is for {runtime.spec.task}, not {args.command}."
        )
    phase_started = perf_counter()
    image = load_image(args.input)
    load_ms = (perf_counter() - phase_started) * 1000
    phase_started = perf_counter()
    prepared = preprocess(image, runtime.spec)
    preprocessing_ms = (perf_counter() - phase_started) * 1000
    phases: dict[str, float] = {}
    y, inference_ms = infer_y(prepared, runtime, phases)
    phase_started = perf_counter()
    output = postprocess(y, prepared)
    postprocessing_ms = (perf_counter() - phase_started) * 1000
    phase_started = perf_counter()
    save_image(output, args.output)
    save_ms = (perf_counter() - phase_started) * 1000
    total_ms = (perf_counter() - started) * 1000
    if args.comparison_dir:
        from PIL import Image

        original = prepared.rgb.copy()
        if prepared.alpha is not None:
            original.putalpha(prepared.alpha)
        names = (
            ("original.png", "denoised.png")
            if args.command == "denoise"
            else ("original.png", "bicubic_2x.png", "npu_sr_2x.png")
        )
        paths = [args.comparison_dir / name for name in names]
        if any(path.resolve() == args.input.resolve() for path in paths):
            raise SRException("Comparison directory would overwrite the source image.")
        save_image(original, paths[0])
        if args.command == "upscale":
            save_image(original.resize(output.size, Image.Resampling.BICUBIC), paths[1])
        save_image(output, paths[-1])
    print(f"Input:      {prepared.size[0]} × {prepared.size[1]}")
    print(f"Output:     {output.width} × {output.height}")
    print(f"Scale:      {runtime.spec.scale}×")
    print(f"Model:      {runtime.spec.identifier}")
    print(f"Backend:    {runtime.label}")
    print(f"Inference:  {inference_ms:.1f} ms ({prepared.tile_count} tiles)")
    print(f"Startup:    {runtime.startup_ms:.0f} ms (includes NPU proof when selected)")
    print(f"Load:       {load_ms:.1f} ms")
    print(f"Preprocess: {preprocessing_ms:.1f} ms")
    print(f"Tile copy:  {phases['tile_extraction_ms']:.1f} ms")
    print(f"Stitch:     {phases['stitching_ms']:.1f} ms")
    print(f"Postprocess:{postprocessing_ms:8.1f} ms")
    print(f"Save:       {save_ms:.1f} ms")
    print(f"Total:      {total_ms:.0f} ms (load through save, includes startup)")
    print(f"Saved:      {args.output}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .benchmark import benchmark, save_report
    from .image import load_image, preprocess

    if args.json and args.json.resolve() == args.input.resolve():
        raise SRException("JSON report must differ from the input image.")
    from .model import manifest_spec, validate_model

    path = resolve_model(args.model)
    prepared = preprocess(load_image(args.input), manifest_spec(validate_model(path)))
    report = benchmark(prepared, path, args.runs, args.warmups, args.verbose, args.gpu)
    print("Backend       Median      Mean       p95       Theoretical images/sec")
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
        available = "CPU/GPU" if args.gpu else "CPU"
        print(f"\nNPU unavailable — {available} measurements only: {report['npu_unavailable']}")
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

    runtime = Runtime(resolve_model(args.model), args.device, args.verbose, args.cache_dir)
    prepared = preprocess(load_image(args.input), runtime.spec)
    reference = load_image(args.reference)
    expected = (prepared.size[0] * 2, prepared.size[1] * 2)
    if reference.size != expected:
        raise SRException(
            "Ground truth must be exactly 2x the input dimensions (after EXIF rotation)."
        )
    output = postprocess(infer_y(prepared, runtime)[0], prepared)
    bicubic = prepared.rgb.resize(output.size, Image.Resampling.BICUBIC)
    print(f"Backend: {runtime.label}")
    print(f"SR RGB PSNR:      {psnr(output, reference):.2f} dB")
    print(f"Bicubic RGB PSNR: {psnr(bicubic, reference):.2f} dB")
    from .evaluate import quality_metrics

    print(f"SR luminance: {quality_metrics(output, reference)}")
    print(f"Bicubic luminance: {quality_metrics(bicubic, reference)}")
    return 0


def _models(args: argparse.Namespace) -> int:
    import json

    from .acquire_models import SOURCES, acquire_model

    if args.action == "list":
        for spec in MODELS.values():
            print(
                f"{spec.identifier:<20} {spec.task:<8} {spec.scale}x  "
                f"core={spec.core}  {spec.license}"
            )
        return 0
    if not args.identifier:
        raise SRException("Supply a model identifier. Run: npu-sr models list")
    spec = model_spec(args.identifier)
    if args.action == "download":
        print(f"Model ready: {acquire_model(spec.identifier, model_directory())}")
    else:
        source, digest = SOURCES[spec.architecture]
        print(json.dumps(spec.info() | {"source": source, "source_sha256": digest}, indent=2))
    return 0


def _suite(args: argparse.Namespace) -> int:
    from .benchmark import save_report
    from .suite import performance_suite, quality_suite

    if args.json.resolve() == args.input.resolve() or (
        args.quality and args.json.is_file() and args.json.parent.resolve() == args.input.resolve()
    ):
        raise SRException("JSON report would overwrite benchmark input data.")
    if args.quality:
        report = quality_suite(args.input, args.models, args.devices)
    else:
        report = performance_suite(
            args.input,
            args.models,
            args.devices,
            args.runs,
            args.warmups,
            args.trials,
            cache_dir=args.cache_dir,
            checkpoint=args.json,
        )
    save_report(report, args.json)
    print(f"Saved measured results: {args.json}")
    return 0


def _video(args: argparse.Namespace) -> int:
    from .benchmark import save_report
    from .video import VideoSettings, process_video

    if args.json and args.json.resolve() in {args.input.resolve(), args.output.resolve()}:
        raise SRException("Video JSON would overwrite input or output.")
    settings = VideoSettings(
        **{key: getattr(args, key) for key in VideoSettings.__dataclass_fields__}
    )

    def progress(frames: int) -> None:
        if frames % 30 == 0:
            print(f"Processed {frames} frames", file=sys.stderr)

    if args.command == "benchmark-video":
        from .video_benchmark import benchmark_video

        report = benchmark_video(args.input, args.output, settings, args.trials, args.json)
        if args.json:
            save_report(report, args.json)
        return 0
    report = process_video(args.input, args.output, settings, progress)
    width, height = report["input"]["resolution"]
    out_width, out_height = report["output"]["resolution"]
    print(f"Input:       {width} × {height} @ {report['input']['frame_rate']} FPS")
    print(f"Output:      {out_width} × {out_height} @ {report['output']['frame_rate']} FPS")
    print(f"Frames:      {report['frames_processed']}")
    print(f"SR backend:  {report['backend']} ({report['model']})")
    for kind in ("decode", "encode"):
        print(f"{kind.title()}:      {report['codec_evidence'][kind]}")
    print(f"End-to-end:  {report['end_to_end_fps']:.1f} FPS")
    print(f"Real-time factor: {report['real_time_factor']:.2f} (processing / source duration)")
    print(f"Saved:       {args.output}")
    if args.json:
        save_report(report, args.json)
    return 0


def _video_quality(args: argparse.Namespace) -> int:
    from .benchmark import save_report
    from .video_benchmark import evaluate_video

    if args.json.resolve() in {args.input.resolve(), args.reference.resolve()}:
        raise SRException("Quality JSON would overwrite video input.")
    report = evaluate_video(args.input, args.reference, args.ffmpeg, args.stride, args.vmaf)
    save_report(report, args.json)
    print(f"PSNR Y: {report['mean_psnr_y_db']} dB; SSIM Y: {report['mean_ssim_y']:.4f}")
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
            "denoise": _upscale,
            "models": _models,
            "benchmark-suite": _suite,
            "benchmark": _benchmark,
            "evaluate": _evaluate,
            "video": _video,
            "benchmark-video": _video,
            "evaluate-video": _video_quality,
        }[args.command](args)
    except KeyboardInterrupt:
        print("Cancelled; child processes cleaned up.", file=sys.stderr)
        return 130
    except (SRException, OSError, ValueError, ImportError) as exc:
        print(f"Error: {exc}\n\nRun: npu-sr doctor\nSee: docs/troubleshooting.md", file=sys.stderr)
        if args.verbose:
            logging.exception("Detailed failure")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
