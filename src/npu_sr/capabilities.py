"""Executed capability probes; provider/encoder listings alone are not proof."""

import tempfile
from fractions import Fraction
from pathlib import Path

from .errors import SRException
from .ffmpeg import (
    VideoInfo,
    hardware_decode_probe,
    hardware_encode_probe,
    run_tool,
    tool_path,
)
from .model import model_path, validate_model

USER_MODELS = ("quicksrnet-small-y-x2", "quicksrnet-medium-y-x2", "dncnn-25")


def model_inventory() -> dict:
    result = {}
    for identifier in USER_MODELS:
        try:
            manifest = validate_model(model_path(identifier=identifier))
            result[identifier] = {"status": "ready", "sha256": manifest["sha256"]}
        except SRException as exc:
            result[identifier] = {"status": "not ready", "reason": str(exc)}
    return result


def video_capabilities(explicit: Path | None = None) -> dict:
    """Probe SDR 540p H.264 decode and 1080p encode, not every media profile."""
    report: dict = {"ready": False, "decode": {}, "encode": {}, "errors": []}
    try:
        ffmpeg = tool_path(explicit)
        report["ffmpeg_version"] = (
            run_tool([str(ffmpeg), "-version"]).stdout.decode("utf-8", "replace").splitlines()[0]
        )
    except (SRException, IndexError) as exc:
        report["errors"].append(str(exc))
        return report
    info = VideoInfo(960, 540, Fraction(30), 0.1, 3, False, "h264")
    with tempfile.TemporaryDirectory(prefix="npu-sr-codec-proof-") as directory:
        source = Path(directory) / "h264.mp4"
        try:
            run_tool(
                [
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=size=960x540:rate=30",
                    "-frames:v",
                    "3",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ]
            )
            report["decode"]["h264"] = {
                "status": "proven",
                "evidence": hardware_decode_probe(ffmpeg, source),
            }
        except SRException as exc:
            report["decode"]["h264"] = {"status": "not proven", "reason": str(exc)}
    for codec in ("h264", "hevc", "av1"):
        try:
            evidence = hardware_encode_probe(ffmpeg, info, 2, codec, "8M")
            report["encode"][codec] = {"status": "proven", "evidence": evidence}
        except SRException as exc:
            report["encode"][codec] = {"status": "not proven", "reason": str(exc)}
    report["scope"] = "three SDR frames, H.264 960x540 decode and 1920x1080 encode at 30 FPS"
    report["ready"] = (
        report["decode"].get("h264", {}).get("status") == "proven"
        and report["encode"]["av1"]["status"] == "proven"
    )
    return report


def gpu_capability(path: Path, verbose: bool = False) -> dict:
    from .runtime import Runtime

    try:
        runtime = Runtime(path, "gpu", verbose)
        return {"status": "proven", "evidence": runtime.evidence}
    except SRException as exc:
        return {"status": "not proven", "reason": str(exc)}
