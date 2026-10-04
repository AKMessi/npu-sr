"""Hash-checked sources and controlled paired video fixtures for quality research."""

import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .errors import SRException
from .ffmpeg import PipeProcess, probe, run_tool, validate_cfr, write_frame
from .model import sha256


def acquire_source(spec: dict, directory: Path, identifier: str) -> Path:
    """Verify existing bytes or atomically acquire a pinned public source.

    ZIP contents are read by exact member name, never extracted by filesystem paths.
    Only this operation's temporary files are cleaned up on failure.
    """
    directory.mkdir(parents=True, exist_ok=True)
    archived = "member" in spec
    downloaded = directory / f"{identifier}.{'zip' if archived else 'mp4'}"
    if not downloaded.exists():
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as output:
            partial = Path(output.name)
            try:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "npu-sr"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    copied = 0
                    while block := response.read(1024 * 1024):
                        copied += len(block)
                        if copied > 1_000_000_000:
                            raise SRException("Benchmark source exceeds the one-GB limit.")
                        output.write(block)
                output.flush()
                if sha256(partial) != spec["sha256"]:
                    raise SRException("Benchmark source SHA256 changed; no fixture published.")
            except BaseException:
                output.close()
                partial.unlink(missing_ok=True)
                raise
        partial.replace(downloaded)
    if sha256(downloaded) != spec["sha256"]:
        raise SRException(f"Cached benchmark source hash changed: {identifier}")
    if not archived:
        return downloaded
    media = directory / f"{identifier}.mov"
    if not media.exists():
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as target:
            partial = Path(target.name)
            try:
                with zipfile.ZipFile(downloaded) as archive:
                    if archive.getinfo(spec["member"]).file_size > 1_000_000_000:
                        raise SRException("Extracted benchmark movie exceeds the one-GB limit.")
                    with archive.open(spec["member"]) as source:
                        shutil.copyfileobj(source, target)
                target.flush()
                if sha256(partial) != spec["media_sha256"]:
                    raise SRException("Extracted benchmark movie hash changed.")
            except BaseException:
                target.close()
                partial.unlink(missing_ok=True)
                raise
        partial.replace(media)
    if sha256(media) != spec["media_sha256"]:
        raise SRException(f"Extracted benchmark movie hash changed: {identifier}")
    return media


def validate_definition(definition: dict) -> None:
    """Reject ambiguous pairing and unsafe fixture identifiers before file creation."""
    if not definition["clips"]:
        raise SRException("Corpus must declare at least one clip.")
    for name, source in definition["sources"].items():
        url = urlsplit(source["url"])
        if (
            not re.fullmatch(r"[a-z0-9-]+", name)
            or url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])
        ):
            raise SRException("Invalid pinned corpus source identifier, URL or hash.")
    identifiers = set()
    for clip in definition["clips"]:
        name = clip["identifier"]
        if (
            not isinstance(name, str)
            or not name
            or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in name)
        ):
            raise SRException("Corpus identifiers must contain lowercase letters, digits or '-'.")
        if name in identifiers:
            raise SRException("Corpus identifiers must be unique.")
        identifiers.add(name)
        if clip["split"] not in {"development", "holdout"} or not 1 <= clip["frames"] <= 3600:
            raise SRException("Invalid corpus split or frame count.")
        if clip["source"] not in definition["sources"] and not clip["geometry"].startswith(
            "synthetic-"
        ):
            raise SRException("Corpus references an unknown source.")
        from .video_quality import validate_roi

        validate_roi(tuple(clip["metric_roi"]), 1920, 1080)


def synthetic_ui(frame: int, vertical: bool = False) -> Image.Image:
    """Original text/edge stress artwork, deterministic at 1920x1080."""
    canvas = Image.new("RGB", (2048, 1200), "#20232b")
    draw = ImageDraw.Draw(canvas)
    for row, size in enumerate((16, 20, 28, 40, 64)):
        font = ImageFont.load_default(size=size)
        draw.text(
            (80, 80 + row * 150),
            "npu-sr 0123456789 | Neural video / AaBbCc | 540p to 1080p",
            fill=(235, 235, 235),
            font=font,
        )
    for column in range(20):
        x = 60 + column * 90
        draw.line((x, 880, x, 1040), fill=(80 + column * 7, 180, 210), width=1 + column % 4)
    offset = frame % 60
    return canvas.crop(
        (
            0 if vertical else offset,
            offset if vertical else 0,
            1920 + (0 if vertical else offset),
            1080 + (offset if vertical else 0),
        )
    )


def prepare_clip(clip: dict, source: Path | None, directory: Path, ffmpeg: Path) -> dict:
    """Preserve selected reference pixels and derive exactly 2x paired LR inputs."""
    directory.mkdir(parents=True, exist_ok=True)
    name = clip["identifier"]
    reference, low = directory / f"{name}-reference.mkv", directory / f"{name}-low.mp4"
    common = [
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
        "-bitexact",
    ]
    geometry = clip["geometry"]
    if geometry.startswith("synthetic-ui"):
        encoder = PipeProcess(
            ffmpeg,
            [
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1920x1080",
                "-r",
                "30",
                "-i",
                "pipe:0",
                "-vf",
                "scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                *common,
                str(reference),
            ],
            input_pipe=True,
        )
        try:
            for index in range(clip["frames"]):
                write_frame(
                    encoder.process.stdin, synthetic_ui(index, "vertical" in geometry).tobytes()
                )
            encoder.process.stdin.close()
            encoder.finish()
        finally:
            encoder.close()
    elif geometry == "synthetic-texture":
        run_tool(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1920x1080:rate=30",
                "-frames:v",
                str(clip["frames"]),
                *common,
                str(reference),
            ],
            timeout=180,
        )
    else:
        if source is None:
            raise SRException("Filmed corpus clip requires a source.")
        info = probe(source, ffmpeg)
        if geometry == "letterbox" and (info.width, info.height) == (1920, 800):
            spatial = "pad=1920:1080:0:140:black"
        elif (
            geometry == "center-crop"
            and info.width >= 1920
            and info.height * 1920 >= 1080 * info.width
        ):
            spatial = "scale=1920:-2:flags=bicubic,crop=1920:1080"
        elif geometry == "native" and (info.width, info.height) == (1920, 1080):
            spatial = "null"
        else:
            raise SRException("Corpus native reference must be 1920x1080; no silent enlargement.")
        ratio = Fraction(info.fps, 30)
        filters = (
            f"setpts=(PTS-STARTPTS)*{ratio.numerator}/{ratio.denominator},{spatial},format=yuv420p"
        )
        run_tool(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-y",
                "-ss",
                str(clip["start_seconds"]),
                "-i",
                str(source),
                "-vf",
                filters,
                "-r",
                "30",
                "-frames:v",
                str(clip["frames"]),
                *common,
                str(reference),
            ],
            timeout=180,
        )
    run_tool(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-i",
            str(reference),
            "-vf",
            "scale=960:540:flags=bicubic:param0=0:param1=0.5",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(clip["lr_crf"]),
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-bitexact",
            str(low),
        ],
        timeout=180,
    )
    for path, resolution in ((reference, (1920, 1080)), (low, (960, 540))):
        info = probe(path, ffmpeg, count_frames=True)
        if (
            (info.width, info.height) != resolution
            or info.fps != 30
            or info.frames != clip["frames"]
        ):
            raise SRException("Prepared corpus fixture has incorrect geometry, FPS or frame count.")
        if validate_cfr(path, ffmpeg, info) != clip["frames"]:
            raise SRException("Prepared corpus cadence validation failed.")
    return clip | {
        "reference_file": reference.name,
        "reference_sha256": sha256(reference),
        "input_file": low.name,
        "input_sha256": sha256(low),
    }


def prepare_corpus(manifest: Path, directory: Path, ffmpeg: Path, split: str) -> dict:
    if split not in {"development", "holdout", "all"}:
        raise SRException("Unknown corpus split.")
    definition = json.loads(manifest.read_text(encoding="utf-8"))
    validate_definition(definition)
    clips = [row for row in definition["clips"] if split == "all" or row["split"] == split]
    sources = {
        name: acquire_source(spec, directory / "sources", name)
        for name, spec in definition["sources"].items()
        if any(row["source"] == name for row in clips)
    }
    version = run_tool([str(ffmpeg), "-version"]).stdout.decode().splitlines()[0]
    report = {
        "definition_sha256": sha256(manifest),
        "definition": definition,
        "preparation_software": {
            "ffmpeg_version": version,
            "ffmpeg_sha256": sha256(ffmpeg),
            "Pillow": Image.__version__,
            "numpy": np.__version__,
        },
        "complete": False,
        "clips": [],
    }
    destination = directory / f"provenance-{split}.json"
    for clip in clips:
        row = prepare_clip(clip, sources.get(clip["source"]), directory, ffmpeg)
        report["clips"].append(row)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {clip['identifier']}: {clip['frames']} aligned frames", flush=True)
    report["complete"] = True
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
