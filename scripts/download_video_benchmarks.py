"""Acquire Tears of Steel and prepare small aligned 30 FPS research clips.

Film: Blender Foundation / mango.blender.org, CC BY 3.0. Source files and clips
are not redistributed. This explicitly retimes the 24 FPS source by 1.25x to
30 FPS without intentionally duplicating frames, and center-crops to 16:9.
References preserve the decoded movie pixels losslessly after those transforms.
"""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from npu_sr.ffmpeg import run_tool, tool_path
from npu_sr.model import sha256

SOURCE = "https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip"
SHA256 = "d87a41de040d3814dbde143e9ab85ef122caf22265f660b0bebf476cd8b357a5"
MOVIE_SHA256 = "99486359be7e3681168a0fe94e1cbb0284c48b57b2a3ce7df4fa75e185987a15"
CLIPS = {"faces": 32, "scene": 10, "motion": 124, "texture": None}


def preparation_software(ffmpeg: Path) -> dict:
    return {
        "ffmpeg_version": run_tool([str(ffmpeg), "-version"]).stdout.decode().splitlines()[0],
        "ffmpeg_sha256": sha256(ffmpeg),
        "bitexact": True,
    }


def source_movie(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "tears_of_steel_1080p.mov.zip"
    if not archive.exists():
        with urllib.request.urlopen(SOURCE, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(
            "Tears of Steel SHA256 mismatch; inspect/remove incomplete archive and retry"
        )
    movie = directory / "tears-of-steel" / "tears_of_steel_1080p.mov"
    if not movie.exists():
        movie.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open("tears_of_steel_1080p.mov") as source, movie.open("wb") as target:
                shutil.copyfileobj(source, target)
    if sha256(movie) != MOVIE_SHA256:
        raise ValueError(
            "Extracted source movie SHA256 changed; inspect the local benchmark source"
        )
    return movie


def acquire(directory: Path, duration: int, ffmpeg: Path) -> None:
    movie = source_movie(directory)
    provenance = {
        "source": SOURCE,
        "source_sha256": SHA256,
        "movie_sha256": MOVIE_SHA256,
        "preparation_software": preparation_software(ffmpeg),
        "attribution": "Blender Foundation / mango.blender.org",
        "license": "CC-BY-3.0",
        "duration_seconds": duration,
        "preparation": (
            "24 FPS -> 30 FPS by 1.25x retiming; center crop after aspect-preserving scale; "
            "FFV1 reference, H264 CRF10 LR"
        ),
        "clips": [],
    }
    for name, start in CLIPS.items():
        reference = directory / f"{name}-reference.mkv"
        if start is None:
            run_tool(
                [
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=1920x1080:rate=30",
                    "-t",
                    str(duration),
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",
                    "-pix_fmt",
                    "yuv420p",
                    "-bitexact",
                    str(reference),
                ],
                timeout=300,
            )
        else:
            run_tool(
                [
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(start),
                    "-i",
                    str(movie),
                    "-vf",
                    "setpts=(PTS-STARTPTS)*0.8,scale=-2:1080:flags=bicubic,crop=1920:1080",
                    "-af",
                    "atempo=1.25",
                    "-r",
                    "30",
                    "-t",
                    str(duration),
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-bitexact",
                    str(reference),
                ],
                timeout=300,
            )
        row = {
            "name": name,
            "source_start_seconds": start,
            "reference_sha256": sha256(reference),
            "inputs": {},
        }
        for width, height in ((640, 360), (960, 540), (1280, 720)):
            # Each LR size has its own matching 2x reference; 720p tests are not quality claims.
            low = directory / f"{name}-{width}x{height}.mp4"
            run_tool(
                [
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(reference),
                    "-vf",
                    f"scale={width}:{height}:flags=bicubic",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "10",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "copy",
                    "-bitexact",
                    str(low),
                ],
                timeout=300,
            )
            row["inputs"][f"{width}x{height}"] = {"file": low.name, "sha256": sha256(low)}
        provenance["clips"].append(row)
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Aligned clips prepared: {directory}")


def prepare_sustained(
    directory: Path, duration: int, ffmpeg: Path, source: Path | None = None
) -> None:
    """Prepare a varied real-film 540p input without storing a large HR reference."""
    movie = source or source_movie(directory)
    if sha256(movie) != MOVIE_SHA256:
        raise ValueError("Sustained source movie SHA256 changed")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"sustained-960x540-{duration}s.mp4"
    run_tool(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-ss",
            "32",
            "-i",
            str(movie),
            "-vf",
            "setpts=(PTS-STARTPTS)*0.8,scale=-2:540:flags=bicubic,crop=960:540",
            "-af",
            "atempo=1.25",
            "-r",
            "30",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-bitexact",
            str(output),
        ],
        timeout=1200,
    )
    from npu_sr.ffmpeg import probe, validate_cfr

    info = probe(output, ffmpeg, count_frames=True)
    count = validate_cfr(output, ffmpeg, info)
    if count != duration * 30 or info.frames != count:
        raise ValueError("Sustained input frame count changed")
    (directory / "sustained-provenance.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "source_sha256": SHA256,
                "movie_sha256": MOVIE_SHA256,
                "preparation_software": preparation_software(ffmpeg),
                "license": "CC-BY-3.0",
                "attribution": "Blender Foundation / mango.blender.org",
                "source_start_seconds": 32,
                "input_file": output.name,
                "input_sha256": sha256(output),
                "input": info.public(),
                "preparation": (
                    "24 FPS film retimed 1.25x to 30 FPS, aspect-preserving scale/center crop, "
                    "H264 CRF10 LR"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Sustained input ready: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument(
        "--source-movie", type=Path, help="Reuse a hash-verified movie for sustained input"
    )
    parser.add_argument("--duration", type=int)
    parser.add_argument("--sustained", action="store_true")
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    duration = args.duration if args.duration is not None else (120 if args.sustained else 4)
    if not 1 <= duration <= 300:
        parser.error("duration must be 1–300 seconds")
    default = (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
        / "npu-sr"
        / "benchmarks"
        / "video"
    )
    if args.source_movie and not args.sustained:
        parser.error("--source-movie is only used with --sustained")
    if args.sustained:
        prepare_sustained(
            args.directory or default, duration, tool_path(args.ffmpeg), args.source_movie
        )
    else:
        acquire(args.directory or default, duration, tool_path(args.ffmpeg))
