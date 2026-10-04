"""Acquire a pinned native FFmpeg build; binaries and datasets stay outside git.

FFmpeg's official download page links the BtbN builder. This GPL shared build
contains its licenses/notices. The project invokes it as a separate executable,
does not link its libraries, and does not redistribute its binaries.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "autobuild-2026-08-31-13-27"
VERSION = "n9.0.1-11-ge47273f4d9"
HASHES = {
    "arm64": "4419887ac2c5585d909d532944843756f67e430d4f3822e9062bff1455498abc",
    "x64": "00d78694632f17a1de325c639d0acf04a6b3ab8f20ce0a2e5bedd2d5e21e3adb",
}


def download(architecture: str, directory: Path) -> Path:
    variant = "winarm64" if architecture == "arm64" else "win64"
    name = f"ffmpeg-{VERSION}-{variant}-gpl-shared-9.0.zip"
    url = f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{RELEASE}/{name}"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / name
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 100_000_000:
                    raise ValueError("FFmpeg archive exceeds expected size limit")
                output.write(chunk)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != HASHES[architecture]:
        raise ValueError("FFmpeg SHA256 mismatch; remove the incomplete archive and retry")
    target = directory / "extracted"
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = target / entry.filename
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError("Unsafe FFmpeg archive path")
        bundle.extractall(target)
    executable = target / name.removesuffix(".zip") / "bin" / "ffmpeg.exe"
    # A matching archive hash does not guarantee a working upstream build.
    ready = directory / "ready.json"
    ready.unlink(missing_ok=True)
    result = subprocess.run([str(executable), "-version"], capture_output=True, timeout=15)
    if result.returncode:
        raise ValueError(
            f"Pinned FFmpeg failed startup (exit {result.returncode:#x}); not selected"
        )
    with executable.open("rb") as stream:
        executable_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    ready.write_text(json.dumps({"executable": executable.name, "sha256": executable_hash}) + "\n")
    return executable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=list(HASHES),
        default="arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64",
    )
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error(
            "Windows only; install native FFmpeg with your OS package manager on other systems"
        )
    directory = (
        args.directory
        or Path(os.environ.get("LOCALAPPDATA", Path.home()))
        / "npu-sr"
        / "tools"
        / f"ffmpeg-{args.architecture}-monthly-202608"
    )
    try:
        print(download(args.architecture, directory))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Error: {exc}\nSee docs/ffmpeg.md; no binary was accepted.\n")
