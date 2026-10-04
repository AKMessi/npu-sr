"""Acquire a pinned native FFmpeg build; binaries and datasets stay outside git.

FFmpeg's official download page links the BtbN builder. This GPL shared build
contains its licenses/notices. The project invokes it as a separate executable,
does not link its libraries, and does not redistribute its binaries.
"""

import argparse
import hashlib
import os
import platform
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "autobuild-2026-10-03-18-14"
VERSION = "n9.0.2-22-g46d8f462ee"
HASHES = {
    "arm64": "82b7eef78a79fdc93a2835154e753b4b175797712f707cc4f419405c0e9f6a2c",
    "x64": "6b2621d2f833cd94ae56371ce154bd0cab4aa504d802d661431093753a5f031f",
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
    return target / name.removesuffix(".zip") / "bin" / "ffmpeg.exe"


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
        / f"ffmpeg-{args.architecture}-9.0"
    )
    print(download(args.architecture, directory))
