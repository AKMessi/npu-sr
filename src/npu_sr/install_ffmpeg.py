"""Acquire a pinned native FFmpeg build; binaries and datasets stay outside git.

FFmpeg's official download page links the BtbN builder. This GPL shared build
contains its licenses/notices. The project invokes it as a separate executable,
does not link its libraries, and does not redistribute its binaries.
"""

import hashlib
import json
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
    if architecture not in HASHES:
        raise ValueError("FFmpeg architecture must be arm64 or x64")
    variant = "winarm64" if architecture == "arm64" else "win64"
    name = f"ffmpeg-{VERSION}-{variant}-gpl-shared-9.0.zip"
    url = f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{RELEASE}/{name}"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / name
    if not archive.exists():
        partial = archive.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100_000_000:
                        raise ValueError("FFmpeg archive exceeds expected size limit")
                    output.write(chunk)
            with partial.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != HASHES[architecture]:
                    raise ValueError("FFmpeg SHA256 mismatch; no binary accepted")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != HASHES[architecture]:
        raise ValueError("FFmpeg SHA256 mismatch; remove the incomplete archive and retry")
    target = directory / "extracted"
    with zipfile.ZipFile(archive) as bundle:
        if (
            len(bundle.infolist()) > 20_000
            or sum(e.file_size for e in bundle.infolist()) > 500_000_000
        ):
            raise ValueError("FFmpeg archive exceeds expanded size limit")
        for entry in bundle.infolist():
            path = target / entry.filename
            if (
                not path.resolve().is_relative_to(target.resolve())
                or "\\" in entry.filename
                or (entry.external_attr >> 16) & 0o170000 == 0o120000
            ):
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
