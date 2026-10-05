"""Clone-compatible wrapper for the packaged, hash-checked FFmpeg installer."""

import argparse
import os
import platform
import subprocess
from pathlib import Path

from npu_sr.install_ffmpeg import HASHES, download

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
        parser.error("Windows only; install FFmpeg with your OS package manager elsewhere")
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
