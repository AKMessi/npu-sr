"""Acquire and prepare the predeclared video quality corpus outside git."""

import argparse
import os
from pathlib import Path

from npu_sr.corpus import prepare_corpus
from npu_sr.ffmpeg import tool_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/corpus-v1.json"))
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
        / "npu-sr/benchmarks/v1-quality",
    )
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    prepare_corpus(args.manifest, args.directory, tool_path(args.ffmpeg), args.split)
