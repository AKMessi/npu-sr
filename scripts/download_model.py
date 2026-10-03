"""Download and reproducibly export the Apache-2.0 ESPCN x2 weights."""

import argparse
from pathlib import Path

from npu_sr.acquire import acquire
from npu_sr.errors import SRException
from npu_sr.model import model_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=model_directory())
    args = parser.parse_args()
    try:
        path = acquire(args.directory)
        print(f"Model ready: {path}")
        return 0
    except (SRException, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
