"""Download and reproducibly export a pinned model (ESPCN baseline by default)."""

import argparse
from pathlib import Path

from npu_sr.acquire_models import acquire_model
from npu_sr.errors import SRException
from npu_sr.model import MODELS, model_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=model_directory())
    parser.add_argument("--model", choices=list(MODELS), default="espcn-x2")
    args = parser.parse_args()
    try:
        path = acquire_model(args.model, args.directory)
        print(f"Model ready: {path}")
        return 0
    except (SRException, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
