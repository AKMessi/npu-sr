"""Create a reproducible photographic denoising comparison from a local clean image.

Use your own photograph, or the locally downloaded BSDS research subset. No
third-party images are copied into the repository. Added noise is synthetic,
sigma 25/255 with seed 2026; this is not a claim of general camera-noise removal.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from npu_sr.image import infer_y, load_image, postprocess, preprocess, save_image
from npu_sr.model import model_path
from npu_sr.runtime import Runtime


def create(input_path: Path, directory: Path, device: str) -> None:
    clean = load_image(input_path).convert("L").convert("RGB")
    noise = np.random.default_rng(2026).normal(0, 25, (clean.height, clean.width, 1))
    noisy = Image.fromarray(
        np.clip(np.rint(np.asarray(clean, np.float64) + noise), 0, 255).astype(np.uint8)
    )
    runtime = Runtime(model_path(identifier="dncnn-25"), device)
    prepared = preprocess(noisy, runtime.spec)
    result = postprocess(infer_y(prepared, runtime)[0], prepared)
    for name, image in [
        ("clean-reference.png", clean),
        ("noisy.png", noisy),
        ("denoised.png", result),
    ]:
        path = directory / name
        if path.resolve() == input_path.resolve():
            raise ValueError("Comparison path would overwrite input")
        save_image(image, path)
    print(f"Backend: {runtime.label}\nPhotographic comparison saved: {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", type=Path, default=Path("outputs/denoise-example"))
    parser.add_argument("--device", choices=["cpu", "npu", "gpu"], default="npu")
    args = parser.parse_args()
    create(args.input, args.directory, args.device)
