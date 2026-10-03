"""Acquire a five-image BSDS300 research subset. Images are NOT redistributed.

Berkeley permits non-commercial research and educational downloads. See
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
Citation: Martin, Fowlkes, Tal, Malik. ICCV 2001, A Database of Human Segmented
Natural Images and its Application to Evaluating Segmentation Algorithms.
"""

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

SOURCE = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
SHA256 = "7f855a00491a3f1ca2609b3c96651b88b265fe59b70a7b007cc7a2850a22b386"
IMAGES = ["101085", "101087", "102061", "103070", "105025"]
BSD68_SOURCE = (
    "https://raw.githubusercontent.com/cszn/DnCNN/"
    "e93b27812d3ff523a3a79d19e5e50d233d7a8d0a/testsets/BSD68/"
)
BSD68_HASHES = {
    "test001.png": "94ddd662377347ca7b07d5a11c2a83feff658723cf31f79bfa75f65713e94ee4",
    "test002.png": "7dabadb169c463d7bd7da110c36d486846c8c787883b6fc979257d44e09b3a29",
    "test003.png": "d27b8117a1b175b594f80de162e81776c8587e2e512fa4d07fbaeacb67c97949",
    "test004.png": "3ceebbbf583e355fefcdb9e653bff65451b20a512f1c4ca12a0518f4569b481c",
    "test005.png": "d187d8b58e5d19b4aa6e055b849064fb974cad7b18083c25f6580a1ba17d6bb3",
}


def download_bsd68(directory: Path) -> None:
    """Author-designated denoising test images, pinned individually; research only."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in BSD68_HASHES.items():
        with urllib.request.urlopen(BSD68_SOURCE + name, timeout=60) as response:
            data = response.read(1_000_001)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"BSD68 source hash mismatch: {name}")
        (directory / name).write_bytes(data)
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "dataset": "BSD68-author-test-first-five",
                "source": BSD68_SOURCE,
                "terms": "BSDS-derived, non-commercial research; images not redistributed",
                "files": BSD68_HASHES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def download(directory: Path) -> None:
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        data = response.read(23_000_001)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Dataset source hash changed; refusing to extract")
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for identifier in IMAGES:
            name = f"BSDS300/images/test/{identifier}.jpg"
            member = archive.getmember(name)
            if not member.isfile() or member.size > 1_000_000:
                raise ValueError("Unexpected dataset member")
            # Write only explicit filenames. Never extract arbitrary tar paths or links.
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing dataset image")
            image = stream.read()
            (directory / f"{identifier}.jpg").write_bytes(image)
            files[f"{identifier}.jpg"] = hashlib.sha256(image).hexdigest()
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "dataset": "BSDS300-test-first-five",
                "source": SOURCE,
                "source_sha256": SHA256,
                "terms": "non-commercial research and educational use; not redistributed",
                "files": files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bsds300-five", "bsd68-five"], default="bsds300-five")
    parser.add_argument("--directory", type=Path)
    arguments = parser.parse_args()
    try:
        directory = arguments.directory or Path("datasets") / arguments.dataset
        (download_bsd68 if arguments.dataset == "bsd68-five" else download)(directory)
        print(f"Research subset ready: {directory}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
