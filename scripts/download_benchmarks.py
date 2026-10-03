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
    parser.add_argument("--directory", type=Path, default=Path("datasets/bsds300-five"))
    arguments = parser.parse_args()
    try:
        download(arguments.directory)
        print(f"Research subset ready: {arguments.directory}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
