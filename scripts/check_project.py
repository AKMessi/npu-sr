"""Validate public metadata, local README links, and distributable artifact contents."""

import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "0.1.0"
    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["scripts"]["npu-sr"] == "npu_sr.cli:main"
    required = [
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "docs/architecture.md",
        "docs/qnn.md",
        "docs/benchmarking.md",
        "docs/troubleshooting.md",
    ]
    assert all((ROOT / path).is_file() for path in required)
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if target.startswith(("https://", "http://", "#")):
                continue
            assert (path.parent / target.split("#")[0]).exists(), f"Broken link: {path}: {target}"
    for artifact in (ROOT / "dist").glob("*"):
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
        elif artifact.name.endswith(".tar.gz"):
            with tarfile.open(artifact) as archive:
                names = archive.getnames()
        else:
            continue
        forbidden = {".dll", ".bin", ".onnx", ".pb", ".pyc"}
        assert not any(Path(name).suffix in forbidden or ".venv" in name for name in names)
        assert any(name.endswith("TF-ESPCN-APACHE-2.0.txt") for name in names)
    print("Project metadata, documentation links, and package contents passed.")


if __name__ == "__main__":
    check()
