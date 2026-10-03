# Contributing

Keep changes focused on reproducible single-image inference. Include actual local
evidence for claims about NPU assignment, speed, quality, or compatibility.

## Development

Use Python 3.12 ARM64 on the target laptop. Install the prerequisites in README,
then `python -m pip install -e ".[dev]"` and download the model if needed.
On Linux/macOS, standard ORT runs CPU tests; no Windows ML or NPU is needed.

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest
python scripts/check_project.py
python -m build
python -m twine check dist/*
```

Normal tests create a tiny ONNX fixture locally. The pretrained CPU integration
test skips if weights are absent. NPU tests are separate and require the real model:

```powershell
python scripts/download_model.py
python -m pytest -m npu
npu-sr doctor
npu-sr upscale examples/input.png -o outputs/npu.png --device npu --verbose
npu-sr benchmark examples/input.png --runs 30 --json outputs/results.json
```

## Pull requests

- Describe the concrete behavior change and relevant validation.
- Preserve strict `--device npu` behavior and explicit auto fallback reporting.
- Keep preprocessing deterministic and type hints useful.
- Include tests for new error handling and backend selection changes.
- Check model operator support, provenance, SHA256, and redistribution terms.
- Never add vendor DLLs, SDK files, downloaded model weights, credentials, or
  logs containing usernames or personal paths.
- Update docs and changelog for user-visible changes.

Before sharing a benchmark, include input dimensions, tile count, warmups,
iterations, software versions, and timing scope. Report failures honestly.
