# Installation

The setup command is included in release wheels and source installations (v0.8+).

## Snapdragon Windows

Use native Python 3.12 ARM64 and Windows 11 build 26100 or newer. Check
`python -c "import platform; print(platform.machine())"` prints ARM64.
Install the stable ARM64 Windows App Runtime 2.x (minimum 2.3.0.0, tested 2.5.1.0)
from [Microsoft's downloads](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads)
and the [VC++ ARM64 redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
Update NPU drivers through Windows Update/OEM support. Python bindings installed
by pip are not the system App Runtime. Microsoft's [Python deployment instructions](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app)
use framework-dependent deployment; no SDK DLL copying is required.

From a clone:

```powershell
py -3.12-arm64 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
npu-sr setup --download-ffmpeg
npu-sr doctor
npu-sr video input.mp4 -o enhanced.mp4
```

For a release containing setup, install its wheel into the same virtual
environment with `python -m pip install <downloaded-wheel>` instead of `-e .`.
Release artifacts and their SHA256 file are published on the
[releases page](https://github.com/AKMessi/npu-sr/releases).
There is no PyPI publication claim. Do not install another ONNX Runtime wheel
alongside `onnxruntime-windowsml`; they share the Python namespace.

`setup` acquires Small and Medium models, verifies source/export hashes,
registers the catalog QNN provider into Python ORT, runs strict proof inference,
and tests hardware H.264 decode and H.264/HEVC/AV1 encode. Normal NPU setup is
ready only when NPU proof and the H.264-decode/AV1-encode video profile pass.
Individual codec failures remain visible. A provider or encoder listing is not proof.

FFmpeg acquisition is optional: `--download-ffmpeg` downloads a pinned native
GPL build from the BtbN builder linked by [FFmpeg](https://ffmpeg.org/download.html).
Its licenses stay with its files in the local cache; this repository and wheel
do not redistribute those binaries or link their libraries. Otherwise use an
existing tool via PATH, `--ffmpeg <executable>` or `NPU_SR_FFMPEG`.
A changed archive hash or failed startup is rejected. The newer September build
crashed on the tested machine; the tested August build remains pinned.

Models and tools go to a local application cache, or clone-local `models/`.
`NPU_SR_MODEL_DIR` chooses an explicit model directory. QNN contexts are local
and checked/proven again before use; never publish compiled contexts.
Initial network acquisition may take a minute; later inference is local.

## CPU-only Linux

Install native FFmpeg with your distribution's package manager, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
npu-sr setup --device cpu
npu-sr video input.mp4 -o enhanced.mp4 --device cpu --decode software --encode software
```

CPU setup performs real CPU inference. It reports unavailable Windows hardware
paths without claiming them. The default balanced path uses NV12 luminance;
explicit `--model espcn-x2-256 --frame-format rgb24` retains historical behavior.
CPU/GPU speed is not covered by the Snapdragon realtime NPU claim.

## Diagnose

`npu-sr doctor --gpu --json diagnostics.json` adds a strict DirectML proof and
machine-readable results. `--image-only` skips codec probes. Default doctor
requires the Small model and returns success only when its NPU proof passes;
video readiness is reported separately. `setup --models dncnn-25` acquires the
optional denoiser. See [troubleshooting](troubleshooting.md) for actionable errors.
