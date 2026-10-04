# npu-sr

> practical image and video enhancement on Qualcomm Hexagon NPUs.

[![CI](https://github.com/AKMessi/npu-sr/actions/workflows/ci.yml/badge.svg)](https://github.com/AKMessi/npu-sr/actions/workflows/ci.yml)

**2× image/video super-resolution and image luminance denoising on Snapdragon X
Windows laptops.** Uses the Hexagon NPU for neural enhancement, with verified
hardware video decode/encode where available. CPU and DirectML GPU comparison
remain available. An NPU is a processor designed to run neural networks efficiently.

This project explores practical local workloads for the NPU, with reproducible
models, strict execution checks and measurements rather than utilization claims.
**v0.4 adds sustained realtime video processing.** Its final measured configuration
and quality tradeoffs are in the [v0.4 report](benchmarks/v0.4/summary.md).

**960×540 → 1920×1080 @ 30 FPS: 34.3 FPS sustained**,
three 120-second sources / about 105 seconds processing each, no lost frames.
QNN and hardware codecs verified. Fixed 0.5 neural blend, AV1 8M, two-frame queues.
Streaming processing excludes the separately disclosed post-encode file audit,
which makes total CLI elapsed time longer than the source. Power is not measured.

## Results and examples

The included MIT test card and actual QNN outputs:

| Input | Bicubic 2× | ESPCN 2×, QNN NPU |
| --- | --- | --- |
| ![Input](examples/input.png) | ![Bicubic](examples/bicubic_2x.png) | ![NPU result](examples/npu_sr_2x.png) |

Actual video output crops (input / bicubic / QNN realtime):

![Measured video comparison](examples/realtime-comparison.gif)

This is a 6 FPS preview of the measured 30 FPS outputs; the input is enlarged
with nearest-neighbor sampling for display. Source: *Tears of Steel*, Blender
Foundation / [mango.blender.org](https://mango.blender.org/),
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
Crop, retiming, enhancement and GIF conversion are adaptations.
[Reproduction and attribution](examples/README.md).

Lightweight models sharpen some edges but can ring, blur texture or flicker.
No missing-detail reconstruction or state-of-the-art quality is promised.

The realtime network beats bicubic on the five-image BSDS research subset:
**28.31 dB / 0.8463 SSIM**, versus **27.95 dB / 0.8399**. FSRCNN reaches
**28.48 dB / 0.8517**. Delivered video quality is content dependent; consult the
paired PSNR/SSIM/VMAF results before choosing a preset. Power is **not measured**.

All results describe one machine, declared models/inputs/settings, and three
performance trials; they are not general Snapdragon claims. Historical reports:
[v0.1](benchmarks/snapdragon-x-plus.json) · [v0.2](benchmarks/v0.2/summary.md) ·
[v0.3](benchmarks/v0.3/summary.md) · [v0.4](benchmarks/v0.4/summary.md).

## Hardware and prerequisites

Target: Snapdragon X Plus / X Elite Windows 11 ARM64, build 26100 (24H2) or newer,
with supported Qualcomm drivers. Other machines require local validation.
CPU image/video processing also works on Linux; generic CI uses CPU only.

Tested locally on 2026-10-04:

| Component | Configuration |
| --- | --- |
| Processor | Snapdragon X Plus X1P-42-100, 8 logical CPUs, 16 GB memory |
| NPU | Qualcomm Hexagon, 45 TOPS; driver 30.0.219.1000 |
| OS / Python | Windows 11 ARM64 build 26200 / Python 3.12.0 ARM64 |
| ONNX Runtime Windows ML | package 1.25.2.202605110140, API 1.25.2 |
| Windows ML / App Runtime | wasdk 2.3.0 / installed runtime 2.5.1.0 |
| QNN catalog package | 2.2480.53.0 |
| GPU | Qualcomm Adreno X1-45, strict DirectML |
| FFmpeg | native ARM64 n9.0.1-11-ge47273f4d9 |

Install native **Python 3.12 ARM64**, the stable **ARM64 Windows App Runtime 2.x
(version 2.3.0.0 or newer)** from Microsoft's [download page](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads),
the [VC++ ARM64 redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist),
and current OEM/Windows Update NPU drivers. App Runtime is separate from pip's
Python bindings. Python 3.11–3.13 is allowed; 3.12 ARM64 is locally tested.
No Qualcomm SDK, Visual Studio, copied runtime DLLs or cloud credentials are needed.
Internet is needed for acquisition; inference itself is local.

## Install and quickstart

```powershell
git clone https://github.com/AKMessi/npu-sr.git
cd npu-sr
py -3.12-arm64 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/download_model.py
npu-sr models download espcn-x2-256
npu-sr doctor
npu-sr upscale examples/input.png -o output.png --device npu
```

If activation is blocked, invoke `.venv\Scripts\python.exe` and
`.venv\Scripts\npu-sr.exe` directly. On Linux use `python3 -m venv .venv` and
`source .venv/bin/activate`. Do not install another ORT wheel alongside
`onnxruntime-windowsml`: the wheels share the same Python import namespace.

Weights remain ignored. Downloads verify source hashes and write an adjacent
ONNX integrity/provenance manifest. Outside the clone, use `NPU_SR_MODEL_DIR` or
acquire models in the user's application cache. See [models](docs/models.md).

### Images and diagnostics

```powershell
npu-sr doctor
npu-sr models list
npu-sr models info fsrcnn-x2
npu-sr models download fsrcnn-x2
npu-sr upscale input.jpg -o output.png --model fsrcnn-x2 --device npu --verbose
npu-sr upscale input.webp -o output.png --device cpu
npu-sr upscale input.png -o output.png --comparison-dir outputs/compare
npu-sr models download dncnn-25
npu-sr denoise noisy.png -o clean.png --device npu
npu-sr benchmark examples/input.png --runs 30 --gpu --json outputs/image.json
npu-sr evaluate low.png high_ground_truth.png --device npu
```

PNG/JPEG/still WebP, EXIF orientation and alpha are handled. SR preserves aspect
ratio and doubles dimensions. DnCNN denoises luminance at the original size,
targeting Gaussian sigma 25/255; it does not remove chroma noise. Quality metrics
require aligned HR/clean references. ICC/metadata and animation are outside scope.

### Video and realtime

```powershell
python scripts/download_ffmpeg.py
npu-sr video input.mp4 -o output.mp4 --preset realtime --json outputs/video.json
npu-sr video input.mp4 -o output.mp4 --preset quality --device npu --codec av1
npu-sr video input.mp4 -o cpu.mp4 --device cpu --decode software --encode software
npu-sr benchmark-video input.mp4 -o outputs/trials --device npu --trials 3 --json outputs/video.json
npu-sr evaluate-video enhanced.mp4 high-reference.mkv --vmaf --json outputs/quality.json
```

The realtime preset materializes documented settings and prints them. Explicit
options override a preset; changed settings do not inherit its benchmark claim.
Audio is copied by default (`--audio none` to disable). MP4/MKV, CFR SDR, square
pixels and even dimensions are supported. HDR, full-range planar video, rotation
and variable framerates require explicit conversion or the documented RGB path.
Existing outputs require `--overwrite`.

FFmpeg is acquired outside git with a pinned hash and separate GPL notices.
`--ffmpeg path/to/ffmpeg` or `NPU_SR_FFMPEG` selects an explicit installation.
Hardware modes require executed-path evidence. Software AV1 selects an installed
SVT-AV1 or libaom encoder. See [FFmpeg](docs/ffmpeg.md), [video pipeline](docs/video-pipeline.md)
and [realtime reproduction](docs/realtime.md).

## Architecture and execution proof

```mermaid
flowchart TD
    V[Compressed video] --> D[FFmpeg decoder / D3D11VA when proven]
    D --> P[Bounded native NV12 or RGB frame stream]
    P --> T[CPU normalization and fixed tiles]
    W[Windows ML catalog and Python ORT registration] --> O[Persistent ONNX Runtime session]
    T --> O
    O --> Q[QNN HTP / Hexagon NPU]
    Q --> S[CPU stitching, chroma and declared postprocessing]
    S --> E[FFmpeg encoder / hardware QCOM MFT when proven]
    E --> U[Enhanced video / copied audio]
```

Windows ML installs/discovers QNN and explicitly registers its library in Python's
ORT environment. It is not another inference engine. Fixed-shape, overlapping
tiles discard their receptive-field halo. FP32 ONNX uses QNN HTP FP16 math.
One session remains alive across all video frames; proof and warmups precede timing.

| Device / codec mode | Behavior |
| --- | --- |
| `--device npu` | Requires an NPU hardware device, HTP, disabled CPU fallback and executed QNN kernels |
| `--device gpu` | Requires strict DirectML GPU assignment and executed GPU kernels |
| `--device cpu` | CPUExecutionProvider only |
| `--device auto` | Prefers strict NPU; explicitly reports CPU fallback on initialization failure |
| `--decode/--encode hardware` | Fails if the actual hardware path cannot be established |
| `--decode/--encode auto` | Explicitly reports an unavailable hardware path and chosen software fallback |

Strict proof profiles the **actual session** and rejects CPU/other-provider
kernels. Provider lists alone are not evidence; CPU may still appear in the list.
No midstream fallback occurs. Hardware decode requires D3D11 frames and explicit
hardware download. Encoding requires hardware-only Media Foundation selection,
an activated transform, and validated output counts/cadence. H.264/HEVC/AV1
selected QCOM hardware transforms locally. No internal VPU/NPU utilization is inferred.

Task Manager → Performance → NPU is an external sanity check. `--verbose` reports
initialization evidence. “100% NPU” is not claimed: IO, tiling, chroma, stitching,
transfers and orchestration use CPU. See [QNN](docs/qnn.md) and [architecture](docs/architecture.md).

## Benchmarks, limitations and development

[Benchmark methodology](docs/benchmarking.md) separates startup/proof, inference,
complete-image processing and actual video throughput. [Realtime](docs/realtime.md)
defines the sustained gate, latency scope and settings. All raw release results
are committed without datasets, machine identifiers or private paths.

- Per-frame models have no temporal context; ringing/flicker and content-dependent regressions remain.
- Real-time validation covers a declared 540p workload on one laptop, not arbitrary codecs/resolutions.
- Native NV12 assumes limited-range SDR; RGB remains available for other supported SDR input.
- No zero-copy, GUI, webcam/OBS integration, power measurement or thermal sensor claim.
- Driver/provider updates can change compatibility; proof runs for every new session/cache load.
- Models, FFmpeg, proprietary runtimes and benchmark datasets are acquired separately.

Run `doctor` for missing components. See [troubleshooting](docs/troubleshooting.md).

## Roadmap and license

- [x] v0.1: image SR, strict Qualcomm NPU, CPU benchmark, diagnostics
- [x] v0.2: improved models, denoising, tiled inference, strict GPU comparison, quality suite
- [x] v0.3: in-memory video, FFmpeg, proven hardware decode/encode, audio and output validation
- [x] v0.4: sustained realtime video, bounded pipeline, measured quality/speed tradeoffs
- [ ] Future: temporal models, live sources, perceptual preprocessing, further codec/NPU experiments

Our code is [MIT](LICENSE). ESPCN/FSRCNN/LapSRN upstream weights are Apache-2.0;
KAIR DnCNN is MIT. Weights are not committed. Microsoft, Qualcomm and FFmpeg
retain their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

[Contributing](CONTRIBUTING.md) · [security reporting](SECURITY.md)

```powershell
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m pytest -m npu
python -m pytest -m video_hw
python -m build
```

Ordinary CI needs no NPU. Hardware tests and sustained benchmarks run locally.
