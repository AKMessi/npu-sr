# Benchmarking

```powershell
npu-sr benchmark examples/input.png --runs 30 --warmups 5 --json outputs/results.json
```

The command prepares the image once, constructs an explicit CPU session and then
an explicit NPU session, warms each up, and measures repeated complete-image
inference. Each sample is the sum of `perf_counter` intervals around individual
ORT `run` calls, including output transfer and validation. It excludes tile input
copying, stitching, preprocessing, postprocessing, file IO and startup.
The SR luminance output is actually computed, not simulated.

Median, arithmetic mean, and NumPy's interpolated 95th percentile are reported.
Theoretical images/sec is 1000/median; it is an inference-only calculation and does not
represent an implemented video pipeline. Speedup is CPU median/NPU median.
For no NPU, only actual CPU results appear, together with the NPU failure reason.

Startup covers provider preparation, session construction, graph preparation,
and NPU proof inference. The v0.2 startup timer includes artifact validation;
Python import and image preparation happen outside startup. CPU session creation
does not include an NPU-style proof call, so its startup scope is inherently shorter.
First-time provider download may increase NPU startup substantially. The optional v0.2 cache can reuse compiled QNN contexts. Each session still runs
strict proof. The benchmark matrix records first and subsequent session startup
and their constituent phases; a cache hit must not be described as cold compilation.

## Reproducibility

Use the same image/model hash, runs, warmups, power mode, and runtime/driver versions.
Close background workloads, record whether plugged in, and let temperatures settle.
The v0.1 sample is one local run in the existing desktop environment; power mode,
thermal state, and background activity were not controlled or instrumented.
It demonstrates execution and supplies raw samples; it is not a rigorous hardware study.

The machine-readable report includes UTC timestamp, model name/version/hash/precision,
OS/architecture/build, image dimensions, tile count, iteration count, warmups,
raw latency samples, statistics, startup, software versions, and assignment evidence.
It excludes hostname, serial number, user name and image paths. Review failure
strings before publicly sharing results because a dependency's error may include
a local path. Detailed verbose logs can include DLL paths and should be sanitized.

The checked-in [sample report](../benchmarks/snapdragon-x-plus.json) uses the included
MIT test card. Benchmark images and model weights are not implicitly packaged into JSON.

## Quality

Use `upscale --comparison-dir` to inspect the low-resolution source, bicubic result,
and neural result side by side. For a real aligned high-resolution reference, use:

```powershell
npu-sr evaluate low.png high_ground_truth.png --device npu
```

Evaluation computes unweighted full-image RGB PSNR for both SR and bicubic output.
It requires exactly 2Ã— dimensions and opaque images. No border cropping or alignment
is performed. This is a metric with a specific definition, not a perceptual score.
Do not use the low-resolution source as ground truth. v0.2 also reports full-range
Rec.601 luminance PSNR and SSIM after a two-HR-pixel border shave. SSIM uses
11x11 Gaussian windows (sigma 1.5), population covariance and valid windows.


## Model suite (v0.2)

```powershell
python scripts/download_benchmarks.py
npu-sr models download fsrcnn-x2
npu-sr models download dncnn-25
npu-sr benchmark-suite datasets/bsds300-five --quality --models espcn-x2 fsrcnn-x2 dncnn-25 --devices cpu npu gpu --json outputs/quality.json
npu-sr benchmark-suite examples/input.png --models espcn-x2 fsrcnn-x2 --runs 5 --warmups 3 --trials 3 --cache-dir .cache/qnn --json outputs/performance.json
```

Download every model specified in `--models` first. Linux CPU-only users should
pass `--devices cpu`. Explicitly requested backends must work: this suite fails
instead of omitting a GPU/NPU result. The original `benchmark IMAGE` command
retains its v0.1 CPU-only-on-unavailable-NPU behavior and now accepts `--gpu`.

Performance uses five resolutions, from 256x160 to 1920x1080. The input test card
is resized deterministically to those dimensions; this is a timing workload,
not a quality dataset. Three independent measured trials, each after warmups,
produce median/mean/p95 phase statistics. The headline is the median of trial
medians. Warm total includes preprocessing, tile copies, validated ORT calls,
stitching and RGB reconstruction; it excludes session startup and disk IO. PNG
encoding is measured separately into memory. It is not video FPS.

Windows process CPU time is measured across a complete trial because its timer
has coarse granularity. Working set and cumulative process peak come from the OS;
peak values are not isolated model allocations. Power mode/AC line are recorded.
Power, temperatures and accelerator utilization are not measured. Background
processes remain uncontrolled, so modest differences require repeat measurements.

Quality uses the first five sorted BSDS300 test images, cropped to even HR sizes,
with Pillow bicubic LR generation. This differs from MATLAB's canonical BSD100
protocol; report it as this explicit subset and degradation. Denoising converts
the reference to gray and adds clipped Gaussian luminance noise at sigma 25 with
seed 2026. Each output is compared with the clean reference, never the LR/noisy
source. The JSON includes baseline metrics, image hashes, model hashes and
execution evidence. No dataset images or private machine identifiers are published.

[Research notes](research-notes.md) explain the dataset's research-only terms.
[Release summary](../benchmarks/v0.2/summary.md) links actual measurements and
records optimizations and their limits.
