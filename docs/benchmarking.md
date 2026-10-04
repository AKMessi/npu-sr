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
It requires exactly 2× dimensions and opaque images. No border cropping or alignment
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
python scripts/download_benchmarks.py --dataset bsd68-five
npu-sr benchmark-suite datasets/bsd68-five --quality --models dncnn-25 --devices cpu npu gpu --json outputs/denoising-quality.json
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

BSD68's first five author-designated test images provide a separate denoising
validation subset. Its immutable upstream files are individually hash checked.
BSDS300 photographic demonstrations alone do not establish a held-out denoising
score or absence of overlap with a model's original training data.

[Research notes](research-notes.md) explain the dataset's research-only terms.
[Release summary](../benchmarks/v0.2/summary.md) links actual measurements and
records optimizations and their limits.

## Video (v0.3)

```powershell
python scripts/download_ffmpeg.py
python scripts/download_video_benchmarks.py
python scripts/run_video_benchmarks.py --trials 3
python scripts/run_video_benchmarks.py --quality
npu-sr benchmark-video input.mp4 -o outputs/trials --device npu --trials 3 --json outputs/video.json
npu-sr evaluate-video enhanced.mp4 high-reference.mkv --vmaf --json outputs/video-quality.json
```

The small suite uses three four-second CC BY Tears of Steel clips (faces, outdoor
scene, motion) and a synthetic texture/text sequence. The original 24 FPS film is
retimed 1.25× to 30 FPS without intentionally duplicating frames, scaled with
preserved aspect ratio and center-cropped to 1920×1080. That is benchmark input
preparation, not a runtime enhancement shortcut. FFV1 references preserve those
decoded pixels losslessly; LR uses bicubic downscale and H.264 CRF10. This source
is already compressed film, not raw camera footage. Dataset acquisition records
the immutable source hash and generated clip hashes. Videos are not committed.

Performance tests 360p/540p/720p inputs, CPU versus NPU neural enhancement with
the same hardware H.264 codec configuration. Three complete trials retain their
results. Standalone decoding includes hardware download and raw pipe transfer.
Standalone encoding cycles up to sixteen predecoded reference frames through
RGB conversion, hardware encode and flush. These separate tests are not a
decomposition of a parallel pipeline's wall time. Startup is measured separately.

Quality compares delivered AV1 output at the same 8M target rate with aligned
1080p references. Every twelfth frame receives Rec.601 Y PSNR/SSIM, with the same
two-pixel shave as image quality. Bicubic is FFmpeg's YUV scale path; neural models
use the documented RGB/luminance path. This compares complete delivered pipelines,
not only isolated network architecture. Results include codec artifacts.

The temporal diagnostic is mean absolute consecutive change in reconstruction
residual at 256×144, divided by 255. It is not a standard perceptual metric, is
not motion compensated, and can reward smoothing. Scene motion/cuts affect it.
Inspect flicker visually too. `--vmaf` requires the FFmpeg libvmaf filter and uses
the explicit built-in `vmaf_v0.6.1` model, two worker threads and the same sampling
stride. JSON records the actual library version, sample count and measured mean.
The validated native ARM64 build runs it; other builds may not include the filter.
Four-second results cannot establish sustained real-time
behavior; v0.4 requires at least sixty seconds of measured useful processing.


## Sustained video (v0.4)

See [realtime reproduction](realtime.md). Three long trials use the median of
complete-video trial FPS, retain every result, and require >=60 seconds wall
processing each. Processing includes decoder launch through encoder flush;
provider startup/proof, final output inspection and hashes are separate. A
120-second source does not imply a 120-second benchmark when faster than real time.
No frame pacing, skipping or model bypass is used.

Headline measurements record AC line, Balanced scheme and energy saver off.
First/final and minimum rolling ten-second rates detect a throughput drop without
claiming temperature or power measurement. One-second process CPU/working-set
samples cover Python and its codec children; queue/sample storage is bounded.
Completion intervals and read-to-encoder-submission latency are reported with
their limited scope, independently of end-to-end FPS.

Delivered video quality compares the same codec/rate with FFV1 HR references.
Native-Y and RGB paths are explicitly distinguished. VMAF now assigns CFR frame
ordinals on a shared AVTB clock, disables repeat-last frames and checks the
expected sample count. Historical v0.3 VMAF used its original framesync setup;
v0.4 recomputes all baselines, rather than mixing those protocols. PSNR/SSIM
already compare frame ordinals. Different metrics/content can disagree.

The realtime network also passes a paired five-image quality regression gate:
mean luminance PSNR and SSIM must exceed bicubic, with strict QNN proof. This
isolated image-model check does not promise universal delivered-video gains.
Performance thresholds apply only to the declared local hardware/workload;
generic CI tests gate logic and correctness, without tight timing thresholds.

## Expanded quality protocol (v1 development)

The [predeclared corpus](../benchmarks/corpus-v1.json) contains 24 clips, split
equally into development and holdout. Model/fusion selection uses development
only; lock settings before holdout evaluation. Download/preparation and delivered
quality scripts are `download_quality_corpus.py` and `run_quality_corpus.py`.
Media lives in the user cache, not git. Prepared provenance records source and
fixture hashes, source time ranges, native active regions, degradation and tools.

Native decoded 8-bit coded Y, peak 255, is evaluated directly. PSNR is calculated
from mean sampled MSE per clip (every third frame), including perfect frames.
SSIM uses an 11x11 Gaussian with sigma 1.5 and population covariance. Both shave
two pixels from the declared active region; VMAF evaluates its full declared
region on every frame. Report primary VMAF 0.6.1 and secondary NEG 0.6.1, which
limits enhancement gain. Identical codec/rate settings apply to every method.
This native-Y protocol differs from v0.4's RGB/Rec.601 conversion; absolute scores
across the two protocols must not be presented as model improvements.

`summarize_quality_corpus.py` rejects missing, duplicate or mismatched pairs.
It reports equal per-clip averages and source/category/split groups, retaining
negative results. Positive average PSNR, SSIM and VMAF are required for both
all clips and holdout. A numeric pass does not replace artifact/temporal review
or sustained performance validation. The temporal residual-change diagnostic is
project-specific, does not compensate for motion and may reward smoothing;
it is not a standard perceptual quality score.
