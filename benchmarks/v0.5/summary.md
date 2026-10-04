# v0.5: quality-first realtime selection

## Scope and selection

Measured on Snapdragon X Plus X1P-42-100, Windows 11 ARM64 build 26200,
QNN catalog 2.2480.53.0, driver 30.0.219.1000, ORT 1.25.2, native ARM64 FFmpeg.
Power is **not measured**. Sustained trials ran on battery (38–40%), Balanced,
energy saver off; background processes were uncontrolled. No concurrent npu-sr
benchmark ran during sustained trials. These are one-machine results.

The corpus was declared before scoring: 24 clips, five pinned legal video sources
plus original synthetic text/texture tests, 12 development and 12 holdout clips.
Every clip has 60 frames. It covers faces/skin, animation, city scenes, foliage,
motion, pans, night detail, text and compression stress. Sources are acquired,
not redistributed. Hashes, crop/time, range/matrix and degradation settings are
in [corpus-v1.json](../corpus-v1.json) and [quality.json](quality.json).
References retain native detail: 1920×800 film is padded, not enlarged; metric
ROIs exclude padding. NASA 4240×2832 footage is downscaled before cropping.

Selection used development data only, then froze model hash and settings in
[selection-lock.json](selection-lock.json). Holdout was evaluated with those
settings. Equal per-clip averages include synthetic and compressed stress cases;
no negative case was removed. Holdout is now consumed and must not be described
as untouched evidence for future tuning.

## Delivered quality

Same QCOM AV1 hardware encode, 8 Mbit/s, limited-range BT709. Metrics use native
coded 8-bit Y, not the v0.4 RGB/Rec.601 protocol. Absolute cross-release scores
are not comparable. PSNR uses mean sampled MSE; SSIM uses an 11×11 Gaussian
window; both sample every third frame. VMAF 0.6.1 and NEG measure every frame.
Temporal difference error is a reference-residual diagnostic without motion
compensation, not a standard perceptual metric; it may reward smoothing.

| All 24 clips | PSNR dB | SSIM | VMAF | VMAF NEG |
| --- | ---: | ---: | ---: | ---: |
| Bicubic | 37.8653 | 0.961910 | 85.6160 | 81.9471 |
| QuickSRNet Small Y | 39.8693 | 0.971032 | 91.5656 | 87.7279 |
| Difference | 2.0040 | 0.009123 | 5.9496 | 5.7808 |

Holdout alone improves PSNR by 1.8132 dB, SSIM by 0.008691 and VMAF by 5.4721.
The temporal diagnostic increases 2.17% overall; the largest per-clip ratio is
1.064. Static and six-consecutive-frame crops were inspected for faces, text,
animation, foliage and night detail. This limited contact-sheet inspection is not a full playback audit. The small
48-second corpus is not proof
of universal quality or absence of flicker. Per-frame models still lack temporal
context. VMAF saturation on a synthetic pattern is reported, not excluded.

### Every paired clip

| Clip | Bicubic PSNR | Small PSNR | Bicubic SSIM | Small SSIM | Bicubic VMAF | Small VMAF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conversation | 37.541 | 39.624 | 0.96182 | 0.96785 | 85.794 | 92.112 |
| dark-city | 42.806 | 44.801 | 0.98568 | 0.98894 | 87.623 | 93.717 |
| action | 35.775 | 37.648 | 0.95456 | 0.96457 | 92.212 | 96.169 |
| animated-landscape | 33.939 | 36.689 | 0.92795 | 0.95766 | 83.475 | 91.249 |
| animated-leaves | 41.493 | 43.500 | 0.98098 | 0.98543 | 86.847 | 93.336 |
| animated-face | 33.481 | 34.825 | 0.91831 | 0.93848 | 82.321 | 90.930 |
| snow-pan | 44.816 | 47.124 | 0.98843 | 0.99126 | 86.156 | 91.841 |
| aerial-landscape | 39.027 | 39.686 | 0.96182 | 0.96706 | 92.670 | 95.690 |
| starfield | 35.799 | 38.741 | 0.96540 | 0.97862 | 78.532 | 88.759 |
| ui-horizontal | 30.979 | 36.850 | 0.98340 | 0.99578 | 79.714 | 94.733 |
| moving-texture | 35.557 | 37.767 | 0.98217 | 0.98881 | 100.000 | 100.000 |
| compressed-faces | 34.108 | 34.404 | 0.93084 | 0.93158 | 64.238 | 68.172 |
| street-foliage | 40.664 | 41.488 | 0.97375 | 0.97668 | 99.552 | 99.855 |
| city-pan | 43.500 | 44.778 | 0.98512 | 0.98718 | 96.170 | 97.843 |
| skin-closeup | 45.501 | 45.739 | 0.98420 | 0.98444 | 93.143 | 95.266 |
| animated-fur | 37.933 | 40.700 | 0.96396 | 0.97710 | 86.215 | 93.550 |
| animated-motion | 37.365 | 38.957 | 0.95250 | 0.96351 | 95.107 | 99.540 |
| animated-forest-pan | 35.039 | 36.601 | 0.93311 | 0.95134 | 85.013 | 91.593 |
| animated-skin | 43.190 | 44.835 | 0.98103 | 0.98393 | 93.080 | 96.808 |
| aerial-foliage | 43.017 | 44.000 | 0.97647 | 0.98018 | 89.944 | 93.653 |
| ground-forest | 33.392 | 35.230 | 0.91921 | 0.94396 | 81.932 | 89.983 |
| night-silhouettes | 36.284 | 39.223 | 0.96651 | 0.97922 | 77.832 | 87.871 |
| ui-vertical | 31.065 | 37.108 | 0.98374 | 0.99610 | 78.609 | 93.934 |
| compressed-foliage | 36.495 | 36.545 | 0.92485 | 0.92511 | 58.604 | 60.970 |

### Alternatives and rejected approaches

Development comparisons retain ESPCN unblended and its published 0.5 blend,
FSRCNN, QuickSRNet Medium and small strengths 0.5/0.75. Medium reaches 91.82
mean VMAF versus Small 91.39 on development, but roughly doubles neural cost.
It remains downloadable, not the realtime default. Small strengths 0.5 and 0.75
lose all three mean spatial scores versus 1, so no blend is selected. Full RGB
small/medium graphs proved strict QNN but cost more than the luminance adaptation.
Adaptive/coarse-frequency fusion did not justify complexity in early probes;
those untagged probes are not release evidence.

An initial animation PSNR loss was isolated to missing raw-input BT709 metadata,
not neural brightness bias. Hardware/software decoded NV12 bytes matched. Tagging
both raw encoder input and output changed first-frame delivered neural MSE from
15.56 to 3.14 and mean coded-Y bias from +2.43 to +0.05. All release comparisons
were rerun after the fix. Do not compare untagged experiments with tagged release
results or credit the entire gain to a new architecture.

## Sustained performance

The locked Small Y model uses 256-pixel cores, halo 4, 12 neural calls per frame,
QNN HTP FP16 math, burst mode, unblended NV12, depth-two bounded queues,
D3D11VA decode and QCOM hardware AV1. Three independent 120-second sources:

| Trial | Streaming FPS | Min rolling 10s FPS | Neural median ms | Post median ms | Processing s | Whole command s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52.632 | 51.70 | 8.834 | 7.067 | 68.40 | 155.58 |
| 2 | 52.658 | 51.50 | 8.956 | 7.004 | 68.37 | 149.35 |
| 3 | 52.145 | 50.40 | 9.050 | 7.086 | 69.04 | 151.00 |

Median of trial throughput is 52.63 FPS. Each delivers 3,600 frames and executes
43,200 neural tiles, with zero application drops; strict profiling rejects CPU
kernels and fallback. Audio, delivered frame counts and CFR timestamps passed.
Inference alone is not video throughput. Queued decoder-read-to-encoder-submit
latency is about 76 ms, distinct from approximately 19 ms worker frame processing
and from encoded/display latency. Proof/warmup and full-file audit are separate.

Postprocessing falls from the reproduced v0.4 median 16.75 ms to about 7.07 ms:
removing the blend was selected for quality and also removes its CPU cost.
This compares explicit different models/settings, not an isolated microbenchmark.
No native extension is warranted for the <=8 ms target. The forensic final audit
still costs roughly 69 seconds; total command runtime is **not realtime**. v0.6
will preserve an explicit full audit while reducing normal finalization overhead.
No power, temperature or hardware engine utilization was measured.

## Reproduction

```powershell
npu-sr models download quicksrnet-small-y-x2
npu-sr models download quicksrnet-medium-y-x2
python scripts/download_ffmpeg.py
python scripts/download_quality_corpus.py --split development
python scripts/run_quality_corpus.py --json outputs/development.json
python scripts/download_quality_corpus.py --split holdout
python scripts/run_quality_corpus.py --split holdout --json outputs/holdout.json
python scripts/summarize_quality_corpus.py outputs/development.json outputs/holdout.json --model quicksrnet-small-y-x2 --json outputs/paired.json
python scripts/download_video_benchmarks.py --sustained --duration 120
python scripts/run_selected_sustained.py --directory <prepared-directory> --output-directory <trial-directory> --json outputs/sustained.json
```

Preparation and output files go outside git. Consult each script's `--help` for
explicit directories. Output hashes may differ across FFmpeg versions; recorded
source/model hashes are mandatory. JSON records actual measurement code/version,
including development 0.4.0 metadata and source hashes before the 0.5 version bump;
these fields are not relabeled to pretend they were measured at a later commit.
The release changes the preset/version after those locked-setting measurements.

[Environment](environment.json) · [Performance](performance.json) ·
[Quality and execution evidence](quality.json)

## Release checks

153 ordinary tests passed. The separately selected hardware group passed 19
tests (QNN and DirectML); four hardware video tests passed. Lint, formatting,
build, metadata, public artifact contents and fresh ARM64 wheel installation
passed. Fresh-install strict NPU image inference also passed. The local dependency
audit found no known advisories across 121 packages after updating pip to 26.2.1;
application runtime versions were unchanged. The publication scan reviewed 132
tracked files without detected private paths, credentials or forbidden artifacts.
Generic CI does not verify Qualcomm hardware.
