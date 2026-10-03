# Changelog

## 0.2.0 — 2026-10-04

- Static model registry, pinned FSRCNN/FSRCNN-small/LapSRN exports and learned DnCNN denoising.
- Strict DirectML GPU execution and CPU/NPU/GPU image comparisons.
- Shape-aware receptive-field tiles, reused contiguous input buffers and cheaper RGB quantization.
- Reusable runtimes, optional integrity-checked QNN contexts, separate proof/startup timings.
- Reproducible five-image BSDS300 research subset acquisition, PSNR/SSIM quality evaluation.
- Multi-resolution, three-trial performance suite with phase timings and JSON checkpointing.
- CPU process time and working-set measurements; power and accelerator utilization not measured.
- Existing v0.1 command syntax and exact baseline export retained. No video implementation yet.

## 0.1.0 — 2026-10-04

- Single-image 2× ESPCN super-resolution with a reproducible Apache-2.0 model export.
- Windows ML bootstrap, catalog acquisition and Python ORT QNN library registration.
- Explicit NPU hardware selection, disabled CPU fallback and profiled proof inference.
- CPU/auto modes, diagnostics, warm benchmarks with JSON, and RGB PSNR evaluation.
- Fixed-shape tiles with a receptive-field halo, alpha handling and EXIF orientation.
- CPU unit/integration tests, separate manual NPU test, CI, MIT license and notices.
- Validated locally on Snapdragon X Plus X1P-42-100, Windows build 26200.
