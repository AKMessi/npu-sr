# Changelog

## 0.4.0 — 2026-10-04

- Persistent native NV12 luminance enhancement with reusable buffers and declared fixed neural/bicubic blending.
- Bounded ordered decoder/enhancer/encoder overlap, with cancellation and output validation preserved.
- Transparent quality/balanced/realtime presets, measured QNN performance modes and mode-aware context caches.
- Three-trial sustained acceptance gates, rolling rates, scoped frame handoff latency and process CPU/memory samples.
- Paired image quality regression gate, delivered preset quality/speed suite and corrected VMAF clock alignment.
- Native software AV1 selection, strict planar hardware/cache/cancellation tests and realtime reproduction docs.
- Power, accelerator utilization and temperature are not measured.

## 0.3.0 — 2026-10-04

- In-memory ordered video streaming with one persistent strict enhancement session.
- Native ARM64 FFmpeg acquisition with a pinned archive hash and separate licensing.
- Executed D3D11VA decoding and QCOM Media Foundation H.264/HEVC/AV1 encoding evidence.
- Explicit strict/auto/software codec modes, audio copy, atomic output validation and cancellation cleanup.
- Actual CFR timestamp validation, frame-count/framerate/duration/resolution checks and bounded diagnostics.
- Complete-video trials, aligned delivered-video PSNR/SSIM and a documented temporal residual diagnostic.
- Ordinary CPU video integration and separate Snapdragon video hardware tests.
- Explicit Media Foundation unconstrained VBR / camera_record avoids observed scene-cut frame loss;
  delivered AV1 quality, native VMAF, three-trial video data and image regression are published.
- Battery saver activation is disclosed with individual trials; no energy claims.
- No sustained real-time claim; that remains the v0.4 acceptance gate.

## 0.2.0 — 2026-10-04

- Static model registry, pinned FSRCNN/FSRCNN-small/LapSRN exports and learned DnCNN denoising.
- Strict DirectML GPU execution and CPU/NPU/GPU image comparisons.
- Shape-aware receptive-field tiles, reused contiguous input buffers and cheaper RGB quantization.
- Reusable runtimes, optional integrity-checked QNN contexts, separate proof/startup timings.
- Reproducible BSDS300 SR and author-designated BSD68 denoising subsets, PSNR/SSIM evaluation.
- Multi-resolution, three-trial performance suite with phase timings and JSON checkpointing.
- CPU process time and working-set measurements; power and accelerator utilization not measured.
- Existing v0.1 command syntax and exact baseline export retained. No video implementation yet.
- Published CPU/NPU/DirectML measurements for five resolutions and three trials;
  FSRCNN improves the tested SR subset and DnCNN improves synthetic luminance noise.

## 0.1.0 — 2026-10-04

- Single-image 2× ESPCN super-resolution with a reproducible Apache-2.0 model export.
- Windows ML bootstrap, catalog acquisition and Python ORT QNN library registration.
- Explicit NPU hardware selection, disabled CPU fallback and profiled proof inference.
- CPU/auto modes, diagnostics, warm benchmarks with JSON, and RGB PSNR evaluation.
- Fixed-shape tiles with a receptive-field halo, alpha handling and EXIF orientation.
- CPU unit/integration tests, separate manual NPU test, CI, MIT license and notices.
- Validated locally on Snapdragon X Plus X1P-42-100, Windows build 26200.
