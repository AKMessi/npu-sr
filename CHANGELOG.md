# Changelog

## Unreleased

- Experimental original two-frame residual trained on real licensed sequences;
  fuse spatial and temporal enhancement into one static graph with full halo-7
  context. The published spatial realtime default remains unchanged.
- Bounded previous-frame planes, tested first-frame and scene-cut/black resets,
  direct shared tile submission, and temporal frame/state accounting.
- Reproducible preparation, training and export scripts; original MIT correction
  checkpoints remain outside the package, separate from acquired BSD spatial weights.
- Temporal candidates did not clear development quality gates; retain all paired
  metrics and keep the spatial default. No v0.7 release or temporal quality claim.

## 0.6.0 — 2026-10-04

- Validate every video packet's presentation cadence with bounded reordering;
  compare counts with actual processed frames and container metadata.
- Keep decoded-frame forensic audits behind `--verify-full`; avoid double decode.
- Reduce measured 120-second-source function time from 149–156 seconds to
  76–83 seconds. Default output validation takes about 0.2 seconds.
- Profile chroma, quantization, fusion/finite checks and output copy separately.
  CPU postprocessing remains below 8 ms median with the unchanged v0.5 model.
- Preserve strict QNN, codec proof, every-frame accounting and atomic publication.
  Three delivered-quality regressions have identical PSNR/SSIM/VMAF to v0.5.

## 0.5.0 — 2026-10-04

- Reproducible BSD-licensed QuickSRNet small/medium neutral-Y exports. Realtime
  now selects unblended small after frozen development/holdout validation.
- Predeclared 24-clip development/holdout video corpus, native-Y paired quality
  evaluation, secondary VMAF NEG and complete-pair acceptance summaries.
- Preserve color range/matrix on raw NV12 encoder input as well as output,
  avoiding unintended color conversion when source BT709 metadata is known.
- All 24 declared clips: average +2.00 dB PSNR, +0.00912 SSIM and +5.95 VMAF
  versus bicubic; holdout alone +1.81 dB / +0.00869 / +5.47. Secondary NEG also wins.
- Three 120-second sustained sources: about 52.6 FPS median, strict QNN,
  D3D11VA and QCOM AV1, no application drops, 43,200 neural calls per trial.
- Median postprocessing about 7 ms without the old quality-compromise blend.
  Whole-command time still includes an expensive full-file audit; not yet v1.

## 0.4.0 — 2026-10-04

- Primary 960×540 → 1920×1080 at 30 FPS sustained 34.3 FPS across three 120-second
  sources, with strict QNN and hardware AV1 encoding; post-encode audit is timed separately.

- Persistent native NV12 luminance enhancement with reusable buffers and declared fixed neural/bicubic blending.
- Bounded ordered decoder/enhancer/encoder overlap, with cancellation and output validation preserved.
- Transparent quality/balanced/realtime presets, measured QNN performance modes and mode-aware context caches.
- Three-trial sustained acceptance gates, rolling rates, scoped frame handoff latency and process CPU/memory samples.
- Paired image quality regression gate, delivered preset quality/speed suite and corrected VMAF clock alignment.
- Native software AV1 selection, strict planar hardware/cache/cancellation tests and realtime reproduction docs.
- Retained monthly FFmpeg pin with startup/integrity cache checks; broken September ARM64 builds rejected.
- Actual successful neural tile-call counting and separately timed post-encode safety audits.
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
