# v0.3 measured video release

Measured on Snapdragon X Plus X1P-42-100, Windows ARM64 build 26200. Native FFmpeg n9.0.2-22-g46d8f462ee, D3D11VA decode and activated QCOM hardware Media Foundation encoders. Strict neural profiles contain QNN kernels and no CPU kernels. Power, temperatures and accelerator utilization are not measured.

## Complete-video performance

Each row is the median of three complete 120-frame, four-second, 30 FPS face-clip trials. ESPCN-256, H.264 hardware encoding at target 8M with explicit unconstrained VBR / camera_record. Timing includes codec startup, transfers, preprocessing, inference, reconstruction and encoder flush; neural/provider startup and final output validation are separate. All outputs preserve 120 frames, 30 FPS, 2x size, expected duration and audio.

| Input → output | CPU FPS | NPU FPS | NPU real-time factor | Battery condition |
| --- | ---: | ---: | ---: | --- |
| 640×360 → 1280×720 | 12.33 | 21.77 | 1.38 | 22%, before saver |
| 960×540 → 1920×1080 | 5.51 | 5.59 | 5.36 | 21–20%, saver activated mid NPU trials |
| 1280×720 → 2560×1440 | 1.68 | 2.83 | 10.59 | 19–18%, saver active |

These battery conditions are not a controlled cross-resolution speed comparison. Balanced scheme stayed selected, but Windows reported energy saver after crossing 20%; it was not sampled per frame. The individual 540p NPU trials were 9.08, 5.45 and 5.59 FPS. All trials are retained; no fastest-trial headline. **No sustained real-time claim.**

The 540p hot path is dominated by validated neural calls and CPU RGB reconstruction. v0.4 must optimize those phases and demonstrate sustained end-to-end throughput. Standalone codec tests include CPU transfer/conversion and cannot be added together as a pipeline decomposition. Python peak working set is cumulative for the benchmark process and excludes codec children.

## Delivered-output quality

Four aligned research sequences: face conversation, outdoor scene with a cut, motion/action, and a generated fine-texture test pattern. Film attribution: Blender Foundation / mango.blender.org, Tears of Steel, CC BY 3.0. Source and prepared clips are acquired outside git; hashes, retiming/crop/degradation and model hashes are in JSON. Outputs use AV1 hardware encoding, 8M VBR target, camera_record. References are aligned FFV1 1080p. Every 12th frame: Rec.601 Y PSNR/SSIM with shave2 and native libvmaf vmaf_v0.6.1. Temporal residual diagnostic examines every frame; it is unregistered and not motion compensated.

| Clip | Pipeline | PSNR Y (dB) | SSIM Y | VMAF |
| --- | --- | ---: | ---: | ---: |
| faces | bicubic | 39.204 | 0.9669 | 89.96 |
| faces | espcn-x2-256 | 39.569 | 0.9671 | 90.74 |
| faces | fsrcnn-x2 | 39.528 | 0.9672 | 91.69 |
| scene | bicubic | 43.785 | 0.9813 | 94.89 |
| scene | espcn-x2-256 | 43.059 | 0.9807 | 93.03 |
| scene | fsrcnn-x2 | 43.136 | 0.9808 | 94.50 |
| motion | bicubic | 45.894 | 0.9864 | 96.79 |
| motion | espcn-x2-256 | 45.550 | 0.9861 | 96.55 |
| motion | fsrcnn-x2 | 45.100 | 0.9857 | 96.95 |
| texture | bicubic | 35.637 | 0.9908 | 85.67 |
| texture | espcn-x2-256 | 36.471 | 0.9761 | 78.36 |
| texture | fsrcnn-x2 | 36.803 | 0.9880 | 79.19 |

Neural enhancement is content dependent: it improves the face clip, but can hurt the outdoor scene, fine pattern SSIM/VMAF, or motion scores. This is a comparison of delivered pipelines: FFmpeg bicubic scales YUV directly; image SR reconstructs RGB. Compression and colorspace conversion contribute. These are not canonical model-only benchmark scores. Per-frame networks can ring or flicker.

## Encoder failure investigated

Default H.264 rate control returned exit code zero but lost two frames at a cut. Explicit u_vbr preserved all 120; camera_record is also set to request CFR. The application rejects invalid counts or timestamp cadence before publishing. The regression JSON records the actual scene-cut application run. Default AV1 settings also harmed delivered quality; all final quality rows use explicit rate control. No internal driver root cause is claimed.

## Reproduce

```powershell
python scripts/download_ffmpeg.py
python scripts/download_video_benchmarks.py
python scripts/run_video_benchmarks.py --trials 3
python scripts/run_video_benchmarks.py --quality
```

Use stable power conditions for a new comparison. The historical release was run on battery as disclosed. JSON identifies measured clean commit bd389bb and source hash; subsequent documentation/energy-saver reporting changes do not alter the enhancement or codec settings. Image regression uses three trials, five runs and three warmups at 256×160 across CPU/DirectML/NPU for ESPCN baseline/larger tiles, FSRCNN and DnCNN; it is not video throughput. v0.2 quality model data remains applicable because the models and image algorithms are unchanged.

Validation: 85 ordinary tests, 18 local NPU/video hardware tests, Ruff, wheel/sdist build, metadata and fresh wheel installation. Generic CI runs real software video integration without an NPU.
