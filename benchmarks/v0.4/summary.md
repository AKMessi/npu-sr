# v0.4 validation — Snapdragon X Plus X1P-42-100

**YES: the declared 960×540 → 1920×1080 neural/video pipeline sustained 30 FPS.**
All three trials processed a 120-second, 30 FPS source without skipped/duplicated application frames.
Each completed 3,600 frames and 43,200 neural tile calls, with matching delivered frame counts, cadence and audio.

Headline: **34.2 FPS**, real-time factor **0.878** (processing/source duration).
Actual hardware decode/download, preprocessing, every neural tile, stitching, postprocessing, encode and flush are included.
Provider/input inspection startup and the final full-file safety audit are separate. This is a file-processing pipeline, not a live player.

## Reproduce and configuration

See [realtime instructions](../../docs/realtime.md). Source SHA and preparation are in the JSON provenance.
Pinned native FFmpeg n9.0.1-11-ge47273f4d9, ESPCN x2 256-core/halo4, 264×264 graph, FP32 ONNX/HTP FP16 math,
12 tiles/frame, requested QNN burst mode, native limited-range NV12, **fixed 0.5 neural strength**, two-frame queues,
AV1 hardware encoding at 8M unconstrained VBR / camera_record, three tensor warmups.
Every frame still receives all neural calls; the fixed CPU Catmull–Rom blend is disclosed and not adaptive.
AC power, Balanced scheme, energy saver off; background processes uncontrolled.

## Sustained trials

| Trial | Processing s | Actual FPS | Min rolling 10s FPS | First/final 10s FPS | Handoff p50/p95 ms | Sampled peak MB |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 105.3 | 34.17 | 33.8 | 33.8 / 34.0 | 116.9 / 119.1 | 361.6 |
| 2 | 105.0 | 34.30 | 33.8 | 33.8 / 34.0 | 116.4 / 118.2 | 364.7 |
| 3 | 105.5 | 34.11 | 33.7 | 33.7 / 34.1 | 117.1 / 119.7 | 374.0 |

Frame handoff latency is decoder read start → encoder pipe submission, including queues. It is **not** encoder/display latency.
Minimum rolling rates use submissions on one-second boundaries; complete processing FPS includes actual encoder flush.
Queues peaked at two frames. OS samples cover Python and codec children, not drivers or all machine memory.

| Trial | Startup/input inspection s | Post-encode audit s | Total function elapsed s | Process CPU share of 8 logical CPUs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12.7 | 68.4 | 189.6 | 14.0% |
| 2 | 10.0 | 68.5 | 186.6 | 14.0% |
| 3 | 9.9 | 68.4 | 187.1 | 14.0% |

**The whole CLI takes longer than the source duration because it decodes the finished file again to audit counts and timestamps.**
The realtime claim covers the sustained decode-to-encode pipeline, excluding that post-encode audit. Both times are reported.
Temperature, accelerator utilization and power are **not measured**. Stable throughput is not a thermal or energy measurement.

## Execution evidence

All trials selected NPU hardware / QNN HTP; strict proof found QNN kernels and no CPU kernels, with CPU fallback disabled.
Decode selected D3D11 hardware frames with explicit download. Encode forced hardware-only Media Foundation enumeration
and activated QCOM Hardware Encoder - AV1; the resulting files passed actual full decode count/cadence/audio checks.
These are reasonable Qualcomm hardware codec evidence, not an internal VPU utilization measurement.

## Quality and speed

Delivered 8M AV1 output compared with aligned FFV1 HR references: three CC BY film clips plus a synthetic texture sequence.
PSNR/SSIM use Rec.601 Y, two-HR-pixel shave and every twelfth frame. VMAF v0.6.1 uses corrected ordinal/common clocks.
RGB and native-Y paths, interpolation kernels and codec loss differ; these compare delivered pipelines, not isolated neural gain.

| Preset / pipeline | Mean Y PSNR dB | Mean Y SSIM | Mean VMAF | 4s clip median FPS (3 trials) |
| --- | ---: | ---: | ---: | ---: |
| bicubic | 41.130 | 0.981344 | 91.59 | 105.4 |
| realtime | 41.217 | 0.981538 | 90.17 | 32.6 |
| balanced | 41.162 | 0.977520 | 89.35 | 11.9 |
| quality | 41.142 | 0.980404 | 90.34 | 9.1 |

The realtime blend modestly improves aggregate PSNR/SSIM, while **bicubic wins average VMAF**. Outdoor scene/detail can regress.
Faces improve; synthetic texture can ring. No universal perceptual improvement is claimed. Quality preset names reflect image scores.
The short speed table includes codec startup and is not the long sustained gate; bicubic uses efficient FFmpeg CPU scaling with the same hardware codecs.

### Standard image-model regression

Five BSDS300 test images, declared Pillow bicubic degradation, two-pixel shave; not canonical MATLAB BSD100 scores.
| Model / strict NPU | Mean PSNR dB | Mean SSIM |
| --- | ---: | ---: |
| Bicubic | 27.947 | 0.839886 |
| espcn-x2-256 | 28.314 | 0.846285 |
| fsrcnn-x2 | 28.479 | 0.851679 |

The unblended ESPCN network passes mean PSNR and SSIM gains over bicubic with strict QNN evidence. CPU/GPU paired results are included.
This image-model gate does not imply universal delivered-video gains.

## Profiling and rejected alternatives

- Original RGB 540p path: median postprocessing about 58.5 ms; native unblended NV12 reduced it to about 7.5 ms in short trials.
- QNN 256-core inference: about 34.9 ms default → 12.4 ms sustained → 10.2 ms burst in three-trial compute experiments.
- 512-core and whole-frame graphs were slower (about 13.8 and 30.4 ms sustained); CPU/NPU agreement and strict proof passed.
- Queue depth 2 improved short unblended NV12 trials from about 39.7 to 41.7 FPS; depth 4 brought little benefit.
- Unblended long runs reached about 45 FPS but lost delivered SSIM. Fixed blending recovers aggregate PSNR/SSIM at a clear compute cost.
- Float bicubic buffers reduced its CPU microbenchmark from about 10.6 to 6.9 ms. Final trials remeasure the full selected path.
- September retained ARM64 FFmpeg builds crashed at startup. August monthly works; daily October pins have limited retention.
- Remaining bottleneck: CPU blend/chroma/postprocessing, followed by NPU inference. No native extension or quantization was required.

Exploratory numbers describe declared earlier configurations; final performance/quality files above are the release evidence.
All six prior model hashes and image/denoising behavior are preserved. See the [v0.2 matrix](../v0.2/summary.md) for CPU/GPU/NPU image results.

## Validation and limits

- 115 ordinary tests and 19 local hardware tests (15 NPU, 4 video hardware) passed; generic CI excludes hardware.
- Strict planar cache reuse, all three hardware encoders, temporal ordering, cancellation and neural-call counting passed.
- Publication requires package build, metadata, fresh installation and main/tag CI; outcomes are recorded in release notes.
- One laptop/runtime/configuration and a small content suite; not a general Snapdragon performance or quality claim.
- No HDR, variable framerate, live preview/player, zero-copy or temporal neural model. No energy-efficiency conclusion.
- Post-encode full-file audit adds substantial elapsed time; source FPS is preserved and every neural frame is processed.
- No weights, datasets, codec/runtime binaries or private machine identifiers are published.

## Raw evidence

- [Environment](environment.json)
- [Sustained performance and provenance](realtime-performance.json)
- [Resource/sustained samples](thermal-or-sustained.json)
- [Delivered and image quality / preset trials](quality.json)
- [Hardware-codec bicubic timing](bicubic-performance.json)
