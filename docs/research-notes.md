# Research notes

## v0.2 â€” checked 2026-10-04

- [Microsoft's Python catalog flow](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers)
  still requires explicit library registration in Python's ORT environment. The
  working v0.1 bootstrap and catalog implementation is retained.
- [Current QNN provider documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
  supports static floating-point HTP graphs; QAIRT 2.35+ uses FP16 math on supported
  chips. The older main ORT page still contains a conflicting quantization-only
  statement. Local strict execution determines compatibility here.
- Conv, PRelu, Relu, Add, Sub and DepthToSpace are suitable candidates, but an
  operator listing does not prove a particular graph works. FSRCNN, LapSRN and
  DnCNN exports passed actual strict QNN profiles on this laptop.
- LapSRN's LeakyRelu export was rejected by the installed QNN package. Equivalent
  channel slopes through PRelu work. Even 4x4 TF SAME kernels require asymmetric
  padding; an independent TF graph interpreter tests this conversion.
- QNN GPU graphs compiled but execution failed with code 6999 for ESPCN,
  FSRCNN-small and DnCNN. No successful QNN GPU claim is made.
  [DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
  succeeds with sequential execution and memory patterns disabled. Strict
  profiles establish DirectML kernel execution, with no CPU kernels.
- Embedded QNN EPContext files work with `ep.context_enable=1`,
  `ep.context_embed_mode=1` and an explicit local path. Each loaded context still
  receives a proof run. Cache keys include model, ORT, catalog package, processor
  class, architecture and NPU driver version. Compiled files are never shipped.
- [FSRCNN upstream](https://github.com/Saafke/FSRCNN_Tensorflow) and
  [LapSRN upstream](https://github.com/fannymonori/TF-LapSRN) provide Apache 2.0
  weights. FSRCNN uses a subpixel layer, unlike the paper's transposed convolution.
  [KAIR](https://github.com/cszn/KAIR) provides MIT DnCNN weights with batch
  normalization already folded. Acquisition uses pinned source hashes; no
  TensorFlow/PyTorch runtime dependency is added.
- [Berkeley BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
  explicitly permits non-commercial research and educational downloads. The
  first five sorted test images provide a fixed natural-image research subset.
  Dataset images are not redistributed. Scores use disclosed Pillow bicubic
  degradation and must not be described as canonical MATLAB BSD100 scores.
- Windows process CPU time and working set are available through OS APIs. Peak
  working set is cumulative for the benchmark process. Accelerator utilization,
  temperature and power are not measured. TOPS does not imply watts or throughput.

Video API research and implementation belong to the subsequent gated releases.
## v0.3 video IO investigation (2026-10-04)

- [FFmpeg documentation](https://ffmpeg.org/ffmpeg-all.html) documents
  `h264_mf`/`hevc_mf` with `-hw_encoding 1`. Its
  [transform selection source](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/mf_utils.c)
  enumerates with `MFT_ENUM_FLAG_HARDWARE` when that flag is requested;
  successful encoding and the selected MFT name are stronger evidence than an encoder listing.
- Microsoft's [hardware MFT](https://learn.microsoft.com/en-us/windows/win32/medfound/hardware-mfts)
  and [MFTEnumEx](https://learn.microsoft.com/en-us/windows/win32/api/mfapi/nf-mfapi-mftenumex)
  documentation explain hardware codec proxies. This does not measure the internal
  silicon's power or utilization.
- D3D11VA decode should retain hardware surfaces until an explicit `hwdownload`.
  Requiring that filter plus successful frame output prevents a software pixel-format
  fallback from being mislabeled as hardware decode.
- Existing FFmpeg on this laptop is x64 (PE machine 0x8664). Prefer native ARM64:
  [FFmpeg's download page](https://ffmpeg.org/download.html) links BtbN builds;
  [platform documentation](https://ffmpeg.org/platform.html) supports native ARM64
  toolchains, while ARM64EC is explicitly unsupported.
- Investigation build: BtbN `autobuild-2026-10-03-18-14`, native ARM64 GPL shared
  FFmpeg `n9.0.2-22-g46d8f462ee`; archive SHA256
  `82b7eef78a79fdc93a2835154e753b4b175797712f707cc4f419405c0e9f6a2c`.
  It is downloaded outside git, with its notices and licenses. No binaries are redistributed.
  Actual tests completed H.264, HEVC and AV1 encoding with activated
  `QCOM Hardware Encoder` transforms. D3D11VA H.264 decode completed with
  selected `d3d11` frames and explicit download. The driver rejected a 64Ã—48
  fixture, while 640Ã—360 worked; strict mode rejected that failure.
- Native libvmaf executed using the explicit `vmaf_v0.6.1` built-in model; a
  same-reference sanity run measured 99.39267 over ten sampled frames.
  [Netflix's integration documentation](https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md)
  and [FFmpeg's filter source](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_libvmaf.c)
  define distorted/reference order, sampling and model selection. Full release
  quality numbers come from actual delivered outputs, not the sanity score.
- Default H.264-MF rate control lost two display frames at a scene cut (118
  instead of 120), despite exit code zero. The application rejected the output.
  Explicit unconstrained VBR preserved all 120; increasing the target to 40M
  also worked, while `archive` scenario alone did not. Default AV1 rate control
  measured 34.18 dB on the face clip; unconstrained VBR measured 39.23 dB on a
  two-frame diagnostic sample (different sample counts, not headline scores).
  The [current FFmpeg encoder source](https://github.com/FFmpeg/FFmpeg/blob/n9.0.2/libavcodec/mfenc.c)
  documents Qualcomm frame dropping and `camera_record` for CFR. The project
  uses explicit `u_vbr` / `camera_record`, then validates encoded counts and
  actual timestamp cadence. Final quality measurements use the explicit settings.
- Balanced power scheme alone does not describe Windows energy saver. The
  [SYSTEM_POWER_STATUS SystemStatusFlag](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-system_power_status)
  reports that separately; new environment reports record it without estimating watts.

## v0.4 hot-path experiments (2026-10-04)

- The [current QNN provider documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
  describes `htp_performance_mode`. The installed catalog provider accepts
  `sustained_high_performance` and `burst`; strict proof remains required. The
  540p ESPCN compute experiment changed about 35 ms default to 12 ms sustained
  and 10 ms burst. These short measurements do not establish thermal behavior.
- Larger graphs were slower: 512-pixel tiles took about 14 ms sustained, and a
  full 960x540 ESPCN graph took about 30 ms. A full-frame FSRCNN-small graph took
  about 37 ms. All passed strict QNN proof and CPU/NPU agreement (max difference
  below 0.002 on the test workload), but are rejected for the real-time preset.
  QNN logs showed more DDR traffic on larger graphs; this is consistent with a
  memory/locality cost, not a measured silicon-level diagnosis.
- Native NV12 avoids the RGB round trip. Limited-range Y is normalized 16â€“235
  to 0â€“1; native chroma is resized on CPU. Explicit full-range input needs RGB.
  Unknown YUV range is assumed limited. This changes the delivered colorspace
  path and requires new quality measurements; no zero-copy claim is made.
- FFmpeg's [Media Foundation documentation](https://ffmpeg.org/ffmpeg-all.html#MediaFoundation)
  supports NV12 encoder input. Executed paths still require hardware format/MFT
  evidence. Queue depth 2 improved short 540p trials from about 40 to 42 FPS;
  depth 4 had little extra benefit and holds more frames. Long trials are the gate.
- VMAF framesync can repeat a frame when MP4 and millisecond Matroska clocks
  differ. v0.4 assigns aligned CFR frame ordinals on a common AVTB clock, disables
  repeated-last frames, and verifies the sampling count. Historical v0.3 VMAF
  numbers used its disclosed original setup; v0.4 recomputes comparisons with
  the corrected alignment. Ordinal PSNR/SSIM comparisons were already aligned.
- [GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes)
  and [GetProcessMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo)
  provide process CPU and working set for Python and codec children. One-second
  samples exclude other apps; they are not system energy or accelerator telemetry.

- Unblended NV12 reached about 45 FPS in three long trials, but the four-clip
  delivered suite lost average SSIM versus bicubic. Fixed strengths 0.25/0.5/0.75
  all run neural inference on every frame. Strength 0.5 improves average paired
  PSNR and SSIM in that suite; VMAF still favors bicubic. It is a disclosed quality
  compromise, not a general perceptual improvement. Pillow baseline resizing
  cost about 10.6 ms at 540p; reusable separable float Catmullâ€“Rom buffers measured
  about 6.9 ms in a 50-call CPU experiment. Edge/rounding semantics differ and
  are independently tested against Pillow's float resampler in the interior.
  Final sustained release trials remeasure the selected blend and QNN mode.
- Native FFmpeg includes SVT-AV1 but no libaom. Software AV1 now selects an
  installed implementation, recording the actual encoder; the native CPU
  integration test completed with SVT. This is explicitly software encoding.

- [BtbN retention policy](https://github.com/BtbN/FFmpeg-Builds#release-retention-policy)
  keeps daily builds fourteen days and monthly builds two years. September's
  retained 9.0/8.1 ARM64 packages crashed at `-version` (0xc0000005). The
  [October 3 COFF stripping fix](https://github.com/BtbN/FFmpeg-Builds/commit/9acad4a9ef1583096af7836cc1e9c8cbcb4d3950)
  explains missing COMDAT relocation symbols; the observed failure is consistent
  with it, rather than independently proven by a native debugger. August's
  retained `n9.0.1-11-ge47273f4d9` passed startup and hardware codec checks.
  v0.4 uses that pinned build, adds startup/integrity cache acceptance and
  remeasures the final workload. No failing package or vendor binary is published.

## v1 quality investigation

- Qualcomm's official QuickSRNet small/medium x2 checkpoints use Conv, clipped
  activations and pixel shuffle. The AI Hub manifest assigns BSD 3-Clause model
  licensing; pinned checkpoint URLs/hashes and license attribution are retained.
  Full RGB and neutral-RGB luminance graphs both passed strict QNN. On three
  short model-only trials, 12 small-Y tile calls took about 8.7 ms, RGB 13.8 ms;
  medium-Y 16.0 ms. These are not video FPS or sustained headline measurements.
- The neutral-Y export sums the first RGB input weights and mixes output RGB
  phases *after* their Clip. Mixing before Clip changes the function. All arrays
  match independent Torch extraction; CPU output matches the source RGB network
  on repeated Y within 1.32e-6. No Torch runtime dependency is introduced.
- Expanded delivered-video testing exposed a color metadata bug: NV12 bytes
  crossing a raw pipe lacked their BT709 input tag. FFmpeg's
  [color options](https://www.ffmpeg.org/ffmpeg-all.html) apply to decoding/input
  as well as encoding/output. Hardware/software decode matched byte-for-byte.
  Native pre-encode neural output had low MSE, while the encoded result acquired
  content-dependent brightness error. Tagging raw input and output reduced one
  animation frame's MSE from 15.56 to 3.14. Correct the boundary and rerun paired
  metrics instead of treating the error as a model defect or learning around it.
- Global-bias subtraction and coarse lowpass-residual correction made little
  difference in the untagged three-clip probe. Adaptive residual clipping reduced
  VMAF in all three. These provisional experiments neither select a default nor
  establish a fusion ranking; correct color negotiation precedes further search.

## v0.6 validation and CPU profiling

- Official [FFprobe packet/frame options](https://www.ffmpeg.org/ffprobe-all.html)
  distinguish coded packets from decoded frames. Packet PTS inspection avoids
  neural-independent full-file decode. A local 3,600-frame H.264 input and AV1
  output each inspected in about 0.2 seconds, before pipeline integration.
- Default validation now inspects every packet with a bounded 64-PTS reorder
  heap, rejects bad cadence/flags/missing timestamps, and requires counts to
  agree with actual decoded/processed frames and container metadata when present.
  This proves checked access-unit timing/accounting, not pixel integrity.
- `--verify-full` explicitly decodes every frame for count/timestamp evidence.
  One decoded pass provides both; the previous output path decoded twice, once
  for `-count_frames` and again for timestamp inspection.
- The new model already removes the measured blend bottleneck. Detailed timers
  split finite/fusion checks, quantization, chroma scaling and owned output copy
  before considering native code or a changed resampling algorithm.
## Temporal research for v1

Small early-fusion networks and implicit recurrent feature propagation are
supported research directions, but published speedups on other devices do not
establish Snapdragon performance. Sources: [spatio-temporal SR paper](https://arxiv.org/abs/1611.05250),
[RLSP paper](https://arxiv.org/abs/1909.08080).

The current upstream [QNN operator documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
includes GridSample. That does not establish support or efficient execution in
our installed Windows ML catalog package. The first local candidate uses only
Conv, Relu, Clip and DepthToSpace, and proves its actual installed-runtime graph
through the existing strict execution/profile checks. Working runtime versions
remain pinned; newer upstream package documentation is not an upgrade mandate.

An original 1,028-parameter two-frame correction trained on licensed film
sequences passes QNN, but its first four delivered development comparisons lose
VMAF versus the spatial default. No quality success or production default change
is claimed from training loss or random-input execution proof.

The feature-input revision uses 1,316 original parameters and gains modest
delivered VMAF. Its spatial-plus-correction graph requires halo seven, not four;
full-context/tiled equality tests enforce this. On four fresh corrected
validation clips it gains 0.143 VMAF, loses 0.028 dB/0.000649 SSIM, and disagrees
across coarse versus native-resolution temporal residual diagnostics. The
native-resolution diagnostic gets slightly worse. That is not a temporal gate
pass. Investigate a stronger temporal objective on development data before
consuming another fresh validation split; do not select the favorable metric.

Development-only weight-one, output-rounding-aware, wider and weight-four
corrections are recorded in [paired research results](../benchmarks/temporal-research/development.json).
The best native temporal reduction among those development runs is only 0.59%;
the stronger loss worsens both spatial quality and temporal error. None selects
a new default. Output-rounding-aware training is straight-through 8-bit limited
range rounding, not an INT8/QDQ inference claim.

Upstream QNN lists Slice, and installed-runtime strict proof succeeds for a
static current-channel Slice. Its twelve-call model-only latency regresses
(about 19.9 versus 18.6 ms), so it remains a rejected graph experiment. A
320x270 core instead cuts padded work for the primary resolution: six-call
model-only median about 12.4 ms, with independent full-context equality tests.
Short delivered-video comparisons retain identical quality. Sustained testing
must establish actual video throughput before a realtime claim.


### Temporal development outcome (2026-10-05)

Two-frame feature correction and two-step recurrent enhanced-state correction
execute through strict QNN but do not establish better temporal video quality.
Native residual-difference gains are below one percent with spatial tradeoffs;
recurrent VMAF declines. Stronger loss worsens results. Keep the spatial default,
retain all paired metrics, and do not release v0.7 from these probes. Details:
[temporal research](temporal-model.md). No unsupported-GridSample claim: current
[QNN operator documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
lists it, but a usable motion-warp model still requires installed-runtime proof.
