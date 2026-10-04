# Research notes

## v0.2 — checked 2026-10-04

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
  selected `d3d11` frames and explicit download. The driver rejected a 64×48
  fixture, while 640×360 worked; strict mode rejected that failure.
- Native libvmaf executed using the explicit `vmaf_v0.6.1` built-in model; a
  same-reference sanity run measured 99.39267 over ten sampled frames.
  [Netflix's integration documentation](https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md)
  and [FFmpeg's filter source](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_libvmaf.c)
  define distorted/reference order, sampling and model selection. Full release
  quality numbers come from actual delivered outputs, not the sanity score.
- A direct FFmpeg H.264-MF bicubic transcode lost two display frames at a scene
  cut (118 instead of 120, with a timestamp gap), despite an exit code of zero.
  For the same clip, HEVC and AV1 delivered all 120. The quality suite therefore
  compares AV1 hardware outputs. This is an observed path-specific failure;
  its internal driver cause is not established. Project video output always
  checks actual encoded-frame count and rejects such loss.
