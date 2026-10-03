# Architecture

`cli.py` parses commands and presents actionable failures. `model.py` supplies
the model registry and checks generated manifests. `acquire.py` retains the exact
v0.1 baseline; `acquire_models.py` constructs additional static ONNX graphs from
hash-checked upstream parameters without TensorFlow/PyTorch dependencies.
`qnn.py` owns Windows ML's process-lifetime bootstrap and
provider registration. `runtime.py` owns strict session creation and inference proof.
`image.py` owns preprocessing, tiling, reconstruction and file IO.
`benchmark.py` measures inference independently of acquisition/startup;
`suite.py` adds full-image phase measurements and reference-based quality evaluation.

## Model

The export preserves the pretrained TF-ESPCN parameters and function: 5×5 Conv
(1→64), Relu, 3×3 Conv (64→32), Relu, 3×3 Conv (32→4), DepthToSpace x2, Tanh.
Bias addition is folded into each Conv, weights change HWIO→OIHW, and the external
layout changes NHWC→NCHW. With one output channel, DCR pixel shuffle matches the
upstream TensorFlow depth-to-space ordering. No retraining occurs.

Input: `[1,1,136,136]`, float32 luminance in [0,1]. Output: `[1,1,272,272]`, float32.
Model opset is 13 and IR version is 8 for broad runtime compatibility.
CPU uses FP32; the selected QNN HTP device uses FP16 math for floating operators.
The CPU and NPU consume the same ONNX file, identified by SHA256 in benchmark JSON.

## Image path

Load PNG/JPEG/still WebP, reject oversized inputs, apply EXIF orientation, and split
RGB/alpha. Compute full-range BT.601-style luminance/chroma in float32:
`Y=.299R+.587G+.114B`, `Cb=.564(B-Y)+.5`, `Cr=.713(R-Y)+.5`.
This matches the upstream normalized luminance convention. The output reconstruction
uses the inverse coefficients, bicubic chroma and alpha, and clamps to uint8.
Color management and metadata copying remain outside scope.

## Why tiles exist in v0.1

The QNN EP requires fixed input shapes. Resizing every source to one arbitrary
model size would lose aspect ratio or detail. We instead use a single compiled
136×136 shape for all sources, with 128×128 cores and a four-pixel halo.
The three convolution kernels have total receptive radius `2+1+1=4`.
Halo outputs are discarded. Image edges repeat the outermost pixel; internal
tile boundaries use their actual image neighbors. The final partial tiles are
padded and cropped, producing exactly `2W×2H`, including for a 1×1 source.
There is no multiscale selection, overlap blending, parallel tile scheduler, or batching.

## Backend contract

NPU readiness requires catalog installation, Python ORT registration, selection
of a QNN device of hardware type NPU, session compilation without CPU fallback,
valid proof output, and executed QNN kernels in a profile with no other providers.
The proof uses the same session that subsequently processes the image. The temporary
profile is consumed and deleted; stopped profiling does not affect warm measurements.
Every session has a measured startup field separate from inference latency.

Auto mode can create a new CPU session if NPU initialization/proof fails, with an
explicit message. A later run failure never triggers silent backend switching.
Benchmark always creates explicit CPU and NPU sessions, and reports unavailable
NPU status instead of generating measurements for a nonexistent backend.

## v0.2 contracts and reuse

Registry entries provide task, scale, core, halo and color space; runtime shape
checks and image padding/stitching consume that shared contract. Learned denoising
uses scale one. Inputs are copied into one reused contiguous tile buffer.
Independent TF graph reference tests validate additional SR export semantics.

One Runtime can serve many images. QNN contexts can be cached locally using
`cache.py`; model/runtime/package/driver changes select another cache key.
Integrity checks precede loading, and strict proof follows every cache load.
DirectML provides a separately verified GPU comparison. See [models](models.md),
[image pipeline](image-pipeline.md) and [research notes](research-notes.md).
Video belongs to the next gated release and is not implemented in v0.2.
