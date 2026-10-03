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
FPS equivalent is 1000/median; it is an inference-only calculation and does not
represent an implemented video pipeline. Speedup is CPU median/NPU median.
For no NPU, only actual CPU results appear, together with the NPU failure reason.

Startup covers provider preparation, session construction, graph preparation,
and NPU proof inference. The benchmark's timer starts after artifact validation;
Python import and image preparation happen outside startup. CPU session creation
does not include an NPU-style proof call, so its startup scope is inherently shorter.
First-time provider download may increase NPU startup substantially. No persistent
context cache exists in v0.1; a new process creates a fresh session.

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
Do not use the low-resolution source as ground truth. SSIM is deferred.
