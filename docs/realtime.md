# Realtime video

The acceptance gate measures sustained **end-to-end** throughput, including
codec startup, transfers, preprocessing, every neural tile, stitching, chroma,
pipe waits and encoder flush. Provider/model startup and final validation/hashing
are reported separately. Model-only throughput is not video FPS.

## Reproduce

```powershell
python scripts/download_ffmpeg.py
npu-sr models download espcn-x2-256
python scripts/download_video_benchmarks.py --sustained --duration 120
python scripts/run_realtime_benchmarks.py --trials 3
python scripts/download_benchmarks.py
npu-sr models download fsrcnn-x2
python scripts/download_video_benchmarks.py
python scripts/run_realtime_benchmarks.py --quality --trials 3
```

By default, large video files and caches use the local application cache; reports
use ignored `outputs/v04-validation`. Source movie and generated inputs are hash
verified. The 24 FPS film is explicitly retimed to a 30 FPS benchmark fixture;
no runtime frames are skipped, intentionally duplicated or conditionally bypassed.
A 120-second source gives at least 60 seconds of wall processing on this machine.

For another CFR source:

```powershell
npu-sr benchmark-realtime long-input.mp4 -o outputs/trials --trials 3 --json outputs/realtime.json
```

A nonzero exit indicates a failed gate, with measured results retained in JSON.
The gate requires three complete trials, at least 854×480 at 30 FPS, exactly 2×
output, >=60 seconds processing per trial, average and minimum rolling ten-second
throughput >= source FPS, strict QNN proof, hardware encoding, matching frame
counts and validated timestamp cadence. Short clips cannot pass this gate.

## Presets

All settings are printed; explicit options override them. Consult the release
[quality/speed report](../benchmarks/v0.4/summary.md) before choosing one.

| Preset | Model | Frame path | NPU mode | Queue depth |
| --- | --- | --- | --- | ---: |
| realtime | ESPCN 256-pixel cores | native NV12, fixed 0.5 neural strength | burst | 2 |
| balanced | ESPCN 256-pixel cores | RGB | sustained_high_performance | 2 |
| quality | FSRCNN | RGB | sustained_high_performance | 2 |

Realtime additionally requires NPU, hardware decode, hardware AV1 encode. Other
presets retain the default auto device/codecs unless explicitly selected. The
quality name reflects the model's image-subset score, not universal video quality.
Unspecified presets preserve the v0.3 RGB/default-performance/serial behavior.
No automatic per-frame model switch, dropped frame or dynamically disabled SR exists.

NV12 receives native limited-range Y (16–235 normalized to 0–1), avoiding the
RGB/Rec.601 round trip. Chroma is bicubic resized on CPU; output remains limited
range. Unknown YUV range is assumed limited; explicitly full-range input fails
with an RGB-path suggestion. It is not zero-copy and changes the quality path.

`--neural-strength` is a fixed NV12-only luminance blend with a CPU Catmull–Rom
bicubic baseline (0 < strength <= 1). Every tile still runs through the network.
This experimental override has separate quality and speed costs; it must not be
represented by a benchmark obtained at another strength. It uses float buffers,
replicated edges and no intermediate uint8 rounding, which differ slightly from
Pillow/FFmpeg bicubic. The preset selects fixed strength 0.5; strength 1 is unblended. The selection
recovers mean PSNR/SSIM on the delivered four-clip suite at a measured compute cost;
VMAF still favors bicubic on average. No universal perceptual gain is claimed.

## Parallelism and measurements

A decoder reader and encoder writer overlap with a single enhancement worker.
Two bounded FIFO queues apply backpressure and preserve order. The enhancement
worker alone owns the ORT session/buffers. Frames sent to queues own their bytes.
Depth 0 is the sequential control, 1–4 are explicit alternatives. Every process
is joined/terminated on errors or Ctrl+C; a watchdog handles stalled children.

Completion intervals end at encoder pipe submission. Reported frame latency
starts at decoder read and ends after that submission; it includes queues and
is **not encoded-frame or display latency**. Average end-to-end time/FPS includes
encoder flush. First/final/minimum rolling ten-second rates use encoder submissions;
minimum windows cover the complete processing interval, including codec startup.
Final encoded counts/cadence separately verify the actual delivered file.

One-second OS samples measure Python plus the two codec processes, including
normalized process CPU time and simultaneous working sets. They exclude other
applications and drivers. Queue/timing/diagnostic/sample storage is bounded.
Temperature, accelerator utilization and power are **not measured**; throughput
stability is evidence of sustained performance, not a thermal sensor reading.
