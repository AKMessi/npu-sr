# npu-sr v1.0 validation

## Primary profile

Snapdragon X Plus X1P-42-100 / Windows 11 ARM64 build 26200 / driver 30.0.219.1000.
Python 3.12 ARM64, ORT 1.25.2, Windows ML bindings 2.3.0, App Runtime 2.5.1.0,
QNN catalog 2.2480.53.0, native ARM64 FFmpeg n9.0.1-11-ge47273f4d9.
Exact versions, hashes and state snapshots are in [environment.json](environment.json).

**960×540@30 →1920×1080@30**, QuickSRNet Small neutral-Y, 256-pixel cores,
halo 4 / input 264², FP32 ONNX / QNN HTP FP16 math, strength 1 / no blend, NV12,
twelve neural tiles/frame, depth-two FIFO, D3D11VA H.264 decode, QCOM AV1
Media Foundation hardware encode / 8M, copied AAC. No internal VPU utilization claim.

The **fresh ARM64 installed 1.0.0 wheel** at frozen commit
`b17680c1d8fa90ec61a7336ad70218ed01d5ab7e` processed three independent
1,200-second sources. Film content deliberately loops; every application frame
is enhanced. Each trial exceeds ten minutes of **actual processing**.

| Trial | End-to-end FPS | Processing seconds | Whole function seconds | Minimum rolling 10s FPS | Peak sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
|1|52.73|682.76|696.73|50.9|368.6|
|2|52.77|682.19|692.27|50.3|398.4|
|3|52.74|682.63|692.13|51.4|370.7|

**Median: 52.74 FPS**, processing/source factor about **0.569**. All **108,000
frames /1,296,000 neural calls** accounted for, zero application skips/drops,
no duplicated/reordered application frame indices, matching encoded counts,
2× dimensions, cadence, duration and audio offsets/channels. Strict QNN proof
succeeds on each actual session; CPU inference fallback stays disabled.
Hardware transform and D3D11 executed-frame evidence retained in every trial.
[All records and production gate](realtime-performance.json).

Working-set first/final window means are 348→363 /362→372 /356→366 MiB,
within the declared tolerance; no runaway growth observed. These are one-second
Python+decoder+encoder samples, excluding driver memory and other applications.
The two queues stay bounded at two. This does not prove indefinite operation.

Whole-run histogram median-of-trial quantiles, with0.05 ms bins:

| Stage/scope | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| QNN run calls, includes host submission/output checks |8.88|9.28|9.53|
| CPU postprocessing |7.13|7.58|8.13|
| Enhancement |18.83|19.73|20.53|
| Completion interval |18.88|19.78|20.58|
| Frame handoff, including queues |75.78|78.13|81.53|

Handoff ends at encoder pipe submission, **not encoder completion/display**.
Pipe waits are not isolated hardware codec compute. The p50 encoder pipe wait
is0.28 ms; no independent encode latency is claimed. All 36,000 samples/trial
contribute to bounded histogram quantiles (up to 0.025 ms binning error), actual
means/extrema and disclosed overflow. Legacy last 8,192 windows remain separate.
Complete throughput includes encoder flush. Whole function adds startup/probes,
validation/hash/report preparation, but excludes CLI imports/printing.
Default final validation is 0.70–1.35 s; no full-file decode unless `--verify-full`.

AC/Balanced/saver off at start/end and monitoring snapshots. No concurrent project
training/tests/builds. Other background activity uncontrolled. **Power, temperature,
accelerator utilization not measured**. Stable first/final rolling throughput is
not a temperature reading or a performance-per-watt claim.

## Quality

The unchanged realtime weights/frame math inherit the **complete rerun of all 24
frozen clips** in v0.9, not a newly selected or cherry-picked subset. The twelve
originally held-out clips remain excluded from tuning; repeated evaluation is a
regression check, not fresh unseen data. [Pinned full report](quality.json) links
its exact SHA256 and contains complete per-clip/group summary. Final installed
wheel separately rechecks conversation, animated leaves and aerial detail.

| Delivered method | PSNR dB | SSIM | VMAF | VMAF NEG |
|---|---:|---:|---:|---:|
| Bicubic / AV1 |37.8653|0.961910|85.6160|81.9471|
| Small / QNN / AV1 |39.8693|0.971032|91.5656|87.7279|
| Difference |+2.0040|+0.009123|+5.9496|+5.7808|

Coarse temporal residual error worsens 2.17%, native-resolution error improves
8.93%. Both project diagnostics lack motion compensation and may reward
smoothing/bias; neither is a standard flicker metric. Limited consecutive native
crop inspection found no severe instability in those inspected regions, not
full playback or blinded validation. Independent-frame ringing/flicker remains
possible. [Metric details](../../docs/quality.md),
[inspection evidence](../v0.9/visual-review.json),
[rejected temporal candidates](../../docs/temporal-model.md).

Realtime:Small/burst/AV1. Balanced:Small/sustained/H.264. Quality:Medium/sustained/
AV1; actual three-development-clip default-preset checks are in
[quality-preset.json](quality-preset.json). Medium showed modest development
quality gains under equal AV1, at higher compute cost; no universal ranking.
See the [controlled v0.8 preset trials](../v0.8/presets.json) for measured speeds,
whose explicit AV1 overrides must not be presented as default-H.264 throughput.

## Reliability and remaining limits

**280 ordinary tests, 12 NPU tests, 8 DirectML tests and 7 hardware video tests passed.**
Generic Windows/Linux CI verifies CPU/failure paths, not Snapdragon execution.
Fresh wheel setup runs actual NPU and H.264 decode/H.264/HEVC/AV1 encode proofs.
Output/report collision and hardlink guards, atomic image/video publication,
disk-write failure, cancellation, broken codec and bounded audits are tested.

[Format attempts](../v0.9/formats.json) cover integer/fractional rates 23.976–60,
H.264/HEVC/AV1, portrait, no audio, stereo/surround AAC and Opus/MKV, 360p–1080p.
Tiny fixtures are correctness evidence, not every cross-product or sustained FPS.
H.264/HEVC hardware MKV is unsupported on this stack; use MP4/software.
QCOM AV1 pads 1708 to 1712; strict geometry rejects it. Explicit HEVC preserved
1708×960 in a **single** 98.5 s 480p60 exploration at 73.07 FPS; it is not ten-minute/
three-trial 60 FPS qualification. AV1/MKV requires tested sequence-header extraction.
HDR/VFR/anamorphic/rotation metadata remain unsupported. First audio stream only;
no subtitles/chapters/other streams, live/player integration or zero-copy.
Only X1P-42-100 verified; [other SKUs unknown](../../docs/hardware-matrix.md).

## Reproduce

Install a release wheel or clone, official App Runtime/native Python prerequisites,
then `npu-sr setup --download-ffmpeg`. Prepare the frozen quality corpus and licensed
film using the documented acquisition scripts. Models/datasets/codec/runtime
binaries stay outside git. Cache loads always repeat strict proof.

```powershell
python scripts/download_video_benchmarks.py --sustained --loop-source --duration 1200 --directory datasets/longrun
python scripts/validate_longrun.py datasets/longrun/sustained-960x540-1200s.mp4 --directory outputs/trials --provenance datasets/longrun/sustained-provenance.json --json outputs/realtime.json --trials 3
```

Output videos are validated/hashed then individually deleted to bound disk use;
`--retain-videos` retains them. Source loops are declared; no application neural
bypass, dropped/duplicated frame or dynamic quality switch exists.
[Benchmarking](../../docs/benchmarking.md), [installation](../../docs/installation.md).
Final CI/artifact hashes are checked at publication; no untested tag is declared green.

## Exact final wheel user command

The ordinary video command, without model/device/preset flags, selected realtime
NPU/D3D11VA/QCOM AV1 and finished a 120-second source in **75.82 seconds
from process launch to exit**, including imports, printing and JSON. All 3,600
frames and 43,200 neural calls, audio/cadence and strict proof passed; exit 0.
[Measured command](cli-wall-time.json), [executed doctor/GPU proofs](capabilities.json).
The final wheel Python source fingerprint exactly matches the three sustained
trials; final metadata/documentation changes do not alter the measured core.

Dependency audit reported no known advisories for 120 installed distributions;
editable project metadata was skipped. This does not assess drivers, vendor runtime
internals or undisclosed vulnerabilities. [Audit scope](dependency-audit.json).
