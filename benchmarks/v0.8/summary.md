# v0.8 validation

Packaged setup, executed capability diagnostics, simple video defaults and
throttled progress. Inference weights/math and the realtime configuration remain
unchanged. Temporal research did not clear its gate; no v0.7 release was made.

## Qualified realtime trials

Frozen candidate commit `a853086d0a4b92e0ce0e1cd2189cebb9487b15d2`, version 0.8.0,
native ARM64 fresh wheel; final release additionally fixes UTF-8 pipe output and
updates documentation. The included source fingerprint describes the timed code.
Three 180-second sources, 5,400 frames and 64,800 neural calls each. Source
acquisition, retiming, license and hashes are in [performance.json](performance.json).

| Trial | End-to-end FPS | Processing seconds | Whole function seconds | Minimum rolling 10s FPS |
|---|---:|---:|---:|---:|
| 1 | 53.71 | 100.54 | 108.42 | 52.4 |
| 2 | 54.54 | 99.01 | 102.98 | 53.8 |
| 3 | 53.75 | 100.47 | 104.38 | 53.0 |

**Median: 53.75 FPS**, processing/source factor about 0.56. All frames processed;
strict QNN proof, CPU inference fallback disabled, D3D11VA hardware H.264 decode,
QCOM AV1 Media Foundation hardware encode, copied audio, packet cadence/counts,
and bounded depth-two queues pass. This does not measure internal VPU utilization.
Power state snapshots: AC, Balanced, energy saver off. Power, temperature and
accelerator utilization are not measured; background processes are uncontrolled.
Different power/input conditions from v0.6 do not establish a code speedup.

An initial three-trial series used 0.8 source with stale 0.6 installed metadata.
It is retained in [preliminary.json](preliminary.json), excluded from the headline
before repeating all three qualified trials. No trial was selected by speed.

Frame latency is raw-read start through encoder pipe submission, including queues;
it is not display or encoder-completion latency. Complete throughput includes
encoder flush. Per-stage p50/p95/p99 and sampled process memory are in JSON.
Peak aggregate sampled working sets were approximately 723/473/464 MiB; these
are one-second samples of Python plus codec processes, excluding drivers/other apps.

## Default command and presets

The ordinary command (no model/device/preset flags) processes the 120-second
fixture in **72.73 seconds from process launch through exit**, including imports,
printing and JSON. It selected realtime/Small/QNN/D3D11VA/QCOM AV1, enhanced all
3,600 frames in 43,200 neural calls, and recorded zero application drops.
[Actual CLI report](cli-wall-time.json). Default final packet validation is about
0.2 seconds; decoded forensic validation remains opt-in with `--verify-full`.

Controlled secondary comparisons explicitly use strict NPU and hardware AV1
for both presets (overriding their default auto H.264 codec):

| Configuration | Trials | End-to-end FPS | NPU-call p50 ms/frame | CPU post p50 ms/frame |
|---|---:|---:|---:|---:|
| realtime, Small, burst | 3 | 53.75 median | see per-trial JSON | see per-trial JSON |
| balanced, Small, sustained request | 1 | 50.00 | 10.25 | 6.86 |
| quality, Medium, sustained request | 1 | 34.84 | 18.78 | 6.93 |

[Exact preset reports](presets.json). Single trials are not headline qualification.
Requested performance modes do not establish actual power savings. Medium had
higher mean PSNR/SSIM/VMAF than Small on the twelve v0.5 development clips under
equal AV1 encoding; that is not a new holdout or universal preset quality claim.

## Quality and tests

Conversation, animated leaves and aerial landscape delivered outputs are
**byte-identical** to their v0.5 Small baseline; PSNR/SSIM/VMAF deltas are zero.
[Regression evidence](quality-regression.json). The complete locked 24-clip
bicubic comparison remains in the [v0.5 report](../v0.5/summary.md); no new
24-clip evaluation is claimed here. Its temporal diagnostic tradeoff remains.

246 ordinary tests passed, including redirected UTF-8 output. Twenty tests in the manual
NPU/DirectML marker group and six video hardware tests passed. Executed doctor
and setup proofs, wheel installation, packaging and metadata checks pass.
CI runs CPU/failure paths on Windows/Linux; it does not prove Snapdragon hardware.
[Executed capability report](capabilities.json), [environment](environment.json).

## Reproduce

```powershell
npu-sr setup --download-ffmpeg
python scripts/download_video_benchmarks.py --sustained --duration 180
npu-sr benchmark-realtime <prepared-input.mp4> -o outputs/trials --trials 3 --json outputs/performance.json
python scripts/check_video_regression.py --directory <prepared-quality-cache> --json outputs/quality.json
```

Models, films, outputs and compiled contexts remain outside git. Other devices,
codecs, color formats and power states require their own validation. Long-run
release-candidate testing and wider format coverage remain future work; this
release does not meet the v1 ten-minute sustained acceptance gate yet.

Dependency audit: 120 installed distributions checked against the audit service;
no known advisories reported. Editable project entries are skipped; this is not a
claim that all dependencies are vulnerability-free.
