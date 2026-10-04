# v0.6: cheaper finalization and detailed CPU profiling

The model, weights, NV12 math and strict QNN/hardware codec requirements are
unchanged from v0.5. Every video packet is now inspected for CFR cadence with
bounded coded-order reordering. Counts must agree with actual processed frames
and container metadata. Resolution, duration, audio, successful codec flush and
neural calls per frame remain checked before atomic publication.

`--verify-full` explicitly decodes every frame for forensic timestamp/count
inspection. The old output audit decoded twice; one pass now supplies both.
Packet counts are not universally picture counts, and do not prove pixel
integrity. Unsupported layouts fail with a full-audit suggestion, not a timing
assumption. Full mode still rejects VFR or corrupt streams.

## Three sustained trials

Same declared 120-second 960×540@30 source, QuickSRNet Small Y, 12 tiles/frame,
QNN burst, unblended NV12, queue depth two, D3D11VA and QCOM hardware AV1 8M.
Three trials deliver 3,600 frames each, execute 43,200 neural calls each and
pass strict QNN profiling, frame counts/cadence, audio and hardware evidence.

| Trial | Streaming FPS | Min rolling 10s FPS | Post median ms | Output check s | Function elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.728 | 47.30 | 7.721 | 0.201 | 82.63 |
| 2 | 48.336 | 46.90 | 7.731 | 0.180 | 78.65 |
| 3 | 49.896 | 48.10 | 7.538 | 0.158 | 76.13 |

Median streaming throughput is 48.73 FPS. Median complete function time is
78.65 seconds for a 120-second source, versus v0.5's 149–156 seconds. These
function timers include initialization, validation and hashing, but exclude CLI
imports/printing/JSON writing. A separately timed CLI run is recorded below.
Background conditions were uncontrolled; battery 31/29/29%, Balanced, saver off.
The slower streaming rate than v0.5's 52.63 FPS is reported; no isolated neural
speed gain or thermal/power cause is claimed. Power and temperature are not measured.

CPU postprocessing medians remain below 8 ms. First-trial medians:

| Component | ms |
| --- | ---: |
| Finite/fusion check | 0.271 |
| Luminance quantization/write | 2.967 |
| Chroma scaling | 4.256 |
| Owned output copy | 0.122 |

Medians of components need not sum to the total median. Chroma/quantization now
dominate postprocessing; a native extension is not justified merely to meet the
8 ms target already met by the quality-driven removal of blending in v0.5.

## Correctness and quality regression

The packet check and explicit full decoded audit both count all 3,600 frames
in the same sustained output: 0.107 seconds versus 33.91 seconds. Generic tests
exercise real B-frame MP4 and fractional-rate MKV, duplicate/missing/bad packet
PTS, VFR duration, corruption flags, bounded reorder, timeouts and cancellation.
Both modes generate identical decoded pixels on the CPU integration fixture.

Faces, animation and aerial-detail delivered regressions are bit-for-bit identical to v0.5 and have exactly unchanged
PSNR/SSIM/VMAF. This is a regression smoke suite, not a new expanded
quality ranking or a fresh holdout. The [v0.5 full quality report](../v0.5/summary.md)
remains the quality evidence for the unchanged model. Per-clip differences and
new execution proofs are retained in [quality-regression.json](quality-regression.json).

## Reproduce

```powershell
python scripts/run_selected_sustained.py --directory <prepared-directory> --output-directory <trial-directory> --json outputs/sustained.json
python scripts/check_video_regression.py --directory <quality-directory> --json outputs/regression.json
npu-sr video input.mp4 -o enhanced.mp4 --preset realtime
npu-sr video input.mp4 -o audited.mp4 --preset realtime --verify-full
```

Recorded development versions/source hashes are preserved, including 0.5.0
metadata before the 0.6 version bump. They are not relabeled as later measurements.
The detailed timer additions do not change output pixels; normal validation
changes the audit scope explicitly. [Performance](performance.json),
[environment](environment.json), [audit agreement](audit-agreement.json).

## Actual CLI wall time

A separate launch-to-exit run measured **74.71 seconds**, exit code 0, for the
120-second source. The internal total was 74.3 seconds; all 3,600 frames and
43,200 neural tile calls were verified, with hardware decode/AV1 and audio.
This run is separate from the three-trial headline, not substituted as a faster
trial. Default audit is packet-based. See [timed CLI evidence](cli-wall-time.json).

## Release checks

175 ordinary tests pass; 19 separately selected QNN/DirectML tests and four
hardware video tests pass. An additional targeted hardware test checks full-mode
audit with cached strict QNN and cancellation. Lint, formatting, build/metadata
and a fresh ARM64 wheel installation pass, including strict NPU image inference.
Generic CI remains CPU-only. The added code uses only the standard library.
