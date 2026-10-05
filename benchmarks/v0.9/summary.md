# v0.9 reliability validation

## Sustained primary profile

Snapdragon X Plus X1P-42-100, Windows ARM64 build 26200. Strict QuickSRNet Small
Y / QNN HTP, 256-pixel useful cores, twelve neural calls per frame, NV12,
unblended luminance, depth-two pipeline, D3D11VA H.264 decode and QCOM AV1
Media Foundation hardware encode at 8M. No internal VPU utilization claim.

Three independent 1,200-second sources, 36,000 frames and 432,000 neural calls
per trial. The licensed film content loops; every application input frame is
processed. Each trial exceeds **ten minutes of actual processing**, rather than
merely having a ten-minute source. Frozen candidate `7082d614faeb00ff25c2123b7583aba25e519dbf`;
final initialization additionally probes the requested output container, with
unchanged weights, frame math and streaming path. Exact fingerprints retained.

| Trial | End-to-end FPS | Processing seconds | Whole function seconds | Minimum rolling 10s FPS | Peak sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 52.81 | 681.66 | 693.16 | 51.4 | 360.6 |
| 2 | 53.06 | 678.45 | 687.72 | 51.6 | 366.4 |
| 3 | 53.05 | 678.56 | 688.17 | 52.2 | 368.2 |

**Median throughput: 53.05 FPS.** Zero application drops; all expected frames,
neural calls, 2× dimensions, cadence and copied AAC offset/channel checks pass.
Both queues are bounded at two. Resource samples include Python and codec
processes, excluding drivers and other applications. Working-set window means
increase modestly within the declared 15% plus 64 MB tolerance; no runaway growth
is observed. This is not a proof of indefinite operation or driver memory usage.

AC, Balanced power scheme, saver off at trial start/end. No project tests,
training or builds ran concurrently. Other background processes are uncontrolled.
Power, temperature and accelerator utilization are **not measured**. First/final
ten-second throughput is stable; that does not establish sensor temperatures.

Whole-run approximate quantiles use fixed 0.05 ms histogram bins; actual
means/extrema and sample counts include all 36,000 frames. The legacy last-8,192
window remains separate. Frame handoff p50 is about 75 ms including queues,
**not display or encoder-completion latency**. Enhancement p50 is about 18.8 ms;
NPU calls about 8.9 ms and CPU postprocessing about 7.1 ms. Pipe blocking times
are not isolated hardware codec compute times.

Default final packet checks take 0.66–0.84 seconds. Whole function time includes
startup, probes, validation and hashing, and excludes CLI import/printing. The
older full-file forensic decode remains opt-in with `--verify-full`.

[All three trial records, proofs, timings and samples](realtime-performance.json),
[environment](environment.json). Outputs were hashed after successful validation
and deleted individually to bound disk usage; hashes and reproduction remain.

## Quality recheck

All 24 frozen clips were rerun without model/fusion retuning. Twelve development
and twelve originally held-out clips, equal per-clip means, same delivered AV1
encoding, native coded-Y protocol, every-third-frame PSNR/SSIM and every-frame
VMAF/NEG. The repeated holdout is a regression recheck, not newly unseen data.

| Method | PSNR dB | SSIM | VMAF | VMAF NEG |
|---|---:|---:|---:|---:|
| Bicubic | 37.8653 | 0.961910 | 85.6160 | 81.9471 |
| QuickSRNet Small / QNN | 39.8693 | 0.971032 | 91.5656 | 87.7279 |
| Difference | +2.0040 | +0.009123 | +5.9496 | +5.7808 |

The coarse 256×144 residual-change diagnostic worsens 2.17%, while the native
resolution diagnostic improves 8.93% (0.00552331 → 0.00503009). Both are project
diagnostics without motion compensation, not standard perceptual flicker scores.
Both are retained; neither substitutes for visual inspection. Six-consecutive-frame
crops of faces, foliage, aerial texture, a night scene and text show sharpening
and blur tradeoffs without severe instability in the inspected regions. This is
limited crop inspection, not full playback or a blinded perceptual study.
Synthetic VMAF saturation and all unfavorable diagnostics remain in the averages.

[Every clip, source/hash, metric and proof](quality.json),
[inspection coordinates and hashes](visual-review.json). Three earlier regression
clips also reproduced v0.5 metrics exactly. No temporal model became the default.

## Formats and reliability

Executed generated SDR cases cover H.264/HEVC/AV1 input, 23.976/24/25/29.97/30/
50/59.94/60 FPS, 360p through 1080p, portrait, no audio, stereo/surround AAC,
and Opus/MKV. Twelve-frame cases are correctness checks, not sustained FPS claims
or a promise for every codec/rate/resolution cross-product. [All attempts](formats.json).

- QCOM AV1 changes requested 1708×960 into 1712×960: reject; explicit HEVC/H.264
  MP4 alternatives preserve geometry.
- AV1/MKV requires sequence-header extraction, now tested in the application.
- H.264/HEVC hardware MKV headers fail on this stack: use MP4 or explicit software.
  Final startup proof tests the requested container before processing.
- Real generated VFR and explicitly PQ-tagged fixtures are rejected. HDR/VFR,
  anamorphic pixels and rotation metadata remain unsupported.
- Copied audio offsets are rebased relative to the first video frame; leading
  audio is trimmed. Actual AAC decoded samples match the correctly trimmed source.
- Disk-write failure, broken encoder and cancellation preserve existing outputs
  and clean child processes. Full decoded audit has bounded lines and timeout.

[Negative input checks](negative-inputs.json), [codec details](../../docs/hardware-codecs.md),
[honest SKU matrix](../../docs/hardware-matrix.md). Only X1P-42-100 is locally verified.

## Reproduce

```powershell
python scripts/download_video_benchmarks.py --sustained --loop-source --duration 1200 --directory datasets/longrun
python scripts/validate_longrun.py datasets/longrun/sustained-960x540-1200s.mp4 --directory outputs/longrun-video --provenance datasets/longrun/sustained-provenance.json --json outputs/longrun.json --trials 3
python scripts/validate_video_formats.py --directory datasets/formats --json outputs/formats.json
python scripts/run_quality_corpus.py --split development --json outputs/development.json
python scripts/run_quality_corpus.py --split holdout --json outputs/holdout.json
python scripts/summarize_quality_corpus.py outputs/development.json outputs/holdout.json --model quicksrnet-small-y-x2 --json outputs/quality.json
```

Acquire models/native FFmpeg with `npu-sr setup --download-ffmpeg` and prepare the
frozen corpus with `scripts/download_quality_corpus.py` first. Long-run headline
trials require AC and saver off, consistent inputs, enough disk and no concurrent
project workloads. No dataset, tool/runtime binary or model weights are published.
269 ordinary tests and 20 separately selected NPU/DirectML tests passed.
Seven actual hardware video tests passed after the final container-proof change.
Fresh wheel installation, packaging and main/tag CI are checked before publication.

## Optional 60 FPS exploration

One 120-second 854×480@60 →1708×960@60 source measured **73.07 FPS**
over 98.5 seconds of processing, minimum rolling
71.9 FPS. All 7,200 frames and 57,600 neural calls,
strict QNN/D3D11VA and QCOM HEVC hardware encode/cadence pass. Explicit HEVC
avoids the AV1 geometry constraint. This is one exploratory trial, not three
independent trials or ten-minute 60 FPS qualification.
[Actual report and preparation](480p60-exploration.json).
