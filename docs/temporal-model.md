# Temporal research

The released realtime default remains spatial QuickSRNetSmall. The temporal
experiments below did **not** demonstrate a convincing improvement. No v0.7
release was made. Failed checkpoints are not included in model downloads,
presets or package assets. The reusable state, export and diagnostics code is
retained for further research; strict execution does not establish quality.

## Models and state

All candidates fuse the pinned spatial graph with a bounded residual CNN:
Conv/ReLU/Conv/ReLU/Conv/Clip(±0.05)/DepthToSpace/Add. They use fixed shapes,
full halo-seven context (spatial radius four plus correction radius three),
one persistent session, and unchanged strict QNN profiling with CPU fallback
disabled. The two-frame version consumes previous/current LR luminance and
four current spatial subpixel features internally. Corrections have 1,316 or
3,780 learned parameters. The recurrent version has 4,356 correction parameters;
its external input also includes four previous enhanced-output subpixel phases.

State owns one previous LR plane and, for the recurrent variant, four previous
output phase planes. Output is quantized to limited-range coded Y before saving
history. First-frame and scene-cut initialization use the current LR plane in
all history phases. A 32x32 sample detects black transitions and large brightness,
histogram or decorrelation changes. Motion can reset history; subtle cuts can be
missed. This is not optical flow or a universal shot classifier. Geometry changes
reset owned buffers. Tests verify phase order, bounded memory, independent
returned bytes, resets and full-context/tiled equivalence.

The 256x256 core requires twelve calls for 960x540. A 320x270 core requires six:
input 284x334, output 568x668, trim fourteen HR border pixels. The wide two-frame
model's measured model-only cost falls from about 18.6 ms to 12.4 ms, with identical
four-clip delivered quality. Static current-channel Slice was supported by the
installed QNN runtime but slower (about 19.9 ms for twelve calls), so it was rejected.
The recurrent six-channel graph costs about 18 ms for six calls. These are not
sustained video throughput results.

## Evaluation and rejection

See [all paired development results](../benchmarks/temporal-research/development.json).
Every row retains PSNR, SSIM, VMAF, VMAF NEG, both temporal diagnostics, hashes,
frame counts, actual QNN profiling and hardware codec evidence. Four development
clips cover conversation, animated leaves, aerial landscape and moving text.
They are development probes, not an unbiased release corpus. Both temporal
measures are project diagnostics; neither is a standard perceptual metric or
motion-aligned measure, and both can reward smoothing or constant bias.

| Candidate | PSNR Δ dB | SSIM Δ | VMAF Δ | Native temporal error change |
|---|---:|---:|---:|---:|
| 8-wide, temporal loss 1 | -0.028 | -0.000461 | +0.051 | -0.13% |
| 8-wide, rounding-aware | -0.016 | -0.000385 | +0.180 | -0.30% |
| 16-wide, rounding-aware | +0.026 | -0.000389 | +0.187 | -0.59% |
| 16-wide, loss 4 | -0.077 | -0.000479 | -0.138 | +0.39% |
| 16-wide, QNN teacher | +0.019 | -0.000572 | -0.041 | -0.94% |
| 16-wide, recurrent state | +0.017 | +0.000005 | -0.117 | -0.14% |

The earlier 8-wide loss-0.25 halo-seven candidate was tested on four fresh
240-frame validation clips: +0.143 VMAF, -0.028 dB PSNR, -0.000649 SSIM,
coarse temporal error -0.14%, but native temporal error about +0.37% worse.
Earlier halo-four results are superseded. Increasing loss, capacity, matching
QNN teacher outputs, and adding recurrent enhanced state did not clear the gate.
No further holdout was consumed to rescue unfavorable development results.

## Reproduction

[Training definition](../benchmarks/temporal-training-v1.json): twenty licensed
Blender snippets, separated in time from the original quality corpus; 880 adjacent
patch triples and 10% reset examples. Preparation records source/data hashes,
teacher backend and strict proof when QNN is selected. Training uses separate
Python 3.13 x64 PyTorch 2.9.1 CPU, four threads, seed 20261004, 2,500 Adam steps,
batch eight and learning rate 0.001. PyTorch is not an application dependency.
Ground truth is the HR sequence, not LR inputs. Rounding-aware training uses
straight-through limited-range 8-bit quantization, not INT8 inference. Recurrent
training unrolls two predictions with gradients through prior output and excludes
twelve HR edge pixels from the loss. The original correction excludes six.

```powershell
python scripts/prepare_temporal_training.py --directory <work> --source-directory <sources> --teacher-device npu --recurrent-state
py -3.13 scripts/train_temporal_residual.py --directory <work> --output <work>/weights.npz --teacher-features --output-rounding --hidden-channels 16 --temporal-weight 1 --recurrent-state
python scripts/export_fused_temporal.py <work>/weights.npz models/quicksrnet-small-y-x2.onnx <work>/fused.onnx --geometry 320x270
python scripts/probe_temporal_model.py <work>/fused.onnx --fused --geometry 320x270 --state-features --json <work>/proof.json
python scripts/run_temporal_experiment.py <work>/fused.onnx --geometry 320x270 --state-features --directory <quality-cache> --clips conversation animated-leaves aerial-landscape ui-horizontal --json <work>/quality.json
```

Acquire the separately licensed spatial model first. Ordinary application CLI
model registry does not expose these failed candidates; the research scripts
explicitly register their contracts. All checkpoints and prepared datasets remain
ignored. Preparation receipts preserve original CRLF definition hashes alongside
published LF hashes; parsed settings are identical. Original small-weight training
reproduced exactly on the recorded stack; cross-platform identity is not promised.
See [third-party notices](../THIRD_PARTY_NOTICES.md) for source terms.
