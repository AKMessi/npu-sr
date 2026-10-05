# Performance engineering record

Measurements apply to declared configurations and input/power conditions.
They do not isolate a code speedup unless the comparison actually does so.

| Work | Baseline / bottleneck | Decision | Measured outcome |
|---|---|---|---|
| v0.5 quality/model selection | Delivered v0.4 neural VMAF lost to bicubic on its original small corpus | Expanded/frozen 24 clips; QuickSRNet Small neutral Y; remove fixed 0.5 blend after scalar trials | Recomputed paired protocol: +2.00 dB PSNR/+0.00912 SSIM/+5.95 VMAF; 52.63 FPS median on three 120s sources |
| v0.6 final audit | A full-file decode nearly doubled command time | Bounded packet cadence/count audit by default; opt-in full audit | Default final check ~0.2s versus 33.9s full audit on one 120s output; agreement on 3,600 frames |
| v0.6 CPU stages | Blend/chroma/postprocessing had been ~15.7ms in the old realtime profile | Selected unblended model removes bicubic Y fusion; reuse planes, quantify chroma/quantization | New profile postprocessing p50 7.54–7.73ms; weights, power/input conditions differ from v0.4 |
| Temporal research | Independent image model may flicker | Small early-fusion/recurrent residuals, correct halo, NPU teacher, rounding-aware loss | Insignificant temporal gains or spatial regressions; reject default, no v0.7 release |
| Temporal rectangular graph | Twelve padded square calls | Six useful 320×270 cores, matching full-context CPU semantics | Model-only ~18.6→12.4ms, identical four-clip quality; research candidate still fails quality gate |
| v0.8 productization | Clone scripts/provider/tool setup required technical steps | Packaged setup/proofs, capability diagnostics, persistent contexts/progress | Fresh wheel proof passed; AC median 53.75 FPS on three 180s sources; different conditions do not establish isolated gain |
| v0.9 reliability | Last-8,192 timing window cannot represent long runs | Fixed all-frame histograms, start/end power, memory windows, audio offset/container proofs | Three actual 678–682s processing trials: median 53.05 FPS; min rolling 51.4; all 108,000 frames accounted for |

The current CPU postprocessing is about 7.1ms/frame and QNN calls about 8.9ms
for the primary profile. Quantization/chroma are still CPU work. No native extension,
zero-copy claim, accelerator utilization, watts or temperature estimate is added.
Frame handoff includes queue residence; enhancement time is not display latency.

Sources: [v0.5](../benchmarks/v0.5/summary.md),
[v0.6](../benchmarks/v0.6/summary.md), [temporal research](temporal-model.md),
[v0.8](../benchmarks/v0.8/summary.md), [v0.9](../benchmarks/v0.9/summary.md).
The detailed experimental record is [development history](development-history.md).
