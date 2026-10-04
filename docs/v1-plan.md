# v1 engineering gates

This is a temporary working record, not a release promise. Published v0.4 stays
the default until the stronger quality gates pass. Earlier tags remain immutable.

## Reproduced baseline (2026-10-04)

- Clean `886e23fdcc583066b91c4a5e74bbef0afb9114e7`, exact v0.4 runtime fingerprint.
- 116 ordinary, 15 NPU and 4 video hardware tests passed separately.
- One original 120-second fixture: 3,600 delivered frames, 43,200 neural tile calls;
  strict QNN/no CPU kernels, D3D11VA and QCOM hardware AV1 evidence preserved.
- About 32.1 FPS, minimum rolling ten-second rate 31.1 FPS; battery 90%, Balanced,
  saver off, background conditions uncontrolled. This is not a new AC headline.
- Median postprocessing 16.75 ms, inference 10.87 ms; forensic output audit 72.5 s.
- [Raw baseline](../benchmarks/v0.5/baseline-v0.4.json).

## A / v0.5 — quality first (active)

1. Predeclare a diverse legal corpus and development/holdout split before candidate
   scoring. Preserve source hashes, native references, crop/time, degradation,
   color format, preparation tool hash and codec settings. Keep media outside git.
2. Include faces, motion, city/nature detail, foliage, animation, low light, pans,
   texture and text. Separate synthetic stress tests from filmed content.
3. Compare bicubic, published ESPCN and FSRCNN, then multiple small candidates.
   Start with Qualcomm QuickSRNet small/medium: official x2 checkpoints and BSD-3
   model attribution are available. No cloud inference or extra runtime dependency.
4. Independently verify exports; strict QNN proof precedes timing/quality claims.
   Test full RGB and documented luminance adaptation without pretending they are identical.
5. Compare unblended, fixed scalar, frequency/adaptive and residual approaches.
   Select on development data, lock settings, evaluate untouched holdout once.
6. Report paired PSNR/SSIM/VMAF, VMAF NEG as a secondary anti-sharpening check,
   and reference-based temporal residual diagnostics. Inspect output manually.
7. Release only if aggregate PSNR, SSIM and VMAF all beat bicubic, temporal behavior
   has no severe regression and measured neural cost leaves a credible 30 FPS budget.

## Later gates — sequential, not concurrent implementation

- v0.6: profile CPU phases, reduce postprocessing <=8 ms, preserve quality; move
  forensic audit behind `--verify-full` with meaningful streaming/lightweight checks.
- v0.7: temporal candidate research, scene-cut reset, bounded state, measured
  stability/quality and 30 FPS; retain spatial default if temporal does not win.
- v0.8: setup/doctor, fixed transparent presets, progress/errors and install docs.
- v0.9: meaningful format matrix, 150+ ordinary tests, hardware checks, >=10-minute
  runs (prefer one 30-minute run), truthful hardware matrix and regression evidence.
- v1: three >=10-minute sustained primary trials, all quality/product/reliability
  gates, clean license/security review, fresh artifacts, main/tag CI and verified release.

## Hypotheses and decisions

- Existing delivered quality mixes color paths/chroma/interpolation/codec effects.
  Expanded comparisons must control these and disclose any unavoidable differences.
- QuickSRNet's small plain Conv/Clip/pixel-shuffle graph is a plausible candidate;
  official timing on another device is not evidence for this laptop.
- Do not add complexity or change the default based on a single example or metric.
- No power, temperature or accelerator utilization claim without actual telemetry.
- QuickSRNet neutral-Y small/medium exports match independent PyTorch weight
  extraction exactly; integrated small export max error 1.32e-6 against source
  RGB-on-repeated-Y reference. Both pass strict QNN and DirectML hardware tests.
- Early expanded clips expose low-frequency luminance bias in the neutral-Y
  delivered path: unblended small gains VMAF/SSIM, but loses 5.3 dB PSNR on one
  animation clip. Native-Y pixel inspection confirms a roughly +2.4 coded-Y
  mean error on its first frame. This is not hidden or removed from the corpus.
  Investigate scalar fusion and frequency-separated residual correction on
  development data; holdout remains untouched. Do not choose by VMAF alone.
- Follow-up isolated the cause to a raw NV12 color metadata boundary, not the
  network: software/hardware decoded bytes match; pre-encode neural MSE 2.50
  versus cubic 4.05 on the first animation frame. Tagging the raw encoder input
  BT709 reduces delivered neural MSE from 15.56 to 3.14 and mean error from
  +2.43 to +0.05 coded-Y levels. Preserve tags on raw input and output; rerun
  the full paired suite. Earlier untagged results remain investigation evidence,
  not release quality evidence. Global/frequency correction is not justified
  merely to compensate for an avoidable color conversion bug.
- Corrected 12-clip development mean, unblended small-Y: bicubic
  37.1101 dB / 0.961781 SSIM / 84.9652 VMAF, candidate
  39.3049 dB / 0.971336 / 91.3923. NEG gains 6.4686; temporal residual
  diagnostic increases 2.4% on average, with individual ratios below 1.065.
  This is development evidence only, not a holdout/release pass. Same-size
  delivered crops of faces, text, animation, forest and night detail reviewed.
  Strength 0.5 loses mean PSNR/SSIM/VMAF versus unblended small-Y; 0.75 and
  additional model comparisons are still running. Do not tune on holdout.
- Publication checks: checkpoint globals constrained; source hashes precede
  deserialization; exact ZIP member streamed to an atomically published verified
  file; manifest paths validated; FFmpeg subprocesses use argument arrays.
  The initial dependency audit identified old local pip 23.2.1. Updated this
  environment's installer to official 26.2.1; application runtimes unchanged.

## Current official sources checked

- [Windows ML catalog/Python registration](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers)
- [QNN operators and configuration](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
- [Qualcomm QuickSRNetSmall source and checkpoint](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/quicksrnetsmall/model.py)
- [QuickSRNet paper](https://openaccess.thecvf.com/content/CVPR2023W/MobileAI/papers/Berger_QuickSRNet_Plain_Single-Image_Super-Resolution_Architecture_for_Faster_Inference_on_Mobile_CVPRW_2023_paper.pdf)
- [Netflix VMAF and enhancement-gain caveat](https://github.com/Netflix/vmaf)
