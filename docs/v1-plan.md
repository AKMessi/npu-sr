# v1 engineering gates

This is a temporary working record, not a release promise. The v0.5 realtime selection follows the stronger quality gates. Earlier tags remain immutable.

## Reproduced baseline (2026-10-04)

- Clean `886e23fdcc583066b91c4a5e74bbef0afb9114e7`, exact v0.4 runtime fingerprint.
- 116 ordinary, 15 NPU and 4 video hardware tests passed separately.
- One original 120-second fixture: 3,600 delivered frames, 43,200 neural tile calls;
  strict QNN/no CPU kernels, D3D11VA and QCOM hardware AV1 evidence preserved.
- About 32.1 FPS, minimum rolling ten-second rate 31.1 FPS; battery 90%, Balanced,
  saver off, background conditions uncontrolled. This is not a new AC headline.
- Median postprocessing 16.75 ms, inference 10.87 ms; forensic output audit 72.5 s.
- [Raw baseline](../benchmarks/v0.5/baseline-v0.4.json).

## A / v0.5 — released

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

## Locked selection results

- All 24 clips: +2.004 dB PSNR, +0.009123 SSIM, +5.9496 VMAF versus bicubic.
- Holdout alone: +1.813 dB, +0.008691 SSIM, +5.4721 VMAF.
- VMAF NEG improves; temporal residual diagnostic increases 2.17% on average,
  maximum individual ratio 1.064. Preserve this limitation and inspect sequences.
- Three sustained trials: 52.63, 52.66, 52.15 FPS, 3,600 frames each, zero lost
  frames, strict QNN and hardware codecs. Postprocessing medians 7.07/7.00/7.09 ms.
- Removing the blend was selected for quality and also removes its CPU cost.
  No native extension is justified to meet the <=8 ms postprocessing goal.
- Whole-command times 155.58/149.35/151.00 seconds remain dominated by the final
  audit. v0.6 must separate lightweight validation from explicit forensic decode.

## B / v0.6 — released

- Three trials: 48.73/48.34/49.90 streaming FPS, whole function times
  82.63/78.65/76.13 seconds; 3,600 frames/43,200 neural calls each.
- Default output checks 0.18–0.20 seconds; full decoded audit agrees on all
  3,600 frames and takes 33.91 seconds. No frame-count assumption from nominal FPS.
- CPU post medians 7.72/7.73/7.54 ms. First trial chroma 4.26 ms, quantization
  2.97 ms, finite/fusion check 0.27 ms, copy 0.12 ms. Preserve this profile before
  choosing a native kernel; target is already met without one.
- Three selected delivered regression files are bit-identical to v0.5.
- Packet versus decoded audit, bad cadence, corruption, reordering, process failure,
  timeout and cancellation are tested. Strict QNN and hardware codec tests pass.
- External CLI launch-to-exit measurement: 74.71 seconds for the 120-second
  source. Fresh installation, downloaded artifact hashes, Windows/Linux main
  and tag CI, strict hardware tests and publication review passed.

## C / temporal experiments — not a release pass

- Predeclared 20 training snippets from CC BY 3.0 Blender films, separated in
  time from the original quality corpus; 880 real adjacent-frame patch triples.
  Initial validation was subsequently consumed by the prototype; see revisions below.
- Original 1,028-parameter two-frame bounded residual trained locally for
  2,500 steps, with temporal weight 0.25 and a spatial-only loss control.
  Training is CPU PyTorch; application inference still uses native ARM64 ORT.
- Both exports pass the unchanged strict QNN runtime. Primary CPU/PyTorch
  maximum difference 4.0e-8; CPU/QNN maximum difference 1.67e-4. Previous-frame
  input affects output. Twelve correction calls cost about 4.3 ms, model-only.
- Primary four-clip delivered development probe: VMAF changes -0.074, -0.149,
  -0.059 and -0.187; PSNR changes +0.002, -0.015, -0.047 and +0.049 dB.
  Temporal residual diagnostic improves slightly on three, worsens on text.
  This does not justify replacing the spatial default. Probe only initializes
  previous state on the first frame; no scene-cut handling claim yet.
- Compare the control and revise or reject this candidate before consuming the
  new validation split. Original spatial production path remains unchanged.
- The spatial-loss control also fails to justify a default change. A second
  development experiment adds the teacher's four subpixel output phases as
  input features (six channels total, 1,316 parameters). Loss excludes the
  receptive radius of replicate-padded training feature edges; delivered
  inference uses actual halo features. Same training samples, seed, steps and
  temporal weight 0.25. It gains four-clip development VMAF, with spatial and
  temporal diagnostic tradeoffs; remains an experiment.
- Independent context analysis revealed that fused radius four plus radius
  three requires halo seven. The halo-four probes are superseded. A new frozen
  validation split, separate in time, was prepared for halo seven.
- Four corrected validation clips: +0.143 average VMAF, -0.028 dB PSNR,
  -0.000649 SSIM versus spatial Small. Coarse temporal interval ratio improves
  only 0.14%; native-resolution residual difference worsens about 0.37% overall.
  Neither supports a convincing temporal improvement. No v0.7 release allowed
  from these numbers. Keep the published spatial default unchanged.
- The next development-only experiment increases temporal-loss weight to one
  with the same six-feature architecture, samples, seed and step count. Original
  and revised diagnostics will both be retained; new validation follows only
  after development selection. Do not optimize for whichever metric wins.
- Weight-one development results: mean native-resolution temporal error improves
  only 0.13%; aerial footage gets worse. PSNR deltas are -0.018/-0.052/-0.118/
  +0.075 dB and VMAF +0.082/-0.158/-0.267/+0.546. Still not a temporal pass.
- Training with straight-through limited-range 8-bit output rounding (not INT8
  inference) gives native temporal ratios 0.994/0.960/1.010/0.973. Mean native
  temporal error improves 0.30%, with aerial regression. PSNR deltas +0.012/
  -0.037/-0.097/+0.059 dB; VMAF +0.188/-0.026/+0.116/+0.443. These are four
  development clips, not a new holdout success or a perceptual stability claim.
- Test one width-16 correction (3,780 parameters) with the same rounding-aware
  objective, samples, seed and steps before deciding whether further capacity
  helps. No new validation split is consumed by these development experiments.
- Default ordinary test suite currently passes 207 tests; 27 hardware tests are
  deselected, not implicitly validated by that invocation. Wide-shape export and
  full-context tile tests were added subsequently and pass separately.
- Width-16 rounding-aware correction: development mean +0.026 dB PSNR,
  +0.187 VMAF, -0.000389 SSIM; native temporal error improves only 0.59%.
  Increasing its temporal weight to four loses 0.077 dB/0.000479 SSIM/0.138
  VMAF and worsens native temporal error by 0.39%. Reject that stronger loss;
  do not consume fresh holdout data to rescue unfavorable development results.
- Static current-channel Slice is supported by the installed QNN runtime and
  CPU outputs are identical, but model-only twelve-call medians are about 19.9
  ms versus 18.6 ms without Slice. Keep it out of the production export.
- A 320x270 useful core exactly covers the primary 960x540 profile in six calls,
  compared to twelve 256x256 calls with padded unused areas. Strict QNN passes;
  model-only six-call medians about 12.4 ms. Full-context CPU tile equivalence
  tests pass for both widths/geometries. Initial four-clip delivered quality is
  identical to the square graph; this is not sustained realtime evidence yet.
- Next development experiment uses a strictly proven QNN teacher instead of CPU
  FP32 teacher outputs. Hypothesis: fitting the CPU teacher's subpixel features
  can mismatch HTP FP16 features near final 8-bit rounding boundaries. Reuse the
  hash-verified original training clips and sample coordinates, keep both dataset
  hashes, and measure the disagreement before training. No fresh validation used.

- Native QNN teacher trial: original raw/GT arrays and sampled coordinates are
  identical; baseline output mean normalized disagreement 0.000130, max 0.001953.
  Delivered native temporal error improves 0.94% but SSIM/VMAF decline. No default.
- Actual recurrent enhanced-phase state, two-step unrolled training: 4,356 learned
  parameters, six external channels. Strict QNN passes, six calls about 18 ms.
  Four development clips give +0.017 dB, +0.000005 SSIM, -0.117 VMAF and only
  -0.14% native temporal error. Reject as default; no v0.7 release. Keep all scores.
- Remove failed pretrained temporal checkpoints from the package/model download
  list. Retain explicit research training/export and bounded state implementation;
  test hardware paths with seeded test-only corrections, never as quality evidence.
- Proceed to installation/capability/preset/reliability work. Temporal acceptance
  remains unfulfilled; do not claim a release pass merely to fill a version number.


## D / productization candidate — release pending

- Packaged setup acquires verified Small/Medium and optional pinned native FFmpeg;
  CPU setup performs CPU inference and does not request NPU. System App Runtime
  remains an explicit official prerequisite, not copied DLLs.
- Doctor now reports executed hardware H.264 decode and H.264/HEVC/AV1 encode,
  optional strict DirectML proof, model inventory and JSON. Probe scope is explicit.
- Windows ARM64 one-command video chooses realtime; CPU/non-ARM defaults use
  balanced Small NV12. Explicit legacy model/frame options preserve custom usage.
  Quality now uses the measured Medium NV12 path; no generic power-saving claim.
- Throttled progress reports submitted frames; final throughput includes flush.
  Context reuse is automatic locally and every loaded context still requires proof.
- 245 ordinary / 20 manual NPU+DirectML / 6 video hardware tests passed before
  version promotion. Lint passes; actual setup and doctor proofs pass on AC.
- Freeze candidate code, then repeat three sustained realtime trials, delivered
  regression checks, fresh wheel installation and main/tag CI before a v0.8 tag.
  Do not claim temporal v0.7 acceptance; it remains unfulfilled research.


## E / reliability candidate

- v0.8 released at 871400c8c26f5071c1220588082d959e500e4046; main CI
  37281638670 and tag CI 37281898027 passed Windows/Linux. Downloaded wheel,
  sdist and hash file match tested builds. Final suite 246 ordinary / 20 manual
  NPU+DirectML / 6 video hardware tests.
- v0.9 adds bounded whole-run timing histograms, not just last-8192-frame
  percentiles. 0.05 ms quantile bins, actual means/extrema; overflow is explicit.
  Start/end power snapshots; custom power-plan names/GUIDs are not serialized.
- Real reproduction: source video began at 0.5 s, copied audio at 0.0 s. Old
  output lost that offset. Compensate copied input timestamps by container start
  minus first video timestamp. Trimmed decoded AAC samples match exactly.
- Real AV1/MKV mux failure fixed by extract_extradata bitstream filter. Decoder
  and hardware transform execution still require their existing proofs.
- Actual QCOM AV1 1708x960 request becomes 1712x960. H.264 and HEVC preserve
  1708x960. Encoder proof now encodes/probes three actual frames before processing;
  reject changed geometry/cadence, offer explicit HEVC/H.264/software alternatives.
- Generated matrix exercises H.264/HEVC/AV1 inputs, fractional and integer frame
  rates, 360p through 1080p, portrait, stereo/surround AAC, Opus/MKV, no audio.
  It is correctness evidence, not sustained throughput or every cross-product.
- Prepare a hash-verified CC-BY film loop for 1200 source seconds. Content repeats;
  every application input frame must still be processed. At ~54 FPS this should
  exceed ten minutes of actual processing; source duration alone is insufficient.
- Freeze candidate code after ordinary/hardware/regression checks, then run three
  independent sustained trials without concurrent project training/tests. Gate
  actual >=600 seconds/trial, exact 540p30->1080p30, strict QNN, hardware encode,
  all calls/counts/cadence and bounded queues/working sets. No release claim yet.

- Frozen 7082d61 completed three independent actual 678–682-second processing
  trials at 52.81/53.06/53.05 FPS; full counts/proofs, bounded memory and queues
  pass. Full 24-clip recheck reproduces v0.5 spatial metrics exactly. Native
  temporal diagnostic improves 8.93% while coarse diagnostic worsens 2.17%.
- Strengthen initialization to test the requested container and retain its proof;
  primary MP4 neural/streaming path remains unchanged. Seven hardware video tests
  pass afterward. H.264/HEVC MF-MKV is explicitly unsupported on this stack.
