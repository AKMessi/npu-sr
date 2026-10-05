# Quality and its limits

The default realtime model is a neutral-Y adaptation of QuickSRNet Small, chosen
using twelve development clips. Twelve other clips were originally held out of
selection. The frozen 24-clip corpus is rerun for regressions without retuning;
repeated holdout evaluation is not newly unseen test data.

## Ground truth

References are high-quality 1080p source frames, not enlarged low-resolution
inputs. Source/license/hash, exact times, retiming, crop/padding, input degradation
and output encoding are declared in [the corpus](../benchmarks/corpus-v1.json).
Blender films, NASA scenes and project-generated UI/texture sequences cover faces,
motion, foliage, text, animation, low light and compression stress. Films/datasets
remain outside git. Rights and attribution are recorded per source. NASA usage
guidelines include exceptions; no blanket public-domain or endorsement claim is made.
References are lossless FFV1, with declared limited-range BT709 and active film
regions. The corpus is 24 two-second clips, not a full-movie perceptual study.

The controlled degradation uses antialiased bicubic downscale to 540p and H.264
CRF10, with declared CRF26 stress cases. Both reconstruction methods use QCOM AV1
at the same 8M setting. This does not represent every real streaming degradation.

## Metrics

- Native decoded coded Y, peak 255; predeclared active ROI excludes film padding.
- PSNR derives from the clip's mean sampled MSE; every third frame, two-pixel shave.
- SSIM uses an 11×11 Gaussian, sigma 1.5, population covariance, the same sampling/shave.
- VMAF 0.6.1 and enhancement-limited NEG 0.6.1 evaluate every frame of the declared ROI.
- Average all clips equally, retain every per-clip result and also group by source,
  category and original split. A VMAF-saturated synthetic clip remains included.

The [v0.9 recheck](../benchmarks/v0.9/quality.json) gives:

| Method | PSNR dB | SSIM | VMAF |
|---|---:|---:|---:|
| Bicubic + AV1 | 37.8653 | 0.961910 | 85.6160 |
| QuickSRNet Small / QNN + AV1 | 39.8693 | 0.971032 | 91.5656 |
| Difference | +2.0040 | +0.009123 | +5.9496 |

These are luma-focused measurements, not complete color/perceptual correctness
or a promise for all content. Colored input is not identical to the original RGB
QuickSRNet: the export evaluates the network at R=G=B=Y and mixes output RGB to Y.
CPU chroma resize is separately visible in delivered output. See [models](models.md).

## Temporal evidence

Two project diagnostics compare consecutive reconstruction-error changes to the
reference: a coarse 256×144 version and a native-resolution version. Neither uses
motion compensation or is a standard perceptual flicker metric. They can reward
smoothing or constant bias and must be read with spatial scores and inspection.

In the frozen recheck, coarse error worsens **2.17%**, while native error improves
**8.93%**. Both are reported. Hash-verified six-consecutive-frame native crops of
faces, foliage, aerial texture, a night scene and text show sharpening/blur tradeoffs
without severe instability in those inspected regions. This is limited crop
inspection, not full playback or a blinded study; artifacts elsewhere remain possible.

Small trained temporal corrections did not pass their development gates, so none
became the default. See [the rejected temporal experiments](temporal-model.md).
The production model still processes frames independently and may ring or flicker.

## Presets and reproduction

Realtime uses Small/burst/NV12/AV1; balanced uses Small/sustained/NV12/H.264;
quality uses Medium/sustained/NV12/AV1. A performance request is not an energy
measurement. Medium had modestly higher mean development scores under equal AV1
encoding, with higher neural compute cost; that is not a universal ranking.
Use identical codecs/settings when comparing models. [Settings and speeds](realtime.md).

Prepare with `scripts/download_quality_corpus.py`; run `run_quality_corpus.py`
for development/holdout and aggregate using `summarize_quality_corpus.py`.
The scripts require exact input/reference/model hashes and complete paired results.
Weights, software changes and different degradation require fresh validation.


The paired numeric gate rejects missing/duplicate comparisons, mismatched
references/frames/ROI, nonfinite metrics, CPU fallback and neural-frame bypass.
VMAF uses a shared ordinal CFR clock and expected frame count; no repeated final
reference frame is allowed. `evaluate-video` retains its legacy RGB protocol for
compatibility, while the frozen corpus scripts use the native-Y evaluator.
Absolute earlier RGB-protocol scores must not be treated as model improvements
against the later native-Y protocol. Future tuning requires fresh holdout material.
