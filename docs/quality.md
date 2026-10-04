# Quality evaluation

The v0.5 protocol compares delivered video with a native high-resolution reference,
not with its enlarged low-resolution input. The corpus and split were frozen
before expanded scoring. Development selected QuickSRNet Small Y; its hash and
settings were locked before evaluating holdout. Future tuning needs new holdout
material rather than reusing these clips as supposedly unseen data.

## Sources and preparation

[Corpus declaration](../benchmarks/corpus-v1.json) records source URL, source
license/usage terms, SHA256, exact time/crop, split and categories. Media is
acquired in the application cache. Blender CC BY 3.0 permits attributed
adaptations. NASA footage remains subject to its media usage guidelines;
no endorsement or blanket public-domain claim is made. Original UI/texture
stress tests are distinct from photographed detail. Scripts verify hashes,
extract only a declared ZIP member and avoid enlarging low-resolution references.

References use lossless FFV1, 1920×1080 limited-range BT709. Letterboxed material
is padded and scored only in its active ROI. Input is 960×540 bicubic downscale
and H.264 CRF 10; declared compression-stress cases use CRF 26. Bicubic and neural
outputs use the same QCOM AV1 hardware encoder and 8 Mbit/s target.
Known color range/matrix are preserved on both raw encoder input and output.

## Metrics

- PSNR: native decoded coded-Y, peak 255, two-pixel shave, mean sampled MSE per clip.
- SSIM: native Y, 11×11 Gaussian, sigma 1.5, population covariance, valid support.
- PSNR/SSIM sample every third frame; VMAF 0.6.1 and 0.6.1 NEG sample every frame.
- VMAF uses an active ROI and shared ordinal CFR clock; no repeated last frame.
- Temporal difference error: absolute change in reconstruction residual between
  consecutive frames, normalized by 255, ROI reduced to 256×144. Every frame is
  used. This is a project diagnostic without motion compensation, not a standard
  perceptual metric; it can reward smoothing and cannot establish absence of flicker.

Equal clip averages include all declared cases. Per-clip, per-source, per-category,
development and holdout scores are retained. The paired numeric gate rejects
missing/duplicate comparisons, mismatched references/frames/ROI, nonfinite scores,
CPU fallback and neural-frame bypass. Manual artifact review and sustained
performance are additional gates. Saturated VMAF values are not excluded.

## Reproduce and interpret

See [v0.5 results and commands](../benchmarks/v0.5/summary.md).
`evaluate-video` retains its earlier RGB protocol for compatibility; the expanded
corpus scripts explicitly call the native-Y evaluator. Absolute v0.4 and v0.5
scores are therefore not directly comparable. New-model gains are paired within
the same declared protocol. Model-only inference throughput is not video FPS.

The 24 two-second clips cover diverse cases but are still a limited corpus.
Six-consecutive-frame contact sheets and same-size delivered crops were inspected;
this is not a comprehensive playback study. Longer motion sequences and temporal
models are future validation work. No universal enhancement guarantee is made.
