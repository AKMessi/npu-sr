# v0.2 image benchmark validation

Validated on Snapdragon X Plus X1P-42-100, Windows ARM64 build 26200,
driver 30.0.219.1000, QNN package 2.2480.53.0 and ORT 1.25.2.
See [environment](environment.json), [performance](performance.json) and
[quality](quality.json). No dataset images or model weights are published.

## Performance

The final matrix contains 60 configurations: four models, CPU/DirectML GPU/QNN
NPU and five input sizes (256×160 through 1920×1080). Each configuration has
three trials, five measured images per trial after three warmups. Headline values
are the median of trial medians. Warm total includes preprocessing, tile copying,
validated inference, stitching and RGB reconstruction; it excludes startup and
disk IO. PNG encoding is a separate in-memory probe. These are images, not video FPS.

960×540 warm complete-image totals (SR produces 1920×1080; denoising stays 960×540):

| Model | CPU | Adreno GPU / DirectML | Hexagon NPU / QNN |
| --- | ---: | ---: | ---: |
| espcn-x2 | 193.0 ms | 122.0 ms | 108.1 ms |
| espcn-x2-256 | 153.0 ms | 126.9 ms | 89.4 ms |
| fsrcnn-x2 | 162.4 ms | 174.3 ms | 147.7 ms |
| dncnn-25 | 2548.1 ms | 1293.5 ms | 413.2 ms |

ESPCN core 256 reduces NPU calls from 40 to 12 at this size: inference is
36.2 ms versus 50.6 ms for core 128, while total falls from 108.1 to 89.4 ms.
This still misses a 30 FPS full-frame budget. DnCNN's heavier workload gains
more from the NPU: about 6.17× versus CPU for the measured complete-image total.
DirectML is a genuine GPU comparison, but is slower than QNN for these workloads.

Every NPU model/session passed strict proof with QNN kernels and no CPU kernels;
every GPU model/session passed equivalent DirectML proof on the Adreno adapter.
Cached QNN contexts were independently verified. Startup and proof are recorded
separately from steady inference. Cumulative process peak working set is an OS
measurement, not an isolated model or accelerator memory allocation.

The final performance and quality runs used clean commit
`3e1390a7c4d1f32b2cb71def333981a1be70860c`; the subsequent release commit adds
results, documentation and a missing-model error message. Performance JSON also
retains 30 earlier FSRCNN-small/LapSRN candidate configurations with explicitly
weaker source provenance; they are exploratory, not the headline validation.

## Quality

SR uses five pinned BSDS300 test photographs, even-size HR crops, Pillow bicubic
LR generation, full-range Rec.601 luminance and two HR pixels shaved. Arithmetic
mean per-image NPU metrics:

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Bicubic | 27.9475 | 0.839886 |
| ESPCN (both tile sizes) | 28.3141 | 0.846285 |
| FSRCNN-small | 28.2484 | 0.844477 |
| FSRCNN | 28.4793 | 0.851679 |
| LapSRN | 28.2165 | 0.844115 |

FSRCNN is the improved quality option, modestly ahead of ESPCN and bicubic.
FSRCNN-small and LapSRN are retained as reproducible comparisons but are not
recommended upgrades: both score below ESPCN on this subset. The original
LapSRN LeakyRelu export failed QNN compilation; equivalent channel PRelu succeeds.
All final exports pass strict QNN proof. QNN GPU execution failed (error 6999),
so the working GPU implementation uses DirectML instead.

DnCNN uses five author-designated BSD68 test images, gray clean references,
clipped Gaussian noise sigma 25/255, seed 2026, no border shave:

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Noisy input | 20.5805 | 0.388098 |
| DnCNN / QNN NPU | 28.1999 | 0.817885 |

These are explicit small-subset results, not canonical MATLAB BSD100/BSD68
scores or claims about arbitrary camera noise. CPU and GPU quality results are
also in JSON. This protocol establishes aligned reference comparisons; it does
not establish generalization across unseen datasets.

## Optimization probes

Earlier single-session probes used ten steady samples after two warmups;
they are not the final three-trial matrix.

| Before | Bottleneck | Change | After |
| --- | --- | --- | --- |
| 49.5 ms / 40 calls | 540p ESPCN inference | Core 128 → 256 | 35.9 ms / 12 calls |
| 72.3 ms | 540p RGB reconstruction | Quantize planes in place | 44.4 ms |
| 118.0 ms | Warm 540p image | Reconstruction change | 90.5 ms |
| 1808 ms | QNN graph session construction | Embedded context reuse | 259 ms |

Pixel-parity tests cover the reconstruction change. Cache startup is workload
and runtime dependent; the matrix supplies fresh context and cache-hit phases.

## Limits and reproduction

The performance run was on battery, Balanced power scheme, battery 53% → 41%.
Background processes were uncontrolled. Power, accelerator utilization and
temperature are **not measured**. No broad energy-efficiency or video throughput
claim follows. Repeat on the same model/input/software and power configuration.

See [benchmark commands and metric definitions](../../docs/benchmarking.md).
Local validation: 71 ordinary tests and 15 separate hardware tests passed;
hardware tests cover six SR/denoising models, strict QNN/DirectML, numerical
agreement, repeated inference, denoising quality and context reuse.
