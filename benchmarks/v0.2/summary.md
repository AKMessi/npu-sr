# v0.2 image benchmark validation

Validation is in progress locally. This file will be finalized with the complete
measured matrix before v0.2 is pushed or released.

## Measured model exploration

On the fixed five-image BSDS300 research subset, Pillow bicubic LR generation,
full-range Rec.601 luminance, two HR pixels shaved for SR:

| Method | Mean PSNR (dB) | Mean SSIM |
| --- | ---: | ---: |
| Bicubic | 27.9475 | 0.839886 |
| ESPCN | 28.3141 | 0.846285 |
| FSRCNN-small | 28.2484 | 0.844477 |
| FSRCNN | 28.4793 | 0.851679 |
| LapSRN | 28.2165 | 0.844115 |

DnCNN sigma 25, clipped Gaussian noise seed 2026, gray references, no border shave:
noisy 20.5805 dB / 0.388098 SSIM; NPU denoised 28.1999 dB / 0.817886 SSIM.
These are subset results, not canonical MATLAB BSD100 scores. Images are not shipped.

## Measured optimization probes

These exploratory single-session probes used 10 steady samples after two warmups;
they are not the final three-trial headline matrix.

| Before | Bottleneck | Change | After |
| --- | --- | --- | --- |
| 49.5 ms / 40 calls | 540p ESPCN tile inference | Core 128 → 256 | 35.9 ms / 12 calls |
| 72.3 ms | 540p RGB reconstruction | Quantize planes in place | 44.4 ms |
| 118.0 ms | Full warm 540p image processing | Reconstruction change | 90.5 ms |
| 1808 ms | QNN graph session construction | Embedded context reuse | 259 ms |

Strict QNN proof passed for every candidate; cached contexts also passed proof.
DirectML strict proof passed. QNN GPU failed execution (6999), so GPU measurements
use DirectML. Power and accelerator utilization are not measured. The laptop is
running on battery with the Balanced power scheme; background work is uncontrolled.

The image totals above exclude disk IO and startup. No video throughput or
real-time claim follows from these measurements.
