# Models

`npu-sr models list`, `models info IDENTIFIER` and `models download IDENTIFIER`
describe and acquire static ONNX exports. `--model` also accepts the v0.1 ONNX
path plus adjacent manifest syntax. Hash and registry contract checks happen
before session creation. No model weights are included in git or wheels.

| Identifier | Task | Useful core | Halo | Notes |
| --- | --- | --- | --- | --- |
| espcn-x2 | 2x SR | 128 | 4 | Exact v0.1 baseline and default |
| espcn-x2-256 | 2x SR | 256 | 4 | Same weights, fewer ORT calls |
| fsrcnn-small-x2 | 2x SR | 256 | 4 | Small subpixel FSRCNN candidate |
| fsrcnn-x2 | 2x SR | 256 | 6 | Modest quality improvement in the tested subset |
| lapsrn-x2 | 2x SR | 256 | 12 | Residual Laplacian candidate; slower, not recommended over FSRCNN |
| quicksrnet-small-y-x2 | 2x SR | 256 | 4 | v0.5 realtime default, neutral-RGB luminance adaptation |
| quicksrnet-medium-y-x2 | 2x SR | 256 | 7 | Higher quality, roughly twice small model compute |
| dncnn-25 | Denoise | 256 | 17 | 17 layers; luminance, Gaussian sigma 25/255 |

All use normalized full-range luminance, float32 ONNX, batch one, and opset 13.
Input size is core + twice halo; SR output is twice input size. QNN HTP executes
float math in FP16; CPU/DirectML results may differ slightly. Chroma and alpha
use CPU bicubic resizing for SR and remain at native size for denoising.

DnCNN does not remove color noise and is not a general real-world noise model.
It can smooth details or fail on noise unlike its training distribution.

The original ESPCN is retained for compatibility. FSRCNN improves average PSNR
and SSIM modestly on our five natural images; it is slower than ESPCN on this
NPU. FSRCNN-small and LapSRN remain reproducible comparison candidates, rather
than defaults. The selection is specific to the recorded workload, not a claim
that any architecture is universally best.

On the five-image research subset, initial strict NPU evaluation measured mean
Y PSNR/SSIM of 27.95/0.8399 for bicubic, 28.31/0.8463 for ESPCN,
28.48/0.8517 for FSRCNN, 28.25/0.8445 for FSRCNN-small and 28.22/0.8441 for
LapSRN. These figures are specific to our Pillow degradation; the release JSON
records per-image results and the exact methodology. They do not establish a
standard benchmark ranking.

Sources, immutable revisions, licenses and SHA256 values are in
[`acquire_models.py`](../src/npu_sr/acquire_models.py) and
[third-party notices](../THIRD_PARTY_NOTICES.md). Exports transpose TF kernels,
fold biases and replace equivalent activations without retraining. The DnCNN
reader only accepts its known legacy float storage layout and an explicit
allowlist of pickle globals; acquisition verifies the exact hash first.


The v0.5 realtime preset selects the small QuickSRNet adaptation after locked
holdout evaluation. Image commands retain their existing ESPCN default; explicit
`--model` selects any registry option. Balanced uses the same Small model through NV12 with a sustained performance
request; quality uses Medium through NV12. Realtime uses Small with burst.
These requests do not establish measured power savings. Medium has slightly
better development quality and higher compute cost, not a universal ranking.

## QuickSRNet luminance exports

`quicksrnet-small-y-x2` and `quicksrnet-medium-y-x2` are reproducible neutral-RGB
luminance adaptations of Qualcomm's BSD 3-Clause x2 checkpoints. Download with
`npu-sr models download <identifier>`. The small graph has three 32-channel
Conv/Clip feature stages, a 12-channel Conv/Clip output, luminance phase mixing
and DepthToSpace. Medium adds three feature stages. Core size is 256, with halo
4 for small and 7 for medium. FP32 files use QNN HTP FP16 computation.

The transformation evaluates the RGB network at R=G=B=Y, then converts its
clipped RGB output to Y. It avoids RGB runtime traffic but does not model colored
input identically to the original network. A NumPy reference test independently
checks convolution, Clip order and RGB pixel-shuffle phases against ONNX CPU.
No PyTorch dependency is added. Strict QNN tests still apply to every candidate.

Small has 22,860 parameters in the original RGB checkpoint. Folding repeated-Y
input leaves 22,284 learned values, plus 48 fixed luminance mixing coefficients
and two Clip constants. Medium has 50,604 original parameters. Model provenance,
export hashes, comparative results and limitations are in the
[v0.5 report](../benchmarks/v0.5/summary.md).
