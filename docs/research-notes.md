# Research notes

## v0.2 — checked 2026-10-04

- [Microsoft's Python catalog flow](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers)
  still requires explicit library registration in Python's ORT environment. The
  working v0.1 bootstrap and catalog implementation is retained.
- [Current QNN provider documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
  supports static floating-point HTP graphs; QAIRT 2.35+ uses FP16 math on supported
  chips. The older main ORT page still contains a conflicting quantization-only
  statement. Local strict execution determines compatibility here.
- Conv, PRelu, Relu, Add, Sub and DepthToSpace are suitable candidates, but an
  operator listing does not prove a particular graph works. FSRCNN, LapSRN and
  DnCNN exports passed actual strict QNN profiles on this laptop.
- LapSRN's LeakyRelu export was rejected by the installed QNN package. Equivalent
  channel slopes through PRelu work. Even 4x4 TF SAME kernels require asymmetric
  padding; an independent TF graph interpreter tests this conversion.
- QNN GPU graphs compiled but execution failed with code 6999 for ESPCN,
  FSRCNN-small and DnCNN. No successful QNN GPU claim is made.
  [DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
  succeeds with sequential execution and memory patterns disabled. Strict
  profiles establish DirectML kernel execution, with no CPU kernels.
- Embedded QNN EPContext files work with `ep.context_enable=1`,
  `ep.context_embed_mode=1` and an explicit local path. Each loaded context still
  receives a proof run. Cache keys include model, ORT, catalog package, processor
  class, architecture and NPU driver version. Compiled files are never shipped.
- [FSRCNN upstream](https://github.com/Saafke/FSRCNN_Tensorflow) and
  [LapSRN upstream](https://github.com/fannymonori/TF-LapSRN) provide Apache 2.0
  weights. FSRCNN uses a subpixel layer, unlike the paper's transposed convolution.
  [KAIR](https://github.com/cszn/KAIR) provides MIT DnCNN weights with batch
  normalization already folded. Acquisition uses pinned source hashes; no
  TensorFlow/PyTorch runtime dependency is added.
- [Berkeley BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
  explicitly permits non-commercial research and educational downloads. The
  first five sorted test images provide a fixed natural-image research subset.
  Dataset images are not redistributed. Scores use disclosed Pillow bicubic
  degradation and must not be described as canonical MATLAB BSD100 scores.
- Windows process CPU time and working set are available through OS APIs. Peak
  working set is cumulative for the benchmark process. Accelerator utilization,
  temperature and power are not measured. TOPS does not imply watts or throughput.

Video API research and implementation belong to the subsequent gated releases.
