# Windows ML and Qualcomm QNN

```text
Python
  → ONNX Runtime (Windows ML wheel)
  → QNN Execution Provider (selected NPU device, HTP backend)
  → Qualcomm AI runtime
  → Hexagon NPU
```

Windows ML supplies the provider catalog and package bootstrap. It prepares the
QNN library before ORT creates the inference session. It is not a second inference
pass. Python projections are published as `wasdk-*` packages but currently import
under `winui3.*`; use the pinned tested versions in pyproject.toml.
We explicitly install the ApplicationModel, Foundation, and Foundation.Collections
projections used by these APIs. Microsoft's convenient `[all]` extra recursively
pulls many unrelated Windows namespaces; the minimal set was separately installed
and verified with `doctor` and strict NPU upscaling in a fresh environment.

## Bootstrap and registration

`qnn.py` keeps the bootstrap, catalog and provider alive until process exit.
The installed App Runtime must satisfy the Python bootstrap's matching runtime
family/minimum. For our 2.3.0 binding, the validated installed runtime is 2.5.1.0.
Python's ORT environment is separate from Windows ML's native ORT environment.
Therefore `EnsureAndRegisterCertifiedAsync` / `TryRegister` alone are insufficient
for Python. Follow Microsoft's [Python registration instructions](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/register-execution-providers).

The implementation:

1. Calls the Windows App SDK `initialize()` bootstrap without interactive UI.
2. Gets `ExecutionProviderCatalog.get_default()` and calls `find_all_providers()`.
3. Finds `QNNExecutionProvider` and calls `ensure_ready_async().get()`.
4. Checks the ready result; reports diagnostic text/HRESULT on failure.
5. Calls `ort.register_execution_provider_library(provider.name, provider.library_path)`.
6. Enumerates `ort.get_ep_devices()` and selects `QNNExecutionProvider` with
   `d.device.type == ort.OrtHardwareDeviceType.NPU`.
7. Uses `SessionOptions.add_provider_for_devices` before creating the session.

Catalog acquisition follows [Microsoft's specific-provider installation flow](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers).
The device API matches Microsoft's [explicit selection guidance](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers).
The shipped ORT Python binding uses `d.device`, although some documentation snippets
say `hardware_device`; the installed API was inspected and validated locally.

## Compatibility and proof

QNN distinguishes CPU, GPU, and HTP backends. We select both hardware type NPU
and `backend_type=htp`; a QNN provider name alone would be ambiguous.
`enable_htp_fp16_precision=1` expresses the intended floating precision; modern
QAIRT versions use FP16 on supported HTP hardware regardless of that setting.
Current [QNN EP documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
lists Conv, Relu, DepthToSpace, Tanh and fixed-shape requirements, and supports
floating as well as quantized HTP models. Older ORT QNN documentation may describe
quantization as mandatory. That is not the behavior of the tested catalog package.

The strict path disables CPU EP fallback using `session.disable_cpu_ep_fallback=1`,
then disables the Python session's fallback feature. It requires a valid proof
inference and ORT kernel events exclusively attributed to QNN. For ESPCN x2 we
observed one fused QNN kernel. This establishes neural graph execution through the
selected HTP device, with ordinary host/runtime work still using CPU. It is not a
claim about every CPU instruction or NPU utilization percentage.

If a new model has unsupported operations, compilation or proof fails. There is
no relaxed partial-assignment NPU mode. Auto mode explicitly reports the failure
and selects CPU for the whole model. Use Task Manager NPU activity as a secondary
check during longer benchmark runs, not as the only evidence.

## Context caching

The documented ORT QNN context mechanism uses `ep.context_enable=1` and optionally
`ep.context_file_path`; it produces an EPContext ONNX plus a context binary.
v0.2 provides an opt-in local cache: `--cache-dir .cache/qnn`. It enables embedded
contexts (`ep.context_embed_mode=1`), so one generated ONNX contains its binary.
Cache keys include model SHA256, QNN package, ORT, architecture, processor class,
NPU driver and backend/precision options. Integrity metadata is published only
after strict proof. Every cache hit gets another strict proof; a failed context
is invalidated and the original graph is recompiled. CPU fallback remains disabled.
The first cache setup also queries hardware; startup reports this separately from
steady inference. No context binary belongs in git; `.bin` and `.onnx` are ignored.

## GPU comparison

`--device gpu` requires DirectML, sequential session execution and disabled memory
patterns. It performs the same successful-output and profiling checks, accepting
only `DmlExecutionProvider` kernels. It never retries on CPU. This is distinct from
QNN HTP inference: the GPU is not the NPU. QNN GPU execution failed locally; see
[research notes](research-notes.md). DirectML availability varies across installs.

## Distribution

Do not copy QAIRT SDK folders, QNN DLLs or NPU drivers into the repository.
The application obtains the provider through Microsoft's catalog. Microsoft and
Qualcomm packages retain their own licenses. Windows driver updates go through
Windows Update or the laptop vendor. No Qualcomm AI Hub account is required.


## Performance modes in v0.4

The installed provider accepts `htp_performance_mode` via Windows ML device
session options. Video exposes default, burst and sustained_high_performance;
the selected request is recorded. Cache identities include nondefault modes, so
a context from another configuration is not silently reused. Every cache load
still performs strict assignment/profile proof. Mode names do not establish power
consumption; actual throughput is measured independently. No provider/runtime
upgrade was needed for the validated path.
