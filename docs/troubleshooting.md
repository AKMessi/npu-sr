# Troubleshooting

Start with `npu-sr doctor`, then retry the failing command with `--verbose`.
Normal failures show an error and return a nonzero exit code; verbose failures
also show a traceback and ORT logs. Redact personal paths before sharing them.

| Symptom | Action |
| --- | --- |
| Unsupported OS / Windows build too old | NPU mode needs Windows 11 24H2 build 26100+; use CPU on other supported systems |
| ARM64 Python required | Check `python -c "import platform; print(platform.machine())"`; recreate venv with `py -3.12-arm64` |
| Windows ML unavailable / bootstrap fails | Install stable ARM64 Windows App Runtime 2.x ≥2.3.0.0 from the official page; the tested version is 2.5.1.0 |
| Missing native runtime / DLL load failure | Install Microsoft's VC++ ARM64 redistributable; recreate the venv to remove conflicting ORT wheels |
| QNN absent from catalog | Confirm Snapdragon hardware, OS build, OEM firmware and driver updates |
| QNN ready result failure | Read its diagnostic text/HRESULT; check Windows package installation restrictions and network access; retry after updates |
| QNN registered but no NPU device | Update the actual Qualcomm Hexagon NPU driver through Windows Update/OEM; a QNN GPU device is insufficient |
| Strict session cannot compile / no QNN profile kernels | Redownload the pinned model, inspect verbose graph logs and installed QNN version; NPU mode intentionally fails |
| Partial assignment rejected | The profile contains another provider's kernel. Adjust the model/export; do not relabel CPU work as NPU |
| Model missing / manifest mismatch | Run `python scripts/download_model.py`; use `NPU_SR_MODEL_DIR` or `--model` for other directories |
| Image load failure | Supply a valid still PNG/JPEG/WebP under 16 million input pixels; EXIF orientation is supported |
| Output path failure | Use a writable directory and PNG/JPEG/WebP extension; input/output must be different files |
| Small Task Manager NPU graph | A small model finishes quickly; try a longer benchmark. ORT profiling and strict assignment are the primary check |
| Slow first command | Provider acquisition/bootstrap and graph preparation are startup costs; inspect warm benchmark separately |

## Official installations

- [Windows App Runtime downloads](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads)
- [Windows ML Python installation](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app)
- [VC++ redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- [Windows ML provider download troubleshooting](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/execution-provider-errors)

No step requires manually copying DLLs out of a Qualcomm SDK.

## Runtime messages observed on the tested machine

The pinned ORT Windows ML wheel prints `Init provider bridge failed` at import
even when explicit catalog registration, NPU device selection and profiled QNN
inference subsequently succeed. It appears before application session creation
and is not evidence of CPU fallback. We leave vendor logging visible, rather than
suppressing failures. If `doctor` does not pass, treat its specific error as a real
failure. QNN can also print native graph preparation/bandwidth messages despite
the session's error-level logging setting; these are not timing measurements.

Microsoft's [Python API notes](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/api-reference)
describe a possible conflict with `winrt-runtime`'s bundled `msvcp140.dll` and advise
removing that copy while installing the VC++ redistributable. The CLI never deletes
dependency DLLs automatically. If diagnosing that specific conflict, inside your
virtual environment only, follow the official instructions and reinstall the
environment if needed. Removing it locally did not eliminate the generic bridge
warning above, and the strict NPU proof passed with and without it.

## Exit behavior

`doctor`: 0 only when the pretrained model passes strict QNN proof; otherwise 1.
`upscale`/`evaluate`: 0 on success; 1 on a runtime/input/output failure.
`benchmark`: 0 when valid measurements were produced, even if NPU is unavailable;
inspect `npu_unavailable` in JSON to distinguish a CPU-only result.
Argparse usage errors return 2. Auto fallback is printed explicitly.
## v0.2 models and GPU

- Missing registry model: run `npu-sr models download IDENTIFIER`. Paths still
  require adjacent hash/contract manifests. Do not rename model identifiers in a manifest.
- DnCNN cleans luminance Gaussian noise at sigma 25/255. Colored or structured
  real-world noise may remain; this is a model limitation, not a backend failure.
- GPU mode requires the DirectML provider supplied by the Windows ML ORT wheel.
  It fails explicitly on CPU-only installations. QNN GPU execution was unsuccessful
  on the tested package; changing `backend_type` alone does not establish GPU usage.
- Cache invalid/corrupt: source models must remain available. Cache metadata checks
  trigger recompilation; a cached context that fails proof is invalidated and retried
  from the source graph. Never publish compiled contexts or native verbose logs.
- The research dataset downloader requires access to Berkeley's official server.
  It refuses changed archives rather than using unverified data. Dataset images
  are research-only and excluded from git; benchmark your own aligned images if needed.
