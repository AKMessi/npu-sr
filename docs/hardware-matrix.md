# Hardware validation

Only actual executed tests qualify a device as verified. Processor families and
advertised TOPS do not establish compatibility or performance.

| Device | State | Evidence |
|---|---|---|
| Snapdragon X Plus X1P-42-100 | Verified | Published v0.5–v0.8 QNN/video tests and benchmarks |
| X1P-64-100 / X1P-66-100 | Unknown | No device access or submitted runs |
| X1E-78-100 / X1E-80-100 / X1E-84-100 | Unknown | No device access or submitted runs |
| Non-Qualcomm / non-Windows ARM64 | CPU usage supported | Generic Windows/Linux CI; strict NPU unavailable |

The verified laptop has Windows 11 ARM64 build 26200, driver 30.0.219.1000,
QNN catalog 2.2480.53.0, ORT 1.25.2 and Windows ML bindings 2.3.0. Exact software,
input hashes, power-state snapshots and scopes are attached to each benchmark.
No separate SKU, battery life or internal accelerator utilization claim follows.

The Adreno DirectML path is separately assignment/profile verified on this laptop.
The Qualcomm QNN GPU backend was not usable; it is not listed as verified.

Community validation can use `doctor --gpu --json ...`, separately selected
hardware pytest groups and the documented benchmark scripts. Review reports for
private paths and identifiers before attaching them to an issue. We do not mark
unsubmitted or unexecuted machines as community verified.
