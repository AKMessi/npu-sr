# Security

The latest published minor release receives fixes. Older experimental releases
are retained for reproducibility and do not have a separate maintenance guarantee.

For a vulnerability, use GitHub's private vulnerability reporting on the published
repository if enabled. If it is not enabled, open an issue requesting a private
contact without posting exploit details, secrets, or personal paths.

We verify pinned model downloads with SHA256 and check local artifact manifests.
Model acquisition reads allowlisted, bounded tensor formats and does not execute
downloaded Python or arbitrary pickle. Research NumPy checkpoints disable pickle.
An adjacent manifest is an integrity check, not a signature or a trust guarantee
for arbitrary third-party models. Use only models you trust.

Windows ML acquires QNN from Microsoft's catalog and manages package servicing.
The application does not download vendor DLLs from arbitrary URLs. Inference is
local; images are not submitted to a cloud service. ORT telemetry is explicitly
disabled, but vendor/OS package installation and servicing have their own behavior.

Malformed images are rejected by Pillow, animation is unsupported, and images
above the documented pixel limit are rejected before decoding. Keep Pillow,
Windows, drivers and runtime packages current; retest strict NPU execution after
updates. The project is not intended as an untrusted public upload service.

Video runs local FFmpeg subprocesses using argument lists, never a command shell.
Frame buffers, queues, diagnostic tails and packet reorder storage are bounded.
Packet verification is the default; `--verify-full` adds a timeout-bounded decoded
audit. FFmpeg and driver parsers remain separate attack surfaces: use trusted
sources and current compatible tools. Model/tool hashes establish reproducibility,
not immunity from defects. CI audits installed dependencies for known advisories;
the audit cannot assess proprietary drivers or undisclosed vulnerabilities.
