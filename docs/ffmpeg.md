# FFmpeg and video hardware

Install FFmpeg separately, including its adjacent `ffprobe` executable. The
Python package invokes those processes; it does not link or redistribute FFmpeg.

On Windows, acquire the validated pinned native build:

```powershell
python scripts/download_ffmpeg.py
npu-sr video input.mp4 -o output.mp4 --device npu --decode hardware --encode hardware --codec av1
```

The script checks the archive SHA256 and keeps the build, DLLs and bundled notices
under the user's local application cache. It supports native ARM64 and x64 CI.
The BtbN builder is linked from [FFmpeg's official download page](https://ffmpeg.org/download.html).
The selected shared build is GPL; its terms apply to FFmpeg and its libraries.
No FFmpeg binary is part of this MIT repository or package. On Linux, use your
distribution's native FFmpeg package. `--ffmpeg path/to/ffmpeg` or
`NPU_SR_FFMPEG` explicitly selects a different installation.

## Strict paths

- Decode requests D3D11VA and keeps `d3d11` hardware frames until an explicit
  `hwdownload,format=nv12` (plus `format=rgb24` for the RGB path). Successful output and the selected
  hardware pixel format are required. Software frames cannot satisfy that filter.
- Encode requests `h264_mf`, `hevc_mf` or `av1_mf` with `-hw_encoding 1`.
  FFmpeg's source enumerates hardware-only Media Foundation transforms for that
  option. The application requires successful output and the activated MFT name.
  It explicitly selects `u_vbr` rate control and `camera_record` scenario.
  Vendor defaults lost two frames at a scene cut and degraded AV1 quality locally.
  Encoded counts and actual timestamp cadence are checked before publishing output;
  bitrate is a VBR target, not a guaranteed maximum.
- `hardware` fails on any missing proof. `auto` tries a real initialization probe,
  prints a software fallback when necessary, and never changes codecs midstream.
- `software` is explicit: libx264, libx265 or an installed libsvtav1/libaom-av1. AV1 selects SVT first
  and records the actual software encoder. Availability depends on
  the chosen build. Hardware bitrate and software CRF quality are separate settings.

Local native ARM64 FFmpeg `n9.0.2-22-g46d8f462ee` selected
`QCOM Hardware Encoder - H264`, `QCOM Hardware Encoder - HEVC` and
`QCOM Hardware Encoder - AV1`, and completed the encoded-frame validation tests.
H.264 D3D11VA decoding also completed. A 64×48 fixture was rejected by the driver;
the 640×360 hardware tests succeeded. Other resolutions/codecs still need proof
on each machine. An encoder appearing in `ffmpeg -encoders` is not enough.

These QCOM transforms are reasonable evidence of Qualcomm hardware codec use.
The project does not measure which internal video block is active or its power.
It does not claim zero-copy: NV12 or RGB frames cross CPU pipes and CPU processing.
See [research notes](research-notes.md) for Microsoft and FFmpeg sources.


The cache selector prefers the process architecture when both ARM64 and x64 tools
exist. Explicit paths override this preference. Software SVT-AV1 uses preset 10
and software CRF, as described in [FFmpeg encoder documentation](https://ffmpeg.org/ffmpeg-codecs.html#libsvtav1).
It is not a hardware encoder. The native validated build has SVT-AV1 but no libaom;
this is why software encoder selection checks the installed build.
