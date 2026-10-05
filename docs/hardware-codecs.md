# Hardware codec evidence

Strict decode requires an actual selected D3D11 hardware frame and successful
hardware download. Strict encode requires Media Foundation hardware enumeration,
an activated transform, and three actual encoded frames in the requested container matching
dimensions, framerate and count. Every completed video also checks packet cadence,
actual processed/encoded counts, geometry and copied audio metadata.

These checks establish hardware codec paths, not internal Qualcomm VPU counters.
FFmpeg provider/encoder lists alone are insufficient. The decoder transfers NV12
to system memory; this is not zero-copy.

## Observed constraints

On driver 30.0.219.1000, an AV1 1708×960 request is encoded as 1712×960 with an
aspect-ratio adjustment. Strict mode rejects it before processing. H.264 and HEVC
preserve 1708×960. Use explicit `--codec hevc` or `--codec h264` for that profile;
do not remove the geometry checks or silently substitute codecs.

Media Foundation supplies AV1 sequence headers in-band. MKV output needs FFmpeg's
`extract_extradata` bitstream filter; this extracts codec private data without
reencoding. MP4 output uses the existing path. Actual Opus/MKV and hardware AV1
processing was tested after this fix. Short generated correctness cases do not
establish sustained performance.

H.264/HEVC Media Foundation MKV headers fail on the tested stack, including after
trying sequence-header extraction and explicit global-header flags. Use MP4 or
explicit software encoding for those codec/container pairs. The startup probe now
tests the requested container so strict mode fails before processing the source.

Copied audio is the first audio stream. Its timestamps are shifted relative to
the first video frame; leading audio outside the video interval is trimmed and
delayed audio remains delayed. Codec, channel count and relative start are checked
with a 30 ms allowance for codec priming/container rounding. Subtitles, chapters,
attachments and other audio streams are not copied. Use a separate muxing workflow
if those are required.

HDR, VFR, anamorphic pixels and rotation metadata remain unsupported. Normalize
them explicitly first; the application does not silently rewrite their timing or
color interpretation. SDR color/range requirements are in the video pipeline docs.

References: [FFmpeg option ordering and input offsets](https://ffmpeg.org/ffmpeg.html),
[FFprobe stream metadata](https://ffmpeg.org/ffprobe.html),
[Microsoft hardware transforms](https://learn.microsoft.com/en-us/windows/win32/medfound/hardware-mfts).
