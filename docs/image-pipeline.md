# Image pipeline

1. Pillow loads one oriented PNG/JPEG/WebP image; animation is rejected.
2. NumPy converts RGB to normalized full-range Rec.601 luminance and chroma.
3. Padding repeats exterior image pixels. Interior tiles include real neighbors.
4. One contiguous float input buffer is reused across synchronous tile calls.
5. A persistent runtime executes each static tile on the explicitly chosen backend.
6. The halo is cropped, preserving each useful pixel exactly once.
7. CPU bicubic chroma/alpha and reconstructed luminance produce the output.
8. PNG/JPEG/WebP saving uses Pillow. JPEG alpha is flattened onto white.

Model contracts provide task, scale, core and receptive-field halo. Odd image
sizes preserve their exact dimensions times scale. Tests cover small boundaries,
large odd sizes, seams, alpha, normalization and source overwrite prevention.

The tile generator reuses its input buffer: consume each yielded tensor before
advancing it. It is not an asynchronous queue of independently owned tensors.
One Runtime can enhance many images; it performs proof only at initialization.

`benchmark-suite` times preprocessing, tile extraction/copy, ORT invocation
(including output validation), stitching and reconstruction separately. PNG
encoding is a separate probe. Disk IO and startup are excluded from warm total.
`upscale` includes image loading, initialization and saving in its total.

The first profile found CPU reconstruction dominated large images. Quantizing
planes in place avoids several full-size float RGB arrays and preserves the
previous pixel formula exactly. See the release benchmark summary for measured
before/after results. Larger tiles reduce calls but also increase padding on
partial tiles; they are not always faster for small images.

Encoded images are written to an owned temporary file in the output directory
and atomically renamed only after successful encoding. Disk/encode failure
preserves existing output. CLI rejects input/model/report filesystem aliases.
