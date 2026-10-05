# Third-party notices

NPU-SR's original code, documentation, and generated test card are MIT licensed.
The repository's MIT license does not replace any dependency or model license.

## Temporal research

Training scripts use adjacent frames from Blender Foundation's *Tears of Steel*
and *Big Buck Bunny*, CC BY 3.0; the [training definition](benchmarks/temporal-training-v1.json)
records source URLs, hashes and attribution. Footage, prepared arrays and learned
experimental checkpoints are not distributed. The frozen QuickSRNet teacher is
acquired separately under BSD-3-Clause. Original correction code is MIT;
exported graphs combining its learned arrays with the teacher require
BSD-3-Clause AND MIT. These experiments did not justify replacing the spatial
realtime default; no pretrained temporal model is included in the package.

## ESPCN weights

- Author/upstream: Fanny Monori, [TF-ESPCN](https://github.com/fannymonori/TF-ESPCN).
- Work: pretrained `export/ESPCN_x2.pb`, developed during Google Summer of Code 2019 with OpenCV.
- Revision: `5c628eca82028161a53e1265cc3a5b571ab8625f`.
- License: [Apache License 2.0](licenses/TF-ESPCN-APACHE-2.0.txt), as provided in upstream's LICENSE.
- Original SHA256: `59f77351e1d7c0057bf6fe088b4a8a07e42c468c8c8aebb674a6b4ea1823221d`.
- Acquisition: `scripts/download_model.py` downloads only from that immutable upstream URL.
- Modifications: ONNX opset 13, fixed NCHW shape, HWIO-to-OIHW weight layout,
  Conv bias folding; the pretrained parameters are not retrained.

Weights and derived ONNX files are downloaded/generated locally in ignored `models/`.
The pretrained model retains its Apache 2.0 license. If you redistribute a derived
model, include the upstream license and attribution, and identify your changes.
DIV2K training data and upstream sample images are not distributed here.

## Additional v0.2 models

- FSRCNN and FSRCNN-small: [Saafke/FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow),
  revision `6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d`,
  [Apache 2.0](licenses/TF-FSRCNN-APACHE-2.0.txt). The upstream implementation
  uses a subpixel output layer. Exports change layout and fold bias/PRelu expressions.
- LapSRN x2: Fanny Monori's [TF-LapSRN](https://github.com/fannymonori/TF-LapSRN),
  revision `fc51c90af1b5801a357abc919160d7ff4f24b997`,
  [Apache 2.0](licenses/TF-LapSRN-APACHE-2.0.txt). Exports preserve SAME padding
  and express leaky activation using equivalent PRelu slopes for QNN compatibility.
- DnCNN sigma 25: Kai Zhang's [KAIR](https://github.com/cszn/KAIR), release v1.0
  `dncnn_25.pth`, [MIT](licenses/KAIR-MIT.txt). License copy pinned at
  `fc1732f4a4514e42ce15e5b3a1e18c828af47a1e`. The upstream weights already fold
  batch normalization. The ONNX export computes input minus predicted noise.

Exact source SHA256 values and download URLs are in
[`acquire_models.py`](src/npu_sr/acquire_models.py) and the generated manifests.
Parameters are not retrained. Original and exported weights remain ignored;
their model licenses apply independently of this repository's MIT code license.

## QuickSRNet candidate checkpoints and adaptation

- Qualcomm AIMET Model Zoo / AI Hub Models, QuickSRNet small and medium x2.
- Model license: [BSD 3-Clause](licenses/AIMET-MODEL-ZOO-BSD-3-CLAUSE.txt),
  confirmed by the official AI Hub model manifest. License/source reference:
  AIMET Model Zoo revision `1bd2bf5b17cdda9251437c444009b29e1a25054b`.
- Checkpoints: official `quic/aimet-model-zoo` release
  `phase_2_january_artifacts`; URLs and pinned hashes are in `acquire_models.py`.
- Changes: fixed NCHW ONNX export, sum the first convolution's RGB input weights
  to evaluate neutral repeated-Y input, retain every Clip, mix clipped RGB output
  phases with Rec.601 luminance coefficients, then pixel shuffle. These luminance
  adaptations are not the original RGB model and do not inherit its published scores.
- No retraining. Checkpoints, exported weights and optimizer metadata are not shipped.
  Acquisition verifies SHA256 before an allowlisted, inert checkpoint reader runs.

## Expanded video corpus

The predeclared [corpus manifest](benchmarks/corpus-v1.json) records source hashes,
license links, attribution, time ranges, active image regions and degradation.
Big Buck Bunny, Sintel video and Tears of Steel are Blender Foundation CC BY 3.0
works; benchmark audio is omitted. NASA source use follows its media usage
guidelines for informational benchmarking, with no endorsement implied. NASA's
guidelines include exceptions for third-party content and are not a blanket
public-domain license. Source movies, prepared clips and output movies remain
outside git and are not distributed in the package. Text/UI artwork is original
MIT content; the moving FFmpeg test pattern is explicitly synthetic.

## Image benchmark data

The [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) images
are available for non-commercial research and educational use; copyright remains
with their owners. This project downloads a fixed five-image test subset for
research and does not redistribute the images or their derivatives.
Citation: D. Martin, C. Fowlkes, D. Tal and J. Malik, *A Database of Human Segmented
Natural Images and its Application to Evaluating Segmentation Algorithms and
Measuring Ecological Statistics*, ICCV 2001. Download provenance and hashes are
written beside local data. Committed benchmark JSON contains measurements only.

## Runtime and Python dependencies

| Component | License / terms | Distribution in this repository |
| --- | --- | --- |
| NumPy | BSD 3-Clause; bundled components have additional notices | Package reference only |
| Pillow | HPND / Pillow license; bundled codecs have additional notices | Package reference only |
| ONNX | Apache 2.0 | Package reference only |
| ONNX Runtime open-source code | MIT | Package reference only |
| ONNX Runtime Windows ML wheel, Windows App Runtime, Windows ML bindings | Their published package licenses and Microsoft terms apply | Package reference / official installation only |
| Qualcomm QNN / QAIRT runtime and Hexagon drivers | Qualcomm and vendor package terms; proprietary components | Installed through Windows ML / Windows Update; no binaries shipped |
| pytest | MIT | Development dependency only |
| Ruff | MIT | Development dependency only |
| setuptools, build, wheel, twine | Their published package licenses | Development/build dependencies only |

The minimal protobuf reader implements public protobuf wire-format parsing; it
does not embed TensorFlow code or generated TensorFlow protobuf files.
Review installed packages' own LICENSE/NOTICE files when distributing an application
or environment. No permission to redistribute Microsoft or Qualcomm binaries is
granted by NPU-SR. Vendor names identify compatible hardware; no endorsement is implied.
## FFmpeg and video benchmark sources (v0.3)

FFmpeg is invoked as a separate executable. The optional acquisition script uses
the BtbN GPL shared build linked by FFmpeg's official download page, retaining
its bundled license/notices outside git. No codec binaries or libraries are
redistributed by this repository/package. Microsoft Media Foundation and Qualcomm
driver transforms retain their own installed-system terms.

Tears of Steel benchmark source: Blender Foundation / mango.blender.org,
Creative Commons Attribution 3.0. The pinned source is downloaded from Blender's
official server. Prepared research clips, audio and reference videos remain
outside git. Scripts document retiming, crop, degradation and source hash.
The generated FFmpeg test pattern is a separate synthetic workload.

The small `examples/realtime-comparison.gif` is a redistributed adaptation of
*Tears of Steel*, Blender Foundation / [mango.blender.org](https://mango.blender.org/),
under [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/).
It retains attribution in the image and adjacent documentation. Changes include
retiming, crop, degradation, neural enhancement, labels and GIF conversion.
Its license is CC BY 3.0; the repository's MIT license does not replace it.
