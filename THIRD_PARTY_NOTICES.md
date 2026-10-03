# Third-party notices

NPU-SR's original code, documentation, and generated test card are MIT licensed.
The repository's MIT license does not replace any dependency or model license.

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
