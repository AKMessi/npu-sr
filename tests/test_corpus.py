import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from npu_sr.corpus import acquire_source, synthetic_ui, validate_definition
from npu_sr.errors import SRException
from npu_sr.video_quality import validate_roi


def test_predeclared_corpus_is_diverse_and_partitioned():
    definition = json.loads(Path("benchmarks/corpus-v1.json").read_text())
    validate_definition(definition)
    clips = definition["clips"]
    assert sum(row["split"] == "development" for row in clips) == 12
    assert sum(row["split"] == "holdout" for row in clips) == 12
    assert len({row["source"] for row in clips}) >= 5
    duplicate = definition | {"clips": [clips[0], clips[0]]}
    with pytest.raises(SRException, match="unique"):
        validate_definition(duplicate)
    bad = definition | {"clips": [clips[0] | {"identifier": "../unsafe"}]}
    with pytest.raises(SRException, match="identifiers"):
        validate_definition(bad)


def test_download_failure_does_not_publish_partial_source(tmp_path, monkeypatch):
    from npu_sr import corpus

    monkeypatch.setattr(corpus.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(SRException, match="SHA256"):
        acquire_source({"url": "https://example.invalid", "sha256": "0" * 64}, tmp_path, "film")
    assert not list(tmp_path.iterdir())


def test_exact_zip_member_and_atomic_extraction(tmp_path):
    data = b"movie pixels"
    archive = tmp_path / "film.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../movie.mp4", data)
    spec = {
        "member": "../movie.mp4",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "media_sha256": "0" * 64,
    }
    with pytest.raises(SRException, match="hash"):
        acquire_source(spec, tmp_path, "film")
    assert list(tmp_path.iterdir()) == [archive]
    spec["media_sha256"] = hashlib.sha256(data).hexdigest()
    media = acquire_source(spec, tmp_path, "film")
    assert media == tmp_path / "film.mov" and media.read_bytes() == data
    assert not (tmp_path.parent / "movie.mp4").exists()


def test_existing_source_tampering_is_detected(tmp_path):
    (tmp_path / "film.mp4").write_bytes(b"tampered")
    with pytest.raises(SRException, match="hash changed"):
        acquire_source({"sha256": "0" * 64}, tmp_path, "film")


def test_ui_frames_are_deterministic_with_actual_motion():
    a, b = synthetic_ui(0), synthetic_ui(1)
    assert a.size == (1920, 1080)
    assert a.tobytes() == synthetic_ui(0).tobytes()
    np.testing.assert_array_equal(np.asarray(a)[:, 1:], np.asarray(b)[:, :-1])
    assert a.tobytes() != synthetic_ui(1, vertical=True).tobytes()


@pytest.mark.parametrize("roi", [(-1, 0, 20, 20), (0, 0, 15, 20), (1900, 0, 40, 20)])
def test_metric_roi_rejects_invalid_bounds(roi):
    with pytest.raises(SRException, match="ROI"):
        validate_roi(roi, 1920, 1080)
