import io
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npu_sr import video_quality
from npu_sr.errors import SRException
from npu_sr.ffmpeg import VideoInfo


def install_streams(monkeypatch, actual, truth):
    info = VideoInfo(32, 32, Fraction(30), len(truth) / 30, len(truth), False, "raw")
    streams = iter((actual, truth))
    children = []

    class Child:
        def __init__(self, *_args):
            planes = next(streams)
            data = b"".join(
                np.full((32, 32), value, np.uint8).tobytes() + b"\x80" * 512 for value in planes
            )
            self.process = SimpleNamespace(stdout=io.BytesIO(data))
            self.closed = False
            children.append(self)

        def finish(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(video_quality, "PipeProcess", Child)
    monkeypatch.setattr(video_quality, "probe", lambda *_args: info)
    monkeypatch.setattr(video_quality, "sha256", lambda *_args: "0" * 64)
    calls = []

    def vmaf(*_args, **kwargs):
        calls.append(kwargs)
        return {"mean": 99}

    monkeypatch.setattr(video_quality, "measure_vmaf", vmaf)
    return children, calls


def evaluate():
    return video_quality.evaluate_native_video(
        Path("actual.mp4"), Path("truth.mkv"), Path("ffmpeg"), (0, 0, 32, 32), stride=1
    )


def test_native_identity_and_secondary_vmaf(monkeypatch):
    children, calls = install_streams(monkeypatch, [16, 100, 235], [16, 100, 235])
    report = evaluate()
    assert report["psnr_y_db"] is None
    assert report["ssim_y"] == pytest.approx(1)
    assert report["temporal_difference_error"] == 0
    assert report["decoded_frames"] == 3
    assert len(calls) == 2 and calls[1]["model"] == "vmaf_v0.6.1neg"
    assert all(child.closed for child in children)


def test_psnr_uses_clip_mse_including_perfect_frames(monkeypatch):
    install_streams(monkeypatch, [100, 110], [100, 100])
    report = evaluate()
    assert report["mean_mse_y"] == 50
    assert report["psnr_y_db"] == pytest.approx(10 * np.log10(255**2 / 50))


def test_temporal_diagnostic_distinguishes_stable_bias_from_flicker(monkeypatch):
    install_streams(monkeypatch, [110, 110, 110], [100, 100, 100])
    stable = evaluate()
    install_streams(monkeypatch, [90, 110, 90], [100, 100, 100])
    flicker = evaluate()
    assert stable["temporal_difference_error"] == 0
    assert flicker["temporal_difference_error"] == pytest.approx(20 / 255)
    assert flicker["mean_mse_y"] == stable["mean_mse_y"]


def test_unequal_frames_fail_and_close_both_children(monkeypatch):
    children, _ = install_streams(monkeypatch, [100], [100, 100])
    with pytest.raises(SRException, match="unequal"):
        evaluate()
    assert all(child.closed for child in children)


def test_invalid_stride_fails_before_launch():
    with pytest.raises(SRException, match="stride"):
        video_quality.evaluate_native_video(Path("a"), Path("b"), Path("ffmpeg"), stride=0)
