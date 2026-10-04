"""Planar range, chroma, edges and buffer ownership without NPU hardware."""

from fractions import Fraction

import numpy as np
import pytest

from npu_sr.errors import SRException
from npu_sr.ffmpeg import VideoInfo, decode_args
from npu_sr.image import infer_y, preprocess
from npu_sr.model import ModelSpec
from npu_sr.planar import NV12Enhancer


class NearestRuntime:
    spec = ModelSpec("test", "ESPCN", core=16, core_height=8, halo=2)

    def run(self, tensor):
        return tensor.repeat(2, axis=2).repeat(2, axis=3)


def test_nv12_luma_range_chroma_and_owned_buffers():
    info = VideoInfo(20, 12, Fraction(30), 1, 30, False, "h264", "tv")
    enhancer = NV12Enhancer(info, NearestRuntime())
    y = np.arange(240, dtype=np.uint8).reshape(12, 20)
    uv = np.full((6, 10, 2), [67, 177], np.uint8)
    raw = y.tobytes() + uv.tobytes()
    data, timings = enhancer.process(raw)
    output = np.frombuffer(data, np.uint8)
    expected = np.clip(y, 16, 235).repeat(2, 0).repeat(2, 1)
    np.testing.assert_array_equal(output[:960].reshape(24, 40), expected)
    np.testing.assert_array_equal(output[960:].reshape(12, 20, 2), np.full((12, 20, 2), [67, 177]))
    assert all(value >= 0 for value in timings.values())
    enhancer.process(bytes([16] * 240) + uv.tobytes())
    np.testing.assert_array_equal(np.frombuffer(data, np.uint8)[:960].reshape(24, 40), expected)
    with pytest.raises(SRException, match="byte count"):
        enhancer.process(raw[:-1])


def test_planar_rejects_explicit_full_range():
    info = VideoInfo(20, 12, Fraction(30), 1, 30, False, "h264", "pc")
    with pytest.raises(SRException, match="limited-range"):
        NV12Enhancer(info, NearestRuntime())


def test_rectangular_tiles_preserve_edges_and_reusable_output():
    from PIL import Image

    image = Image.fromarray(np.random.default_rng(12).integers(0, 256, (23, 39, 3), np.uint8))
    prepared = preprocess(image, NearestRuntime.spec)
    reference = prepared.padded_y[2:25, 2:41].repeat(2, 0).repeat(2, 1)
    output = np.empty_like(reference)
    actual, _ = infer_y(prepared, NearestRuntime(), output=output)
    assert actual is output and prepared.tile_count == 9
    np.testing.assert_array_equal(actual, reference)
    with pytest.raises(SRException, match="reusable"):
        infer_y(prepared, NearestRuntime(), output=np.empty((1, 1)))


def test_nv12_decode_retains_explicit_hardware_download(tmp_path):
    args = decode_args(tmp_path / "in.mp4", True, pixel_format="nv12")
    assert "hwdownload,format=nv12" in args
    assert args[args.index("-pix_fmt") + 1] == "nv12"
