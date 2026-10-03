import math

import pytest
from PIL import Image

from npu_sr.benchmark import measure, statistics
from npu_sr.errors import SRException
from npu_sr.evaluate import psnr
from npu_sr.image import preprocess
from npu_sr.runtime import Runtime


def test_statistics():
    result = statistics([10, 20, 30, 40, 50])
    assert result["median_ms"] == 30 and result["mean_ms"] == 30
    assert result["p95_ms"] == 48
    assert result["fps_equivalent"] == pytest.approx(1000 / 30)
    for values in ([], [0], [math.nan], [math.inf], [-1]):
        with pytest.raises(SRException):
            statistics(values)


def test_measure_runs(tiny_model):
    runtime = Runtime(tiny_model, "cpu")
    report = measure(preprocess(Image.new("RGB", (15, 7))), runtime, 3, 2)
    assert len(report["samples_ms"]) == 3
    assert report["iterations"] == 3 and report["warmups"] == 2
    assert report["latency"]["median_ms"] > 0


def test_psnr_requires_matching_ground_truth():
    image = Image.new("RGB", (8, 8), "black")
    assert psnr(image, image) == math.inf
    assert psnr(image, Image.new("RGB", (8, 8), "white")) == 0
    with pytest.raises(SRException, match="Ground truth"):
        psnr(image, Image.new("RGB", (4, 4)))
