import numpy as np
import pytest

from npu_sr.errors import SRException
from npu_sr.video_quality import full_temporal_difference


def test_temporal_signed_extrema_do_not_wrap():
    high, low = np.full((5, 7), 255, np.int16), np.full((5, 7), -255, np.int16)
    assert full_temporal_difference(high, low) == pytest.approx(2)
    assert full_temporal_difference(low, high) == pytest.approx(2)


def test_temporal_constant_bias_and_correctly_tracked_scene_change():
    previous_reference = np.zeros((5, 7), np.int16)
    current_reference = np.full((5, 7), 100, np.int16)
    previous_output = previous_reference + 5
    current_output = current_reference + 5
    assert (
        full_temporal_difference(
            current_output - current_reference, previous_output - previous_reference
        )
        == 0
    )
    # A frozen/smoothed output fails to track the genuine reference change.
    assert full_temporal_difference(
        previous_output - current_reference, previous_output - previous_reference
    ) == pytest.approx(100 / 255)


@pytest.mark.parametrize("case", ["uint8", "invalid-range", "empty", "geometry"])
def test_temporal_diagnostic_rejects_bad_error_planes(case):
    current = np.zeros((5, 7), np.int16)
    previous = current.copy()
    if case == "uint8":
        current = current.astype(np.uint8)
    elif case == "invalid-range":
        current[0, 0] = 256
    elif case == "empty":
        current = previous = np.empty((0, 7), np.int16)
    else:
        previous = previous[:, :6]
    with pytest.raises(SRException):
        full_temporal_difference(current, previous)
