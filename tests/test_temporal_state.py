import numpy as np
import pytest

from npu_sr.errors import SRException
from npu_sr.temporal import SceneDetector


def test_temporal_pipeline_orders_planes_reuses_state_and_resets():
    from fractions import Fraction

    from npu_sr.ffmpeg import VideoInfo
    from npu_sr.model import ModelSpec
    from npu_sr.temporal import TemporalNV12Enhancer

    class Counting:
        spec = ModelSpec("test-temporal", "test", core=16, halo=7, input_channels=2)

        def __init__(self):
            self.calls = []

        def run(self, tensor):
            self.calls.append(tensor[:, :, 7:-7, 7:-7].copy())
            return tensor[:, 1:].repeat(2, 2).repeat(2, 3)

    runtime = Counting()
    info = VideoInfo(20, 12, Fraction(30), 1, 30, False, "h264", "tv")
    enhancer = TemporalNV12Enhancer(info, runtime)
    original_history = enhancer.previous
    for value in (80, 85, 200):
        raw = bytes([value] * 240 + [128] * 120)
        output, _ = enhancer.process(raw)
        np.testing.assert_array_equal(np.frombuffer(output, np.uint8)[:960], value)
    assert len(runtime.calls) == 6  # Two tiles, every frame, including the reset.
    np.testing.assert_allclose(runtime.calls[0][:, 0], runtime.calls[0][:, 1])
    assert float(runtime.calls[2][:, 0].mean()) == pytest.approx((80 - 16) / 219)
    assert float(runtime.calls[2][:, 1].mean()) == pytest.approx((85 - 16) / 219)
    np.testing.assert_allclose(runtime.calls[4][:, 0], runtime.calls[4][:, 1])
    assert enhancer.previous is original_history
    assert enhancer.evidence()["frames_processed"] == 3
    assert enhancer.evidence()["scene_reset_counts"] == {"initial": 1, "abrupt-change": 1}


def test_temporal_pipeline_requires_valid_context_and_unblended_graph():
    from fractions import Fraction

    from npu_sr.ffmpeg import VideoInfo
    from npu_sr.model import ModelSpec
    from npu_sr.temporal import TemporalNV12Enhancer

    info = VideoInfo(20, 12, Fraction(30), 1, 30, False, "h264", "tv")
    for channels, halo, strength in [(1, 7, 1), (2, 4, 1), (2, 7, 0.5)]:

        class Invalid:
            spec = ModelSpec("test", "test", input_channels=channels, halo=halo)

        with pytest.raises(SRException, match="Temporal NV12"):
            TemporalNV12Enhancer(info, Invalid(), strength)


def test_two_plane_tile_input_requires_history_and_rejects_still_image():
    from PIL import Image

    from npu_sr.image import LuminanceInput, preprocess
    from npu_sr.model import ModelSpec

    spec = ModelSpec("test", "test", core=8, halo=7, input_channels=2)
    data = LuminanceInput((8, 8), np.zeros((22, 22), np.float32), spec)
    with pytest.raises(SRException, match="previous-frame"):
        next(data.tiles())
    with pytest.raises(SRException, match="still image"):
        preprocess(Image.new("RGB", (8, 8)), spec)


def test_initial_stationary_and_bounded_history():
    detector = SceneDetector()
    frame = np.full((540, 960), 0.5, np.float32)
    assert detector.observe(frame) == "initial"
    for _ in range(100):
        assert detector.observe(frame) is None
    assert detector.previous.shape == (32, 32)
    assert detector.previous.nbytes == 4096
    assert detector.resets == 1
    frame[:] = 0
    assert detector.previous.mean() == 0.5  # Owned state, no alias to caller.


def test_recurrent_features_preserve_phase_order_owned_buffers_and_scene_reset():
    from fractions import Fraction

    from npu_sr.ffmpeg import VideoInfo
    from npu_sr.model import ModelSpec
    from npu_sr.temporal import TemporalNV12Enhancer

    class Phases:
        spec = ModelSpec("test-state", "test", core=16, halo=7, input_channels=6)

        def __init__(self):
            self.calls = []

        def run(self, tensor):
            self.calls.append(tensor[:, :, 7:-7, 7:-7].copy())
            result = tensor[:, 1:2].repeat(2, 2).repeat(2, 3)
            for phase in range(4):
                result[..., phase // 2 :: 2, phase % 2 :: 2] += phase / 219
            return result

    runtime = Phases()
    enhancer = TemporalNV12Enhancer(
        VideoInfo(20, 12, Fraction(30), 1, 30, False, "h264", "tv"), runtime
    )
    state = enhancer.previous_features
    first, _ = enhancer.process(bytes([80] * 240 + [128] * 120))
    owned = bytes(first)
    second, timings = enhancer.process(bytes([85] * 240 + [128] * 120))
    assert first == owned and first != second
    for phase in range(4):
        np.testing.assert_allclose(
            runtime.calls[2][:, phase + 2], (80 + phase - 16) / 219, atol=1e-7
        )
    enhancer.process(bytes([200] * 240 + [128] * 120))
    np.testing.assert_allclose(runtime.calls[4][:, 2:], (200 - 16) / 219, atol=1e-7)
    assert enhancer.previous_features is state
    assert len(runtime.calls) == 6 and enhancer.evidence()["frames_processed"] == 3
    assert enhancer.evidence()["scene_reset_counts"] == {"initial": 1, "abrupt-change": 1}
    assert enhancer.evidence()["feature_state_bytes"] == state.nbytes
    assert timings["temporal_state_update_ms"] >= 0


def test_recurrent_tiling_rejects_missing_phase_history():
    from npu_sr.image import LuminanceInput
    from npu_sr.model import ModelSpec

    spec = ModelSpec("test", "test", core=8, halo=7, input_channels=6)
    plane = np.zeros((22, 22), np.float32)
    data = LuminanceInput((8, 8), plane, spec, previous_y=plane.copy())
    with pytest.raises(SRException, match="four matching"):
        next(data.tiles())


def test_hard_cut_with_equal_histogram_resets_history():
    detector = SceneDetector()
    pattern = np.indices((32, 32)).sum(axis=0) % 2
    detector.observe(pattern.astype(np.float32))
    assert detector.observe((1 - pattern).astype(np.float32)) == "abrupt-change"


def test_constant_brightness_cut_and_geometry_change():
    detector = SceneDetector()
    detector.observe(np.full((32, 32), 0.2, np.float32))
    assert detector.observe(np.full((32, 32), 0.7, np.float32)) == "abrupt-change"
    assert detector.observe(np.full((16, 32), 0.7, np.float32)) == "geometry-change"


def test_fade_resets_on_black_entry_and_exit():
    detector = SceneDetector()
    reasons = [
        detector.observe(np.full((32, 32), value, np.float32)) for value in np.linspace(0.5, 0, 21)
    ]
    assert reasons[0] == "initial"
    assert reasons[-1] == "black-transition"
    assert all(reason is None for reason in reasons[1:-1])
    assert detector.observe(np.zeros((32, 32), np.float32)) is None
    assert detector.observe(np.full((32, 32), 0.05, np.float32)) == "black-transition"


@pytest.mark.parametrize(
    "frame", [np.zeros(0), np.empty((0, 3)), np.full((2, 2), np.nan), np.full((2, 2), 2)]
)
def test_invalid_luminance_rejected(frame):
    with pytest.raises(SRException):
        SceneDetector().observe(frame)
