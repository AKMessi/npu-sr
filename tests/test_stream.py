"""Ordering, backpressure and failure propagation without codecs or hardware."""

from time import perf_counter, sleep

import pytest

from npu_sr.errors import SRException
from npu_sr.stream import Frame, process_stream


def source(count=60):
    for index in range(count):
        yield Frame(index, bytes([index]), 0, perf_counter())


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_bounded_ordered_stream(depth):
    output, progress = [], []

    def sink(data):
        sleep(0.001)  # Deliberately apply encoder backpressure.
        output.append(data)

    peaks = process_stream(
        source(),
        lambda raw: (raw, {}),
        sink,
        lambda count, phases: progress.append(count),
        depth,
    )
    assert output == [bytes([n]) for n in range(60)]
    assert progress == list(range(1, 61))
    assert max(peaks.values()) <= depth


@pytest.mark.parametrize("depth", [0, 2])
def test_source_order_violation(depth):
    bad = iter([Frame(1, b"x", 0, perf_counter())])
    with pytest.raises(SRException, match="ordering"):
        process_stream(bad, lambda data: (data, {}), lambda data: None, lambda *args: None, depth)


def test_encoder_error_propagates_and_stops_workers():
    stopped = []

    def sink(data):
        raise BrokenPipeError("encoder died")

    with pytest.raises(BrokenPipeError, match="encoder died"):
        process_stream(
            source(),
            lambda data: (data, {}),
            sink,
            lambda *args: None,
            2,
            lambda: stopped.append(True),
        )
    # In-memory workers can already have exited; no child kill is then needed.


def test_worker_keyboard_interrupt_propagates():
    def progress(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        process_stream(source(), lambda data: (data, {}), lambda data: None, progress, 2)
