"""Bounded ordered frame streams; enhancement stays on one session-owning thread."""

import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from time import perf_counter

from .errors import SRException


@dataclass(frozen=True)
class Frame:
    index: int
    data: bytes
    read_ms: float
    read_started: float


def process_stream(
    source: Iterator[Frame],
    enhance: Callable[[bytes], tuple[bytes, dict[str, float]]],
    sink: Callable[[bytes], None],
    completed: Callable[[int, dict[str, float]], None],
    depth: int = 0,
    stop_children: Callable[[], None] | None = None,
) -> dict[str, int]:
    """Overlap read / enhance / write, with bounded queues and propagated errors.

    depth=0 is the sequential control. Queue depth does not include the frames
    currently read/enhanced/written. Latency ends at encoder pipe submission;
    the caller includes encoder flush in end-to-end wall time.
    """
    if not 0 <= depth <= 4:
        raise SRException("Pipeline depth must be between 0 and 4.")
    peaks = {"decode_queue": 0, "encode_queue": 0}

    def finish(frame: Frame, data: bytes, timings: dict[str, float]) -> None:
        before = perf_counter()
        sink(data)
        timings["encoder_pipe_wait_ms"] = (perf_counter() - before) * 1000
        timings["decoder_pipe_wait_ms"] = frame.read_ms
        timings["frame_latency_ms"] = (perf_counter() - frame.read_started) * 1000
        completed(frame.index + 1, timings)

    if depth == 0:
        expected = 0
        for frame in source:
            if frame.index != expected:
                raise SRException("Source frame ordering changed.")
            started = perf_counter()
            data, timings = enhance(frame.data)
            timings["enhancement_ms"] = (perf_counter() - started) * 1000
            finish(frame, data, timings)
            expected += 1
        return peaks

    decoded: queue.Queue = queue.Queue(maxsize=depth)
    encoded: queue.Queue = queue.Queue(maxsize=depth)
    cancelled = threading.Event()
    errors: list[BaseException] = []
    error_lock = threading.Lock()
    sentinel = object()

    def check() -> None:
        with error_lock:
            error = errors[0] if errors else None
        if error is not None:
            raise error
        if cancelled.is_set():
            raise SRException("Frame pipeline cancelled.") from None

    def put(target: queue.Queue, value: object, label: str) -> None:
        while True:
            check()
            try:
                target.put(value, timeout=0.1)
                peaks[label] = max(peaks[label], target.qsize())
                return
            except queue.Full:
                continue

    def get(target: queue.Queue):
        while True:
            check()
            try:
                return target.get(timeout=0.1)
            except queue.Empty:
                continue

    def guarded(worker: Callable[[], None]) -> None:
        try:
            worker()
        except BaseException as exc:
            with error_lock:
                if not errors:
                    errors.append(exc)
            cancelled.set()

    def read() -> None:
        for frame in source:
            put(decoded, frame, "decode_queue")
        put(decoded, sentinel, "decode_queue")

    def write() -> None:
        expected = 0
        while True:
            item = get(encoded)
            if item is sentinel:
                return
            frame, data, timings = item
            if frame.index != expected:
                raise SRException("Pipeline frame ordering changed.")
            finish(frame, data, timings)
            expected += 1

    reader = threading.Thread(target=lambda: guarded(read), name="npu-sr-read", daemon=True)
    writer = threading.Thread(target=lambda: guarded(write), name="npu-sr-write", daemon=True)
    reader.start()
    writer.start()
    succeeded = False
    try:
        expected = 0
        while True:
            frame = get(decoded)
            if frame is sentinel:
                break
            if frame.index != expected:
                raise SRException("Source frame ordering changed.")
            started = perf_counter()
            data, timings = enhance(frame.data)
            timings["enhancement_ms"] = (perf_counter() - started) * 1000
            put(encoded, (frame, data, timings), "encode_queue")
            expected += 1
        put(encoded, sentinel, "encode_queue")
        while writer.is_alive():
            writer.join(timeout=0.1)
            check()
        check()
        succeeded = True
        return peaks
    finally:
        cancelled.set()
        if not succeeded and (reader.is_alive() or writer.is_alive()):
            if stop_children:
                stop_children()
        reader.join(timeout=5)
        writer.join(timeout=5)
        if reader.is_alive() or writer.is_alive():
            raise SRException("Frame workers failed to stop after child termination.")
