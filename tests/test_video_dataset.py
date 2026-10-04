import hashlib

import pytest

from npu_sr.errors import SRException
from npu_sr.ffmpeg import run_tool, tool_path


def test_bitexact_reference_preparation_is_repeatable(tmp_path):
    try:
        ffmpeg = tool_path()
    except SRException:
        pytest.skip("Acquire FFmpeg for reference preparation test")
    hashes = []
    for index in range(2):
        path = tmp_path / f"ref-{index}.mkv"
        run_tool(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=6",
                "-t",
                "1",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-pix_fmt",
                "yuv420p",
                "-bitexact",
                str(path),
            ]
        )
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]
