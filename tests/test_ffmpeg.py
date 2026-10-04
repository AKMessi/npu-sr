"""Pipe framing and strict evidence checks work without accelerator hardware."""

import io
from fractions import Fraction
from pathlib import Path

import pytest

from npu_sr.errors import SRException
from npu_sr.ffmpeg import (
    VideoInfo,
    check_frame_times,
    codec_args,
    decode_args,
    decode_evidence,
    encode_args,
    encode_evidence,
    read_frame,
    write_frame,
)
from npu_sr.video import VideoSettings, select_codecs


class PartialRead(io.BytesIO):
    def readinto(self, buffer):
        return super().readinto(buffer[:2])


class PartialWrite(io.BytesIO):
    def write(self, data):
        return super().write(data[:2])


def test_frame_partial_reads_and_eof():
    stream = PartialRead(b"abcdef123456")
    assert read_frame(stream, 6) == b"abcdef"
    assert read_frame(stream, 6) == b"123456"
    assert read_frame(stream, 6) is None
    with pytest.raises(SRException, match="truncated"):
        read_frame(PartialRead(b"abc"), 6)


def test_frame_partial_writes():
    stream = PartialWrite()
    write_frame(stream, b"abcdef")
    assert stream.getvalue() == b"abcdef"


def test_actual_timestamps_detect_vfr_and_reordering():
    assert check_frame_times((n / 30 for n in range(100)), Fraction(30)) == 100
    with pytest.raises(SRException, match="Variable or reordered"):
        check_frame_times([0, 0.033333, 0.09], Fraction(30))
    with pytest.raises(SRException):
        check_frame_times([0, 0.033333, 0], Fraction(30))
    with pytest.raises(SRException, match="finite"):
        check_frame_times([float("nan")], Fraction(30))


def test_hardware_evidence_requires_executed_path():
    with pytest.raises(SRException):
        decode_evidence("D3D11 supported", True)
    with pytest.raises(SRException):
        decode_evidence("Format d3d11 chosen by get_format()", False)
    assert decode_evidence("Format d3d11 chosen by get_format()", True)["explicit_hwdownload"]
    with pytest.raises(SRException):
        encode_evidence("h264_mf available")
    assert (
        encode_evidence("MFT name: 'QCOM Hardware Encoder - AV1'")["transform"]
        == "QCOM Hardware Encoder - AV1"
    )


def test_software_av1_selection_uses_installed_encoder(monkeypatch):
    from types import SimpleNamespace

    from npu_sr.ffmpeg import software_av1_encoder

    monkeypatch.setattr(
        "npu_sr.ffmpeg.run_tool",
        lambda _: SimpleNamespace(stdout=b" V..... libsvtav1 software AV1\n"),
    )
    assert software_av1_encoder(Path("ffmpeg")) == "libsvtav1"
    args = codec_args("av1", False, "8M", 20, "libsvtav1")
    assert "libsvtav1" in args and "10" in args and "-cpu-used" not in args
    monkeypatch.setattr(
        "npu_sr.ffmpeg.run_tool", lambda _: SimpleNamespace(stdout=b" V..... libaom-av1 AV1\n")
    )
    assert software_av1_encoder(Path("ffmpeg")) == "libaom-av1"
    monkeypatch.setattr("npu_sr.ffmpeg.run_tool", lambda _: SimpleNamespace(stdout=b""))
    with pytest.raises(SRException, match="neither"):
        software_av1_encoder(Path("ffmpeg"))


@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
def test_strict_hardware_command(codec, tmp_path):
    info = VideoInfo(64, 48, Fraction(30000, 1001), 1.0, 30, True, "h264")
    args = encode_args(
        Path("source.mp4"), tmp_path / "result.mp4", info, 2, codec, True, "copy", "8M", 20
    )
    assert args[args.index("-hw_encoding") + 1] == "1"
    assert args[args.index("-rate_control") + 1] == "u_vbr"
    assert args[args.index("-scenario") + 1] == "camera_record"
    assert args[args.index("-c:v") + 1] == codec + "_mf"
    assert "1:a:0" in args and "copy" in args
    assert "128x96" in args and "30000/1001" in args
    assert "hwdownload,format=nv12,format=rgb24" in decode_args(Path("source.mp4"), True)


def test_bad_codec_options_are_not_shell_arguments():
    with pytest.raises(SRException):
        codec_args("h264", True, "8M -i private", 20)
    with pytest.raises(SRException):
        codec_args("h264", False, "8M", 100)


def test_auto_reports_fallback_and_strict_fails(monkeypatch, caplog):
    def unavailable(*args):
        raise SRException("no hardware")

    monkeypatch.setattr("npu_sr.video.hardware_decode_probe", unavailable)
    monkeypatch.setattr("npu_sr.video.hardware_encode_probe", unavailable)
    info = VideoInfo(64, 48, Fraction(30), 1, 30, False, "h264")
    decode, encode, evidence = select_codecs(
        Path("ffmpeg"), Path("source"), info, VideoSettings(), 2
    )
    assert not decode and not encode
    assert evidence["decode"]["auto_fallback"]
    assert "using software" in caplog.text
    with pytest.raises(SRException, match="Strict hardware decode"):
        select_codecs(Path("ffmpeg"), Path("source"), info, VideoSettings(decode="hardware"), 2)
