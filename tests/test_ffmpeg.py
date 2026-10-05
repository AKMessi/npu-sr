"""Pipe framing and strict evidence checks work without accelerator hardware."""

import hashlib
import io
import json
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


@pytest.mark.parametrize("changed", ["width", "fps", "count"])
def test_encoder_probe_rejects_driver_geometry_or_cadence_changes(monkeypatch, changed):
    from dataclasses import replace
    from types import SimpleNamespace

    from npu_sr.ffmpeg import hardware_encode_probe

    original = VideoInfo(854, 480, Fraction(60000, 1001), 1, 60, False, "h264")
    actual = VideoInfo(1708, 960, original.fps, 3 / float(original.fps), 3, False, "av1")
    actual = replace(
        actual,
        **{
            "width": {"width": 1712},
            "fps": {"fps": Fraction(60)},
            "count": {"frames": 2},
        }[changed],
    )
    monkeypatch.setattr(
        "npu_sr.ffmpeg.run_tool",
        lambda *a, **kw: SimpleNamespace(stderr=b"MFT name: 'QCOM Hardware Encoder - AV1'"),
    )
    monkeypatch.setattr("npu_sr.ffmpeg.probe", lambda *a, **kw: actual)
    with pytest.raises(SRException, match="changed requested"):
        hardware_encode_probe(Path("ffmpeg"), original, 2, "av1", "8M")


def test_av1_matroska_extracts_sequence_header_without_reencoding(tmp_path):
    info = VideoInfo(640, 360, Fraction(30), 1, 30, False, "h264")
    for extension, hardware in [("mkv", True), ("mp4", True), ("mkv", False)]:
        args = encode_args(
            Path("in.mp4"),
            tmp_path / f"out.{extension}",
            info,
            2,
            "av1",
            hardware,
            "none",
            "8M",
            20,
        )
        assert ("extract_extradata" in args) == (extension == "mkv" and hardware)


def test_planar_pipe_color_tags_both_input_and_output(tmp_path):
    info = VideoInfo(64, 48, Fraction(30), 1, 30, False, "h264", "tv", "bt709")
    args = encode_args(
        Path("source.mp4"),
        tmp_path / "result.mp4",
        info,
        2,
        "av1",
        True,
        "none",
        "8M",
        20,
        "nv12",
    )
    before_input = args[: args.index("-i")]
    after_input = args[args.index("-i") + 2 :]
    for group in (before_input, after_input):
        assert group[group.index("-colorspace") + 1] == "bt709"
        assert group[group.index("-color_range") + 1] == "tv"
    rgb = encode_args(
        Path("source.mp4"),
        tmp_path / "rgb.mp4",
        info,
        2,
        "h264",
        False,
        "none",
        "8M",
        20,
        "rgb24",
    )
    assert "-colorspace" not in rgb[: rgb.index("-i")]


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


def test_cached_tool_requires_native_architecture_and_verified_marker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from npu_sr.ffmpeg import tool_path

    root = tmp_path / "npu-sr/tools"
    for architecture in ("arm64", "x64"):
        directory = root / f"ffmpeg-{architecture}-test"
        binary = directory / "extracted/test/bin/ffmpeg.exe"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(architecture.encode())
        (directory / "ready.json").write_text(
            json.dumps(
                {
                    "executable": binary.name,
                    "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                }
            )
        )
    monkeypatch.setattr(
        "npu_sr.ffmpeg.os", SimpleNamespace(name="nt", environ={"LOCALAPPDATA": str(tmp_path)})
    )
    monkeypatch.setattr("npu_sr.ffmpeg.platform.machine", lambda: "ARM64")
    monkeypatch.setattr("npu_sr.ffmpeg.shutil.which", lambda _: None)
    selected = tool_path()
    assert "ffmpeg-arm64-test" in str(selected)
    selected.write_bytes(b"tampered")
    with pytest.raises(SRException, match="unavailable"):
        tool_path()


@pytest.mark.parametrize("exit_code", [0, 0xC0000005])
def test_acquisition_checks_startup_before_accepting_cache(tmp_path, monkeypatch, exit_code):
    import zipfile
    from types import SimpleNamespace

    from npu_sr import install_ffmpeg as module

    name = f"ffmpeg-{module.VERSION}-winarm64-gpl-shared-9.0.zip"
    with zipfile.ZipFile(tmp_path / name, "w") as archive:
        archive.writestr(name.removesuffix(".zip") + "/bin/ffmpeg.exe", b"logic fixture")
    monkeypatch.setitem(
        module.HASHES, "arm64", hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=exit_code)
    )
    if exit_code:
        with pytest.raises(ValueError, match="failed startup"):
            module.download("arm64", tmp_path)
        assert not (tmp_path / "ready.json").exists()
    else:
        assert module.download("arm64", tmp_path).is_file()
        assert (tmp_path / "ready.json").is_file()


def test_interrupted_download_never_leaves_an_accepted_archive(tmp_path, monkeypatch):
    from npu_sr import install_ffmpeg

    class Interrupted:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            raise OSError("connection interrupted")

    monkeypatch.setattr(install_ffmpeg.urllib.request, "urlopen", lambda *a, **k: Interrupted())
    with pytest.raises(OSError, match="interrupted"):
        install_ffmpeg.download("arm64", tmp_path)
    assert not list(tmp_path.iterdir())


def test_download_rejects_archive_traversal_even_with_matching_hash(tmp_path, monkeypatch):
    import zipfile

    from npu_sr import install_ffmpeg

    name = f"ffmpeg-{install_ffmpeg.VERSION}-winarm64-gpl-shared-9.0.zip"
    with zipfile.ZipFile(tmp_path / name, "w") as archive:
        archive.writestr("../outside.exe", b"fixture")
    monkeypatch.setitem(
        install_ffmpeg.HASHES, "arm64", hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="Unsafe"):
        install_ffmpeg.download("arm64", tmp_path)
    assert not (tmp_path.parent / "outside.exe").exists()


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
