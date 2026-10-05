from pathlib import Path

import pytest
from PIL import Image

from npu_sr import __version__
from npu_sr.cli import main, parser


def test_parsing_and_version(capsys):
    assert parser().parse_args(["upscale", "test.jpg", "--device", "npu"]).device == "npu"
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
    with pytest.raises(SystemExit):
        parser().parse_args(["benchmark", "image.png", "--runs", "0"])
    assert parser().parse_args(["video", "in.mp4", "-o", "out.mp4"]).preset is None
    assert parser().parse_args(["benchmark-realtime", "in.mp4", "-o", "trials"]).trials == 3
    assert not parser().parse_args(["video", "in.mp4", "-o", "out.mp4"]).verify_full
    for command in ("video", "benchmark-video", "benchmark-realtime"):
        assert parser().parse_args([command, "in.mp4", "-o", "out", "--verify-full"]).verify_full


def test_cli_cpu_end_to_end(tiny_model, tmp_path: Path, capsys):
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.new("RGB", (129, 31), "orange").save(source)
    assert (
        main(
            [
                "upscale",
                str(source),
                "-o",
                str(output),
                "--device",
                "cpu",
                "--model",
                str(tiny_model),
                "--comparison-dir",
                str(tmp_path / "comparison"),
            ]
        )
        == 0
    )
    assert Image.open(output).size == (258, 62)
    assert "ONNX Runtime CPU" in capsys.readouterr().out
    assert len(list((tmp_path / "comparison").glob("*.png"))) == 3


def test_cli_errors_are_concise(tmp_path, capsys):
    assert main(["upscale", str(tmp_path / "absent.png"), "--device", "cpu"]) == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err and "Traceback" not in captured.err
    assert main(["upscale", "same.png", "-o", "same.png"]) == 1
    assert "Output must differ" in capsys.readouterr().err


@pytest.mark.parametrize(
    "problem,device,preset",
    [
        (None, None, "realtime"),
        ("non-Windows", None, "balanced"),
        (None, "cpu", "balanced"),
    ],
)
def test_one_command_selects_sensible_preset_without_hiding_device(
    monkeypatch, capsys, problem, device, preset
):
    monkeypatch.setattr("npu_sr.qnn.platform_problem", lambda: problem)
    args = ["video", "missing.mp4", "-o", "result.mp4"]
    if device:
        args += ["--device", device]
    assert main(args) == 1
    assert f"Configuration: {preset}" in capsys.readouterr().out


def test_explicit_legacy_model_keeps_custom_video_behavior(capsys):
    assert main(["video", "missing.mp4", "-o", "result.mp4", "--model", "espcn-x2-256"]) == 1
    text = capsys.readouterr().out
    assert "custom/default" in text and "frames=rgb24" in text


def test_diagnostic_json_cannot_overwrite_model_manifest(tiny_model, capsys):
    manifest = tiny_model.with_suffix(".json")
    before = manifest.read_bytes()
    assert main(["doctor", "--model", str(tiny_model), "--json", str(manifest)]) == 1
    assert manifest.read_bytes() == before
    assert "overwrite" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command", ["benchmark", "benchmark-suite", "video", "benchmark-video", "benchmark-realtime"]
)
def test_report_cannot_replace_selected_model_manifest(
    command, tiny_model, tmp_path, capsys, monkeypatch
):
    manifest = tiny_model.with_suffix(".json")
    before = manifest.read_bytes()
    args = [command, str(tmp_path / "source.png"), "--json", str(manifest)]
    if command == "benchmark-suite":
        monkeypatch.setenv("NPU_SR_MODEL_DIR", str(tiny_model.parent))
        args += ["--models", "espcn-x2"]
    else:
        args += ["--model", str(tiny_model)]
    if command in {"video", "benchmark-video", "benchmark-realtime"}:
        args += ["-o", str(tmp_path / "result.mp4")]
    assert main(args) == 1
    assert manifest.read_bytes() == before
    assert "overwrite" in capsys.readouterr().err


def test_report_rejects_a_hardlink_to_model_manifest(tiny_model, tmp_path, capsys):
    manifest = tiny_model.with_suffix(".json")
    alias = tmp_path / "report.json"
    try:
        alias.hardlink_to(manifest)
    except OSError:
        pytest.skip("Filesystem does not support hardlink regression")
    before = manifest.read_bytes()
    assert main(["benchmark", "missing.png", "--model", str(tiny_model), "--json", str(alias)]) == 1
    assert manifest.read_bytes() == before and alias.read_bytes() == before
    assert "overwrite" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["benchmark-video", "benchmark-realtime"])
def test_checkpoint_json_cannot_replace_a_generated_video(command, tmp_path, capsys):
    directory = tmp_path / "trials"
    target = directory / "trial-1.mp4"
    assert main([command, "missing.mp4", "-o", str(directory), "--json", str(target)]) == 1
    assert "overwrite" in capsys.readouterr().err and not directory.exists()


def test_extreme_trial_count_is_rejected_before_allocating_output_paths(tmp_path, capsys):
    assert (
        main(["benchmark-video", "missing.mp4", "-o", str(tmp_path), "--trials", "100000000"]) == 1
    )
    assert "trials must be" in capsys.readouterr().err


def test_image_output_hardlink_cannot_destroy_input(tiny_model, tmp_path, capsys):
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.new("RGB", (16, 16), "gray").save(source)
    try:
        output.hardlink_to(source)
    except OSError:
        pytest.skip("Filesystem does not support hardlink regression")
    before = source.read_bytes()
    assert (
        main(
            [
                "upscale",
                str(source),
                "-o",
                str(output),
                "--device",
                "cpu",
                "--model",
                str(tiny_model),
            ]
        )
        == 1
    )
    assert source.read_bytes() == before and "overwrite" in capsys.readouterr().err


def test_redirected_cli_dimensions_are_utf8(tiny_model, tmp_path):
    import os
    import subprocess
    import sys

    source, output = tmp_path / "source.png", tmp_path / "result.png"
    Image.new("RGB", (37, 23), "gray").save(source)
    # Reproduce a Windows-style pipe encoding even on Linux CI.
    environment = os.environ | {"PYTHONIOENCODING": "cp1252"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npu_sr.cli",
            "upscale",
            str(source),
            "-o",
            str(output),
            "--model",
            str(tiny_model),
            "--device",
            "cpu",
        ],
        capture_output=True,
        env=environment,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "37 × 23" in result.stdout.decode("utf-8", "strict")
    assert Image.open(output).size == (74, 46)
