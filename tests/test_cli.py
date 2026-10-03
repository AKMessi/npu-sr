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
