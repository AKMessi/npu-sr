"""Make a small, attributed GIF from measured video outputs, without rerunning SR."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from npu_sr.ffmpeg import run_tool, tool_path
from npu_sr.model import sha256


def crops(source: Path, rectangle: str, ffmpeg: Path) -> list[Image.Image]:
    result = run_tool(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"crop={rectangle},fps=6",
            "-frames:v",
            "12",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    width, height = map(int, rectangle.split(":")[:2])
    size = width * height * 3
    if len(result.stdout) != size * 12:
        raise ValueError("Expected twelve preview crops from the measured four-second clip")
    return [
        Image.frombytes("RGB", (width, height), result.stdout[i * size : (i + 1) * size])
        for i in range(12)
    ]


def create(directory: Path, report: Path, output: Path) -> None:
    measurements = json.loads(report.read_text(encoding="utf-8"))
    provenance = measurements["provenance"]
    source = directory / "faces-960x540.mp4"
    expected = next(clip for clip in provenance["clips"] if clip["name"] == "faces")
    if sha256(source) != expected["inputs"]["960x540"]["sha256"]:
        raise ValueError("Input does not match the measured fixture hash")
    paths = [source]
    for preset in ("bicubic", "realtime"):
        path = directory / "v04-quality" / f"faces-{preset}.mp4"
        row = next(
            row
            for row in measurements["video"]
            if row["clip"] == "faces" and row["preset"] == preset
        )
        if sha256(path) != row["output_sha256"]:
            raise ValueError(f"{preset} output does not match the measured hash")
        paths.append(path)
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("Preview must not overwrite a measured source")
    ffmpeg = tool_path()
    panels = [crops(paths[0], "180:120:520:50", ffmpeg)]
    panels += [crops(path, "360:240:1040:100", ffmpeg) for path in paths[1:]]
    frames = []
    font = ImageFont.load_default(size=14)
    labels = ("540p input (nearest display)", "FFmpeg bicubic 2x", "QNN realtime 2x (strength 0.5)")
    for index in range(12):
        canvas = Image.new("RGB", (1080, 288), "#111111")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(labels):
            image = panels[column][index]
            if column == 0:
                image = image.resize((360, 240), Image.Resampling.NEAREST)
            canvas.paste(image, (column * 360, 24))
            draw.text((column * 360 + 6, 4), label, font=font, fill="white")
        draw.text(
            (6, 269),
            "Crop / 6 FPS preview | Tears of Steel: "
            "Blender Foundation / mango.blender.org, CC BY 3.0",
            font=font,
            fill="white",
        )
        frames.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output.with_suffix(".png"))
    palette = frames[0].quantize(colors=128)
    indexed = [palette] + [frame.quantize(palette=palette) for frame in frames[1:]]
    indexed[0].save(
        output,
        save_all=True,
        append_images=indexed[1:],
        duration=[170, 170, 160] * 4,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Measured-output crop preview saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", ".")) / "npu-sr/benchmarks/video",
    )
    parser.add_argument("--report", type=Path, default=Path("benchmarks/v0.4/quality.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/video-example.gif"))
    args = parser.parse_args()
    create(args.directory, args.report, args.output)
