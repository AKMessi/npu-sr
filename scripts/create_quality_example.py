"""Make an attributed crop preview from the measured v0.5 conversation outputs."""

import argparse
import json
from pathlib import Path

from create_video_example import crops
from PIL import Image, ImageDraw, ImageFont

from npu_sr.ffmpeg import tool_path
from npu_sr.model import sha256


def create(directory: Path, report: Path, output: Path) -> None:
    measurements = json.loads(report.read_text(encoding="utf-8"))
    comparison = measurements["comparisons"]["selected-development"]
    clip = next(c for c in comparison["preparation"]["clips"] if c["identifier"] == "conversation")
    paths = [directory / clip["reference_file"], directory / clip["input_file"]]
    hashes = [clip["reference_sha256"], clip["input_sha256"]]
    for model in ("bicubic", "quicksrnet-small-y-x2"):
        row = next(
            r for r in comparison["results"] if r["clip"] == "conversation" and r["model"] == model
        )
        candidates = [
            directory / "delivered-tagged/development" / f"conversation-{model}-1{suffix}.mp4"
            for suffix in ("", "-nv12")
        ]
        path = next((p for p in candidates if p.is_file()), candidates[0])
        paths.append(path)
        hashes.append(row["metrics"]["output_sha256"])
    if output.resolve() in {p.resolve() for p in paths}:
        raise ValueError("Preview must not overwrite measured media")
    for path, expected in zip(paths, hashes, strict=True):
        if sha256(path) != expected:
            raise ValueError(f"Measured bytes changed: {path.name}")
    ffmpeg = tool_path()
    panels = [
        crops(path, "180:120:500:130" if i == 1 else "360:240:1000:260", ffmpeg)
        for i, path in enumerate(paths)
    ]
    labels = (
        "1080p reference",
        "540p input (nearest display)",
        "Bicubic + AV1",
        "QuickSRNet Y / QNN + AV1",
    )
    frames = []
    font = ImageFont.load_default(size=14)
    for index in range(12):
        canvas = Image.new("RGB", (1440, 288), "#111111")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(labels):
            panel = panels[column][index]
            if column == 1:
                panel = panel.resize((360, 240), Image.Resampling.NEAREST)
            canvas.paste(panel, (column * 360, 24))
            draw.text((column * 360 + 6, 4), label, font=font, fill="white")
        draw.text(
            (6, 269),
            "6 FPS crop preview | Tears of Steel: "
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
    print(f"Measured preview: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("benchmarks/v0.5/quality.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/quality-v05.gif"))
    args = parser.parse_args()
    create(args.directory, args.report, args.output)
