"""Generate an original MIT-licensed test card, with no third-party image assets."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    width, height = 512, 320
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack([25 + xx * 0.06, 40 + yy * 0.10, 72 + xx * 0.04], axis=-1)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    small = ImageFont.load_default(size=18)
    draw.text((24, 20), "NPU-SR / 2x", fill="#f4f2ec", font=font)
    draw.text((24, 60), "Edges, curves, texture, small text", fill="#c5d4e8", font=small)
    draw.rounded_rectangle((24, 110, 220, 286), radius=24, fill="#eeb864")
    for x in range(44, 200, 12):
        draw.line((x, 130, x, 266), fill="#624a38", width=3)
    draw.ellipse((256, 110, 440, 294), fill="#68c4c1")
    draw.ellipse((282, 136, 414, 268), fill="#214555")
    draw.line((260, 288, 470, 114), fill="#ffffff", width=3)
    for x in range(462, 500, 6):
        draw.line((x, 210, x, 286), fill="#f4f2ec", width=2)
    directory = Path(__file__).resolve().parents[1] / "examples"
    directory.mkdir(exist_ok=True)
    image.save(directory / "reference.png")
    image.resize((width // 2, height // 2), Image.Resampling.BICUBIC).save(directory / "input.png")


if __name__ == "__main__":
    main()
