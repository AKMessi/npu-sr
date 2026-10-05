from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from npu_sr.errors import SRException
from npu_sr.image import infer_y, load_image, postprocess, preprocess, save_image
from npu_sr.model import CORE, INPUT_SHAPE


def test_preprocess_luminance_and_layout():
    image = Image.new("RGB", (31, 17), (255, 0, 0))
    prepared = preprocess(image)
    tensor = next(prepared.tiles())[2]
    assert tensor.shape == tuple(INPUT_SHAPE)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor, 0.299)
    assert prepared.tile_count == 1


@pytest.mark.parametrize("size", [(1, 1), (17, 31), (CORE, CORE), (129, 257)])
def test_tiles_cover_odd_sizes_without_seams(size):
    rng = np.random.default_rng(4)
    pixels = rng.integers(0, 256, (size[1], size[0], 3), np.uint8)
    prepared = preprocess(Image.fromarray(pixels))

    class RepeatRuntime:
        def run(self, tensor):
            return tensor.repeat(2, axis=2).repeat(2, axis=3)

    y, duration = infer_y(prepared, RepeatRuntime())
    expected = (
        ((pixels.astype(np.float32) / 255) * np.array([0.299, 0.587, 0.114], np.float32))
        .sum(axis=-1)
        .repeat(2, axis=0)
        .repeat(2, axis=1)
    )
    np.testing.assert_allclose(y, expected, atol=1e-6)
    assert duration > 0


def test_postprocess_color_alpha_clamp():
    image = Image.new("RGBA", (4, 3), (180, 90, 20, 100))
    prepared = preprocess(image)
    y = np.full((6, 8), (180 * 0.299 + 90 * 0.587 + 20 * 0.114) / 255, np.float32)
    result = postprocess(y, prepared)
    assert result.mode == "RGBA" and result.size == (8, 6)
    assert result.getpixel((3, 3)) == (180, 90, 20, 100)
    assert np.asarray(postprocess(y + 10, prepared))[..., :3].max() == 255
    assert np.asarray(postprocess(y - 10, prepared))[..., :3].min() == 0
    with pytest.raises(SRException, match="Invalid luminance"):
        postprocess(np.full((6, 8), np.nan), prepared)


def test_save_load_and_jpeg_alpha(tmp_path: Path):
    image = Image.new("RGBA", (8, 4), (255, 0, 0, 0))
    path = tmp_path / "nested" / "image.png"
    save_image(image, path)
    assert load_image(path).mode == "RGBA"
    jpeg = tmp_path / "image.jpg"
    save_image(image, jpeg)
    assert load_image(jpeg).getpixel((1, 1)) == (255, 255, 255)
    with pytest.raises(SRException, match="Unsupported output"):
        save_image(image, tmp_path / "x.tiff")


def test_image_encoding_disk_failure_preserves_previous_output(tmp_path, monkeypatch):
    import errno

    path = tmp_path / "previous.png"
    image = Image.new("RGB", (16, 16), "gray")
    image.save(path)
    before = path.read_bytes()

    def fail(self, target, **kwargs):
        Path(target).write_bytes(b"partial encoded bytes")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(SRException, match="No space"):
        save_image(image, path)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_bad_image_and_unsupported_input(tmp_path: Path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(SRException, match="Cannot load image"):
        load_image(path)
    Image.new("RGB", (2, 2)).save(tmp_path / "image.bmp")
    with pytest.raises(SRException, match="Unsupported input"):
        load_image(tmp_path / "image.bmp")


def test_exif_rotation(tmp_path: Path):
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    Image.new("RGB", (10, 5)).save(path, exif=exif)
    assert load_image(path).size == (5, 10)


def test_plane_quantization_matches_original_vectorized_formula():
    from npu_sr.image import resize_plane

    rng = np.random.default_rng(13)
    prepared = preprocess(Image.fromarray(rng.integers(0, 256, (19, 27, 3), np.uint8)))
    y = rng.uniform(-0.2, 1.2, (38, 54)).astype(np.float32)
    cb = resize_plane(prepared.cb, (54, 38)) - 0.5
    cr = resize_plane(prepared.cr, (54, 38)) - 0.5
    red, blue = y + cr / 0.713, y + cb / 0.564
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    expected = np.rint(np.clip(np.stack([red, green, blue], -1), 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(np.asarray(postprocess(y, prepared)), expected)
