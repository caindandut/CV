import numpy as np
import pytest
from PIL import Image
from app.media import annotation, confidence_text, validate_media, safe_path


def test_confidence():
    assert confidence_text(0.952) == "95,2%"


def test_real_format_validation(tmp_path):
    wrong = tmp_path / "fake.jpg"
    wrong.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        validate_media(wrong, "image")
    actual = tmp_path / "real.jpg"
    Image.new("RGB", (100, 80)).save(actual)
    assert validate_media(actual, "image") == "image/jpeg"
    png = tmp_path / "fake.mp4"
    Image.new("RGB", (100, 80)).save(png, format="PNG")
    with pytest.raises(ValueError):
        validate_media(png, "video")


def test_path_escape(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside")
    assert safe_path(tmp_path, "uploads/a.png").is_relative_to(tmp_path)


def test_annotation_colors_and_input_unchanged():
    pixels = np.zeros((200, 500, 3), dtype=np.uint8)
    result = annotation(
        pixels, [{"class_id": 2, "confidence": 0.875, "xyxy": [20, 70, 350, 180]}]
    )
    assert result.shape == pixels.shape
    assert not pixels.any()
    assert result[180, 20, 2] > result[180, 20, 0]  # BGR: red box
