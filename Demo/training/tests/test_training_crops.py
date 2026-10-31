import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_crop_is_exact_and_does_not_modify_original(tmp_path):
    from prepare_dataset import digest
    from training_crops import prepare_review_image

    source = tmp_path / "source.png"
    im = Image.new("RGB", (600, 500), "blue")
    im.putpixel((110, 90), (255, 0, 0))
    im.save(source)
    before = digest(source)
    result, transform = prepare_review_image(
        source,
        {"crop": [100, 80, 400, 380], "canvas": [300, 300]},
        tmp_path / "crops",
        123,
    )
    assert digest(source) == before
    assert transform["source_sha256"] == before
    with Image.open(result) as crop:
        assert crop.size == (300, 300)
        assert crop.getpixel((10, 10)) == (255, 0, 0)
    assert result.suffix == ".png"


@pytest.mark.parametrize(
    "crop,canvas",
    [
        ([-1, 0, 300, 300], [301, 300]),
        ([0, 0, 601, 300], [601, 300]),
        ([0, 0, 0, 300], [0, 300]),
        ([0.5, 0, 300, 300], [300, 300]),
        ([0, 0, 300, 300], [600, 500]),
        ([0, 0, 100, 100], [100, 100]),
    ],
)
def test_invalid_or_too_small_crop_is_rejected(tmp_path, crop, canvas):
    from training_crops import prepare_review_image

    source = tmp_path / "source.png"
    Image.new("RGB", (600, 500)).save(source)
    with pytest.raises(ValueError):
        prepare_review_image(
            source, {"crop": crop, "canvas": canvas}, tmp_path / "crops", 123
        )


def test_existing_crop_cannot_be_silently_replaced(tmp_path):
    from training_crops import prepare_review_image

    source = tmp_path / "source.png"
    Image.new("RGB", (600, 500), "red").save(source)
    prepare_review_image(
        source,
        {"crop": [0, 0, 300, 300], "canvas": [300, 300]},
        tmp_path / "crops",
        123,
    )
    Image.new("RGB", (600, 500), "blue").save(source)
    with pytest.raises(ValueError, match="Existing crop differs"):
        prepare_review_image(
            source,
            {"crop": [0, 0, 300, 300], "canvas": [300, 300]},
            tmp_path / "crops",
            123,
        )


def test_narrow_native_crop_can_exclude_unresolved_edge_objects(tmp_path):
    from training_crops import prepare_review_image

    source = tmp_path / "source.png"
    raw = Image.new("RGB", (400, 400), "blue")
    raw.putpixel((255, 275), (255, 0, 0))
    raw.save(source)
    result, lineage = prepare_review_image(
        source,
        {"crop": [10, 20, 256, 276], "canvas": [246, 256]},
        tmp_path / "crops",
        1,
    )
    with Image.open(result) as crop:
        assert crop.size == (246, 256)
        assert crop.getpixel((245, 255)) == (255, 0, 0)
    assert lineage["crop"] == [10, 20, 256, 276]


def test_crop_below_240_pixel_edge_still_rejected(tmp_path):
    from training_crops import prepare_review_image

    source = tmp_path / "source.png"
    Image.new("RGB", (400, 400)).save(source)
    with pytest.raises(ValueError):
        prepare_review_image(
            source,
            {"crop": [0, 0, 239, 256], "canvas": [239, 256]},
            tmp_path / "crops",
            1,
        )
