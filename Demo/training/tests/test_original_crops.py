import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_original_crop_preserves_source_and_records_comparison_labels(tmp_path):
    from prepare_dataset import digest
    from training_crops import prepare_original_crop

    image = tmp_path / "IMG_1.rf.original.png"
    Image.new("RGB", (600, 500), "red").save(image)
    label = tmp_path / "source.txt"
    label.write_text("0 0.5 0.5 0.5 0.5\n2 0.05 0.05 0.1 0.1\n")
    row = {
        "id": "train/IMG_1.rf.original.png",
        "image": str(image),
        "label": str(label),
        "sha256": digest(image),
        "label_sha256": digest(label),
        "width": 600,
        "height": 500,
        "labels": [[0, 0.5, 0.5, 0.5, 0.5], [2, 0.05, 0.05, 0.1, 0.1]],
        "source": "scene:test",
        "phash": "0",
    }
    result = prepare_original_crop(
        row,
        {"proposal": 1, "crop": [100, 100, 500, 500], "canvas": [400, 400]},
        tmp_path / "crops",
    )
    assert result["id"] == "original_crops/IMG_1.rf.original.png"
    assert result["source_sha256"] == row["sha256"] == digest(image)
    assert result["source_labels"] == row["labels"]
    assert result["source_id"] == row["id"]
    assert result["labels"] == [[0, 0.5, 0.375, 0.75, 0.625]]
    assert row["width"] == 600 and len(row["labels"]) == 2
    assert result["width"] == result["height"] == 400
    assert result["label_sha256"] == digest(label)
