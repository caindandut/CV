import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_new_snapshot_inherits_latest_frozen_holdout(monkeypatch, tmp_path):
    import build_dataset_2000

    monkeypatch.setattr(build_dataset_2000, "ROOT", tmp_path)
    (tmp_path / "acquisition_2000").mkdir()
    (tmp_path / "dataset_v10").mkdir()
    manifest = tmp_path / "dataset_v10/manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    (tmp_path / "acquisition_2000/progress.json").write_text(
        json.dumps({"frozen_dataset": "dataset_v10"}), encoding="utf-8"
    )
    assert build_dataset_2000.latest_parent_manifest() == manifest


def test_snapshot_parent_rejects_path_outside_training(monkeypatch, tmp_path):
    import build_dataset_2000
    import pytest

    monkeypatch.setattr(build_dataset_2000, "ROOT", tmp_path)
    (tmp_path / "acquisition_2000").mkdir()
    (tmp_path / "acquisition_2000/progress.json").write_text(
        json.dumps({"frozen_dataset": "../other"}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        build_dataset_2000.latest_parent_manifest()


def test_progress_counts_exported_frames_without_augmented_duplicates():
    from progress_2000 import source_frame_count

    assert (
        source_frame_count(
            [
                {"id": "train/frame.rf.a.jpg"},
                {"id": "valid/frame.rf.b.jpg"},
                {"id": "commons/123.jpg"},
            ]
        )
        == 2
    )


def test_two_bareheaded_riders_are_retained_in_r080():
    rows = json.loads(
        (ROOT / "acquisition_2000/original_reviews_batch01.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(r for r in rows if r["proposal"] == 80)
    assert Counter(b[0] for b in row["labels"]) == {0: 4, 2: 2}
    assert all(b[2] > 0.25 for b in row["labels"] if b[0] == 2), (
        "Do not retain the false rider on the parked bike"
    )


def test_r020_has_one_helmeted_rider_and_three_motorcycles():
    rows = json.loads(
        (ROOT / "acquisition_2000/original_reviews_batch01.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(r for r in rows if r["proposal"] == 20)
    assert Counter(b[0] for b in row["labels"]) == {0: 3, 1: 1}
