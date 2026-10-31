import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_training_priority_excludes_any_author_also_in_holdout():
    from propose_candidate_crops import train_source_subset

    manifest = [
        {"id": "a", "new_split": "train"},
        {"id": "b", "new_split": "test"},
        {"id": "c", "new_split": "train"},
        {"id": "d", "new_split": "val"},
    ]
    attribution = [
        {"id": "a", "artist": "A"},
        {"id": "b", "artist": "B"},
        {"id": "c", "artist": "Shared"},
        {"id": "d", "artist": "Shared"},
    ]
    sources = [
        {"pageid": i, "extmetadata": {"Artist": {"value": name}}}
        for i, name in enumerate(["A", "B", "Shared"])
    ]
    assert [
        r["pageid"] for r in train_source_subset(sources, manifest, attribution)
    ] == [0]


def test_only_a_spatially_associated_person_and_motorcycle_propose_a_crop():
    from propose_candidate_crops import choose_crop

    detections = [
        {"name": "motorcycle", "xyxy": [400, 400, 700, 700]},
        {"name": "person", "xyxy": [440, 150, 640, 660]},
        {"name": "person", "xyxy": [0, 0, 100, 200]},
    ]
    rect = choose_crop(detections, (1000, 800))
    assert rect[0] <= 400 and rect[1] <= 150 and rect[2] >= 700 and rect[3] >= 700
    assert rect[0] > 100, "Unrelated pedestrian should not enlarge crop"
    assert min(rect[2] - rect[0], rect[3] - rect[1]) >= 256


def test_no_rider_pair_is_not_proposed_and_edges_are_clamped():
    from propose_candidate_crops import choose_crop

    assert (
        choose_crop([{"name": "person", "xyxy": [0, 0, 200, 500]}], (1000, 800)) is None
    )
    rect = choose_crop(
        [
            {"name": "person", "xyxy": [0, 0, 150, 350]},
            {"name": "motorcycle", "xyxy": [0, 250, 200, 500]},
        ],
        (640, 500),
    )
    assert 0 <= rect[0] < rect[2] <= 640 and 0 <= rect[1] < rect[3] <= 500
