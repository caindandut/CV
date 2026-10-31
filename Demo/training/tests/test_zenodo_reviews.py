import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def reviewed(tmp_path):
    folder = tmp_path / "acquisition_2000/research_zenodo"
    folder.mkdir(parents=True)
    video = folder / "original.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (400, 400)
    )
    assert writer.isOpened()
    for color in [(30, 80, 130), (90, 140, 190), (160, 70, 40)]:
        writer.write(np.full((400, 400, 3), color, dtype=np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ok, pixels = cap.read()
    cap.release()
    assert ok
    frame = folder / "frame.png"
    Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).save(frame)
    archive = folder / "data.zip"
    member = "VNTraffic/VNTraffic_Original-video.mp4"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(video, member)
    metadata = folder / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "id": 18195750,
                "metadata": {
                    "title": "Vehicle Tracking Dataset",
                    "access_right": "open",
                    "license": {"id": "cc-by-4.0"},
                    "description": "VnTraffic shows traffic in Vietnam",
                    "creators": [{"name": "Le, Thi-Nhung"}],
                },
                "files": [
                    {
                        "key": "Vehicle_Tracking.zip",
                        "size": archive.stat().st_size,
                        "checksum": "md5:"
                        + hashlib.md5(archive.read_bytes()).hexdigest(),
                        "links": {
                            "self": "https://zenodo.org/api/records/18195750/files/Vehicle_Tracking.zip/content"
                        },
                    }
                ],
            }
        )
    )
    evidence = folder / "overlay.png"
    Image.new("RGB", (300, 300)).save(evidence)
    entry = {
        "provider": "zenodo",
        "record_id": "18195750",
        "subset": "VNTraffic",
        "country": "VN",
        "record_url": "https://zenodo.org/records/18195750",
        "metadata_file": "research_zenodo/metadata.json",
        "metadata_sha256": sha(metadata),
        "archive_file": "research_zenodo/data.zip",
        "archive_sha256": sha(archive),
        "video_archive_member": member,
        "video_file": "research_zenodo/original.mp4",
        "video_sha256": sha(video),
        "frame_file": "research_zenodo/frame.png",
        "frame_sha256": sha(frame),
        "frame_index": 1,
        "frame_time_seconds": 1 / 30,
        "source_canvas": [400, 400],
        "crop": [20, 30, 320, 330],
        "canvas": [300, 300],
        "boxes": [[0, 40, 90, 280, 290], [2, 60, 20, 250, 260]],
        "source_group": "zenodo:18195750:VNTraffic",
        "review_episode": "stationary-greyshirt",
        "location_evidence": "Primary VnTraffic description explicitly establishes Vietnam",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "status": "corrected",
        "visual_reviewed": True,
        "source_verified": True,
        "evidence": "research_zenodo/overlay.png",
        "evidence_sha256": sha(evidence),
        "reason": "One independently reviewed crop of seated rider and motorcycle",
    }
    return tmp_path, entry


def ledger(root, entries):
    (root / "acquisition_2000/zenodo_reviews_batch01.json").write_text(
        json.dumps(entries)
    )


def test_import_redecodes_exact_frame_and_preserves_crop_and_credit(reviewed):
    from ncpedia_reviews import is_external
    from zenodo_reviews import load_zenodo_reviews, verify_zenodo_attribution

    root, entry = reviewed
    ledger(root, [entry])
    rows, reviews, attrs, audit = load_zenodo_reviews(root)
    row = rows[0]
    assert row["id"] == "zenodo-vntraffic/frame_000001.png" and is_external(row)
    assert row["train_only"] is True
    assert len(reviews[row["id"]]["labels"]) == 2
    assert "Le, Thi-Nhung" in attrs[0]["artist"]
    assert attrs[0]["annotation_license"] == "CC BY 4.0"
    assert audit[0]["frame_index"] == 1
    expected = (
        Image.open(root / "acquisition_2000" / entry["frame_file"])
        .convert("RGB")
        .crop(entry["crop"])
    )
    actual = Image.open(row["image"]).convert("RGB")
    assert ImageChops.difference(expected, actual).getbbox() is None
    verify_zenodo_attribution(rows, attrs, root)
    changed = copy.deepcopy(attrs)
    changed[0]["artist"] = "Unrelated uploader"
    with pytest.raises(ValueError):
        verify_zenodo_attribution(rows, changed, root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("record_id", "999"),
        ("subset", "AICC22-Custom"),
        ("country", "TH"),
        ("video_archive_member", "AICC22-Custom/AICC22-Custom_Original-video.mp4"),
        ("frame_index", -1),
        ("frame_index", 2),
        ("frame_index", 99),
        ("frame_index", 1.5),
        ("frame_time_seconds", 99),
        ("archive_sha256", "0" * 64),
        ("video_sha256", "0" * 64),
        ("frame_sha256", "0" * 64),
        ("metadata_sha256", "0" * 64),
        ("frame_file", "../outside.png"),
        ("visual_reviewed", False),
        ("source_verified", False),
        ("evidence_sha256", "0" * 64),
        ("review_episode", ""),
    ],
)
def test_import_rejects_foreign_subset_wrong_frame_or_tampering(reviewed, field, value):
    from zenodo_reviews import load_zenodo_reviews

    root, entry = reviewed
    entry[field] = value
    ledger(root, [entry])
    with pytest.raises((ValueError, FileNotFoundError)):
        load_zenodo_reviews(root)


def test_duplicate_stationary_actor_episode_not_counted_again(reviewed):
    from zenodo_reviews import load_zenodo_reviews

    root, entry = reviewed
    ledger(root, [entry, copy.deepcopy(entry)])
    with pytest.raises(ValueError, match="Duplicate"):
        load_zenodo_reviews(root)


def test_metadata_license_change_is_rejected_even_after_rehash(reviewed):
    from zenodo_reviews import load_zenodo_reviews

    root, entry = reviewed
    metadata = root / "acquisition_2000" / entry["metadata_file"]
    data = json.loads(metadata.read_text())
    data["metadata"]["license"]["id"] = "cc0-1.0"
    metadata.write_text(json.dumps(data))
    entry["metadata_sha256"] = sha(metadata)
    ledger(root, [entry])
    with pytest.raises(ValueError):
        load_zenodo_reviews(root)


def test_wrong_frame_pixels_rejected_even_with_matching_timestamp(reviewed):
    from zenodo_reviews import load_zenodo_reviews

    root, entry = reviewed
    entry.update(frame_index=2, frame_time_seconds=2 / 30)
    ledger(root, [entry])
    with pytest.raises(ValueError, match="Decoded frame"):
        load_zenodo_reviews(root)


def test_same_actor_episode_rejected_across_distinct_valid_frames(reviewed):
    from zenodo_reviews import load_zenodo_reviews

    root, first = reviewed
    folder = root / "acquisition_2000/research_zenodo"
    cap = cv2.VideoCapture(str(folder / "original.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
    ok, pixels = cap.read()
    cap.release()
    assert ok
    frame = folder / "frame2.png"
    Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).save(frame)
    second = copy.deepcopy(first)
    second.update(
        frame_index=2,
        frame_time_seconds=2 / 30,
        frame_file="research_zenodo/frame2.png",
        frame_sha256=sha(frame),
    )
    ledger(root, [first, second])
    with pytest.raises(ValueError, match="Duplicate"):
        load_zenodo_reviews(root)


def test_published_archive_checksum_must_match_actual_bytes(reviewed):
    from zenodo_reviews import load_zenodo_reviews

    root, entry = reviewed
    path = root / "acquisition_2000" / entry["metadata_file"]
    metadata = json.loads(path.read_text())
    metadata["files"][0]["checksum"] = "md5:" + "0" * 32
    path.write_text(json.dumps(metadata))
    entry["metadata_sha256"] = sha(path)
    ledger(root, [entry])
    with pytest.raises(ValueError, match="published checksum"):
        load_zenodo_reviews(root)
