import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def reviewed(tmp_path):
    folder = tmp_path / "acquisition_2000/research_kaggle"
    folder.mkdir(parents=True)
    image = folder / "1953.jpg"
    Image.new("RGB", (400, 400), (80, 120, 170)).save(image)
    archive = folder / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(image, "archive/images/1953.jpg")
    metadata = folder / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "ref": "maitam/vietnamese-traffic-signs",
                "currentVersionNumber": 8,
                "licenseName": "CC BY-SA 4.0",
                "description": "Vietnam images collected by Data Palpating Team",
            }
        )
    )
    evidence = tmp_path / "overlay.png"
    Image.new("RGB", (300, 300)).save(evidence)
    entry = {
        "provider": "kaggle",
        "dataset_ref": "maitam/vietnamese-traffic-signs",
        "version": 8,
        "image_id": "1953",
        "archive_entry": "archive/images/1953.jpg",
        "archive_file": "research_kaggle/data.zip",
        "archive_sha256": sha(archive),
        "metadata_file": "research_kaggle/metadata.json",
        "metadata_sha256": sha(metadata),
        "source_image": "research_kaggle/1953.jpg",
        "source_sha256": sha(image),
        "source_canvas": [400, 400],
        "crop": [20, 30, 320, 330],
        "canvas": [300, 300],
        "boxes": [[0, 40, 70, 220, 290], [1, 60, 20, 210, 250]],
        "country": "VN",
        "location_evidence": "Primary description explicitly establishes Vietnam",
        "source_group": "kaggle:maitam/vietnamese-traffic-signs",
        "page_url": "https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "creator": "Data Palpating Team",
        "maintainer": "Mai Tam (maitam)",
        "visual_reviewed": True,
        "source_verified": True,
        "status": "corrected",
        "evidence": "overlay.png",
        "reason": "Manually checked all visible objects",
    }
    return tmp_path, entry


def ledger(root, entries):
    (root / "acquisition_2000/kaggle_reviews_batch01.json").write_text(
        json.dumps(entries)
    )


def test_import_checks_archive_and_preserves_pixels_and_attribution(reviewed):
    from kaggle_reviews import load_kaggle_reviews, verify_kaggle_attribution
    from ncpedia_reviews import is_external

    root, entry = reviewed
    ledger(root, [entry])
    rows, reviews, attrs, audit = load_kaggle_reviews(root)
    row = rows[0]
    assert row["source"] == entry["source_group"] and is_external(row)
    assert len(reviews[row["id"]]["labels"]) == 2
    assert attrs[0]["annotation_license"] == "CC BY-SA 4.0"
    assert audit[0]["archive_entry"] == entry["archive_entry"]
    expected = (
        Image.open(root / "acquisition_2000" / entry["source_image"])
        .convert("RGB")
        .crop(entry["crop"])
    )
    actual = Image.open(row["image"]).convert("RGB")
    assert ImageChops.difference(expected, actual).getbbox() is None
    verify_kaggle_attribution(rows, attrs, root)
    altered = copy.deepcopy(attrs)
    altered[0]["annotation_license"] = "CC0"
    with pytest.raises(ValueError):
        verify_kaggle_attribution(rows, altered, root)
    altered = copy.deepcopy(attrs)
    altered[0]["artist"] = "Unrelated uploader"
    with pytest.raises(ValueError):
        verify_kaggle_attribution(rows, altered, root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("visual_reviewed", False),
        ("source_verified", False),
        ("country", "TH"),
        ("archive_sha256", "0" * 64),
        ("metadata_sha256", "0" * 64),
        ("source_sha256", "0" * 64),
        ("source_group", "random:test"),
        ("source_image", "../outside.jpg"),
        ("archive_entry", "archive/images/../other.jpg"),
        ("license", "CC0"),
        ("page_url", "https://example.com/data"),
    ],
)
def test_import_rejects_unreviewed_or_tampered_provenance(reviewed, field, value):
    from kaggle_reviews import load_kaggle_reviews

    root, entry = reviewed
    entry[field] = value
    ledger(root, [entry])
    with pytest.raises((ValueError, FileNotFoundError)):
        load_kaggle_reviews(root)


def test_rejects_multiple_crops_of_same_source(reviewed):
    from kaggle_reviews import load_kaggle_reviews

    root, entry = reviewed
    ledger(root, [entry, copy.deepcopy(entry)])
    with pytest.raises(ValueError, match="Duplicate"):
        load_kaggle_reviews(root)


def test_metadata_cannot_claim_a_different_license_even_with_updated_hash(reviewed):
    from kaggle_reviews import load_kaggle_reviews

    root, entry = reviewed
    path = root / "acquisition_2000" / entry["metadata_file"]
    meta = json.loads(path.read_text())
    meta["licenseName"] = "Unknown"
    path.write_text(json.dumps(meta))
    entry["metadata_sha256"] = sha(path)
    ledger(root, [entry])
    with pytest.raises(ValueError):
        load_kaggle_reviews(root)


def test_candidate_count_only_uses_completed_hash_checked_zip(reviewed):
    from kaggle_reviews import count_kaggle_candidates

    root, _entry = reviewed
    folder = root / "acquisition_2000/research_kaggle"
    assert count_kaggle_candidates(root) == 0
    archive = folder / "maitam_vnts_v8.zip"
    archive.write_bytes((folder / "data.zip").read_bytes())
    report = folder / "maitam_vnts_v8_acquisition01.json"
    data = {"status": "downloading", "sha256": sha(archive)}
    report.write_text(json.dumps(data))
    assert count_kaggle_candidates(root) == 0
    data["status"] = "downloaded_crc_verified_candidates_only"
    report.write_text(json.dumps(data))
    assert count_kaggle_candidates(root) == 1
    archive.write_bytes(b"incomplete or changed download")
    with pytest.raises(ValueError):
        count_kaggle_candidates(root)


@pytest.fixture
def camera_review(reviewed):
    root, entry = reviewed
    folder = root / "acquisition_2000/research_kaggle"
    metadata = folder / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "ref": "huanhuyle/vietnam-traffic-detection",
                "currentVersionNumber": 2,
                "licenseName": "Apache 2.0",
                "ownerRef": "huanhuyle",
                "creatorName": "huan huy le",
                "description": "",
            }
        )
    )
    archive = folder / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(folder / "1953.jpg", "images/train/day_cam_24_00129.jpg")
    proof_crop = folder / "proof.png"
    Image.open(folder / "1953.jpg").crop((10, 20, 110, 120)).save(proof_crop)
    gate = folder / "country_gate.json"
    gate.write_text(
        json.dumps(
            {
                "dataset": "huanhuyle/vietnam-traffic-detection",
                "version": 2,
                "camera_prefix": "day_cam_24",
                "country": "VN",
                "country_verified_by": "manual native-image visual review",
                "metadata_sha256": sha(metadata),
                "country_evidence": {
                    "archive_entry": "images/train/day_cam_24_00129.jpg",
                    "source_file": "research_kaggle/1953.jpg",
                    "source_sha256": entry["source_sha256"],
                    "crop_file": "research_kaggle/proof.png",
                    "crop_sha256": sha(proof_crop),
                    "crop": [10, 20, 110, 120],
                    "visible_features": [
                        "Vietnamese destination text",
                        "Vietnamese registration",
                    ],
                },
            }
        )
    )
    entry.update(
        dataset_ref="huanhuyle/vietnam-traffic-detection",
        version=2,
        image_id="00129",
        archive_entry="images/train/day_cam_24_00129.jpg",
        archive_sha256=sha(archive),
        metadata_sha256=sha(metadata),
        source_group="kaggle:huanhuyle/vietnam-traffic-detection",
        page_url="https://www.kaggle.com/datasets/huanhuyle/vietnam-traffic-detection",
        license="Apache 2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        creator="Original video author not supplied",
        maintainer="huan huy le (huanhuyle)",
        country_gate_file="research_kaggle/country_gate.json",
        country_gate_sha256=sha(gate),
    )
    return root, entry


def test_selected_vietnam_camera_import_preserves_provenance(camera_review):
    from kaggle_reviews import load_kaggle_reviews, verify_kaggle_attribution
    from ncpedia_reviews import is_external

    root, entry = camera_review
    ledger(root, [entry])
    rows, reviews, attrs, _ = load_kaggle_reviews(root)
    assert rows[0]["id"] == "kaggle-huan-v2/day_cam_24_00129.png"
    assert is_external(rows[0])
    assert reviews[rows[0]["id"]]["source_verified"] is True
    assert attrs[0]["annotation_license"] == "Apache 2.0"
    assert "original video author not supplied" in attrs[0]["artist"]
    assert rows[0]["country_gate_sha256"] == entry["country_gate_sha256"]
    assert rows[0]["train_only"] is True
    verify_kaggle_attribution(rows, attrs, root)
    altered = copy.deepcopy(attrs)
    altered[0]["country_gate_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_kaggle_attribution(rows, altered, root)


def test_camera_collection_cannot_be_assigned_to_holdout():
    from prepare_dataset import assign_splits

    rows = [
        {"id": "new", "group": "camera", "source_verified": True, "train_only": True}
    ]
    with pytest.raises(ValueError):
        assign_splits(rows, group_assignments={"camera": "test"})
    assert assign_splits(rows)["new"] == "train"


@pytest.fixture
def camera11_review(camera_review):
    root, entry = camera_review
    folder = root / "acquisition_2000/research_kaggle"
    entry["camera_prefix"] = "day_cam_11"
    entry["archive_entry"] = "images/train/day_cam_11_00129.jpg"
    with zipfile.ZipFile(folder / "data.zip", "w") as z:
        z.write(folder / "1953.jpg", entry["archive_entry"])
    entry["archive_sha256"] = sha(folder / "data.zip")
    gate_file = folder / "country_gate.json"
    gate = json.loads(gate_file.read_text())
    gate["camera_prefix"] = "day_cam_11"
    gate["country_evidence"]["archive_entry"] = entry["archive_entry"]
    gate_file.write_text(json.dumps(gate))
    entry["country_gate_sha256"] = sha(gate_file)
    return root, entry


def test_camera11_needs_its_own_country_evidence(camera11_review):
    from kaggle_reviews import load_kaggle_reviews, verify_kaggle_attribution

    root, entry = camera11_review
    ledger(root, [entry])
    rows, _, attrs, _ = load_kaggle_reviews(root)
    assert rows[0]["id"] == "kaggle-huan-v2/day_cam_11_00129.png"
    assert rows[0]["camera_prefix"] == "day_cam_11"
    assert rows[0]["train_only"] is True
    verify_kaggle_attribution(rows, attrs, root)


@pytest.fixture
def camera23_review(camera_review):
    root, entry = camera_review
    folder = root / "acquisition_2000/research_kaggle"
    gate_file = folder / "country_gate.json"
    reference = folder / "reference_gate.json"
    reference.write_bytes(gate_file.read_bytes())
    entry["camera_prefix"] = "day_cam_23"
    entry["archive_entry"] = "images/train/day_cam_23_00129.jpg"
    with zipfile.ZipFile(folder / "data.zip", "w") as z:
        for camera in ("day_cam_23", "day_cam_24"):
            z.write(folder / "1953.jpg", f"images/train/{camera}_00129.jpg")
    entry["archive_sha256"] = sha(folder / "data.zip")
    gate = json.loads(gate_file.read_text())
    gate.update(
        camera_prefix="day_cam_23",
        reference_camera="day_cam_24",
        reference_country_gate_file="research_kaggle/reference_gate.json",
        reference_country_gate_sha256=sha(reference),
        scene_match_reviewed=True,
    )
    gate["country_evidence"]["archive_entry"] = entry["archive_entry"]
    gate_file.write_text(json.dumps(gate))
    entry["country_gate_sha256"] = sha(gate_file)
    return root, entry


def test_camera23_requires_verified_camera24_scene_chain(camera23_review):
    from kaggle_reviews import load_kaggle_reviews, verify_kaggle_attribution

    root, entry = camera23_review
    ledger(root, [entry])
    rows, _, attrs, _ = load_kaggle_reviews(root)
    assert rows[0]["id"] == "kaggle-huan-v2/day_cam_23_00129.png"
    assert rows[0]["train_only"] is True
    verify_kaggle_attribution(rows, attrs, root)


@pytest.mark.parametrize(
    "change", ["missing", "camera", "hash", "unreviewed", "country", "proof"]
)
def test_camera23_rejects_broken_reference_chain(camera23_review, change):
    from kaggle_reviews import load_kaggle_reviews

    root, entry = camera23_review
    folder = root / "acquisition_2000/research_kaggle"
    path = folder / "country_gate.json"
    gate = json.loads(path.read_text())
    if change == "missing":
        del gate["reference_country_gate_file"]
    elif change == "camera":
        gate["reference_camera"] = "day_cam_23"
    elif change == "hash":
        gate["reference_country_gate_sha256"] = "0" * 64
    elif change == "unreviewed":
        gate["scene_match_reviewed"] = False
    elif change == "proof":
        (folder / "proof.png").write_bytes(b"changed reference pixels")
    else:
        reference = folder / "reference_gate.json"
        ref = json.loads(reference.read_text())
        ref["country"] = "TH"
        reference.write_text(json.dumps(ref))
        gate["reference_country_gate_sha256"] = sha(reference)
    path.write_text(json.dumps(gate))
    entry["country_gate_sha256"] = sha(path)
    ledger(root, [entry])
    with pytest.raises(ValueError):
        load_kaggle_reviews(root)


@pytest.mark.parametrize("camera", ["day_cam_24", "day_cam_18"])
def test_camera11_cannot_reuse_another_camera_gate(camera11_review, camera):
    from kaggle_reviews import load_kaggle_reviews

    root, entry = camera11_review
    folder = root / "acquisition_2000/research_kaggle"
    gate_file = folder / "country_gate.json"
    gate = json.loads(gate_file.read_text())
    gate["camera_prefix"] = camera
    gate_file.write_text(json.dumps(gate))
    entry["country_gate_sha256"] = sha(gate_file)
    ledger(root, [entry])
    with pytest.raises(ValueError):
        load_kaggle_reviews(root)


@pytest.mark.parametrize(
    "change",
    ["camera", "gate_hash", "proof_bytes", "proof_member", "metadata", "country"],
)
def test_camera_import_rejects_unverified_subset_or_evidence(camera_review, change):
    from kaggle_reviews import load_kaggle_reviews

    root, entry = camera_review
    folder = root / "acquisition_2000/research_kaggle"
    if change == "camera":
        entry["archive_entry"] = "images/train/day_cam_18_00129.jpg"
    elif change == "gate_hash":
        entry["country_gate_sha256"] = "0" * 64
    elif change == "proof_bytes":
        (folder / "proof.png").write_bytes(b"changed")
    else:
        path = folder / "country_gate.json"
        gate = json.loads(path.read_text())
        if change == "proof_member":
            gate["country_evidence"]["archive_entry"] = (
                "images/train/day_cam_18_00129.jpg"
            )
        elif change == "metadata":
            gate["metadata_sha256"] = "0" * 64
        else:
            gate["country"] = "TH"
        path.write_text(json.dumps(gate))
        entry["country_gate_sha256"] = sha(path)
    ledger(root, [entry])
    with pytest.raises((ValueError, FileNotFoundError)):
        load_kaggle_reviews(root)
