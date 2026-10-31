import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def reviewed_source(tmp_path):
    folder = tmp_path / "acquisition_2000/research_ncpedia"
    folder.mkdir(parents=True)
    source = folder / "vietnam_101.jpg"
    Image.new("RGB", (320, 360), (31, 72, 113)).save(source)
    page = folder / "source.html"
    page.write_text(
        'Traffic at Hué. <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">'
        'Creative Commons BY-NC-SA</a> <img src="/vietnam_101.jpg"> 852094800',
        encoding="utf-8",
    )
    entry = {
        "source_id": "ncpedia-vietnam-101",
        "source_image": "research_ncpedia/vietnam_101.jpg",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "provider_page_snapshot": "research_ncpedia/source.html",
        "provider_page_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
        "page_url": "https://www.ncpedia.org/media/motorcycle-and-bycycle",
        "image_url": "https://www.ncpedia.org/sites/default/files/vietnam_101.jpg",
        "title": "Motorcycle traffic at Hue",
        "license": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "credit_provider": "NCpedia",
        "photographer_named": False,
        "supplied_copyright_field": "852094800",
        "country": "VN",
        "location_evidence": "Provider caption explicitly says Hué, Vietnam.",
        "source_group": "ncpedia-vietnam-collection",
        "visual_reviewed": True,
        "source_verified": True,
        "status": "accepted",
        "crop": [16, 20, 272, 340],
        "canvas": [256, 320],
        "boxes": [[0, 40, 120, 230, 320], [2, 50, 8, 220, 280]],
        "reason": "Manually reviewed cloth-hat rider, one motorcycle.",
    }
    return tmp_path, entry


def export_review(root, entries):
    from ncpedia_reviews import load_ncpedia_reviews

    ledger = root / "acquisition_2000/ncpedia_reviews.json"
    ledger.write_text(json.dumps(entries), encoding="utf-8")
    return load_ncpedia_reviews(root)


def test_provider_crop_preserves_pixels_credit_and_noncommercial_license(
    reviewed_source,
):
    root, entry = reviewed_source
    rows, reviews, attrs, audit = export_review(root, [entry])
    row = rows[0]
    assert row["id"] == "ncpedia/ncpedia_101.png"
    assert Path(row["image"]).name == "ncpedia_101.png"
    assert row["source"] == "ncpedia-collection:vietnam"
    assert row["noncommercial_only"] is True
    assert reviews[row["id"]]["labels"][1][0] == 2
    assert attrs[0]["annotation_license"] == "CC BY-NC-SA 4.0"
    assert attrs[0]["creator"] is None
    assert attrs[0]["credit_provider"] == "NCpedia"
    assert attrs[0]["supplied_copyright_field"] == "852094800"
    assert audit[0]["source_id"] == entry["source_id"]
    with Image.open(row["source_image"]) as source, Image.open(row["image"]) as crop:
        assert source.convert("RGB").crop(entry["crop"]).tobytes() == crop.tobytes()


@pytest.mark.parametrize(
    "change",
    [
        {"source_image": "../outside.jpg"},
        {"source_sha256": "0" * 64},
        {"provider_page_sha256": "0" * 64},
        {"country": "unknown"},
        {"visual_reviewed": False},
        {"source_verified": False},
        {"status": "draft"},
        {"license": "CC BY-NC-ND 4.0"},
        {"page_url": "https://example.com/media/photo"},
        {"source_id": "../../101"},
        {"source_group": "another-split-for-the-same-collection"},
    ],
)
def test_rejects_unsafe_unverified_or_incompatible_review(reviewed_source, change):
    root, entry = reviewed_source
    entry.update(change)
    with pytest.raises((ValueError, FileNotFoundError)):
        export_review(root, [entry])


def test_duplicate_source_is_not_counted_twice(reviewed_source):
    root, entry = reviewed_source
    with pytest.raises(ValueError, match="Duplicate"):
        export_review(root, [entry, copy.deepcopy(entry)])


def test_original_asset_number_keeps_leading_zeroes_in_url(reviewed_source):
    root, entry = reviewed_source
    entry["source_id"] = "ncpedia-vietnam-037"
    entry["image_url"] = "https://www.ncpedia.org/sites/default/files/vietnam_037.jpg"
    page = root / "acquisition_2000" / entry["provider_page_snapshot"]
    page.write_text(
        page.read_text("utf-8").replace("vietnam_101", "vietnam_037"), encoding="utf-8"
    )
    entry["provider_page_sha256"] = hashlib.sha256(page.read_bytes()).hexdigest()
    rows, _, attrs, _ = export_review(root, [entry])
    assert rows[0]["id"] == "ncpedia/ncpedia_37.png"
    assert attrs[0]["image_url"] == entry["image_url"]


def test_frozen_attribution_rejects_relicensing_and_missing_provider_credit(
    reviewed_source,
):
    from ncpedia_reviews import verify_ncpedia_attribution

    root, entry = reviewed_source
    rows, _, attrs, _ = export_review(root, [entry])
    verify_ncpedia_attribution(rows, attrs, root)
    invalid = copy.deepcopy(attrs)
    invalid[0]["annotation_license"] = "CC BY-SA 4.0"
    with pytest.raises(ValueError):
        verify_ncpedia_attribution(rows, invalid, root)
    invalid = copy.deepcopy(attrs)
    invalid[0]["credit_provider"] = ""
    with pytest.raises(ValueError):
        verify_ncpedia_attribution(rows, invalid, root)


def test_noncommons_external_images_are_separate_from_original_progress():
    from ncpedia_reviews import is_external

    assert is_external({"id": "ncpedia/ncpedia_101.png"})
    assert is_external({"id": "commons/commons_123.png"})
    assert not is_external({"id": "original_crops/frame.png"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "ncpedia-collection:another-split"),
        ("id", "ncpedia/ncpedia_999.png"),
    ],
)
def test_frozen_source_identity_and_group_cannot_drift(reviewed_source, field, value):
    from ncpedia_reviews import verify_ncpedia_attribution

    root, entry = reviewed_source
    rows, _, attrs, _ = export_review(root, [entry])
    rows[0][field] = value
    if field == "id":
        attrs[0][field] = value
    with pytest.raises(ValueError):
        verify_ncpedia_attribution(rows, attrs, root)


def test_no_provider_ledger_leaves_existing_export_unchanged(tmp_path):
    from ncpedia_reviews import load_ncpedia_reviews

    assert load_ncpedia_reviews(tmp_path) == ([], {}, [], [])
