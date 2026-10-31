import copy
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_dataset import digest


@pytest.fixture
def originals(tmp_path):
    base = tmp_path / "acquisition_2000"
    images = base / "images"
    folder = base / "commons_original_resolution_review"
    images.mkdir(parents=True)
    folder.mkdir()
    thumb = images / "thumb.png"
    original = folder / "original.png"
    Image.new("RGB", (200, 150), "red").save(thumb)
    Image.new("RGB", (400, 300), "blue").save(original)
    source = {
        "pageid": 123,
        "local_image": "images/thumb.png",
        "download_sha256": digest(thumb),
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/photo.png",
        "width": 400,
        "height": 300,
    }
    receipt = folder / "acquisition.json"
    receipt.write_text(
        json.dumps(
            {
                "pageid": 123,
                "requested_url": source["url"],
                "http_status": 200,
                "status": "downloaded_candidate_only",
                "file": "original.png",
                "sha256": digest(original),
                "canvas": [400, 300],
            }
        )
    )
    entry = {
        "pageid": 123,
        "source_variant": "original",
        "source_file": "commons_original_resolution_review/original.png",
        "image_sha256": digest(original),
        "original_acquisition_file": "commons_original_resolution_review/acquisition.json",
        "original_acquisition_sha256": digest(receipt),
    }
    return tmp_path, source, entry, receipt


def test_original_selection_preserves_thumbnail_and_metadata(originals):
    from commons_sources import resolve_commons_source

    root, source, entry, _ = originals
    before = copy.deepcopy(source)
    thumb = root / "acquisition_2000/images/thumb.png"
    thumb_hash = digest(thumb)
    path, lineage = resolve_commons_source(source, entry, root)
    assert Image.open(path).size == (400, 300)
    assert path.name == "original.png"
    assert lineage["commons_source_variant"] == "original"
    assert lineage["original_download_url"] == source["url"]
    assert source == before and digest(thumb) == thumb_hash


def test_existing_thumbnail_reviews_keep_original_behavior(originals):
    from commons_sources import resolve_commons_source

    root, source, _, _ = originals
    path, lineage = resolve_commons_source(
        source, {"pageid": 123, "image_sha256": source["download_sha256"]}, root
    )
    assert path.name == "thumb.png" and lineage == {}


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "url",
        "canvas",
        "page",
        "status",
        "path",
        "receipt_hash",
        "metadata_canvas",
    ],
)
def test_original_rejects_wrong_source_or_receipt(originals, change):
    from commons_sources import resolve_commons_source

    root, source, entry, receipt = originals
    data = json.loads(receipt.read_text())
    if change == "hash":
        entry["image_sha256"] = "0" * 64
    elif change == "url":
        data["requested_url"] = "https://example.com/unrelated.png"
    elif change == "canvas":
        data["canvas"] = [800, 600]
    elif change == "page":
        data["pageid"] = 456
    elif change == "status":
        data["status"] = "not_downloaded"
    elif change == "path":
        data["file"] = "../images/thumb.png"
    elif change == "metadata_canvas":
        source["width"] = 800
    if change == "receipt_hash":
        entry["original_acquisition_sha256"] = "0" * 64
    else:
        receipt.write_text(json.dumps(data))
        entry["original_acquisition_sha256"] = digest(receipt)
    with pytest.raises(ValueError):
        resolve_commons_source(source, entry, root)


def test_original_rejects_outside_file_even_when_bytes_match(originals):
    from commons_sources import resolve_commons_source

    root, source, entry, _ = originals
    outside = root / "copied.png"
    outside.write_bytes((root / "acquisition_2000" / entry["source_file"]).read_bytes())
    entry["source_file"] = str(outside)
    with pytest.raises(ValueError):
        resolve_commons_source(source, entry, root)


def test_unknown_variant_cannot_fall_back_to_thumbnail(originals):
    from commons_sources import resolve_commons_source

    root, source, entry, _ = originals
    entry["source_variant"] = "upscaled"
    with pytest.raises(ValueError):
        resolve_commons_source(source, entry, root)


@pytest.mark.parametrize("change", [None, "attribution", "downgrade", "url", "pixels"])
def test_snapshot_original_provenance_is_revalidated(originals, change):
    from commons_sources import resolve_commons_source, verify_commons_sources

    root, source, entry, _ = originals
    path, evidence = resolve_commons_source(source, entry, root)
    row = dict(
        evidence,
        id="commons/commons_123.png",
        original_pageid=123,
        source_image=str(path),
        source_sha256=digest(path),
    )
    attr = dict(row, download_url=source["url"], original_url=source["url"])
    if change == "attribution":
        attr["original_acquisition_sha256"] = "0" * 64
    elif change == "downgrade":
        for obj in (row, attr):
            for key in evidence:
                del obj[key]
    elif change == "url":
        source["url"] = "https://example.com/other.png"
    elif change == "pixels":
        Image.new("RGB", (400, 300), "green").save(path)
    (root / "acquisition_2000/sources.json").write_text(json.dumps([source]))
    if change is None:
        assert verify_commons_sources([row], [attr], root) == 1
    else:
        with pytest.raises(ValueError):
            verify_commons_sources([row], [attr], root)


@pytest.mark.parametrize("tamper", [False, True])
def test_legacy_commons_uses_pinned_v3_bytes_not_new_acquisition_folder(
    tmp_path, tamper
):
    from commons_sources import verify_commons_sources

    folder = tmp_path / "external_commons/images"
    folder.mkdir(parents=True)
    image = folder / "commons_456.png"
    Image.new("RGB", (300, 300), "yellow").save(image)
    row = {
        "id": "commons/commons_456.png",
        "image": str(image),
        "sha256": digest(image),
    }
    attr = {"id": row["id"], "sha256": row["sha256"], "license": "CC BY 2.0"}
    pinned = tmp_path / "dataset_v3"
    pinned.mkdir()
    (pinned / "manifest.json").write_text(json.dumps([row]))
    (pinned / "attribution.json").write_text(json.dumps([attr]))
    if tamper:
        Image.new("RGB", (300, 300), "blue").save(image)
        with pytest.raises(ValueError):
            verify_commons_sources([row], [attr], tmp_path)
    else:
        assert verify_commons_sources([row], [attr], tmp_path) == 0
