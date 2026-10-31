"""Verify the current clean snapshot and unchanged user-provided data."""

import json
from itertools import combinations

from commons_sources import verify_commons_sources
from kaggle_reviews import verify_kaggle_attribution
from ncpedia_reviews import is_external, verify_ncpedia_attribution
from PIL import Image, ImageChops
from prepare_dataset import ROOT, digest, parse_labels, write_json
from train_colab import verify_dataset
from zenodo_reviews import verify_zenodo_attribution


def main():
    progress = json.loads(
        (ROOT / "acquisition_2000/progress.json").read_text(encoding="utf-8")
    )
    data = ROOT / progress["frozen_dataset"]
    verify_dataset(data / "data.yaml")
    original = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    for row in original:
        assert (
            digest(row["image"]) == row["sha256"]
            and digest(row["label"]) == row["label_sha256"]
        ), row["id"]
    rows = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    active = [r for r in rows if r["new_split"] != "pending"]
    assert len({r["id"].split(".rf.")[0].split("/", 1)[-1] for r in active}) == len(
        active
    )
    assert len({r["sha256"] for r in active}) == len(active)
    assert len({r.get("source_sha256", r["sha256"]) for r in active}) == len(active), (
        "Repeated source photograph after cropping"
    )
    for row in active:
        parse_labels("\n".join(" ".join(map(str, b)) for b in row["labels"]))
        if "source_image" in row:
            assert digest(row["source_image"]) == row["source_sha256"], (
                "Original supplemental photograph changed"
            )
        if "crop" in row:
            with (
                Image.open(row["source_image"]) as raw,
                Image.open(row["image"]) as cropped,
            ):
                expected = raw.convert("RGB").crop(row["crop"])
                assert expected.size == cropped.size
                assert (
                    ImageChops.difference(expected, cropped.convert("RGB")).getbbox()
                    is None
                ), "Crop pixels differ from original"
    for a, b in combinations(active, 2):
        if a["new_split"] == b["new_split"]:
            continue
        assert a["source"] != b["source"] and a["group"] != b["group"]
        if abs(a["width"] / a["height"] - b["width"] / b["height"]) <= 0.05:
            assert (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count() > 6
    old = json.loads((ROOT / "dataset_v3/manifest.json").read_text(encoding="utf-8"))
    old_test = {r["id"]: r for r in old if r["new_split"] == "test"}
    test = {r["id"]: r for r in active if r["new_split"] == "test"}
    assert old_test.keys() <= test.keys()
    assert all(test[key]["labels"] == row["labels"] for key, row in old_test.items())
    lock = json.loads((data / "split_lock.json").read_text(encoding="utf-8"))
    if "parent_manifest_sha256" in lock:
        parents = [
            p
            for p in ROOT.glob("dataset_v*/manifest.json")
            if digest(p) == lock["parent_manifest_sha256"]
        ]
        assert parents, "Inherited parent manifest missing or changed"
        parent = json.loads(parents[0].read_text(encoding="utf-8"))
        inherited_test = {r["id"]: r for r in parent if r["new_split"] == "test"}
        assert inherited_test.keys() <= test.keys()
        assert all(
            test[key]["labels"] == r["labels"] for key, r in inherited_test.items()
        )
    attrs = json.loads((data / "attribution.json").read_text(encoding="utf-8"))
    assert {r["id"] for r in attrs} == {r["id"] for r in active if is_external(r)}
    assert all(
        r["country"] == "VN" and r["location_evidence"] and r["artist"] and r["license"]
        for r in attrs
    )
    verify_ncpedia_attribution(active, attrs, ROOT)
    verify_kaggle_attribution(active, attrs, ROOT)
    verify_zenodo_attribution(active, attrs, ROOT)
    nc_ids = sorted(r["id"] for r in active if r.get("noncommercial_only"))
    if nc_ids:
        restrictions = json.loads(
            (data / "usage_restrictions.json").read_text(encoding="utf-8")
        )
        assert restrictions["noncommercial_image_ids"] == nc_ids
        assert restrictions["license"] == "CC BY-NC-SA 4.0"
        assert (
            restrictions["scope"]
            == "listed_source_images_crops_and_associated_annotations"
        )
    result = {
        "passed": True,
        "dataset": data.name,
        "active_images": len(active),
        "external_vietnam_images": len(attrs),
        "kaggle_source_images": sum(
            r["id"].startswith(("kaggle-maitam-v8/", "kaggle-huan-v2/"))
            for r in active
        ),
        "noncommercial_source_images": len(nc_ids),
        "zenodo_vntraffic_images": sum(r["id"].startswith("zenodo-vntraffic/") for r in active),
        "provider_attribution_and_usage_conditions_verified": True,
        "original_images_and_labels_unchanged": len(original),
        "unique_source_frames": True,
        "cross_split_hash_group_and_phash_checks": True,
        "old_test_images_and_labels_preserved": True,
        "original_test_images": len(old_test),
        "current_test_images": len(test),
        "cropped_images": sum("crop" in r for r in active),
        "commons_original_resolution_images": verify_commons_sources(active, attrs, ROOT),
        "original_supplemental_bytes_and_crop_pixels_verified": True,
        "latest_parent_test_preserved": True,
        "note": "Heuristic duplicate checks do not prove all unknown provenance independent; counts are not model quality scores.",
    }
    write_json(ROOT / "verification" / f"{data.name}.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
