"""Export only explicit visual review decisions, preserving original files and splits."""

import argparse
import json
import re
import shutil
from collections import Counter

from commons_sources import resolve_commons_source
from curate_v2 import make_gallery, normalize_boxes, plain, validate_external
from kaggle_reviews import load_kaggle_reviews
from ncpedia_reviews import load_ncpedia_reviews
from PIL import Image
from prepare_dataset import (
    ROOT,
    digest,
    export,
    group_rows,
    parse_labels,
    perceptual_hash,
    write_json,
)
from train_colab import verify_dataset
from training_crops import prepare_original_crop, prepare_review_image
from zenodo_reviews import load_zenodo_reviews


def latest_parent_manifest():
    progress = ROOT / "acquisition_2000/progress.json"
    name = (
        json.loads(progress.read_text(encoding="utf-8"))["frozen_dataset"]
        if progress.exists()
        else "dataset_v3"
    )
    if not re.fullmatch(r"dataset_v\d+", name):
        raise ValueError("Invalid frozen dataset name")
    parent = ROOT / name / "manifest.json"
    if not parent.is_file():
        raise FileNotFoundError(parent)
    return parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="New frozen snapshot, e.g. v4")
    args = parser.parse_args()
    progress_path = ROOT / "acquisition_2000/progress.json"
    prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    target = prior_progress.get("target_clean_images", 2000)
    if type(target) is not int or target <= 0:
        raise ValueError("target_clean_images must be a positive integer")
    if not re.fullmatch(r"v\d+", args.version) or int(args.version[1:]) < 4:
        raise ValueError("Expected version v4 or later")
    output = ROOT / f"dataset_{args.version}"
    if output.exists():
        raise FileExistsError("Frozen snapshot exists; use a new version")
    rows = json.loads((ROOT / "inventory_v2.json").read_text(encoding="utf-8"))
    lookup = {r["id"]: r for r in rows}
    reviews = json.loads((ROOT / "reviews_v3.json").read_text(encoding="utf-8"))
    audit = []
    attribution = json.loads(
        (ROOT / "dataset_v3/attribution.json").read_text(encoding="utf-8")
    )
    for path in sorted(
        (ROOT / "acquisition_2000").glob("original_reviews_batch*.json")
    ):
        for entry in json.loads(path.read_text(encoding="utf-8")):
            row = lookup[entry["id"]]
            assert (
                entry["label_sha256"] == row["label_sha256"]
                and entry["image_sha256"] == row["sha256"]
            )
            assert (
                digest(row["image"]) == row["sha256"]
                and digest(row["label"]) == row["label_sha256"]
            )
            assert (ROOT / entry["evidence"]).is_file(), "Missing review evidence"
            assert (
                entry["status"] in {"accepted", "corrected", "pending"}
                and entry["reason"].strip()
            )
            decision = {k: entry[k] for k in ["status", "label_sha256", "reason"]}
            if entry["status"] != "pending":
                assert entry["source_verified"] is True
                decision.update(labels=entry["labels"], source_verified=True)
                parse_labels("\n".join(" ".join(map(str, b)) for b in entry["labels"]))
            if "crop" in entry:
                assert entry["status"] != "pending", (
                    "Crop needs its own complete visual annotation"
                )
                cropped = prepare_original_crop(
                    row, entry, ROOT / "acquisition_2000/curated_original_crops"
                )
                assert cropped["id"] not in lookup, "Repeated original crop"
                rows.append(cropped)
                lookup[cropped["id"]] = cropped
                reviews[cropped["id"]] = decision
                reviews[row["id"]] = {
                    "status": "pending",
                    "label_sha256": row["label_sha256"],
                    "reason": "Ảnh toàn cảnh giữ chờ duyệt; chỉ vùng cắt có nhật ký và nhãn thủ công được đưa vào tập sử dụng.",
                }
            else:
                reviews[row["id"]] = decision
            audit.append(dict(batch=path.name, **entry))
    source_lookup = {}
    for name in ["downloaded_partial.json", "sources.json"]:
        path = ROOT / "acquisition_2000" / name
        if path.exists():
            for source in json.loads(path.read_text(encoding="utf-8")):
                if "download_sha256" in source:
                    source_lookup[source["pageid"]] = source
    baseline = ROOT / "acquisition_2000/unlabeled_baseline"
    baseline.mkdir(exist_ok=True)
    used_pageids = {
        int(r["id"].rsplit("_", 1)[-1].split(".")[0])
        for r in rows
        if r["id"].startswith("commons/")
    }
    for path in sorted(
        (ROOT / "acquisition_2000").glob("external_reviews_batch*.json")
    ):
        for entry in json.loads(path.read_text(encoding="utf-8")):
            validate_external(entry)
            assert entry["pageid"] not in used_pageids, (
                "Only one admitted image per original Commons photograph"
            )
            used_pageids.add(entry["pageid"])
            assert (ROOT / entry["evidence"]).is_file()
            source = source_lookup[entry["pageid"]]
            im_path, source_variant = resolve_commons_source(source, entry, ROOT)
            meta = source["extmetadata"]
            license_name = plain(meta.get("LicenseShortName", {}).get("value", ""))
            artist = entry.get("artist_attribution") or plain(
                meta.get("Artist", {}).get("value", "")
            )
            assert artist and license_name.startswith(("CC BY", "CC0", "Public domain"))
            im_path, lineage = prepare_review_image(
                im_path, entry, ROOT / "acquisition_2000/curated_crops", entry["pageid"]
            )
            identifier = f"commons/commons_{entry['pageid']}{im_path.suffix}"
            assert identifier not in lookup, "Duplicate external review ID"
            label = baseline / f"commons_{entry['pageid']}.txt"
            if not label.exists():
                label.write_text("", encoding="utf-8")
            assert not label.read_bytes(), "Unlabeled baseline changed"
            with Image.open(im_path) as im:
                assert (
                    list(im.size) == entry["canvas"] and im.getexif().get(274, 1) == 1
                )
                width, height = im.size
                phash = f"{perceptual_hash(im):016x}"
            row = {
                "id": identifier,
                "image": str(im_path.resolve()),
                "label": str(label.resolve()),
                "labels": [],
                "sha256": digest(im_path),
                "label_sha256": digest(label),
                "width": width,
                "height": height,
                "phash": phash,
                "source": "commons-author:" + entry.get("source_group", artist),
                "country": "VN",
                "location_evidence": entry["location_evidence"],
                "review_method": "visual_manual_review",
                "review_evidence": entry["evidence"],
                "license": license_name,
                "source_url": source["descriptionurl"],
                "original_pageid": entry["pageid"],
                **source_variant,
                **lineage,
            }
            rows.append(row)
            lookup[identifier] = row
            reviews[identifier] = {
                "status": "corrected",
                "label_sha256": row["label_sha256"],
                "source_verified": True,
                "labels": normalize_boxes(entry["boxes"], entry["canvas"]),
                "reason": entry["reason"],
            }
            attribution.append(
                {
                    "id": identifier,
                    "title": source["title"],
                    "artist": artist,
                    "license": license_name,
                    "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                    "page_url": source["descriptionurl"],
                    "download_url": (
                        source["url"] if source_variant else source.get("thumburl", source["url"])
                    ),
                    "original_url": source["url"],
                    "sha256": row["sha256"],
                    "country": "VN",
                    "location_evidence": entry["location_evidence"],
                    "modifications": (
                        f"Cắt vùng {entry['crop']} từ ảnh nguồn, giữ nguyên pixel PNG; bổ sung nhãn YOLO thủ công."
                        if "crop" in entry
                        else "Ảnh thumbnail Commons giữ nguyên byte; bổ sung nhãn YOLO thủ công."
                    ),
                    **source_variant,
                    **lineage,
                    "annotation_license": "CC BY-SA 4.0",
                }
            )
            audit.append(dict(batch=path.name, id=identifier, **entry))
    provider_rows, provider_reviews, provider_attrs, provider_audit = (
        load_ncpedia_reviews(ROOT)
    )
    for loader in (load_kaggle_reviews, load_zenodo_reviews):
        extra_rows, extra_reviews, extra_attrs, extra_audit = loader(ROOT)
        provider_rows.extend(extra_rows)
        provider_reviews.update(extra_reviews)
        provider_attrs.extend(extra_attrs)
        provider_audit.extend(extra_audit)
    for row in provider_rows:
        if row["id"] in lookup:
            raise ValueError("Duplicate supplemental review ID")
        rows.append(row)
        lookup[row["id"]] = row
    reviews.update(provider_reviews)
    attribution.extend(provider_attrs)
    audit.extend(provider_audit)
    # Second visual QA found a parked motorcycle at the left edge in frame 1406.
    original = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    row = original[1406]
    if reviews[row["id"]]["status"] != "pending":
        # Retain the whole image for re-annotation rather than silently omit the bike.
        reviews[row["id"]] = {
            "status": "pending",
            "label_sha256": row["label_sha256"],
            "reason": "Rà QA lần hai phát hiện xe đỗ bị bỏ sót ở mép trái; giữ cả ảnh chờ bổ sung box chính xác.",
        }
        audit.append(
            {
                "id": row["id"],
                "status": "pending",
                "reason": reviews[row["id"]]["reason"],
                "evidence": "review_round3/expansion/after_08.jpg",
                "batch": "second_visual_QA",
            }
        )
    active = [r for r in rows if reviews[r["id"]]["status"] != "pending"]
    families = Counter(r["id"].split(".rf.")[0].split("/", 1)[-1] for r in active)
    assert all(count == 1 for count in families.values()), (
        "Repeated augmented source frame in clean target"
    )
    group_rows(rows)
    write_json(ROOT / "inventory_2000.json", rows)
    write_json(ROOT / "reviews_2000.json", reviews)
    export(rows, reviews, output, inherit_manifest=latest_parent_manifest())
    for name in ["external_candidates_excluded.json"]:
        shutil.copy2(ROOT / "dataset_v3" / name, output / name)
    write_json(output / "attribution.json", attribution)
    (output / "ATTRIBUTION.md").write_text(
        "# Nguồn ảnh bổ sung Việt Nam\n\nẢnh và nhãn giữ giấy phép ghi ở từng mục. "
        "Nguồn NCpedia có điều kiện phi thương mại và chia sẻ tương tự (CC BY-NC-SA 4.0). "
        "Không áp giấy phép này cho dataset gốc của người dùng.\n\n"
        + "\n\n".join(
            f"- [{a['title']}]({a['page_url']}) — {a['artist']} — [{a['license']}]({a['license_url'] or a['page_url']}). {a['modifications']} Nhãn: {a['annotation_license']}."
            for a in attribution
        ),
        encoding="utf-8",
    )
    write_json(
        output / "usage_restrictions.json",
        {
            "noncommercial_image_ids": sorted(
                a["id"] for a in attribution if a.get("noncommercial_only")
            ),
            "license": "CC BY-NC-SA 4.0",
            "scope": "listed_source_images_crops_and_associated_annotations",
            "note": "Các mục được liệt kê chỉ dùng phi thương mại, giữ ghi công và điều kiện chia sẻ tương tự. Không đổi giấy phép dữ liệu gốc hoặc các nguồn khác.",
        },
    )
    write_json(output / "review_audit_2000.json", audit)
    verify_dataset(output / "data.yaml")
    make_gallery(output, args.version, ROOT / "inventory_2000.json")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    active = [r for r in manifest if r["new_split"] != "pending"]
    sources_path = ROOT / "acquisition_2000/sources.json"
    candidates = (
        json.loads(sources_path.read_text(encoding="utf-8"))
        if sources_path.exists()
        else []
    )
    progress = {
        "target_clean_images": target,
        "accepted_clean_images": len(active),
        "completion_percent": round(len(active) / target * 100, 2),
        "remaining_clean_images": max(0, target - len(active)),
        "frozen_dataset": output.name,
        "unique_source_frames": len(
            {r["id"].split(".rf.")[0].split("/", 1)[-1] for r in active}
        ),
        "downloaded_candidates": sum("download_sha256" in r for r in candidates),
        "metadata_candidates": len(candidates),
        "target_achieved": len(active) >= target,
        "note": "Candidate downloads and pending images are excluded from the clean count.",
    }
    write_json(ROOT / "acquisition_2000/progress.json", progress)
    print(json.dumps(progress))


if __name__ == "__main__":
    main()
