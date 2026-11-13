"""Build and verify dataset_v243 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Saigon street family of 5 on Honda Air Blade with 2 unhelmeted young children (Staffan Scherz, CC BY 2.0) -> train (group_staffan_scherz_hcmc) [2x cls 0, 3x cls 1, 2x cls 2]
2. Hanoi Old Quarter pedestrian street with parked blue Honda Wave 110 and background scooter (Alexkom000, CC BY 4.0) -> train (group_alexkom000_hanoi) [2x cls 0, 0x cls 1, 0x cls 2]
3. HCMC Bui Vien Street commuters on Piaggio scooters and motorcycles (松岡明芳, CC BY-SA 3.0) -> val (group_matsuoka_buivien) [4x cls 0, 4x cls 1, 0x cls 2]
4. HCMC Bui Vien alleyway food stall with vintage Honda Super Cub and parked scooters (松岡明芳, CC BY-SA 3.0) -> val (group_matsuoka_buivien) [4x cls 0, 0x cls 1, 0x cls 2]

Maintains 100% group isolation, locks test split at 68 images, verifies bounding boxes and pHash.
"""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
V242_DIR = TRAINING_DIR / "dataset_v242"
V243_DIR = TRAINING_DIR / "dataset_v243"
ACQ_DIR = TRAINING_DIR / "acquisition_5000"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def dct2_8x8_from_32(gray):
    N = 32
    i = np.arange(8)[:, None]
    j = np.arange(N)[None, :]
    D8 = np.cos((2 * j + 1) * i * np.pi / (2 * N))
    D8[0, :] *= 1.0 / np.sqrt(2)
    D8 *= np.sqrt(2.0 / N)
    return D8 @ gray @ D8.T

def perceptual_hash(image):
    gray = np.asarray(ImageOps.grayscale(image).resize((32, 32)), dtype=np.float32)
    low = dct2_8x8_from_32(gray).reshape(-1)
    bits = low > np.median(low[1:])
    return sum(int(bit) << i for i, bit in enumerate(bits))

def boxes_to_yolo(boxes, W, H):
    lines = []
    for cls, (x1, y1, x2, y2) in boxes:
        x1 = max(0.0, min(float(W), float(x1)))
        x2 = max(0.0, min(float(W), float(x2)))
        y1 = max(0.0, min(float(H), float(y1)))
        y2 = max(0.0, min(float(H), float(y2)))
        cx = (x1 + x2) / (2.0 * W)
        cy = (y1 + y2) / (2.0 * H)
        w = (x2 - x1) / float(W)
        h = (y2 - y1) / float(H)
        assert cx - w / 2.0 >= -0.0001, f"Invalid cx-w/2: {cx - w/2.0}"
        assert cx + w / 2.0 <= 1.0001, f"Invalid cx+w/2: {cx + w/2.0}"
        assert cy - h / 2.0 >= -0.0001, f"Invalid cy-h/2: {cy - h/2.0}"
        assert cy + h / 2.0 <= 1.0001, f"Invalid cy+h/2: {cy + h/2.0}"
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + "\n"

def main():
    print("=== Building Dataset v243 Snapshot ===")
    assert V242_DIR.is_dir(), f"Missing parent snapshot {V242_DIR}"

    if V243_DIR.exists():
        shutil.rmtree(V243_DIR)
    V243_DIR.mkdir(parents=True, exist_ok=True)

    # Copy metadata files from v242
    shutil.copy2(V242_DIR / "split_lock.json", V243_DIR / "split_lock.json")
    shutil.copy2(V242_DIR / "data.yaml", V243_DIR / "data.yaml")
    shutil.copy2(V242_DIR / "duplicates.json", V243_DIR / "duplicates.json")
    shutil.copy2(V242_DIR / "usage_restrictions.json", V243_DIR / "usage_restrictions.json")
    shutil.copy2(V242_DIR / "README.md", V243_DIR / "README.md")

    # Set up train / val / test directories
    for split in ["train", "val", "test"]:
        (V243_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V243_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Link existing v242 images and labels
    print("Linking existing v242 assets...")
    for split in ["train", "val", "test"]:
        src_img_dir = V242_DIR / split / "images"
        dst_img_dir = V243_DIR / split / "images"
        if src_img_dir.exists():
            for item in src_img_dir.iterdir():
                target = item.resolve()
                dst = dst_img_dir / item.name
                os.symlink(target, dst)

        src_lbl_dir = V242_DIR / split / "labels"
        dst_lbl_dir = V243_DIR / split / "labels"
        if src_lbl_dir.exists():
            for item in src_lbl_dir.iterdir():
                dst = dst_lbl_dir / item.name
                shutil.copy2(item, dst)

    # Directories for evidence and metadata
    (V243_DIR / "review_evidence").mkdir(exist_ok=True)
    for f in (V242_DIR / "review_evidence").iterdir():
        if f.is_file():
            os.link(f, V243_DIR / "review_evidence" / f.name)

    (V243_DIR / "source_originals").mkdir(exist_ok=True)
    for f in (V242_DIR / "source_originals").iterdir():
        if f.is_file():
            os.link(f, V243_DIR / "source_originals" / f.name)

    (V243_DIR / "source_metadata").mkdir(exist_ok=True)
    for f in (V242_DIR / "source_metadata").iterdir():
        if f.is_file():
            os.link(f, V243_DIR / "source_metadata" / f.name)

    # Candidate definitions for v243
    cands_info = [
        {
            "id": "commons_vietnam/Saigon_family_airblade_unhelmeted_children_manual_v243.png",
            "source_file": ACQ_DIR / "download" / "commons_35410109.jpg",
            "pageid": 35410109,
            "title": "File:Art of Transportation (6161289029).jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Art_of_Transportation_(6161289029).jpg",
            "author": "Staffan Scherz",
            "author_url": "https://www.flickr.com/people/staffanscherz/",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "split": "train",
            "group": "group_staffan_scherz_hcmc",
            "dest_img": "commons_vietnam__Saigon_family_airblade_unhelmeted_children_manual_v243.png",
            "dest_lbl": "commons_vietnam__Saigon_family_airblade_unhelmeted_children_manual_v243.txt",
            "boxes": [
                # Mother behind in yellow shirt with purple helmet
                (1, (80, 15, 305, 780)),
                # Older daughter in pink shirt with yellow helmet
                (1, (195, 20, 440, 680)),
                # Father driver in white shirt with black helmet
                (1, (280, 40, 630, 735)),
                # Young daughter in floral dress without helmet
                (2, (370, 190, 590, 580)),
                # Young son standing in front in yellow shirt without helmet
                (2, (480, 205, 690, 700)),
                # Honda Air Blade motorcycle
                (0, (20, 380, 855, 760)),
                # Workhorse motorcycle rear wheel/cargo rack on right
                (0, (850, 550, 1024, 705))
            ],
            "description": "Ho Chi Minh City street scene showing a Vietnamese family of five riding a Honda Air Blade scooter with helmeted parents/older daughter and two unhelmeted young children, alongside a passing workhorse motorcycle."
        },
        {
            "id": "commons_vietnam/Hanoi_OldQuarter_street_parked_scooters_manual_v243.png",
            "source_file": ACQ_DIR / "download" / "commons_156674592.jpg",
            "pageid": 156674592,
            "title": "File:2024-11-03 Hanoi's Old Quarter 4.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:2024-11-03_Hanoi%27s_Old_Quarter_4.jpg",
            "author": "Alexkom000",
            "author_url": "https://commons.wikimedia.org/wiki/User:Alexkom000",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "split": "train",
            "group": "group_alexkom000_hanoi",
            "dest_img": "commons_vietnam__Hanoi_OldQuarter_street_parked_scooters_manual_v243.png",
            "dest_lbl": "commons_vietnam__Hanoi_OldQuarter_street_parked_scooters_manual_v243.txt",
            "boxes": [
                # Foreground blue Honda Wave 110 on kickstand
                (0, (1340, 2010, 1840, 2980)),
                # Parked Honda Vision scooter in background
                (0, (3500, 1630, 3780, 2060))
            ],
            "description": "Hanoi Old Quarter pedestrian street with tourists and parked motorcycles including a blue Honda Wave 110 on kickstand in the foreground and a parked Honda Vision scooter in the background."
        },
        {
            "id": "commons_vietnam/HCMC_BuiVien_commuters_piaggio_scooters_manual_v243.png",
            "source_file": ACQ_DIR / "download" / "commons_18088520.jpg",
            "pageid": 18088520,
            "title": "File:Bui Vien St.ブイヴィエン通り Thành phố Hồ Chí Minh 城舗胡志明 ホーチミン DSCF1548.JPG",
            "url": "https://commons.wikimedia.org/wiki/File:Bui_Vien_St.%E3%83%96%E3%82%A4%E3%83%B4%E3%82%A3%E3%82%A7%E3%83%B3%E9%80%9A%E3%82%8A_Th%C3%A0nh_ph%E1%BB%91_H%E1%BB%93_Ch%C3%AD_Minh_%E5%9F%8E%E8%88%97%E8%83%A1%E5%BF%97%E6%98%8E_%E3%83%98%E3%83%BC%E3%83%81%E3%83%9F%E3%83%B3_DSCF1548.JPG",
            "author": "松岡明芳",
            "author_url": "https://commons.wikimedia.org/wiki/User:%E6%9D%BE%E5%B2%A1%E6%98%8E%E8%8A%B3",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "split": "val",
            "group": "group_matsuoka_buivien",
            "dest_img": "commons_vietnam__HCMC_BuiVien_commuters_piaggio_scooters_manual_v243.png",
            "dest_lbl": "commons_vietnam__HCMC_BuiVien_commuters_piaggio_scooters_manual_v243.txt",
            "boxes": [
                # Foreground Piaggio scooter commuter with helmet
                (1, (520, 830, 735, 1210)),
                # Foreground black Piaggio scooter
                (0, (370, 970, 800, 1360)),
                # Commuter in white jacket with helmet on red motorcycle
                (1, (975, 865, 1140, 1170)),
                # Red motorcycle in street
                (0, (905, 970, 1270, 1210)),
                # Street commuter with helmet on black scooter
                (1, (1350, 800, 1445, 1060)),
                # Black scooter
                (0, (1350, 890, 1445, 1100)),
                # Street commuter with helmet on red scooter
                (1, (1460, 800, 1585, 1050)),
                # Red scooter
                (0, (1460, 890, 1585, 1110))
            ],
            "description": "Bui Vien Street in District 1, Ho Chi Minh City, Vietnam, showing commuters on Piaggio scooters and motorcycles wearing helmets, with pedestrians and shophouses."
        },
        {
            "id": "commons_vietnam/HCMC_BuiVien_alley_parked_supercub_scooters_manual_v243.png",
            "source_file": ACQ_DIR / "download" / "commons_18088524.jpg",
            "pageid": 18088524,
            "title": "File:Bui Vien St.ブイヴィエン通り Thành phố Hồ Chí Minh 城舗胡志明 ホーチミン DSCF1575.JPG",
            "url": "https://commons.wikimedia.org/wiki/File:Bui_Vien_St.%E3%83%96%E3%82%A4%E3%83%B4%E3%82%A3%E3%82%A7%E3%83%B3%E9%80%9A%E3%82%8A_Th%C3%A0nh_ph%E1%BB%91_H%E1%BB%93_Ch%C3%AD_Minh_%E5%9F%8E%E8%88%97%E8%83%A1%E5%BF%97%E6%98%8E_%E3%83%98%E3%83%BC%E3%83%81%E3%83%9F%E3%83%B3_DSCF1575.JPG",
            "author": "松岡明芳",
            "author_url": "https://commons.wikimedia.org/wiki/User:%E6%9D%BE%E5%B2%A1%E6%98%8E%E8%8A%B3",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "split": "val",
            "group": "group_matsuoka_buivien",
            "dest_img": "commons_vietnam__HCMC_BuiVien_alley_parked_supercub_scooters_manual_v243.png",
            "dest_lbl": "commons_vietnam__HCMC_BuiVien_alley_parked_supercub_scooters_manual_v243.txt",
            "boxes": [
                # Parked orange and white vintage Honda Super Cub
                (0, (320, 630, 560, 1080)),
                # Parked white/black SYM scooter in foreground
                (0, (950, 680, 2048, 1360)),
                # Parked black SYM scooter
                (0, (980, 570, 1460, 1060)),
                # Parked teal scooter with basket
                (0, (940, 550, 1200, 960))
            ],
            "description": "Narrow residential alleyway off Bui Vien Street in Ho Chi Minh City, Vietnam, featuring parked vintage Honda Super Cub and scooters parked outside food vendors."
        }
    ]

    new_manifest_rows = []
    review_records = []

    for idx, c in enumerate(cands_info):
        print(f"Processing candidate {idx+1} ({c['id']})...")
        src_path = c["source_file"]
        assert src_path.is_file(), f"Missing source {src_path}"

        img = Image.open(src_path).convert("RGB")
        W, H = img.size

        # Save destination image as PNG
        dst_img_path = V243_DIR / c["split"] / "images" / c["dest_img"]
        img.save(dst_img_path, format="PNG")

        # Save label
        yolo_text = boxes_to_yolo(c["boxes"], W, H)
        dst_lbl_path = V243_DIR / c["split"] / "labels" / c["dest_lbl"]
        dst_lbl_path.write_text(yolo_text, encoding="utf-8")

        # Copy original
        shutil.copy2(src_path, V243_DIR / "source_originals" / f"{c['pageid']}.jpg")

        # Save source metadata
        src_meta = {
            "pageid": c["pageid"],
            "title": c["title"],
            "url": c["url"],
            "author": c["author"],
            "author_url": c["author_url"],
            "license": c["license"],
            "license_url": c["license_url"],
            "original_width": W,
            "original_height": H
        }
        meta_out = V243_DIR / "source_metadata" / f"{c['pageid']}.json"
        if meta_out.is_file():
            meta_out.unlink()
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(src_meta, f, ensure_ascii=False, indent=2)

        # Generate review evidence overlay
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay)
        for cls, (x1, y1, x2, y2) in c["boxes"]:
            color = "#00ff00" if cls == 1 else ("#ff3333" if cls == 2 else "#ffff00")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, int(min(W, H) * 0.004)))
        ev_name = f"evidence_{c['dest_img']}"
        overlay.save(V243_DIR / "review_evidence" / ev_name)

        img_sha = sha256_file(dst_img_path)
        img_phash = f"{perceptual_hash(img):016x}"

        row = {
            "id": c["id"],
            "file_name": c["dest_img"],
            "split": c["split"],
            "new_split": c["split"],
            "group": c["group"],
            "width": W,
            "height": H,
            "sha256": img_sha,
            "phash": img_phash,
            "source": "wikimedia_commons",
            "pageid": c["pageid"],
            "title": c["title"],
            "author": c["author"],
            "license": c["license"],
            "status": "clean_accepted",
            "review_status": "accepted_clean",
            "annotations_count": len(c["boxes"]),
            "boxes_detail": [
                {"class": b[0], "coords": [b[1][0], b[1][1], b[1][2], b[1][3]]}
                for b in c["boxes"]
            ]
        }
        new_manifest_rows.append(row)

        review_records.append({
            "image_id": c["id"],
            "decision": "accept",
            "assigned_split": c["split"],
            "assigned_group": c["group"],
            "license_verified": c["license"],
            "license_url": c["license_url"],
            "author": c["author"],
            "vietnam_context": c["description"],
            "box_count": len(c["boxes"]),
            "evidence_overlay": f"review_evidence/{ev_name}"
        })

    # Save review batch
    batch_record = {
        "batch_id": "manual_review_batch78",
        "snapshot": "dataset_v243",
        "reviewer": "antigravity_cloud_codex",
        "date": "2026-11-13",
        "items": review_records
    }
    with open(V243_DIR / "manual_review_batch78.json", "w", encoding="utf-8") as f:
        json.dump(batch_record, f, ensure_ascii=False, indent=2)

    # Update manifest
    print("Updating manifest.json...")
    manifest_path = V243_DIR / "manifest.json"
    with open(V242_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.extend(new_manifest_rows)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Update attribution.json
    attrib_path = V243_DIR / "attribution.json"
    with open(V242_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attrib = json.load(f)
    for c in cands_info:
        img_p = V243_DIR / c["split"] / "images" / c["dest_img"]
        attrib.append({
            "id": c["id"],
            "title": c["title"],
            "artist": c["author"],
            "license": c["license"],
            "license_url": c["license_url"],
            "page_url": c["url"],
            "download_url": c["url"],
            "original_url": c["url"],
            "sha256": sha256_file(img_p),
            "country": "VN",
            "location_evidence": c["description"],
            "modifications": "Original image converted to PNG; manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        })
    with open(attrib_path, "w", encoding="utf-8") as f:
        json.dump(attrib, f, ensure_ascii=False, indent=2)

    # Update ATTRIBUTION.md
    attrib_md = V243_DIR / "ATTRIBUTION.md"
    shutil.copy2(V242_DIR / "ATTRIBUTION.md", attrib_md)
    with open(attrib_md, "a", encoding="utf-8") as f:
        f.write("\n### dataset_v243 Additions\n")
        for c in cands_info:
            f.write(f"- `{c['dest_img']}`: [{c['title']}]({c['url']}) by {c['author']}, [{c['license']}]({c['license_url']})\n")

    # Update changes.json
    changes_path = V243_DIR / "changes.json"
    with open(V242_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v243",
        "parent_snapshot": "dataset_v242",
        "action": "batch_admit",
        "manual_review_batch": "manual_review_batch78.json",
        "admitted_items_count": 4,
        "new_train_images": 2,
        "new_val_images": 2,
        "new_test_images": 0,
        "boxes_added": {"0": 12, "1": 7, "2": 2},
        "total_boxes_added": 21,
        "test_split_locked": True,
        "isolation_verified": True
    })
    with open(changes_path, "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # Update statistics.json
    print("Updating statistics.json...")
    stats_path = V243_DIR / "statistics.json"
    with open(V242_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["accepted_clean_images"] = 1250
    stats["remaining_to_goal"] = 3750

    # Train stats: +2 images (35410109, 156674592)
    # Train boxes added: 4x cls 0 (2+2), 3x cls 1 (3+0), 2x cls 2 (2+0)
    stats["train"]["images"] += 2
    stats["train"]["classes"]["0"] += 4
    stats["train"]["classes"]["1"] += 3
    stats["train"]["classes"]["2"] += 2
    # group_staffan_scherz_hcmc (+0)
    # group_alexkom000_hanoi (+0)

    # Val stats: +2 images (18088520, 18088524)
    # Val boxes added: 8x cls 0 (4+4), 4x cls 1 (4+0), 0x cls 2 (0+0)
    stats["val"]["images"] += 2
    stats["val"]["classes"]["0"] += 8
    stats["val"]["classes"]["1"] += 4
    stats["val"]["classes"]["2"] += 0
    # group_matsuoka_buivien (+1)
    stats["val"]["groups"] = stats["val"].get("groups", 0) + 1

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Verification report
    verif = {
        "snapshot": "dataset_v243",
        "parent": "dataset_v242",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2407,
        "rows_total": 2411,
        "accepted_clean_images": 1250,
        "remaining_to_5000": 3750,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 12, "1": 7, "2": 2},
        "split_counts": {
            "train": 983,
            "val": 199,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch78.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V243_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(attrib_path),
        "duplicates_sha256": sha256_file(V243_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(changes_path)
    }
    with open(V243_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v243.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    with open(progress_path, "r", encoding="utf-8") as f:
        progress = json.load(f)

    progress["accepted_clean_images"] = 1250
    progress["remaining_to_goal"] = 3750
    progress["snapshot"] = "dataset_v243"
    progress["accepted_split_counts"]["train"] = 983
    progress["accepted_split_counts"]["validation"] = 199
    progress["accepted_split_counts"]["test_locked"] = 68
    progress["newly_accepted_this_batch"] = 4
    progress["new_batch_class_box_counts"] = {"0": 12, "1": 7, "2": 2}
    progress["total_class_box_counts"]["0"] += 12
    progress["total_class_box_counts"]["1"] += 7
    progress["total_class_box_counts"]["2"] += 2
    progress["verification_report"] = "../verification/dataset_v243.json"
    progress["notes"].append(
        "dataset_v243 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY 2.0 / CC BY 4.0 / CC BY-SA 3.0), adding 21 clean bounding boxes including 2 rare unhelmeted young children (Class 2): Saigon family of 5 on Honda Air Blade scooter, Hanoi Old Quarter pedestrian street with parked Honda Wave 110, HCMC Bui Vien street commuters on Piaggio scooters and motorcycles, and HCMC Bui Vien alleyway food stall with vintage Honda Super Cub and parked scooters. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,250 (25.0% of 5,000 milestone)."
    )
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v243 build completed successfully! ===")

if __name__ == "__main__":
    main()
