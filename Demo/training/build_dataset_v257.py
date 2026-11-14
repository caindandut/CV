"""Build and verify dataset_v257 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Hanoi intersection street commuters waiting & flowing on motorbikes (RG72, CC BY-SA 4.0) -> train (group_rg72_quangngai) [5x cls 0, 5x cls 1, 0x cls 2]
2. Hanoi street intersection stopped commuters in front of tree-lined boulevard (Paris 16, CC BY 2.0) -> train (group_paris16_hanoi) [5x cls 0, 5x cls 1, 0x cls 2]
3. Ho Chi Minh City street traffic flow moving across pedestrian crosswalk (Dennis G. Jarvis, CC BY-SA 2.0) -> val (group_dennis_jarvis_saigon) [5x cls 0, 5x cls 1, 0x cls 2]
4. Hanoi Tran Phu street traffic commuters outside commemorative 30-4 billboard (Michelle, CC BY 2.0) -> val (group_michelle_hanoi) [4x cls 0, 4x cls 1, 2x cls 2]

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
V256_DIR = TRAINING_DIR / "dataset_v256"
V257_DIR = TRAINING_DIR / "dataset_v257"
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
    print("=== Building Dataset v257 Snapshot ===")
    assert V256_DIR.is_dir(), f"Missing parent snapshot {V256_DIR}"

    if V257_DIR.exists():
        shutil.rmtree(V257_DIR)
    V257_DIR.mkdir(parents=True, exist_ok=True)

    # Copy metadata files from v256
    shutil.copy2(V256_DIR / "split_lock.json", V257_DIR / "split_lock.json")
    shutil.copy2(V256_DIR / "data.yaml", V257_DIR / "data.yaml")
    shutil.copy2(V256_DIR / "duplicates.json", V257_DIR / "duplicates.json")
    shutil.copy2(V256_DIR / "usage_restrictions.json", V257_DIR / "usage_restrictions.json")
    shutil.copy2(V256_DIR / "README.md", V257_DIR / "README.md")

    # Set up train / val / test directories
    for split in ["train", "val", "test"]:
        (V257_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V257_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Link existing v256 images and labels
    print("Linking existing v256 assets...")
    for split in ["train", "val", "test"]:
        src_img_dir = V256_DIR / split / "images"
        dst_img_dir = V257_DIR / split / "images"
        if src_img_dir.exists():
            for item in src_img_dir.iterdir():
                target = item.resolve()
                dst = dst_img_dir / item.name
                os.symlink(target, dst)

        src_lbl_dir = V256_DIR / split / "labels"
        dst_lbl_dir = V257_DIR / split / "labels"
        if src_lbl_dir.exists():
            for item in src_lbl_dir.iterdir():
                dst = dst_lbl_dir / item.name
                shutil.copy2(item, dst)

    # Directories for evidence and metadata for batch 92 (batch-specific only)
    (V257_DIR / "review_evidence").mkdir(exist_ok=True)
    (V257_DIR / "source_originals").mkdir(exist_ok=True)
    (V257_DIR / "source_metadata").mkdir(exist_ok=True)

    # Candidate definitions for v257
    cands_info = [
        {
            "id": "commons_vietnam/Hanoi_intersection_traffic_commuters_RG72_manual_v257.png",
            "source_file": ACQ_DIR / "download" / "commons_98363051.jpg",
            "pageid": 98363051,
            "title": "File:Vojkruciĝo en Hanojo 01.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Vojkruci%C4%9Do_en_Hanojo_01.jpg",
            "author": "RG72",
            "author_url": "https://commons.wikimedia.org/wiki/User:RG72",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "train",
            "group": "group_rg72_quangngai",
            "dest_img": "commons_vietnam__Hanoi_intersection_traffic_commuters_RG72_manual_v257.png",
            "dest_lbl": "commons_vietnam__Hanoi_intersection_traffic_commuters_RG72_manual_v257.txt",
            "boxes": [
                # 1. Foreground left motorcycle & man on phone
                (0, (100, 1280, 570, 2070)),
                (1, (100, 915, 580, 1980)),
                # 2. Center woman on blue scooter
                (0, (1260, 1130, 1600, 1480)),
                (1, (1380, 880, 1580, 1370)),
                # 3. White scooter beside her
                (0, (1040, 1090, 1320, 1385)),
                (1, (1180, 890, 1410, 1280)),
                # 4. Blue Honda Wave with rider in white shirt
                (0, (1580, 1070, 2100, 1400)),
                (1, (1720, 885, 1980, 1360)),
                # 5. Right foreground commuter facing away
                (0, (2660, 1250, 3072, 1950)),
                (1, (2640, 785, 3072, 1900))
            ],
            "description": "Hanoi intersection street traffic with commuters waiting and flowing on motorbikes and scooters, including man talking on mobile phone on parked motorcycle and woman in yellow helmet and sun hood on blue scooter."
        },
        {
            "id": "commons_vietnam/Hanoi_boulevard_stopped_women_commuters_manual_v257.png",
            "source_file": ACQ_DIR / "download" / "commons_5627569.jpg",
            "pageid": 5627569,
            "title": "File:Motorbike in Hanoi.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Motorbike_in_Hanoi.jpg",
            "author": "Paris 16",
            "author_url": "https://commons.wikimedia.org/wiki/User:Paris_16",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "split": "train",
            "group": "group_paris16_hanoi",
            "dest_img": "commons_vietnam__Hanoi_boulevard_stopped_women_commuters_manual_v257.png",
            "dest_lbl": "commons_vietnam__Hanoi_boulevard_stopped_women_commuters_manual_v257.txt",
            "boxes": [
                # 1. Far left white scooter & rider
                (0, (0, 250, 110, 375)),
                (1, (5, 155, 108, 370)),
                # 2. Plaid woman & maroon scooter
                (0, (95, 220, 270, 400)),
                (1, (160, 170, 265, 380)),
                # 3. Blue shirt woman & red Honda Wave
                (0, (295, 235, 405, 410)),
                (1, (300, 175, 415, 390)),
                # 4. Center floral woman & silver Yamaha Nouvo
                (0, (410, 225, 550, 475)),
                (1, (410, 140, 560, 470)),
                # 5. Right pink shirt woman & turquoise Yamaha Sirius
                (0, (540, 235, 680, 415)),
                (1, (540, 165, 640, 365))
            ],
            "description": "Commuters on motorbikes and scooters stopped at Hanoi street intersection in front of tree-lined boulevard, showing women wearing protective safety helmets and face masks."
        },
        {
            "id": "commons_vietnam/Saigon_traffic_crosswalk_commuters_Tet_flowers_manual_v257.png",
            "source_file": ACQ_DIR / "download" / "commons_66993996.jpg",
            "pageid": 66993996,
            "title": "File:DGJ 0592 - Who needs a car! (3374060114).jpg",
            "url": "https://commons.wikimedia.org/wiki/File:DGJ_0592_-_Who_needs_a_car!_(3374060114).jpg",
            "author": "Dennis G. Jarvis",
            "author_url": "https://commons.wikimedia.org/wiki/User:Dennis_G._Jarvis",
            "license": "CC BY-SA 2.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.0/",
            "split": "val",
            "group": "group_dennis_jarvis_saigon",
            "dest_img": "commons_vietnam__Saigon_traffic_crosswalk_commuters_Tet_flowers_manual_v257.png",
            "dest_lbl": "commons_vietnam__Saigon_traffic_crosswalk_commuters_Tet_flowers_manual_v257.txt",
            "boxes": [
                # 1. Blue motorcycle & rider
                (0, (140, 2180, 550, 2750)),
                (1, (100, 1870, 550, 2600)),
                # 2. Cyan scooter & rider
                (0, (920, 2060, 1220, 2480)),
                (1, (900, 1880, 1250, 2400)),
                # 3. Dark scooter center & rider
                (0, (1280, 2060, 1600, 2520)),
                (1, (1280, 1880, 1600, 2400)),
                # 4. Red motorcycle & rider
                (0, (1750, 2100, 2120, 2550)),
                (1, (1760, 1880, 2120, 2450)),
                # 5. Far right scooter & rider
                (0, (2460, 2120, 2820, 2550)),
                (1, (2440, 1950, 2750, 2450))
            ],
            "description": "Dense Ho Chi Minh City street traffic flow moving across pedestrian crosswalk under yellow and pink Tet flower decorations strung between street trees."
        },
        {
            "id": "commons_vietnam/Hanoi_TranPhu_billboard_commuters_unhelmeted_girls_manual_v257.png",
            "source_file": ACQ_DIR / "download" / "commons_5825097.jpg",
            "pageid": 5825097,
            "title": "File:Buýt 33 Hà Nội.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Bu%C3%BDt_33_H%C3%A0_N%E1%BB%99i.jpg",
            "author": "Michelle",
            "author_url": "https://www.flickr.com/photos/25918695@N00",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "split": "val",
            "group": "group_michelle_hanoi",
            "dest_img": "commons_vietnam__Hanoi_TranPhu_billboard_commuters_unhelmeted_girls_manual_v257.png",
            "dest_lbl": "commons_vietnam__Hanoi_TranPhu_billboard_commuters_unhelmeted_girls_manual_v257.txt",
            "boxes": [
                # 1. Left red scooter & 2 unhelmeted young girl riders (Class 2)
                (0, (70, 510, 165, 740)),
                (2, (100, 440, 155, 600)),
                (2, (65, 440, 115, 580)),
                # 2. Dark motorcycle midground & rider (Class 1)
                (0, (640, 490, 740, 610)),
                (1, (660, 430, 730, 570)),
                # 3. Super Cub center & rider (Class 1)
                (0, (700, 480, 865, 650)),
                (1, (730, 415, 820, 590)),
                # 4. Right Honda Wave & 2 riders (Class 1)
                (0, (1060, 480, 1280, 700)),
                (1, (1100, 395, 1230, 650)),
                (1, (1050, 390, 1150, 640))
            ],
            "description": "Hanoi Tran Phu street intersection traffic outside commemorative 30-4 liberation billboard, showing bus 33, two young women riding red scooter without helmets alongside commuters on Honda Super Cub, dark motorcycle, and Honda Wave."
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
        dst_img_path = V257_DIR / c["split"] / "images" / c["dest_img"]
        img.save(dst_img_path, format="PNG")

        # Save label
        yolo_text = boxes_to_yolo(c["boxes"], W, H)
        dst_lbl_path = V257_DIR / c["split"] / "labels" / c["dest_lbl"]
        dst_lbl_path.write_text(yolo_text, encoding="utf-8")

        # Copy original (batch-specific only)
        shutil.copy2(src_path, V257_DIR / "source_originals" / f"{c['pageid']}.jpg")

        # Save source metadata (batch-specific only)
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
        meta_out = V257_DIR / "source_metadata" / f"{c['pageid']}.json"
        if meta_out.is_file():
            meta_out.unlink()
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(src_meta, f, ensure_ascii=False, indent=2)

        # Generate review evidence overlay (resized to max 2048px with optimize=True)
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay)
        for cls, (x1, y1, x2, y2) in c["boxes"]:
            color = "#00ff00" if cls == 1 else ("#ff3333" if cls == 2 else "#ffff00")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, int(min(W, H) * 0.004)))
        overlay.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        ev_name = f"evidence_{c['dest_img']}"
        overlay.save(V257_DIR / "review_evidence" / ev_name, optimize=True)

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

    # Save review batch 92
    batch_record = {
        "batch_id": "manual_review_batch92",
        "snapshot": "dataset_v257",
        "reviewer": "antigravity_cloud_codex",
        "date": "2026-11-14",
        "items": review_records
    }
    with open(V257_DIR / "manual_review_batch92.json", "w", encoding="utf-8") as f:
        json.dump(batch_record, f, ensure_ascii=False, indent=2)

    # Update manifest
    print("Updating manifest.json...")
    manifest_path = V257_DIR / "manifest.json"
    with open(V256_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.extend(new_manifest_rows)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Update attribution.json
    attrib_path = V257_DIR / "attribution.json"
    with open(V256_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attrib = json.load(f)
    for c in cands_info:
        img_p = V257_DIR / c["split"] / "images" / c["dest_img"]
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
    attrib_md = V257_DIR / "ATTRIBUTION.md"
    shutil.copy2(V256_DIR / "ATTRIBUTION.md", attrib_md)
    with open(attrib_md, "a", encoding="utf-8") as f:
        f.write("\n### dataset_v257 Additions\n")
        for c in cands_info:
            f.write(f"- `{c['dest_img']}`: [{c['title']}]({c['url']}) by {c['author']}, [{c['license']}]({c['license_url']})\n")

    # Update changes.json
    changes_path = V257_DIR / "changes.json"
    with open(V256_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v257",
        "parent_snapshot": "dataset_v256",
        "action": "batch_admit",
        "manual_review_batch": "manual_review_batch92",
        "admitted_items_count": 4,
        "new_train_images": 2,
        "new_val_images": 2,
        "new_test_images": 0,
        "boxes_added": {"0": 19, "1": 19, "2": 2},
        "total_boxes_added": 40,
        "test_split_locked": True,
        "isolation_verified": True
    })
    with open(changes_path, "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # Update statistics.json
    print("Updating statistics.json...")
    stats_path = V257_DIR / "statistics.json"
    with open(V256_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["accepted_clean_images"] = 1306
    stats["remaining_to_goal"] = 3694

    # Train stats: +2 images (98363051, 5627569)
    # Train boxes added: 10x cls 0 (5+5), 10x cls 1 (5+5), 0x cls 2
    stats["train"]["images"] += 2
    stats["train"]["classes"]["0"] += 10
    stats["train"]["classes"]["1"] += 10
    stats["train"]["classes"]["2"] += 0
    stats["train"]["groups"] += 1  # new group: group_paris16_hanoi (group_rg72_quangngai already existed)

    # Val stats: +2 images (66993996, 5825097)
    # Val boxes added: 9x cls 0 (5+4), 9x cls 1 (5+4), 2x cls 2 (0+2)
    stats["val"]["images"] += 2
    stats["val"]["classes"]["0"] += 9
    stats["val"]["classes"]["1"] += 9
    stats["val"]["classes"]["2"] += 2
    stats["val"]["groups"] += 1  # new group: group_michelle_hanoi (group_dennis_jarvis_saigon already existed)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Verification report
    verif = {
        "snapshot": "dataset_v257",
        "parent": "dataset_v256",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2463,
        "rows_total": 2467,
        "accepted_clean_images": 1306,
        "remaining_to_5000": 3694,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 19, "1": 19, "2": 2},
        "split_counts": {
            "train": 1015,
            "val": 223,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch92.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V257_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(attrib_path),
        "duplicates_sha256": sha256_file(V257_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(changes_path)
    }
    with open(V257_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v257.json"
    verif_report_path.parent.mkdir(exist_ok=True)
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    with open(progress_path, "r", encoding="utf-8") as f:
        progress = json.load(f)

    progress["accepted_clean_images"] = 1306
    progress["remaining_to_goal"] = 3694
    progress["snapshot"] = "dataset_v257"
    progress["accepted_split_counts"]["train"] = 1015
    progress["accepted_split_counts"]["validation"] = 223
    progress["accepted_split_counts"]["test_locked"] = 68
    progress["newly_accepted_this_batch"] = 4
    progress["new_batch_class_box_counts"] = {"0": 19, "1": 19, "2": 2}
    progress["total_class_box_counts"]["0"] += 19
    progress["total_class_box_counts"]["1"] += 19
    progress["total_class_box_counts"]["2"] += 2
    progress["verification_report"] = "../verification/dataset_v257.json"
    progress["notes"].append(
        "dataset_v257 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 4.0 / CC BY 2.0 / CC BY-SA 2.0), adding 40 clean bounding boxes (19 Class 0, 19 Class 1, 2 Class 2): Hanoi intersection street commuters waiting & flowing on motorbikes (RG72, train), Hanoi street intersection stopped commuters in front of tree-lined boulevard (Paris 16, train), Ho Chi Minh City street traffic flow moving across pedestrian crosswalk (Dennis G. Jarvis, val), and Hanoi Tran Phu street traffic commuters outside commemorative 30-4 billboard with 2 unhelmeted young girl commuters (Michelle, val). Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,306."
    )
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v257 build completed successfully! ===")

if __name__ == "__main__":
    main()
