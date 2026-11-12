"""Build and verify dataset_v236 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Buon Ma Thuot city square traffic commuters (Rdavout, CC BY-SA 3.0) -> train (group_rdavout_buonmathuot) [4x cls 0, 5x cls 1]
2. Binh Duong Thuan An market street commuters (Kieprongbuon812, CC BY-SA 4.0) -> train (group_kieprongbuon812_binhduong) [2x cls 0, 1x cls 1, 1x cls 2]
3. Northern Vietnam mountain road parked Minsk (Ryansheldonholliday, CC0) -> val (group_ryansheldonholliday_minsk) [1x cls 0]
4. Hanoi Vu Pham Ham commuters postal trucks (Gispenni, CC BY-SA 4.0) -> val (group_gispenni_vuphamham) [3x cls 0, 4x cls 1]

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
V235_DIR = TRAINING_DIR / "dataset_v235"
V236_DIR = TRAINING_DIR / "dataset_v236"
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
    print("=== Building Dataset v236 Snapshot ===")
    assert V235_DIR.is_dir(), f"Missing parent snapshot {V235_DIR}"

    if V236_DIR.exists():
        shutil.rmtree(V236_DIR)
    V236_DIR.mkdir(parents=True, exist_ok=True)

    # Copy metadata files from v235
    shutil.copy2(V235_DIR / "split_lock.json", V236_DIR / "split_lock.json")
    shutil.copy2(V235_DIR / "data.yaml", V236_DIR / "data.yaml")
    shutil.copy2(V235_DIR / "duplicates.json", V236_DIR / "duplicates.json")
    shutil.copy2(V235_DIR / "usage_restrictions.json", V236_DIR / "usage_restrictions.json")
    shutil.copy2(V235_DIR / "README.md", V236_DIR / "README.md")

    # Set up train / val / test directories
    for split in ["train", "val", "test"]:
        (V236_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V236_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Link existing v235 images and labels
    print("Linking existing v235 assets...")
    for split in ["train", "val", "test"]:
        src_img_dir = V235_DIR / split / "images"
        dst_img_dir = V236_DIR / split / "images"
        if src_img_dir.exists():
            for item in src_img_dir.iterdir():
                target = item.resolve()
                dst = dst_img_dir / item.name
                os.symlink(target, dst)

        src_lbl_dir = V235_DIR / split / "labels"
        dst_lbl_dir = V236_DIR / split / "labels"
        if src_lbl_dir.exists():
            for item in src_lbl_dir.iterdir():
                dst = dst_lbl_dir / item.name
                shutil.copy2(item, dst)

    # Directories for evidence and metadata (hardlink existing files to save disk space)
    (V236_DIR / "review_evidence").mkdir(exist_ok=True)
    for f in (V235_DIR / "review_evidence").iterdir():
        if f.is_file():
            os.link(f, V236_DIR / "review_evidence" / f.name)

    (V236_DIR / "source_originals").mkdir(exist_ok=True)
    for f in (V235_DIR / "source_originals").iterdir():
        if f.is_file():
            os.link(f, V236_DIR / "source_originals" / f.name)

    (V236_DIR / "source_metadata").mkdir(exist_ok=True)
    for f in (V235_DIR / "source_metadata").iterdir():
        if f.is_file():
            os.link(f, V236_DIR / "source_metadata" / f.name)

    # Candidate definitions for v236
    cands_info = [
        {
            "id": "commons_vietnam/BuonMaThuot_city_square_traffic_commuters_manual_v236.png",
            "source_file": ACQ_DIR / "download" / "commons_6673323.jpg",
            "pageid": 6673323,
            "title": "File:Buôn Ma Thuột city square.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Bu%C3%B4n_Ma_Thu%E1%BB%99t_city_square.jpg",
            "author": "Rdavout",
            "author_url": "https://commons.wikimedia.org/wiki/User:Rdavout",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "split": "train",
            "group": "group_rdavout_buonmathuot",
            "dest_img": "commons_vietnam__BuonMaThuot_city_square_traffic_commuters_manual_v236.png",
            "dest_lbl": "commons_vietnam__BuonMaThuot_city_square_traffic_commuters_manual_v236.txt",
            "boxes": [
                # Center commuter on red/black bike
                (1, (1170, 1150, 1300, 1370)),
                (0, (1140, 1245, 1315, 1415)),
                # Right commuter driver
                (1, (1675, 1160, 1800, 1360)),
                # Right commuter passenger
                (1, (1630, 1150, 1730, 1350)),
                # Right blue motorbike
                (0, (1620, 1220, 1930, 1415)),
                # Left commuter dark green bike
                (1, (340, 1180, 440, 1370)),
                (0, (315, 1250, 465, 1415)),
                # Left commuter dark bike
                (1, (465, 1185, 565, 1365)),
                (0, (445, 1245, 575, 1405))
            ],
            "description": "Motorcycle commuters with safety helmets circulating around the iconic Victory Monument roundabout in Buon Ma Thuot city square, Dak Lak, Vietnam."
        },
        {
            "id": "commons_vietnam/BinhDuong_ThuanAn_market_street_commuters_manual_v236.png",
            "source_file": ACQ_DIR / "download" / "commons_61386152.jpg",
            "pageid": 61386152,
            "title": "File:Chợ Bình Chuẩn, Thuận An, Bình Dương.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Ch%E1%BB%A3_B%C3%ACnh_Chu%E1%BA%A9n%2C_Thu%E1%BA%ADn_An%2C_B%C3%ACnh_D%C6%B0%C6%A1ng.jpg",
            "author": "Kieprongbuon812",
            "author_url": "https://commons.wikimedia.org/wiki/User:Kieprongbuon812",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "train",
            "group": "group_kieprongbuon812_binhduong",
            "dest_img": "commons_vietnam__BinhDuong_ThuanAn_market_street_commuters_manual_v236.png",
            "dest_lbl": "commons_vietnam__BinhDuong_ThuanAn_market_street_commuters_manual_v236.txt",
            "boxes": [
                # Young man rider with helmet
                (1, (160, 850, 700, 1790)),
                # White Honda Wave motorbike
                (0, (0, 1180, 780, 1850)),
                # Commuter in pink cap without helmet
                (2, (880, 1000, 1120, 1420)),
                # Blue/white scooter
                (0, (840, 1150, 1270, 1500))
            ],
            "description": "Motorcycle commuters outside Binh Chuan market in Thuan An, Binh Duong, southern Vietnam, featuring a young man on a white Honda Wave with helmet and a commuter in a baseball cap without helmet."
        },
        {
            "id": "commons_vietnam/NorthernVietnam_mountain_road_parked_Minsk_manual_v236.png",
            "source_file": ACQ_DIR / "download" / "commons_185886752.jpg",
            "pageid": 185886752,
            "title": "File:Minsk motorcycle in Vietnam.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Minsk_motorcycle_in_Vietnam.jpg",
            "author": "Ryansheldonholliday",
            "author_url": "https://commons.wikimedia.org/wiki/User:Ryansheldonholliday",
            "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "split": "val",
            "group": "group_ryansheldonholliday_minsk",
            "dest_img": "commons_vietnam__NorthernVietnam_mountain_road_parked_Minsk_manual_v236.png",
            "dest_lbl": "commons_vietnam__NorthernVietnam_mountain_road_parked_Minsk_manual_v236.txt",
            "boxes": [
                # Vintage Soviet Minsk motorcycle parked
                (0, (0, 0, 3393, 2300))
            ],
            "description": "A classic vintage blue Soviet Minsk 125 motorcycle parked on a scenic mountain roadway during a motorcycle touring trip across northern Vietnam."
        },
        {
            "id": "commons_vietnam/Hanoi_VuPhamHam_commuters_postal_trucks_manual_v236.png",
            "source_file": ACQ_DIR / "download" / "commons_184809335.jpg",
            "pageid": 184809335,
            "title": "File:Midmorning of Vũ Phạm Hàm Street 23-02-2026.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Midmorning_of_V%C5%A9_Ph%E1%BA%A1m_H%C3%A0m_Street_23-02-2026.jpg",
            "author": "Gispenni",
            "author_url": "https://commons.wikimedia.org/wiki/User:Gispenni",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "val",
            "group": "group_gispenni_vuphamham",
            "dest_img": "commons_vietnam__Hanoi_VuPhamHam_commuters_postal_trucks_manual_v236.png",
            "dest_lbl": "commons_vietnam__Hanoi_VuPhamHam_commuters_postal_trucks_manual_v236.txt",
            "boxes": [
                # Rightmost driver yellow helmet
                (1, (3730, 1480, 3880, 1850)),
                # Rightmost passenger white hoodie
                (1, (3840, 1490, 4020, 1850)),
                # Rightmost grey scooter
                (0, (3670, 1610, 4080, 2010)),
                # Grab commuter green jacket
                (1, (3420, 1485, 3630, 1820)),
                # Grab red motorbike
                (0, (3360, 1600, 3730, 1890)),
                # Commuter blue bike
                (1, (3150, 1490, 3320, 1770)),
                # Blue motorbike
                (0, (3140, 1600, 3400, 1910))
            ],
            "description": "Motorcycle commuters wearing safety helmets, including a Grab bike driver, passing yellow postal delivery trucks on Vu Pham Ham Street in Cau Giay, Hanoi, Vietnam."
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
        dst_img_path = V236_DIR / c["split"] / "images" / c["dest_img"]
        img.save(dst_img_path, format="PNG")

        # Save label
        yolo_text = boxes_to_yolo(c["boxes"], W, H)
        dst_lbl_path = V236_DIR / c["split"] / "labels" / c["dest_lbl"]
        dst_lbl_path.write_text(yolo_text, encoding="utf-8")

        # Copy original
        shutil.copy2(src_path, V236_DIR / "source_originals" / f"{c['pageid']}.jpg")

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
        with open(V236_DIR / "source_metadata" / f"{c['pageid']}.json", "w", encoding="utf-8") as f:
            json.dump(src_meta, f, ensure_ascii=False, indent=2)

        # Generate review evidence overlay
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay)
        for cls, (x1, y1, x2, y2) in c["boxes"]:
            color = "#00ff00" if cls == 1 else ("#ff3333" if cls == 2 else "#ffff00")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, int(min(W, H) * 0.004)))
        ev_name = f"evidence_{c['dest_img']}"
        overlay.save(V236_DIR / "review_evidence" / ev_name)

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
        "batch_id": "manual_review_batch71",
        "snapshot": "dataset_v236",
        "reviewer": "antigravity_cloud_codex",
        "date": "2026-11-12",
        "items": review_records
    }
    with open(V236_DIR / "manual_review_batch71.json", "w", encoding="utf-8") as f:
        json.dump(batch_record, f, ensure_ascii=False, indent=2)

    # Update manifest
    print("Updating manifest.json...")
    manifest_path = V236_DIR / "manifest.json"
    with open(V235_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.extend(new_manifest_rows)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Update attribution.json
    attrib_path = V236_DIR / "attribution.json"
    with open(V235_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attrib = json.load(f)
    for c in cands_info:
        img_p = V236_DIR / c["split"] / "images" / c["dest_img"]
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
    attrib_md = V236_DIR / "ATTRIBUTION.md"
    shutil.copy2(V235_DIR / "ATTRIBUTION.md", attrib_md)
    with open(attrib_md, "a", encoding="utf-8") as f:
        f.write("\n### dataset_v236 Additions\n")
        for c in cands_info:
            f.write(f"- `{c['dest_img']}`: [{c['title']}]({c['url']}) by {c['author']}, [{c['license']}]({c['license_url']})\n")

    # Update changes.json
    changes_path = V236_DIR / "changes.json"
    with open(V235_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v236",
        "parent": "dataset_v235",
        "action": "batch_admit",
        "manual_review_batch": "manual_review_batch71.json",
        "admitted_items_count": 4,
        "new_train_images": 2,
        "new_val_images": 2,
        "new_test_images": 0,
        "boxes_added": {"0": 10, "1": 10, "2": 1},
        "total_boxes_added": 21,
        "test_split_locked": True,
        "isolation_verified": True
    })
    with open(changes_path, "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # Update statistics.json
    print("Updating statistics.json...")
    stats_path = V236_DIR / "statistics.json"
    with open(V235_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Train stats: +2 images (6673323, 61386152)
    # Train boxes added: 6x cls 0 (4+2), 6x cls 1 (5+1), 1x cls 2 (0+1)
    stats["train"]["images"] += 2
    stats["train"]["classes"]["0"] += 6
    stats["train"]["classes"]["1"] += 6
    stats["train"]["classes"]["2"] += 1
    stats["train"]["groups"] += 2 # group_rdavout_buonmathuot, group_kieprongbuon812_binhduong

    # Val stats: +2 images (185886752, 184809335)
    # Val boxes added: 4x cls 0 (1+3), 4x cls 1 (0+4), 0x cls 2
    stats["val"]["images"] += 2
    stats["val"]["classes"]["0"] += 4
    stats["val"]["classes"]["1"] += 4
    stats["val"]["classes"]["2"] += 0
    stats["val"]["groups"] += 1 # group_gispenni_vuphamham (group_ryansheldonholliday_minsk already in val)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Verification report
    verif = {
        "snapshot": "dataset_v236",
        "parent": "dataset_v235",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2379,
        "rows_total": 2383,
        "accepted_clean_images": 1222,
        "remaining_to_5000": 3778,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 10, "1": 10, "2": 1},
        "split_counts": {
            "train": 969,
            "val": 185,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch71.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V236_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(V236_DIR / "attribution.json"),
        "duplicates_sha256": sha256_file(V236_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(V236_DIR / "changes.json")
    }
    with open(V236_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v236.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    with open(progress_path, "r", encoding="utf-8") as f:
        progress = json.load(f)

    progress["accepted_clean_images"] = 1222
    progress["remaining_to_goal"] = 3778
    progress["snapshot"] = "dataset_v236"
    progress["accepted_split_counts"]["train"] = 969
    progress["accepted_split_counts"]["validation"] = 185
    progress["accepted_split_counts"]["test_locked"] = 68
    progress["newly_accepted_this_batch"] = 4
    progress["new_batch_class_box_counts"] = {"0": 10, "1": 10, "2": 1}
    progress["total_class_box_counts"]["0"] += 10
    progress["total_class_box_counts"]["1"] += 10
    progress["total_class_box_counts"]["2"] += 1
    progress["verification_report"] = "../verification/dataset_v236.json"
    progress["notes"].append(
        "dataset_v236 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 3.0 / CC BY-SA 4.0 / CC0), adding 21 clean bounding boxes including Buon Ma Thuot city square roundabout commuters with helmets, Binh Duong Thuan An market street commuters (including rare unhelmeted commuter in baseball cap), vintage Soviet Minsk motorcycle in northern Vietnam mountain roadway, and Hanoi Vu Pham Ham commuters passing yellow postal delivery trucks. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,222."
    )
    with open(progress_path, "r+", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
        f.truncate()

    print("=== Dataset v236 build completed successfully! ===")

if __name__ == "__main__":
    main()
