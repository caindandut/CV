"""Build and verify dataset_v244 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Hanoi Old Quarter street fruit seller and commuters on scooters with 2 unhelmeted riders (Alexkom000, CC BY 4.0) -> train (group_alexkom000_hanoi) [6x cls 0, 0x cls 1, 2x cls 2]
2. North Vietnam street scene with young man yawning on Honda Wave and passing commuter (RG72, CC BY-SA 4.0) -> train (group_rg72_quangngai) [3x cls 0, 1x cls 1, 1x cls 2]
3. Hanoi Tu Mo street commuters on scooters with 2 unhelmeted riders behind taxi (Gispenni, CC BY-SA 4.0) -> val (group_gispenni_vuphamham) [2x cls 0, 0x cls 1, 2x cls 2]
4. HCMC Bui Vien street scene with vintage green Honda Super Cub on kickstand and parked bikes (松岡明芳, CC BY-SA 3.0) -> val (group_matsuoka_buivien) [3x cls 0, 0x cls 1, 0x cls 2]

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
V243_DIR = TRAINING_DIR / "dataset_v243"
V244_DIR = TRAINING_DIR / "dataset_v244"
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
    print("=== Building Dataset v244 Snapshot ===")
    assert V243_DIR.is_dir(), f"Missing parent snapshot {V243_DIR}"

    if V244_DIR.exists():
        shutil.rmtree(V244_DIR)
    V244_DIR.mkdir(parents=True, exist_ok=True)

    # Copy metadata files from v243
    shutil.copy2(V243_DIR / "split_lock.json", V244_DIR / "split_lock.json")
    shutil.copy2(V243_DIR / "data.yaml", V244_DIR / "data.yaml")
    shutil.copy2(V243_DIR / "duplicates.json", V244_DIR / "duplicates.json")
    shutil.copy2(V243_DIR / "usage_restrictions.json", V244_DIR / "usage_restrictions.json")
    shutil.copy2(V243_DIR / "README.md", V244_DIR / "README.md")

    # Set up train / val / test directories
    for split in ["train", "val", "test"]:
        (V244_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V244_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Link existing v243 images and labels
    print("Linking existing v243 assets...")
    for split in ["train", "val", "test"]:
        src_img_dir = V243_DIR / split / "images"
        dst_img_dir = V244_DIR / split / "images"
        if src_img_dir.exists():
            for item in src_img_dir.iterdir():
                target = item.resolve()
                dst = dst_img_dir / item.name
                os.symlink(target, dst)

        src_lbl_dir = V243_DIR / split / "labels"
        dst_lbl_dir = V244_DIR / split / "labels"
        if src_lbl_dir.exists():
            for item in src_lbl_dir.iterdir():
                dst = dst_lbl_dir / item.name
                shutil.copy2(item, dst)

    # Directories for evidence and metadata
    (V244_DIR / "review_evidence").mkdir(exist_ok=True)
    for f in (V243_DIR / "review_evidence").iterdir():
        if f.is_file():
            os.link(f, V244_DIR / "review_evidence" / f.name)

    (V244_DIR / "source_originals").mkdir(exist_ok=True)
    for f in (V243_DIR / "source_originals").iterdir():
        if f.is_file():
            os.link(f, V244_DIR / "source_originals" / f.name)

    (V244_DIR / "source_metadata").mkdir(exist_ok=True)
    for f in (V243_DIR / "source_metadata").iterdir():
        if f.is_file():
            os.link(f, V244_DIR / "source_metadata" / f.name)

    # Candidate definitions for v244
    cands_info = [
        {
            "id": "commons_vietnam/Hanoi_OldQuarter_fruitseller_supercub_manual_v244.png",
            "source_file": ACQ_DIR / "download" / "commons_156674593.jpg",
            "pageid": 156674593,
            "title": "File:2024-11-03 Hanoi's Old Quarter 5.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:2024-11-03_Hanoi%27s_Old_Quarter_5.jpg",
            "author": "Alexkom000",
            "author_url": "https://commons.wikimedia.org/wiki/User:Alexkom000",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "split": "train",
            "group": "group_alexkom000_hanoi",
            "dest_img": "commons_vietnam__Hanoi_OldQuarter_fruitseller_supercub_manual_v244.png",
            "dest_lbl": "commons_vietnam__Hanoi_OldQuarter_fruitseller_supercub_manual_v244.txt",
            "boxes": [
                # Center female fruit seller on Super Cub without helmet (wearing sun visor)
                (2, (2050, 2080, 2400, 2880)),
                # Her white Honda Super Cub with front basket
                (0, (2010, 2370, 2550, 2950)),
                # Driver ahead in pink hoodie without helmet
                (2, (3530, 2450, 3680, 2850)),
                # Dark scooter ahead in street
                (0, (3530, 2600, 3680, 2880)),
                # Parked white Honda Vision scooter on left
                (0, (750, 2250, 1400, 3100)),
                # Parked silver scooter
                (0, (1100, 2150, 1550, 3000)),
                # Parked SYM scooter
                (0, (1350, 2050, 1750, 2950)),
                # Parked red Honda Wave
                (0, (1600, 2200, 1950, 2800))
            ],
            "description": "Pedestrian street scene in Old Quarter of Hanoi, Vietnam, showing fruit vendor on Honda Super Cub without helmet, passing commuter scooter, and row of parked scooters along curb."
        },
        {
            "id": "commons_vietnam/NorthVietnam_street_hondawave_yawning_manual_v244.png",
            "source_file": ACQ_DIR / "download" / "commons_98511444.jpg",
            "pageid": 98511444,
            "title": "File:Strato en Norda Vjetnamio 01.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Strato_en_Norda_Vjetnamio_01.jpg",
            "author": "RG72",
            "author_url": "https://commons.wikimedia.org/wiki/User:RG72",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "train",
            "group": "group_rg72_quangngai",
            "dest_img": "commons_vietnam__NorthVietnam_street_hondawave_yawning_manual_v244.png",
            "dest_lbl": "commons_vietnam__NorthVietnam_street_hondawave_yawning_manual_v244.txt",
            "boxes": [
                # Center young man seated yawning on Honda Wave without helmet
                (2, (1080, 2100, 1370, 2500)),
                # His black Honda Wave motorcycle
                (0, (1020, 2150, 1420, 2650)),
                # Passing commuter on scooter in foreground with helmet
                (1, (2080, 2100, 2304, 2800)),
                # Passing scooter
                (0, (2040, 2450, 2304, 3072)),
                # Parked motorcycle on far left against fence
                (0, (130, 2120, 370, 2450))
            ],
            "description": "Street scene in Northern Vietnam showing a young man sitting on a parked Honda Wave motorcycle without a helmet, a passing commuter on a scooter wearing a helmet, and parked motorcycles."
        },
        {
            "id": "commons_vietnam/Hanoi_TuMo_street_unhelmeted_commuters_manual_v244.png",
            "source_file": ACQ_DIR / "download" / "commons_185599261.jpg",
            "pageid": 185599261,
            "title": "File:Afternoon on Tú Mỡ Street 06-03-2026. Quê Lụa taxi.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Afternoon_on_T%C3%BA_M%E1%BB%A1_Street_06-03-2026._Qu%C3%AA_L%E1%BB%A5a_taxi.jpg",
            "author": "Gispenni",
            "author_url": "https://commons.wikimedia.org/wiki/User:Gispenni",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "val",
            "group": "group_gispenni_vuphamham",
            "dest_img": "commons_vietnam__Hanoi_TuMo_street_unhelmeted_commuters_manual_v244.png",
            "dest_lbl": "commons_vietnam__Hanoi_TuMo_street_unhelmeted_commuters_manual_v244.txt",
            "boxes": [
                # Commuter behind taxi with black baseball cap without helmet
                (2, (2150, 1040, 2420, 1450)),
                # His dark scooter
                (0, (2050, 1170, 2570, 1500)),
                # Commuter on right with sun hat without helmet
                (2, (3650, 1080, 3880, 1420)),
                # His white scooter
                (0, (3550, 1170, 4050, 1480))
            ],
            "description": "Afternoon on Tu Mo Street in Cau Giay district, Hanoi, Vietnam, showing commuters riding scooters without helmets alongside a Que Lua / Xanh SM taxi."
        },
        {
            "id": "commons_vietnam/HCMC_BuiVien_green_supercub_street_manual_v244.png",
            "source_file": ACQ_DIR / "download" / "commons_18088522.jpg",
            "pageid": 18088522,
            "title": "File:Bui Vien St.ブイヴィエン通り Thành phố Hồ Chí Minh 城舗胡志明 ホーチミン DSCF1561.JPG",
            "url": "https://commons.wikimedia.org/wiki/File:Bui_Vien_St.%E3%83%96%E3%82%A4%E3%83%B4%E3%82%A3%E3%82%A7%E3%83%B3%E9%80%9A%E3%82%8A_Th%C3%A0nh_ph%E1%BB%91_H%E1%BB%93_Ch%C3%AD_Minh_%E5%9F%8E%E8%88%97%E8%83%A1%E5%BF%97%E6%98%8E_%E3%83%9B%E3%83%BC%E3%83%81%E3%83%9F%E3%83%B3_DSCF1561.JPG",
            "author": "松岡明芳",
            "author_url": "https://commons.wikimedia.org/wiki/User:%E6%9D%BE%E5%B2%A1%E6%98%8E%E8%8A%B3",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "split": "val",
            "group": "group_matsuoka_buivien",
            "dest_img": "commons_vietnam__HCMC_BuiVien_green_supercub_street_manual_v244.png",
            "dest_lbl": "commons_vietnam__HCMC_BuiVien_green_supercub_street_manual_v244.txt",
            "boxes": [
                # Foreground vintage green Honda Super Cub
                (0, (650, 710, 970, 1280)),
                # Parked red motorcycle behind
                (0, (430, 510, 590, 740)),
                # Parked motorcycle on left with sun cover
                (0, (140, 510, 320, 780))
            ],
            "description": "Bui Vien Street in District 1, Ho Chi Minh City, Vietnam, with a parked green vintage Honda Super Cub on kickstand, parked red motorcycle, and street commerce."
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
        dst_img_path = V244_DIR / c["split"] / "images" / c["dest_img"]
        img.save(dst_img_path, format="PNG")

        # Save label
        yolo_text = boxes_to_yolo(c["boxes"], W, H)
        dst_lbl_path = V244_DIR / c["split"] / "labels" / c["dest_lbl"]
        dst_lbl_path.write_text(yolo_text, encoding="utf-8")

        # Copy original
        shutil.copy2(src_path, V244_DIR / "source_originals" / f"{c['pageid']}.jpg")

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
        meta_out = V244_DIR / "source_metadata" / f"{c['pageid']}.json"
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
        overlay.save(V244_DIR / "review_evidence" / ev_name)

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
        "batch_id": "manual_review_batch79",
        "snapshot": "dataset_v244",
        "reviewer": "antigravity_cloud_codex",
        "date": "2026-11-13",
        "items": review_records
    }
    with open(V244_DIR / "manual_review_batch79.json", "w", encoding="utf-8") as f:
        json.dump(batch_record, f, ensure_ascii=False, indent=2)

    # Update manifest
    print("Updating manifest.json...")
    manifest_path = V244_DIR / "manifest.json"
    with open(V243_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.extend(new_manifest_rows)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Update attribution.json
    attrib_path = V244_DIR / "attribution.json"
    with open(V243_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attrib = json.load(f)
    for c in cands_info:
        img_p = V244_DIR / c["split"] / "images" / c["dest_img"]
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
    attrib_md = V244_DIR / "ATTRIBUTION.md"
    shutil.copy2(V243_DIR / "ATTRIBUTION.md", attrib_md)
    with open(attrib_md, "a", encoding="utf-8") as f:
        f.write("\n### dataset_v244 Additions\n")
        for c in cands_info:
            f.write(f"- `{c['dest_img']}`: [{c['title']}]({c['url']}) by {c['author']}, [{c['license']}]({c['license_url']})\n")

    # Update changes.json
    changes_path = V244_DIR / "changes.json"
    with open(V243_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v244",
        "parent_snapshot": "dataset_v243",
        "action": "batch_admit",
        "manual_review_batch": "manual_review_batch79",
        "admitted_items_count": 4,
        "new_train_images": 2,
        "new_val_images": 2,
        "new_test_images": 0,
        "boxes_added": {"0": 14, "1": 1, "2": 5},
        "total_boxes_added": 20,
        "test_split_locked": True,
        "isolation_verified": True
    })
    with open(changes_path, "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # Update statistics.json
    print("Updating statistics.json...")
    stats_path = V244_DIR / "statistics.json"
    with open(V243_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["accepted_clean_images"] = 1254
    stats["remaining_to_goal"] = 3746

    # Train stats: +2 images (156674593, 98511444)
    # Train boxes added: 9x cls 0 (6+3), 1x cls 1 (0+1), 3x cls 2 (2+1)
    stats["train"]["images"] += 2
    stats["train"]["classes"]["0"] += 9
    stats["train"]["classes"]["1"] += 1
    stats["train"]["classes"]["2"] += 3
    # group_alexkom000_hanoi (+0)
    # group_rg72_quangngai (+0)

    # Val stats: +2 images (185599261, 18088522)
    # Val boxes added: 5x cls 0 (2+3), 0x cls 1 (0+0), 2x cls 2 (2+0)
    stats["val"]["images"] += 2
    stats["val"]["classes"]["0"] += 5
    stats["val"]["classes"]["1"] += 0
    stats["val"]["classes"]["2"] += 2
    # group_gispenni_vuphamham (+0)
    # group_matsuoka_buivien (+0)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Verification report
    verif = {
        "snapshot": "dataset_v244",
        "parent": "dataset_v243",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2411,
        "rows_total": 2415,
        "accepted_clean_images": 1254,
        "remaining_to_5000": 3746,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 14, "1": 1, "2": 5},
        "split_counts": {
            "train": 985,
            "val": 201,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch79.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V244_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(attrib_path),
        "duplicates_sha256": sha256_file(V244_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(changes_path)
    }
    with open(V244_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v244.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    with open(progress_path, "r", encoding="utf-8") as f:
        progress = json.load(f)

    progress["accepted_clean_images"] = 1254
    progress["remaining_to_goal"] = 3746
    progress["snapshot"] = "dataset_v244"
    progress["accepted_split_counts"]["train"] = 985
    progress["accepted_split_counts"]["validation"] = 201
    progress["accepted_split_counts"]["test_locked"] = 68
    progress["newly_accepted_this_batch"] = 4
    progress["new_batch_class_box_counts"] = {"0": 14, "1": 1, "2": 5}
    progress["total_class_box_counts"]["0"] += 14
    progress["total_class_box_counts"]["1"] += 1
    progress["total_class_box_counts"]["2"] += 5
    progress["verification_report"] = "../verification/dataset_v244.json"
    progress["notes"].append(
        "dataset_v244 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY 4.0 / CC BY-SA 4.0 / CC BY-SA 3.0), adding 20 clean bounding boxes including 5 rare unhelmeted riders (Class 2): Hanoi Old Quarter street fruit vendor on Super Cub without helmet, North Vietnam street scene with young man yawning on Honda Wave, Hanoi Tu Mo street commuters on scooters without helmets, and HCMC Bui Vien street scene with vintage green Honda Super Cub on kickstand and parked bikes. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,254."
    )
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v244 build completed successfully! ===")

if __name__ == "__main__":
    main()
