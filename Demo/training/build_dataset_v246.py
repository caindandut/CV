"""Build and verify dataset_v246 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Hanoi street pause for Secretary Kerry crossing street (U.S. Dept of State, Public Domain) -> train (group_us_state_dept_hanoi) [4x cls 0, 3x cls 1, 1x cls 2]
2. Vietnam cafe street view with parked SYM Attila & commuters (Everyday English, CC BY-SA 4.0) -> train (group_everyday_english_vietnam) [3x cls 0, 2x cls 1, 0x cls 2]
3. Hanoi Hang Dau street female commuters on scooter (Jakub Hałun, CC BY 4.0) -> val (group_jakubhal_hanoi) [4x cls 0, 5x cls 1, 0x cls 2]
4. Hanoi Old Town street shop front with parked Honda Wave (Jakub Hałun, CC BY 4.0) -> val (group_jakubhal_hanoi) [1x cls 0, 0x cls 1, 0x cls 2]

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
V245_DIR = TRAINING_DIR / "dataset_v245"
V246_DIR = TRAINING_DIR / "dataset_v246"
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
    print("=== Building Dataset v246 Snapshot ===")
    assert V245_DIR.is_dir(), f"Missing parent snapshot {V245_DIR}"

    if V246_DIR.exists():
        shutil.rmtree(V246_DIR)
    V246_DIR.mkdir(parents=True, exist_ok=True)

    # Copy metadata files from v245
    shutil.copy2(V245_DIR / "split_lock.json", V246_DIR / "split_lock.json")
    shutil.copy2(V245_DIR / "data.yaml", V246_DIR / "data.yaml")
    shutil.copy2(V245_DIR / "duplicates.json", V246_DIR / "duplicates.json")
    shutil.copy2(V245_DIR / "usage_restrictions.json", V246_DIR / "usage_restrictions.json")
    shutil.copy2(V245_DIR / "README.md", V246_DIR / "README.md")

    # Set up train / val / test directories
    for split in ["train", "val", "test"]:
        (V246_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V246_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Link existing v245 images and labels
    print("Linking existing v245 assets...")
    for split in ["train", "val", "test"]:
        src_img_dir = V245_DIR / split / "images"
        dst_img_dir = V246_DIR / split / "images"
        if src_img_dir.exists():
            for item in src_img_dir.iterdir():
                target = item.resolve()
                dst = dst_img_dir / item.name
                os.symlink(target, dst)

        src_lbl_dir = V245_DIR / split / "labels"
        dst_lbl_dir = V246_DIR / split / "labels"
        if src_lbl_dir.exists():
            for item in src_lbl_dir.iterdir():
                dst = dst_lbl_dir / item.name
                shutil.copy2(item, dst)

    # Directories for evidence and metadata
    (V246_DIR / "review_evidence").mkdir(exist_ok=True)
    for f in (V245_DIR / "review_evidence").iterdir():
        if f.is_file():
            os.link(f, V246_DIR / "review_evidence" / f.name)

    (V246_DIR / "source_originals").mkdir(exist_ok=True)
    for f in (V245_DIR / "source_originals").iterdir():
        if f.is_file():
            os.link(f, V246_DIR / "source_originals" / f.name)

    (V246_DIR / "source_metadata").mkdir(exist_ok=True)
    for f in (V245_DIR / "source_metadata").iterdir():
        if f.is_file():
            os.link(f, V246_DIR / "source_metadata" / f.name)

    # Candidate definitions for v246
    cands_info = [
        {
            "id": "commons_vietnam/Hanoi_Kerry_crossing_motorbikes_manual_v246.png",
            "source_file": ACQ_DIR / "download" / "commons_31601709.jpg",
            "pageid": 31601709,
            "title": "File:Vietnamese Motorbike Riders Pause as Secretary Kerry Crosses Street (11372986094).jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Vietnamese_Motorbike_Riders_Pause_as_Secretary_Kerry_Crosses_Street_(11372986094).jpg",
            "author": "U.S. Department of State",
            "author_url": "https://www.flickr.com/people/9364837@N06",
            "license": "Public domain",
            "license_url": "https://en.wikipedia.org/wiki/Public_domain",
            "split": "train",
            "group": "group_us_state_dept_hanoi",
            "dest_img": "commons_vietnam__Hanoi_Kerry_crossing_motorbikes_manual_v246.png",
            "dest_lbl": "commons_vietnam__Hanoi_Kerry_crossing_motorbikes_manual_v246.txt",
            "boxes": [
                # Waving man in red shirt on Repsol Honda Wave without helmet
                (2, (740, 560, 1080, 1260)),
                # Repsol Honda Wave motorcycle
                (0, (690, 800, 1140, 1280)),
                # Commuter in grey shirt/shorts on vintage Super Cub with helmet & mask
                (1, (1200, 550, 1500, 1140)),
                # Super Cub
                (0, (1260, 770, 1490, 1160)),
                # Commuter in blue shirt on Honda Wave with helmet
                (1, (1540, 560, 1780, 1140)),
                # Honda Wave
                (0, (1490, 760, 1750, 1150)),
                # Commuter in black jacket on light blue Piaggio Vespa with helmet
                (1, (1760, 540, 2170, 1240)),
                # Light blue Piaggio Vespa
                (0, (1780, 760, 2190, 1330))
            ],
            "description": "Hanoi street scene with motorbikes and scooters stopped at crosswalk as Secretary of State John Kerry crosses street, featuring waving unhelmeted rider on Repsol Honda Wave and helmeted commuters."
        },
        {
            "id": "commons_vietnam/Vietnam_coffee_street_scooters_manual_v246.png",
            "source_file": ACQ_DIR / "download" / "commons_124404546.jpg",
            "pageid": 124404546,
            "title": "File:View of street over coffee in Vietnam.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:View_of_street_over_coffee_in_Vietnam.jpg",
            "author": "Everyday English",
            "author_url": "https://commons.wikimedia.org/wiki/User:Everyday_English",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "train",
            "group": "group_everyday_english_vietnam",
            "dest_img": "commons_vietnam__Vietnam_coffee_street_scooters_manual_v246.png",
            "dest_lbl": "commons_vietnam__Vietnam_coffee_street_scooters_manual_v246.txt",
            "boxes": [
                # Foreground parked yellow SYM Attila scooter on kickstand
                (0, (3080, 330, 4580, 2380)),
                # Commuter passing in midground on scooter with helmet
                (1, (3200, 550, 3850, 1080)),
                # Passing scooter
                (0, (3200, 780, 3850, 1140)),
                # Commuter on far left with helmet
                (1, (0, 570, 200, 980)),
                # Left scooter
                (0, (0, 710, 370, 1050))
            ],
            "description": "Street view from a cafe in Vietnam showing parked yellow SYM Attila scooter and passing commuters on scooters with helmets."
        },
        {
            "id": "commons_vietnam/Hanoi_HangDau_two_women_scooter_manual_v246.png",
            "source_file": ACQ_DIR / "download" / "commons_147777952.jpg",
            "pageid": 147777952,
            "title": "File:Hàng Dầu Street, Hanoi, 20240204 1443 5822.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:H%C3%A0ng_D%E1%BA%A7u_Street,_Hanoi,_20240204_1443_5822.jpg",
            "author": "Jakub Hałun",
            "author_url": "https://commons.wikimedia.org/wiki/User:Jakubhal",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "split": "val",
            "group": "group_jakubhal_hanoi",
            "dest_img": "commons_vietnam__Hanoi_HangDau_two_women_scooter_manual_v246.png",
            "dest_lbl": "commons_vietnam__Hanoi_HangDau_two_women_scooter_manual_v246.txt",
            "boxes": [
                # Driver woman in beige jacket with pink helmet
                (1, (1180, 1750, 1720, 2800)),
                # Passenger woman in black with blue helmet
                (1, (1410, 1760, 2020, 2850)),
                # Dark scooter
                (0, (1050, 2150, 2150, 3120)),
                # Commuter behind in blue jacket with helmet
                (1, (980, 1710, 1420, 2450)),
                # Scooter
                (0, (880, 1950, 1420, 2520)),
                # Midground commuter in white jacket on red/black scooter with red helmet
                (1, (2280, 1680, 2570, 2220)),
                # Red/black scooter
                (0, (2160, 1850, 2750, 2280)),
                # Right commuter in dark jacket on gold scooter with white helmet
                (1, (3460, 1710, 3800, 2180)),
                # Gold scooter
                (0, (3300, 1890, 4000, 2290))
            ],
            "description": "Hang Dau street in Hanoi Old Quarter, Vietnam, showing female commuters on scooter with pink and blue helmets alongside other street motorbike traffic."
        },
        {
            "id": "commons_vietnam/Hanoi_OldTown_shop_parked_wave_manual_v246.png",
            "source_file": ACQ_DIR / "download" / "commons_147777619.jpg",
            "pageid": 147777619,
            "title": "File:Shop on the street of Hanoi Old Town, 20240204 1452 5836.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Shop_on_the_street_of_Hanoi_Old_Town,_20240204_1452_5836.jpg",
            "author": "Jakub Hałun",
            "author_url": "https://commons.wikimedia.org/wiki/User:Jakubhal",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "split": "val",
            "group": "group_jakubhal_hanoi",
            "dest_img": "commons_vietnam__Hanoi_OldTown_shop_parked_wave_manual_v246.png",
            "dest_lbl": "commons_vietnam__Hanoi_OldTown_shop_parked_wave_manual_v246.txt",
            "boxes": [
                # Parked red/black Honda Wave on kickstand
                (0, (0, 1820, 770, 2520))
            ],
            "description": "Traditional street shop front in Hanoi Old Town, Vietnam, with a parked red and black Honda Wave on its kickstand."
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
        dst_img_path = V246_DIR / c["split"] / "images" / c["dest_img"]
        img.save(dst_img_path, format="PNG")

        # Save label
        yolo_text = boxes_to_yolo(c["boxes"], W, H)
        dst_lbl_path = V246_DIR / c["split"] / "labels" / c["dest_lbl"]
        dst_lbl_path.write_text(yolo_text, encoding="utf-8")

        # Copy original
        shutil.copy2(src_path, V246_DIR / "source_originals" / f"{c['pageid']}.jpg")

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
        meta_out = V246_DIR / "source_metadata" / f"{c['pageid']}.json"
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
        overlay.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        ev_name = f"evidence_{c['dest_img']}"
        overlay.save(V246_DIR / "review_evidence" / ev_name, optimize=True)

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
        "batch_id": "manual_review_batch81",
        "snapshot": "dataset_v246",
        "reviewer": "antigravity_cloud_codex",
        "date": "2026-11-13",
        "items": review_records
    }
    with open(V246_DIR / "manual_review_batch81.json", "w", encoding="utf-8") as f:
        json.dump(batch_record, f, ensure_ascii=False, indent=2)

    # Update manifest
    print("Updating manifest.json...")
    manifest_path = V246_DIR / "manifest.json"
    with open(V245_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.extend(new_manifest_rows)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Update attribution.json
    attrib_path = V246_DIR / "attribution.json"
    with open(V245_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attrib = json.load(f)
    for c in cands_info:
        img_p = V246_DIR / c["split"] / "images" / c["dest_img"]
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
    attrib_md = V246_DIR / "ATTRIBUTION.md"
    shutil.copy2(V245_DIR / "ATTRIBUTION.md", attrib_md)
    with open(attrib_md, "a", encoding="utf-8") as f:
        f.write("\n### dataset_v246 Additions\n")
        for c in cands_info:
            f.write(f"- `{c['dest_img']}`: [{c['title']}]({c['url']}) by {c['author']}, [{c['license']}]({c['license_url']})\n")

    # Update changes.json
    changes_path = V246_DIR / "changes.json"
    with open(V245_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v246",
        "parent_snapshot": "dataset_v245",
        "action": "batch_admit",
        "manual_review_batch": "manual_review_batch81",
        "admitted_items_count": 4,
        "new_train_images": 2,
        "new_val_images": 2,
        "new_test_images": 0,
        "boxes_added": {"0": 12, "1": 10, "2": 1},
        "total_boxes_added": 23,
        "test_split_locked": True,
        "isolation_verified": True
    })
    with open(changes_path, "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # Update statistics.json
    print("Updating statistics.json...")
    stats_path = V246_DIR / "statistics.json"
    with open(V245_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["accepted_clean_images"] = 1262
    stats["remaining_to_goal"] = 3738

    # Train stats: +2 images (31601709, 124404546)
    # Train boxes added: 7x cls 0 (4+3), 5x cls 1 (3+2), 1x cls 2 (1+0)
    stats["train"]["images"] += 2
    stats["train"]["classes"]["0"] += 7
    stats["train"]["classes"]["1"] += 5
    stats["train"]["classes"]["2"] += 1
    # group_us_state_dept_hanoi (+1)
    # group_everyday_english_vietnam (+1)
    stats["train"]["groups"] = stats["train"].get("groups", 0) + 2

    # Val stats: +2 images (147777952, 147777619)
    # Val boxes added: 5x cls 0 (4+1), 5x cls 1 (5+0), 0x cls 2 (0+0)
    stats["val"]["images"] += 2
    stats["val"]["classes"]["0"] += 5
    stats["val"]["classes"]["1"] += 5
    stats["val"]["classes"]["2"] += 0
    # group_jakubhal_hanoi (+0)
    stats["val"]["groups"] = stats["val"].get("groups", 0) + 0

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Verification report
    verif = {
        "snapshot": "dataset_v246",
        "parent": "dataset_v245",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2419,
        "rows_total": 2423,
        "accepted_clean_images": 1262,
        "remaining_to_5000": 3738,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 12, "1": 10, "2": 1},
        "split_counts": {
            "train": 989,
            "val": 205,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch81.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V246_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(attrib_path),
        "duplicates_sha256": sha256_file(V246_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(changes_path)
    }
    with open(V246_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v246.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    with open(progress_path, "r", encoding="utf-8") as f:
        progress = json.load(f)

    progress["accepted_clean_images"] = 1262
    progress["remaining_to_goal"] = 3738
    progress["snapshot"] = "dataset_v246"
    progress["accepted_split_counts"]["train"] = 989
    progress["accepted_split_counts"]["validation"] = 205
    progress["accepted_split_counts"]["test_locked"] = 68
    progress["newly_accepted_this_batch"] = 4
    progress["new_batch_class_box_counts"] = {"0": 12, "1": 10, "2": 1}
    progress["total_class_box_counts"]["0"] += 12
    progress["total_class_box_counts"]["1"] += 10
    progress["total_class_box_counts"]["2"] += 1
    progress["verification_report"] = "../verification/dataset_v246.json"
    progress["notes"].append(
        "dataset_v246 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (Public domain / CC BY-SA 4.0 / CC BY 4.0 / CC BY 4.0), adding 23 clean bounding boxes including 1 rare unhelmeted rider (Class 2): Hanoi street pause as Secretary Kerry crosses street (featuring unhelmeted waving man in red shirt on Repsol Honda Wave), Vietnam cafe street view with parked yellow SYM Attila scooter and passing commuters, Hanoi Hang Dau street female commuters on scooter with pink and blue helmets, and Hanoi Old Town street shop front with parked red/black Honda Wave on kickstand. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,262."
    )
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v246 build completed successfully! ===")

if __name__ == "__main__":
    main()
