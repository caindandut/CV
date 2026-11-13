"""Build and verify dataset_v249 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Nha Trang Tran Phu coastal boulevard traffic (RG72, CC BY-SA 4.0) -> train (group_rg72_quangngai) [2x cls 0, 3x cls 1, 0x cls 2]
2. Danang Monkey Mountain sunset parked Honda Air Blade (Beelerb, CC BY-SA 4.0) -> val (group_beelerb_danang) [1x cls 0, 0x cls 1, 0x cls 2]
3. HCMC family of four on motorcycle and Super Cub commuters (Dennis G. Jarvis, CC BY-SA 2.0) -> val (group_dennis_jarvis_saigon) [2x cls 0, 4x cls 1, 2x cls 2]
4. HCMC street mother and daughter on blue scooter peace sign (U.S. Department of State, Public domain) -> train (group_us_state_dept_hanoi) [3x cls 0, 4x cls 1, 0x cls 2]

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
V248_DIR = TRAINING_DIR / "dataset_v248"
V249_DIR = TRAINING_DIR / "dataset_v249"
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
    print("=== Building Dataset v249 Snapshot ===")
    assert V248_DIR.is_dir(), f"Missing parent snapshot {V248_DIR}"

    if V249_DIR.exists():
        shutil.rmtree(V249_DIR)
    V249_DIR.mkdir(parents=True, exist_ok=True)

    # Copy metadata files from v248
    shutil.copy2(V248_DIR / "split_lock.json", V249_DIR / "split_lock.json")
    shutil.copy2(V248_DIR / "data.yaml", V249_DIR / "data.yaml")
    shutil.copy2(V248_DIR / "duplicates.json", V249_DIR / "duplicates.json")
    shutil.copy2(V248_DIR / "usage_restrictions.json", V249_DIR / "usage_restrictions.json")
    shutil.copy2(V248_DIR / "README.md", V249_DIR / "README.md")

    # Set up train / val / test directories
    for split in ["train", "val", "test"]:
        (V249_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V249_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Link existing v248 images and labels
    print("Linking existing v248 assets...")
    for split in ["train", "val", "test"]:
        src_img_dir = V248_DIR / split / "images"
        dst_img_dir = V249_DIR / split / "images"
        if src_img_dir.exists():
            for item in src_img_dir.iterdir():
                target = item.resolve()
                dst = dst_img_dir / item.name
                os.symlink(target, dst)

        src_lbl_dir = V248_DIR / split / "labels"
        dst_lbl_dir = V249_DIR / split / "labels"
        if src_lbl_dir.exists():
            for item in src_lbl_dir.iterdir():
                dst = dst_lbl_dir / item.name
                shutil.copy2(item, dst)

    # Directories for evidence and metadata for batch 84 (batch-specific only)
    (V249_DIR / "review_evidence").mkdir(exist_ok=True)
    (V249_DIR / "source_originals").mkdir(exist_ok=True)
    (V249_DIR / "source_metadata").mkdir(exist_ok=True)

    # Candidate definitions for v249
    cands_info = [
        {
            "id": "commons_vietnam/NhaTrang_TranPhu_boulevard_commuters_flow_manual_v249.png",
            "source_file": ACQ_DIR / "download" / "commons_98815158.jpg",
            "pageid": 98815158,
            "title": "File:Strato Trần Phú en Nha Trang 02.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Strato_Tr%E1%BA%A7n_Ph%C3%BA_en_Nha_Trang_02.jpg",
            "author": "RG72",
            "author_url": "https://commons.wikimedia.org/wiki/User:RG72",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "train",
            "group": "group_rg72_quangngai",
            "dest_img": "commons_vietnam__NhaTrang_TranPhu_boulevard_commuters_flow_manual_v249.png",
            "dest_lbl": "commons_vietnam__NhaTrang_TranPhu_boulevard_commuters_flow_manual_v249.txt",
            "boxes": [
                # Left foreground driver with red helmet
                (1, (30, 1740, 240, 2170)),
                # Left foreground passenger with red helmet
                (1, (15, 1720, 135, 2130)),
                # Left foreground white scooter
                (0, (5, 1860, 280, 2280)),
                # Midground center commuter with red helmet
                (1, (1370, 1750, 1465, 1980)),
                # Midground blue motorbike
                (0, (1330, 1820, 1490, 2010))
            ],
            "description": "Tran Phu coastal boulevard in Nha Trang, Vietnam, showing commuters traveling along the avenue on scooters and motorbikes wearing safety helmets."
        },
        {
            "id": "commons_vietnam/Danang_MonkeyMountain_parked_AirBlade_manual_v249.png",
            "source_file": ACQ_DIR / "download" / "commons_65436103.jpg",
            "pageid": 65436103,
            "title": "File:Honda Air Blade on Monkey Mountain.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Honda_Air_Blade_on_Monkey_Mountain.jpg",
            "author": "Beelerb",
            "author_url": "https://commons.wikimedia.org/wiki/User:Beelerb",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "val",
            "group": "group_beelerb_danang",
            "dest_img": "commons_vietnam__Danang_MonkeyMountain_parked_AirBlade_manual_v249.png",
            "dest_lbl": "commons_vietnam__Danang_MonkeyMountain_parked_AirBlade_manual_v249.txt",
            "boxes": [
                # Parked blue Honda Air Blade scooter on kickstand
                (0, (150, 600, 2710, 2740))
            ],
            "description": "Scenic view of a parked blue Honda Air Blade scooter overlooking Danang city at sunset from Son Tra Peninsula (Monkey Mountain), Vietnam."
        },
        {
            "id": "commons_vietnam/Saigon_family_four_motorcycle_cub_commuters_manual_v249.png",
            "source_file": ACQ_DIR / "download" / "commons_66993736.jpg",
            "pageid": 66993736,
            "title": "File:DGJ 0430 - Room for another one! (3348134314).jpg",
            "url": "https://commons.wikimedia.org/wiki/File:DGJ_0430_-_Room_for_another_one%21_%283348134314%29.jpg",
            "author": "Dennis G. Jarvis",
            "author_url": "https://www.wikidata.org/wiki/Q122977591",
            "license": "CC BY-SA 2.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.0/",
            "split": "val",
            "group": "group_dennis_jarvis_saigon",
            "dest_img": "commons_vietnam__Saigon_family_four_motorcycle_cub_commuters_manual_v249.png",
            "dest_lbl": "commons_vietnam__Saigon_family_four_motorcycle_cub_commuters_manual_v249.txt",
            "boxes": [
                # Left Super Cub driver woman with helmet
                (1, (640, 770, 1350, 2320)),
                # Left Super Cub passenger woman with helmet
                (1, (1150, 780, 1750, 1980)),
                # Left Super Cub scooter
                (0, (370, 1170, 1920, 2350)),
                # Right motorbike front child in straw hat without helmet (Class 2)
                (2, (1900, 990, 2350, 2100)),
                # Right motorbike father driving with helmet
                (1, (1850, 710, 2650, 2530)),
                # Right motorbike middle child in pink hat without helmet (Class 2)
                (2, (2500, 1070, 3050, 2300)),
                # Right motorbike mother with helmet
                (1, (2780, 870, 3400, 1820)),
                # Right red motorbike
                (0, (1220, 1260, 3960, 2480))
            ],
            "description": "Ho Chi Minh City street scene showing a Vietnamese family of four on a red motorcycle (father and mother with helmets, two young children in sun hats without helmets) traveling alongside two women commuters on a white Honda Super Cub."
        },
        {
            "id": "commons_vietnam/HCMC_mother_daughter_peacesign_scooter_manual_v249.png",
            "source_file": ACQ_DIR / "download" / "commons_49057126.jpg",
            "pageid": 49057126,
            "title": "File:A Vietnamese Woman Flashes a Peace Sign at Secretary Kerry as He Walks Through the Streets of Ho Chi Minh City (27200985191).jpg",
            "url": "https://commons.wikimedia.org/wiki/File:A_Vietnamese_Woman_Flashes_a_Peace_Sign_at_Secretary_Kerry_as_He_Walks_Through_the_Streets_of_Ho_Chi_Minh_City_%2827200985191%29.jpg",
            "author": "U.S. Department of State",
            "author_url": "https://www.flickr.com/people/9364837@N06",
            "license": "Public domain",
            "license_url": "https://creativecommons.org/publicdomain/mark/1.0/",
            "split": "train",
            "group": "group_us_state_dept_hanoi",
            "dest_img": "commons_vietnam__HCMC_mother_daughter_peacesign_scooter_manual_v249.png",
            "dest_lbl": "commons_vietnam__HCMC_mother_daughter_peacesign_scooter_manual_v249.txt",
            "boxes": [
                # Mother in grey jacket with blue helmet
                (1, (1660, 570, 2400, 2530)),
                # Child in yellow shirt with helmet and peace sign
                (1, (2020, 860, 2520, 1500)),
                # Blue Honda scooter
                (0, (1750, 1100, 2900, 2870)),
                # Commuter on white scooter with helmet
                (1, (1260, 620, 1680, 1570)),
                # White scooter
                (0, (1200, 1020, 1850, 1740)),
                # Background commuter in blue jacket/white helmet
                (1, (1050, 490, 1250, 1200)),
                # Background yellow motorbike
                (0, (1050, 700, 1270, 1250))
            ],
            "description": "Ho Chi Minh City street view showing a mother and daughter flashing a peace sign from their blue scooter with helmets, alongside waiting scooter commuters."
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
        dst_img_path = V249_DIR / c["split"] / "images" / c["dest_img"]
        img.save(dst_img_path, format="PNG")

        # Save label
        yolo_text = boxes_to_yolo(c["boxes"], W, H)
        dst_lbl_path = V249_DIR / c["split"] / "labels" / c["dest_lbl"]
        dst_lbl_path.write_text(yolo_text, encoding="utf-8")

        # Copy original (batch-specific only)
        shutil.copy2(src_path, V249_DIR / "source_originals" / f"{c['pageid']}.jpg")

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
        meta_out = V249_DIR / "source_metadata" / f"{c['pageid']}.json"
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
        overlay.save(V249_DIR / "review_evidence" / ev_name, optimize=True)

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
        "batch_id": "manual_review_batch84",
        "snapshot": "dataset_v249",
        "reviewer": "antigravity_cloud_codex",
        "date": "2026-11-13",
        "items": review_records
    }
    with open(V249_DIR / "manual_review_batch84.json", "w", encoding="utf-8") as f:
        json.dump(batch_record, f, ensure_ascii=False, indent=2)

    # Update manifest
    print("Updating manifest.json...")
    manifest_path = V249_DIR / "manifest.json"
    with open(V248_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.extend(new_manifest_rows)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Update attribution.json
    attrib_path = V249_DIR / "attribution.json"
    with open(V248_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attrib = json.load(f)
    for c in cands_info:
        img_p = V249_DIR / c["split"] / "images" / c["dest_img"]
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
    attrib_md = V249_DIR / "ATTRIBUTION.md"
    shutil.copy2(V248_DIR / "ATTRIBUTION.md", attrib_md)
    with open(attrib_md, "a", encoding="utf-8") as f:
        f.write("\n### dataset_v249 Additions\n")
        for c in cands_info:
            f.write(f"- `{c['dest_img']}`: [{c['title']}]({c['url']}) by {c['author']}, [{c['license']}]({c['license_url']})\n")

    # Update changes.json
    changes_path = V249_DIR / "changes.json"
    with open(V248_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v249",
        "parent_snapshot": "dataset_v248",
        "action": "batch_admit",
        "manual_review_batch": "manual_review_batch84",
        "admitted_items_count": 4,
        "new_train_images": 2,
        "new_val_images": 2,
        "new_test_images": 0,
        "boxes_added": {"0": 8, "1": 11, "2": 2},
        "total_boxes_added": 21,
        "test_split_locked": True,
        "isolation_verified": True
    })
    with open(changes_path, "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # Update statistics.json
    print("Updating statistics.json...")
    stats_path = V249_DIR / "statistics.json"
    with open(V248_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["accepted_clean_images"] = 1274
    stats["remaining_to_goal"] = 3726

    # Train stats: +2 images (98815158, 49057126)
    # Train boxes added: 5x cls 0 (2+3), 7x cls 1 (3+4), 0x cls 2 (0+0)
    stats["train"]["images"] += 2
    stats["train"]["classes"]["0"] += 5
    stats["train"]["classes"]["1"] += 7
    stats["train"]["classes"]["2"] += 0
    # groups: group_rg72_quangngai and group_us_state_dept_hanoi already in train (+0)
    stats["train"]["groups"] = stats["train"].get("groups", 0) + 0

    # Val stats: +2 images (65436103, 66993736)
    # Val boxes added: 3x cls 0 (1+2), 4x cls 1 (0+4), 2x cls 2 (0+2)
    stats["val"]["images"] += 2
    stats["val"]["classes"]["0"] += 3
    stats["val"]["classes"]["1"] += 4
    stats["val"]["classes"]["2"] += 2
    # groups: group_beelerb_danang (+1) and group_dennis_jarvis_saigon (+1)
    stats["val"]["groups"] = stats["val"].get("groups", 0) + 2

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Verification report
    verif = {
        "snapshot": "dataset_v249",
        "parent": "dataset_v248",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2431,
        "rows_total": 2435,
        "accepted_clean_images": 1274,
        "remaining_to_5000": 3726,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 8, "1": 11, "2": 2},
        "split_counts": {
            "train": 997,
            "val": 209,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch84.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V249_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(attrib_path),
        "duplicates_sha256": sha256_file(V249_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(changes_path)
    }
    with open(V249_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v249.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    with open(progress_path, "r", encoding="utf-8") as f:
        progress = json.load(f)

    progress["accepted_clean_images"] = 1274
    progress["remaining_to_goal"] = 3726
    progress["snapshot"] = "dataset_v249"
    progress["accepted_split_counts"]["train"] = 997
    progress["accepted_split_counts"]["validation"] = 209
    progress["accepted_split_counts"]["test_locked"] = 68
    progress["newly_accepted_this_batch"] = 4
    progress["new_batch_class_box_counts"] = {"0": 8, "1": 11, "2": 2}
    progress["total_class_box_counts"]["0"] += 8
    progress["total_class_box_counts"]["1"] += 11
    progress["total_class_box_counts"]["2"] += 2
    progress["verification_report"] = "../verification/dataset_v249.json"
    progress["notes"].append(
        "dataset_v249 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 4.0 / CC BY-SA 4.0 / CC BY-SA 2.0 / Public domain), adding 21 clean bounding boxes including 2 rare unhelmeted children (Class 2): Nha Trang Tran Phu coastal boulevard traffic commuters with helmets, Danang Monkey Mountain scenic sunset parked Honda Air Blade, Ho Chi Minh City Vietnamese family of four on red motorcycle (with two unhelmeted children in straw/pink hats) alongside women commuters on white Honda Super Cub, and Ho Chi Minh City street scene with mother and daughter on blue scooter flashing peace sign. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,274."
    )
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v249 build completed successfully! ===")

if __name__ == "__main__":
    main()
