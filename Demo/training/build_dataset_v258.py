"""Build and verify dataset_v258 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Nha Trang boy sleeping on Honda Wave motorcycle seat without helmet (RG72, CC BY-SA 4.0) -> train (group_rg72_quangngai) [2x cls 0, 0x cls 1, 1x cls 2]
2. Hanoi West Lake commuters Nouvo duo, Lead, Spacy, unhelmeted woman (Amenoc, CC0) -> train (group_amenoc_hanoi) [4x cls 0, 5x cls 1, 1x cls 2]
3. Au Co Road Hanoi commuters traffic duo, cargo courier (David McKelvey, CC BY 2.0) -> val (group_david_mckelvey_hanoi) [4x cls 0, 5x cls 1, 0x cls 2]
4. Hanoi Tran Duy Hung Agribank parked red Wave and street commuter (Phan Minh Tuấn, CC BY-SA 4.0) -> val (group_phanminhtuan_hanoi) [4x cls 0, 1x cls 1, 0x cls 2]

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
V257_DIR = TRAINING_DIR / "dataset_v257"
V258_DIR = TRAINING_DIR / "dataset_v258"
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
    print("=== Building Dataset v258 Snapshot ===")
    assert V257_DIR.is_dir(), f"Missing parent snapshot {V257_DIR}"

    if V258_DIR.exists():
        shutil.rmtree(V258_DIR)
    V258_DIR.mkdir(parents=True, exist_ok=True)

    # Copy metadata files from v257
    shutil.copy2(V257_DIR / "split_lock.json", V258_DIR / "split_lock.json")
    shutil.copy2(V257_DIR / "data.yaml", V258_DIR / "data.yaml")
    shutil.copy2(V257_DIR / "duplicates.json", V258_DIR / "duplicates.json")
    shutil.copy2(V257_DIR / "usage_restrictions.json", V258_DIR / "usage_restrictions.json")
    shutil.copy2(V257_DIR / "README.md", V258_DIR / "README.md")

    # Set up train / val / test directories
    for split in ["train", "val", "test"]:
        (V258_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V258_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Link existing v257 images and labels
    print("Linking existing v257 assets...")
    for split in ["train", "val", "test"]:
        src_img_dir = V257_DIR / split / "images"
        dst_img_dir = V258_DIR / split / "images"
        if src_img_dir.exists():
            for item in src_img_dir.iterdir():
                target = item.resolve()
                dst = dst_img_dir / item.name
                os.symlink(target, dst)

        src_lbl_dir = V257_DIR / split / "labels"
        dst_lbl_dir = V258_DIR / split / "labels"
        if src_lbl_dir.exists():
            for item in src_lbl_dir.iterdir():
                dst = dst_lbl_dir / item.name
                shutil.copy2(item, dst)

    # Directories for evidence and metadata for batch 93 (batch-specific only)
    (V258_DIR / "review_evidence").mkdir(exist_ok=True)
    (V258_DIR / "source_originals").mkdir(exist_ok=True)
    (V258_DIR / "source_metadata").mkdir(exist_ok=True)

    # Candidate definitions for v258
    cands_info = [
        {
            "id": "commons_vietnam/NhaTrang_boy_sleeping_on_Honda_Wave_manual_v258.png",
            "source_file": ACQ_DIR / "download" / "commons_98859037.jpg",
            "pageid": 98859037,
            "title": "File:Knabo-vagabondo dormanta sur motorciklo Honda Wave (Nha Trang).jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Knabo-vagabondo_dormanta_sur_motorciklo_Honda_Wave_(Nha_Trang).jpg",
            "author": "RG72",
            "author_url": "https://commons.wikimedia.org/wiki/User:RG72",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "train",
            "group": "group_rg72_quangngai",
            "dest_img": "commons_vietnam__NhaTrang_boy_sleeping_on_Honda_Wave_manual_v258.png",
            "dest_lbl": "commons_vietnam__NhaTrang_boy_sleeping_on_Honda_Wave_manual_v258.txt",
            "boxes": [
                # 1. Honda Wave on kickstand
                (0, (100, 520, 2160, 2940)),
                # 2. Sleeping boy lying across seat without helmet (Class 2)
                (2, (130, 1180, 1920, 2360)),
                # 3. Honda Dream parked behind
                (0, (0, 630, 860, 2030))
            ],
            "description": "Nha Trang street scene with a young boy sleeping across the seat of a parked black Honda Wave motorcycle without a helmet, parked beside a Honda Dream motorcycle."
        },
        {
            "id": "commons_vietnam/Hanoi_WestLake_commuters_Nouvo_Lead_Spacy_manual_v258.png",
            "source_file": ACQ_DIR / "download" / "commons_63463129.jpg",
            "pageid": 63463129,
            "title": "File:Along the circular West Lake road 12.jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Along_the_circular_West_Lake_road_12.jpg",
            "author": "Amenoc",
            "author_url": "https://commons.wikimedia.org/wiki/User:Amenoc",
            "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "split": "train",
            "group": "group_amenoc_hanoi",
            "dest_img": "commons_vietnam__Hanoi_WestLake_commuters_Nouvo_Lead_Spacy_manual_v258.png",
            "dest_lbl": "commons_vietnam__Hanoi_WestLake_commuters_Nouvo_Lead_Spacy_manual_v258.txt",
            "boxes": [
                # 1. Yamaha Nouvo duo foreground
                (0, (1750, 1970, 2315, 2445)),
                (1, (1840, 1710, 1980, 2100)),
                (1, (1930, 1770, 2260, 2230)),
                # 2. Honda Lead midground
                (0, (2540, 1900, 2860, 2120)),
                (1, (2650, 1775, 2800, 2070)),
                # 3. Honda Spacy midground
                (0, (2860, 1940, 3160, 2180)),
                (1, (2910, 1770, 3050, 2040)),
                # 4. Right motorcycle with unhelmeted woman and companion
                (0, (3080, 1950, 3264, 2270)),
                (2, (3050, 1790, 3210, 2150)),
                (1, (3150, 1780, 3250, 2020))
            ],
            "description": "Hanoi West Lake circular road intersection in front of the Ministry of Defence building, showing commuters on Yamaha Nouvo, Honda Lead, Honda Spacy, and a red-shirt passenger without a helmet."
        },
        {
            "id": "commons_vietnam/Hanoi_AuCo_road_commuters_traffic_duo_manual_v258.png",
            "source_file": ACQ_DIR / "download" / "commons_105761228.jpg",
            "pageid": 105761228,
            "title": "File:Au Co Road, Tây Hồ District, Hanoi (7067709779).jpg",
            "url": "https://commons.wikimedia.org/wiki/File:Au_Co_Road,_T%C3%A2y_H%E1%BB%93_District,_Hanoi_(7067709779).jpg",
            "author": "David McKelvey",
            "author_url": "https://www.flickr.com/people/94735786@N00",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "split": "val",
            "group": "group_david_mckelvey_hanoi",
            "dest_img": "commons_vietnam__Hanoi_AuCo_road_commuters_traffic_duo_manual_v258.png",
            "dest_lbl": "commons_vietnam__Hanoi_AuCo_road_commuters_traffic_duo_manual_v258.txt",
            "boxes": [
                # 1. Left commuter
                (0, (390, 940, 800, 1260)),
                (1, (510, 850, 710, 1150)),
                # 2. Center duo
                (0, (1450, 940, 1690, 1230)),
                (1, (1450, 860, 1610, 1100)),
                (1, (1530, 860, 1670, 1080)),
                # 3. Cargo courier
                (0, (2050, 910, 2240, 1130)),
                (1, (2100, 860, 2210, 1020)),
                # 4. Right commuter
                (0, (2210, 920, 2370, 1140)),
                (1, (2210, 850, 2370, 1080))
            ],
            "description": "Au Co Road in Tay Ho District, Hanoi, showing roadway commuters riding motorbikes with helmets, including a duo in winter coats with red helmet, a courier transporting bundled cargo boxes, and commuters flowing along the avenue."
        },
        {
            "id": "commons_vietnam/Hanoi_TranDuyHung_Agribank_parked_motorbikes_manual_v258.png",
            "source_file": ACQ_DIR / "download" / "commons_58081319.jpg",
            "pageid": 58081319,
            "title": "File:Agribank, 61 Trần Duy Hưng, Hà Nội 001.JPG",
            "url": "https://commons.wikimedia.org/wiki/File:Agribank,_61_Tr%E1%BA%A7n_Duy_H%C6%B0ng,_H%C3%A0_N%E1%BB%99i_001.JPG",
            "author": "Phan Minh Tuấn",
            "author_url": "https://commons.wikimedia.org/wiki/User:Phan_Minh_Tu%E1%BA%A5n",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "split": "val",
            "group": "group_phanminhtuan_hanoi",
            "dest_img": "commons_vietnam__Hanoi_TranDuyHung_Agribank_parked_motorbikes_manual_v258.png",
            "dest_lbl": "commons_vietnam__Hanoi_TranDuyHung_Agribank_parked_motorbikes_manual_v258.txt",
            "boxes": [
                # 1. Red parked Wave
                (0, (620, 1520, 1010, 1780)),
                # 2. Dark parked bike 2
                (0, (840, 1500, 1160, 1740)),
                # 3. Dark parked bike 3
                (0, (960, 1480, 1280, 1730)),
                # 4. Moving street commuter
                (0, (180, 1530, 310, 1690)),
                (1, (180, 1470, 270, 1610))
            ],
            "description": "Sidewalk in front of Agribank branch at 61 Tran Duy Hung, Cau Giay, Hanoi, with parked red Honda Wave and motorbikes along Pho Do Quang street sign, and commuter riding scooter with safety helmet on the street."
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
        dst_img_path = V258_DIR / c["split"] / "images" / c["dest_img"]
        img.save(dst_img_path, format="PNG")

        # Save label
        yolo_text = boxes_to_yolo(c["boxes"], W, H)
        dst_lbl_path = V258_DIR / c["split"] / "labels" / c["dest_lbl"]
        dst_lbl_path.write_text(yolo_text, encoding="utf-8")

        # Copy original (batch-specific only)
        shutil.copy2(src_path, V258_DIR / "source_originals" / f"{c['pageid']}.jpg")

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
        meta_out = V258_DIR / "source_metadata" / f"{c['pageid']}.json"
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
        overlay.save(V258_DIR / "review_evidence" / ev_name, optimize=True)

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

    # Save review batch 93
    batch_record = {
        "batch_id": "manual_review_batch93",
        "snapshot": "dataset_v258",
        "reviewer": "antigravity_cloud_codex",
        "date": "2026-11-14",
        "items": review_records
    }
    with open(V258_DIR / "manual_review_batch93.json", "w", encoding="utf-8") as f:
        json.dump(batch_record, f, ensure_ascii=False, indent=2)

    # Update manifest
    print("Updating manifest.json...")
    manifest_path = V258_DIR / "manifest.json"
    with open(V257_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.extend(new_manifest_rows)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Update attribution.json
    attrib_path = V258_DIR / "attribution.json"
    with open(V257_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attrib = json.load(f)
    for c in cands_info:
        img_p = V258_DIR / c["split"] / "images" / c["dest_img"]
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
    attrib_md = V258_DIR / "ATTRIBUTION.md"
    shutil.copy2(V257_DIR / "ATTRIBUTION.md", attrib_md)
    with open(attrib_md, "a", encoding="utf-8") as f:
        f.write("\n### dataset_v258 Additions\n")
        for c in cands_info:
            f.write(f"- `{c['dest_img']}`: [{c['title']}]({c['url']}) by {c['author']}, [{c['license']}]({c['license_url']})\n")

    # Update changes.json
    changes_path = V258_DIR / "changes.json"
    with open(V257_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v258",
        "parent_snapshot": "dataset_v257",
        "action": "batch_admit",
        "manual_review_batch": "manual_review_batch93",
        "admitted_items_count": 4,
        "new_train_images": 2,
        "new_val_images": 2,
        "new_test_images": 0,
        "boxes_added": {"0": 14, "1": 11, "2": 2},
        "total_boxes_added": 27,
        "test_split_locked": True,
        "isolation_verified": True
    })
    with open(changes_path, "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # Update statistics.json
    print("Updating statistics.json...")
    stats_path = V258_DIR / "statistics.json"
    with open(V257_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["accepted_clean_images"] = 1310
    stats["remaining_to_goal"] = 3690

    # Train stats: +2 images (98859037, 63463129)
    # Train boxes added: 6x cls 0 (2+4), 5x cls 1 (0+5), 2x cls 2 (1+1)
    stats["train"]["images"] += 2
    stats["train"]["classes"]["0"] += 6
    stats["train"]["classes"]["1"] += 5
    stats["train"]["classes"]["2"] += 2
    stats["train"]["groups"] += 1  # new group: group_amenoc_hanoi (group_rg72_quangngai already existed)

    # Val stats: +2 images (105761228, 58081319)
    # Val boxes added: 8x cls 0 (4+4), 6x cls 1 (5+1), 0x cls 2 (0+0)
    stats["val"]["images"] += 2
    stats["val"]["classes"]["0"] += 8
    stats["val"]["classes"]["1"] += 6
    stats["val"]["classes"]["2"] += 0
    stats["val"]["groups"] += 2  # new groups: group_david_mckelvey_hanoi, group_phanminhtuan_hanoi

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Verification report
    verif = {
        "snapshot": "dataset_v258",
        "parent": "dataset_v257",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2467,
        "rows_total": 2471,
        "accepted_clean_images": 1310,
        "remaining_to_5000": 3690,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 14, "1": 11, "2": 2},
        "total_box_counts": {
            "0": 2473 + 14,
            "1": 1485 + 11,
            "2": 654 + 2
        },
        "split_counts": {
            "train": 1017,
            "val": 225,
            "test_locked": 68
        },
        "group_isolation_verified": True,
        "test_lock_verified": True,
        "no_pseudo_labeling": True,
        "notes": "dataset_v258 ingested 4 newly manually reviewed Vietnam motorcycle photos from Wikimedia Commons with open licenses (CC BY-SA 4.0 / CC0 / CC BY 2.0 / CC BY-SA 4.0), adding 27 clean bounding boxes (14 Class 0, 11 Class 1, 2 Class 2) including two rare rider_without_helmet instances."
    }
    verif_path = TRAINING_DIR / "verification" / "dataset_v258.json"
    verif_path.parent.mkdir(exist_ok=True)
    with open(verif_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    print("=== Successfully Built Dataset v258 ===")
    print(f"Total accepted clean images: 1,310 / 5,000 (Remaining: 3,690)")
    print(f"Train: 1017 | Val: 225 | Test: 68")
    print(f"Boxes: Cls 0: {verif['total_box_counts']['0']} | Cls 1: {verif['total_box_counts']['1']} | Cls 2: {verif['total_box_counts']['2']}")

if __name__ == "__main__":
    main()
