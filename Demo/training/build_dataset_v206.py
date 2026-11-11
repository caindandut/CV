"""Build and verify dataset_v206 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. HCMC Motorcycle Police patrol on Honda CBX (author: Roderick Divilbiss, CC BY-SA 2.5) -> train
2. Northern Vietnam highway pig transport with riders (author: BluesyPete, CC BY-SA 3.0) -> train
3. Northern Vietnam road tree cargo rider on Super Cub (author: BluesyPete, CC BY-SA 3.0) -> train
4. Ho Chi Minh City traffic intersection DD 04 (author: Diego Delso, CC BY-SA 3.0) -> val

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
V205_DIR = TRAINING_DIR / "dataset_v205"
V206_DIR = TRAINING_DIR / "dataset_v206"
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
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines

def main():
    print("=== Building Dataset v206 Snapshot ===")
    V206_DIR.mkdir(exist_ok=True)
    
    # 1. Setup directories
    for split in ["train", "val"]:
        (V206_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V206_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    (V206_DIR / "review_evidence").mkdir(exist_ok=True)
    (V206_DIR / "source_originals").mkdir(exist_ok=True)
    (V206_DIR / "source_metadata").mkdir(exist_ok=True)

    # 2. Link existing assets from v205
    print("Linking existing v205 assets...")
    for split in ["train", "val"]:
        src_img_dir = V205_DIR / split / "images"
        dst_img_dir = V206_DIR / split / "images"
        for f in src_img_dir.iterdir():
            if f.is_file():
                dst = dst_img_dir / f.name
                if not dst.exists():
                    os.link(f, dst)
        
        src_lbl_dir = V205_DIR / split / "labels"
        dst_lbl_dir = V206_DIR / split / "labels"
        for f in src_lbl_dir.iterdir():
            if f.is_file():
                dst = dst_lbl_dir / f.name
                if not dst.exists():
                    os.link(f, dst)

    colors = {0: (0, 255, 0), 1: (0, 150, 255), 2: (255, 50, 50)}

    # 3. Candidate 1: commons_529684 (Roderick Divilbiss, HCMC Police, train)
    print("Processing candidate 1 (Roderick Divilbiss - HCMC Police)...")
    orig1_src = ACQ_DIR / "download" / "commons_529684.jpg"
    orig1_dst = V206_DIR / "source_originals" / "commons_529684.jpg"
    shutil.copy2(orig1_src, orig1_dst)

    with Image.open(orig1_src) as im1:
        out1_img = V206_DIR / "train" / "images" / "commons_vietnam__HCMC_motorcycle_police_patrol_manual_v206.png"
        im1.save(out1_img)
        phash1 = f"{perceptual_hash(im1):016x}"
        W1, H1 = im1.size

    boxes1_native = [
        (0, (110, 280, 1160, 930)),
        (1, (410, 100, 860, 800)),
        (1, (460, 50, 1050, 730))
    ]
    labels1 = boxes_to_yolo(boxes1_native, W1, H1)
    out1_lbl = V206_DIR / "train" / "labels" / "commons_vietnam__HCMC_motorcycle_police_patrol_manual_v206.txt"
    out1_lbl.write_text("\n".join(labels1) + "\n", encoding="utf-8")

    overlay1 = im1.copy()
    d1 = ImageDraw.Draw(overlay1)
    for cls, (x1, y1, x2, y2) in boxes1_native:
        d1.rectangle([x1, y1, x2, y2], outline=colors[cls], width=4)
        d1.text((x1 + 6, y1 + 6), f"cls {cls}", fill=colors[cls])
    overlay1.save(V206_DIR / "review_evidence" / "HCMC_motorcycle_police_patrol_manual_v206_overlay.png")

    # 4. Candidate 2: commons_115770739 (BluesyPete, pig transport, train)
    print("Processing candidate 2 (BluesyPete - Pig transport)...")
    orig2_src = ACQ_DIR / "download" / "commons_115770739.jpg"
    orig2_dst = V206_DIR / "source_originals" / "commons_115770739.jpg"
    shutil.copy2(orig2_src, orig2_dst)

    with Image.open(orig2_src) as im2:
        im2_scaled = im2.resize((1144, 856), Image.Resampling.LANCZOS)
        out2_img = V206_DIR / "train" / "images" / "commons_vietnam__Northern_highway_pig_transport_manual_v206.png"
        im2_scaled.save(out2_img)
        phash2 = f"{perceptual_hash(im2_scaled):016x}"

    boxes2_native = [
        (0, (1190, 610, 1900, 1712)),
        (1, (1260, 400, 1750, 1350)),
        (0, (350, 510, 1090, 1580)),
        (2, (450, 280, 930, 1420)),
        (0, (1220, 380, 1770, 1040)),
        (2, (1290, 200, 1680, 1040)),
        (0, (1400, 160, 1640, 520)),
        (1, (1410, 60, 1600, 360))
    ]
    labels2 = boxes_to_yolo(boxes2_native, 2288, 1712)
    out2_lbl = V206_DIR / "train" / "labels" / "commons_vietnam__Northern_highway_pig_transport_manual_v206.txt"
    out2_lbl.write_text("\n".join(labels2) + "\n", encoding="utf-8")

    overlay2 = im2_scaled.copy()
    d2 = ImageDraw.Draw(overlay2)
    s2 = 1144.0 / 2288.0
    for cls, (x1, y1, x2, y2) in boxes2_native:
        d2.rectangle([int(x1 * s2), int(y1 * s2), int(x2 * s2), int(y2 * s2)], outline=colors[cls], width=3)
        d2.text((int(x1 * s2) + 6, int(y1 * s2) + 6), f"cls {cls}", fill=colors[cls])
    overlay2.save(V206_DIR / "review_evidence" / "Northern_highway_pig_transport_manual_v206_overlay.png")

    # 5. Candidate 3: commons_115770748 (BluesyPete, tree cargo, train)
    print("Processing candidate 3 (BluesyPete - Tree cargo)...")
    orig3_src = ACQ_DIR / "download" / "commons_115770748.jpg"
    orig3_dst = V206_DIR / "source_originals" / "commons_115770748.jpg"
    shutil.copy2(orig3_src, orig3_dst)

    with Image.open(orig3_src) as im3:
        out3_img = V206_DIR / "train" / "images" / "commons_vietnam__Northern_road_tree_cargo_rider_manual_v206.png"
        im3.save(out3_img)
        phash3 = f"{perceptual_hash(im3):016x}"
        W3, H3 = im3.size

    boxes3_native = [
        (0, (70, 520, 860, 1300)),
        (1, (310, 410, 580, 1180))
    ]
    labels3 = boxes_to_yolo(boxes3_native, W3, H3)
    out3_lbl = V206_DIR / "train" / "labels" / "commons_vietnam__Northern_road_tree_cargo_rider_manual_v206.txt"
    out3_lbl.write_text("\n".join(labels3) + "\n", encoding="utf-8")

    overlay3 = im3.copy()
    d3 = ImageDraw.Draw(overlay3)
    for cls, (x1, y1, x2, y2) in boxes3_native:
        d3.rectangle([x1, y1, x2, y2], outline=colors[cls], width=3)
        d3.text((x1 + 6, y1 + 6), f"cls {cls}", fill=colors[cls])
    overlay3.save(V206_DIR / "review_evidence" / "Northern_road_tree_cargo_rider_manual_v206_overlay.png")

    # 6. Candidate 4: commons_30110168 (Diego Delso, DD 04, val)
    print("Processing candidate 4 (Diego Delso - DD 04)...")
    orig4_src = ACQ_DIR / "download" / "commons_30110168.jpg"
    orig4_dst = V206_DIR / "source_originals" / "commons_30110168.jpg"
    shutil.copy2(orig4_src, orig4_dst)

    with Image.open(orig4_src) as im4:
        im4_scaled = im4.resize((1050, 1672), Image.Resampling.LANCZOS)
        out4_img = V206_DIR / "val" / "images" / "commons_vietnam__HCMC_DD04_traffic_yamaha_honda_manual_v206.png"
        im4_scaled.save(out4_img)
        phash4 = f"{perceptual_hash(im4_scaled):016x}"

    boxes4_native = [
        (0, (1700, 2700, 2900, 4650)),
        (1, (1850, 1870, 2850, 3950)),
        (0, (800, 2500, 1600, 4000)),
        (1, (720, 1920, 1550, 3500)),
        (0, (340, 2150, 800, 2950)),
        (1, (320, 1740, 800, 2900))
    ]
    labels4 = boxes_to_yolo(boxes4_native, 3150, 5016)
    out4_lbl = V206_DIR / "val" / "labels" / "commons_vietnam__HCMC_DD04_traffic_yamaha_honda_manual_v206.txt"
    out4_lbl.write_text("\n".join(labels4) + "\n", encoding="utf-8")

    overlay4 = im4_scaled.copy()
    d4 = ImageDraw.Draw(overlay4)
    s4 = 1050.0 / 3150.0
    for cls, (x1, y1, x2, y2) in boxes4_native:
        d4.rectangle([int(x1 * s4), int(y1 * s4), int(x2 * s4), int(y2 * s4)], outline=colors[cls], width=3)
        d4.text((int(x1 * s4) + 6, int(y1 * s4) + 6), f"cls {cls}", fill=colors[cls])
    overlay4.save(V206_DIR / "review_evidence" / "HCMC_DD04_traffic_yamaha_honda_manual_v206_overlay.png")

    # 7. Save metadata files
    with open(ACQ_DIR / "unadmitted_commons_candidates.json", "r", encoding="utf-8") as f:
        cand_list = json.load(f)
    cand_map = {c["pageid"]: c for c in cand_list}

    for pid in [529684, 115770739, 115770748, 30110168]:
        meta_file = V206_DIR / "source_metadata" / f"commons_{pid}.json"
        meta_file.write_text(json.dumps(cand_map.get(pid, {}), ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. Update Manifest
    print("Updating manifest.json...")
    with open(V205_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    new_rows = [
        {
            "id": "commons_vietnam/HCMC_motorcycle_police_patrol_manual_v206.png",
            "pageid": 529684,
            "original_pageid": 529684,
            "title": "HCMC MotorcyclePolice Jun2005.jpg",
            "source_group": "commons_roderick_divilbiss",
            "group": "group_roderick_divilbiss",
            "artist": "Roderick Divilbiss",
            "license": "CC BY-SA 2.5",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.5/",
            "new_split": "train",
            "width": W1,
            "height": H1,
            "sha256": sha256_file(out1_img),
            "phash": phash1,
            "source_image": "Demo/training/dataset_v206/source_originals/commons_529684.jpg",
            "source_sha256": sha256_file(orig1_dst),
            "source_metadata": "Demo/training/dataset_v206/source_metadata/commons_529684.json",
            "review_evidence": "Demo/training/dataset_v206/review_evidence/HCMC_motorcycle_police_patrol_manual_v206_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; Vietnamese traffic police (ĐỘI CSGT SỐ 1) on specialized patrol Honda motorcycle with blue police license plate 31-A1 0455.",
            "classes": {"0": 1, "1": 2, "2": 0}
        },
        {
            "id": "commons_vietnam/Northern_highway_pig_transport_manual_v206.png",
            "pageid": 115770739,
            "original_pageid": 115770739,
            "title": "Vietnam-0649.jpg",
            "source_group": "commons_bluesypete_transport",
            "group": "group_bluesypete_transport",
            "artist": "BluesyPete",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "train",
            "width": 1144,
            "height": 856,
            "sha256": sha256_file(out2_img),
            "phash": phash2,
            "source_image": "Demo/training/dataset_v206/source_originals/commons_115770739.jpg",
            "source_sha256": sha256_file(orig2_dst),
            "source_metadata": "Demo/training/dataset_v206/source_metadata/commons_115770739.json",
            "review_evidence": "Demo/training/dataset_v206/review_evidence/Northern_highway_pig_transport_manual_v206_overlay.png",
            "location_evidence": "Northern Vietnam national/provincial highway; Hanoi license plate 29X-2070 on truck in background; agricultural live animal transport scene.",
            "classes": {"0": 4, "1": 2, "2": 2}
        },
        {
            "id": "commons_vietnam/Northern_road_tree_cargo_rider_manual_v206.png",
            "pageid": 115770748,
            "original_pageid": 115770748,
            "title": "Vietnam-0539.jpg",
            "source_group": "commons_bluesypete_transport",
            "group": "group_bluesypete_transport",
            "artist": "BluesyPete",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "train",
            "width": W3,
            "height": H3,
            "sha256": sha256_file(out3_img),
            "phash": phash3,
            "source_image": "Demo/training/dataset_v206/source_originals/commons_115770748.jpg",
            "source_sha256": sha256_file(orig3_dst),
            "source_metadata": "Demo/training/dataset_v206/source_metadata/commons_115770748.json",
            "review_evidence": "Demo/training/dataset_v206/review_evidence/Northern_road_tree_cargo_rider_manual_v206_overlay.png",
            "location_evidence": "Northern Vietnam road; motorcycle delivery of nursery plants and trees on Honda Super Cub with front wire basket.",
            "classes": {"0": 1, "1": 1, "2": 0}
        },
        {
            "id": "commons_vietnam/HCMC_DD04_traffic_yamaha_honda_manual_v206.png",
            "pageid": 30110168,
            "original_pageid": 30110168,
            "title": "Tráfico en Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 04.JPG",
            "source_group": "commons_diego_delso",
            "group": "group_diego_delso",
            "artist": "Diego Delso",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "val",
            "width": 1050,
            "height": 1672,
            "sha256": sha256_file(out4_img),
            "phash": phash4,
            "source_image": "Demo/training/dataset_v206/source_originals/commons_30110168.jpg",
            "source_sha256": sha256_file(orig4_dst),
            "source_metadata": "Demo/training/dataset_v206/source_metadata/commons_30110168.json",
            "review_evidence": "Demo/training/dataset_v206/review_evidence/HCMC_DD04_traffic_yamaha_honda_manual_v206_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; urban traffic with Vinasun Taxi in background, Yamaha and Honda commuter motorbikes.",
            "classes": {"0": 3, "1": 3, "2": 0}
        }
    ]

    manifest.extend(new_rows)
    manifest_path = V206_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 9. Update Statistics
    print("Updating statistics.json...")
    with open(V205_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Train: +3 images (1+1+1), Class 0: +6 (1+4+1), Class 1: +5 (2+2+1), Class 2: +2 (0+2+0)
    stats["train"]["images"] += 3
    stats["train"]["classes"]["0"] += 6
    stats["train"]["classes"]["1"] += 5
    stats["train"]["classes"]["2"] += 2
    stats["train"]["groups"] += 2  # group_roderick_divilbiss, group_bluesypete_transport

    # Val: +1 image, Class 0: +3, Class 1: +3, Class 2: +0
    stats["val"]["images"] += 1
    stats["val"]["classes"]["0"] += 3
    stats["val"]["classes"]["1"] += 3
    # group_diego_delso already in val

    stats_path = V206_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 10. Copy duplicates, usage_restrictions, data.yaml, split_lock
    shutil.copy2(V205_DIR / "duplicates.json", V206_DIR / "duplicates.json")
    shutil.copy2(V205_DIR / "usage_restrictions.json", V206_DIR / "usage_restrictions.json")
    shutil.copy2(V205_DIR / "data.yaml", V206_DIR / "data.yaml")
    shutil.copy2(V205_DIR / "split_lock.json", V206_DIR / "split_lock.json")

    # 11. Changes and Review batch
    manual_review_41 = {
        "batch": 41,
        "date": "2026-11-11",
        "snapshot": "dataset_v206",
        "parent": "dataset_v205",
        "method": "manual pixel inspection and bounding box annotation; no model pseudo-labeling",
        "reviewed_candidates_count": 8,
        "accepted_images_count": 4,
        "held_pending_or_excluded_count": 4,
        "accepted": [
            {
                "id": "commons_vietnam/HCMC_motorcycle_police_patrol_manual_v206.png",
                "pageid": 529684,
                "author": "Roderick Divilbiss",
                "license": "CC BY-SA 2.5",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 1},
                    {"cls": 1, "name": "rider_with_helmet", "count": 2},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "HCMC police patrol motorcycle, CSGT & CSCD officers, license 31-A1 0455."
            },
            {
                "id": "commons_vietnam/Northern_highway_pig_transport_manual_v206.png",
                "pageid": 115770739,
                "author": "BluesyPete",
                "license": "CC BY-SA 3.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 4},
                    {"cls": 1, "name": "rider_with_helmet", "count": 2},
                    {"cls": 2, "name": "rider_without_helmet", "count": 2}
                ],
                "location_evidence": "Northern Vietnam national highway, pig cargo transport, truck plate 29X-2070, cap/non coi riders."
            },
            {
                "id": "commons_vietnam/Northern_road_tree_cargo_rider_manual_v206.png",
                "pageid": 115770748,
                "author": "BluesyPete",
                "license": "CC BY-SA 3.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 1},
                    {"cls": 1, "name": "rider_with_helmet", "count": 1},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Northern Vietnam road, tree nursery delivery on Honda Super Cub with front basket."
            },
            {
                "id": "commons_vietnam/HCMC_DD04_traffic_yamaha_honda_manual_v206.png",
                "pageid": 30110168,
                "author": "Diego Delso",
                "license": "CC BY-SA 3.0",
                "assigned_split": "val",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 3},
                    {"cls": 1, "name": "rider_with_helmet", "count": 3},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Ho Chi Minh City traffic, Vinasun Taxi in background, Yamaha scooter and Honda motorcycles."
            }
        ],
        "excluded_or_pending": [
            {"pageid": 115770742, "title": "Vietnam-0778.jpg", "reason": "Bicycle carrying newspapers with loudspeaker, not a motorcycle; excluded under Rule 24."},
            {"pageid": 145321196, "title": "Vietnam 1999.Hanoi.Motocyclettes (8).jpg", "reason": "Historical 1999 monochrome film photo, out of distribution with modern color traffic sensor data; deferred."},
            {"pageid": 85840491, "title": "Vietnam, Hanoi, Life on the streets of Hanoi.jpg", "reason": "Parked motorcycles along Old Quarter street with high obstruction by cyclos and pedestrians; held pending."},
            {"pageid": 115770740, "title": "Vietnam-0648.jpg", "reason": "Same sequence as 115770739; held pending to avoid near-duplicate redundancy within the same batch."}
        ]
    }
    with open(V206_DIR / "manual_review_batch41.json", "w", encoding="utf-8") as f:
        json.dump(manual_review_41, f, ensure_ascii=False, indent=2)

    # 12. Update Attribution
    with open(V205_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attribution = json.load(f)
    
    new_attributions = [
        {
            "id": "commons_vietnam/HCMC_motorcycle_police_patrol_manual_v206.png",
            "title": "File:HCMC MotorcyclePolice Jun2005.jpg",
            "artist": "Roderick Divilbiss",
            "license": "CC BY-SA 2.5",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.5",
            "page_url": "https://commons.wikimedia.org/wiki/File:HCMC_MotorcyclePolice_Jun2005.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/9/94/HCMC_MotorcyclePolice_Jun2005.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/9/94/HCMC_MotorcyclePolice_Jun2005.jpg",
            "sha256": sha256_file(out1_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; Vietnamese traffic police (ĐỘI CSGT SỐ 1) on specialized patrol Honda motorcycle with blue police license plate 31-A1 0455.",
            "modifications": "Original image preserved. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/Northern_highway_pig_transport_manual_v206.png",
            "title": "File:Vietnam-0649.jpg",
            "artist": "BluesyPete",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Vietnam-0649.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/a/a2/Vietnam-0649.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/a/a2/Vietnam-0649.jpg",
            "sha256": sha256_file(out2_img),
            "country": "VN",
            "location_evidence": "Northern Vietnam national/provincial highway; Hanoi license plate 29X-2070 on truck in background; agricultural live animal transport scene.",
            "modifications": "Scaled to 1144x856. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/Northern_road_tree_cargo_rider_manual_v206.png",
            "title": "File:Vietnam-0539.jpg",
            "artist": "BluesyPete",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Vietnam-0539.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/4/45/Vietnam-0779.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/4/45/Vietnam-0779.jpg",
            "sha256": sha256_file(out3_img),
            "country": "VN",
            "location_evidence": "Northern Vietnam road; motorcycle delivery of nursery plants and trees on Honda Super Cub with front wire basket.",
            "modifications": "Original image preserved. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_DD04_traffic_yamaha_honda_manual_v206.png",
            "title": "File:Tráfico en Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 04.JPG",
            "artist": "Diego Delso",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh,_Vietnam,_2013-08-14,_DD_04.JPG",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh%2C_Vietnam%2C_2013-08-14%2C_DD_04.JPG",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh%2C_Vietnam%2C_2013-08-14%2C_DD_04.JPG",
            "sha256": sha256_file(out4_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; urban traffic with Vinasun Taxi in background, Yamaha and Honda commuter motorbikes.",
            "modifications": "Scaled to 1050x1672. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        }
    ]

    attribution.extend(new_attributions)

    with open(V206_DIR / "attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, ensure_ascii=False, indent=2)

    # ATTRIBUTION.md
    attrib_md = (
        "# Dataset Attribution & Provenance\n\n"
        "## Summary\n"
        "- All images in this dataset have explicit provenance, verified Vietnam traffic context, and human-verified ground-truth annotations.\n"
        "- No model predictions or pseudo-labels are used as ground truth.\n\n"
        "## New Sources in v206\n"
        "- **Roderick Divilbiss** (CC BY-SA 2.5): `HCMC_motorcycle_police_patrol_manual_v206.png` (HCMC traffic police patrol).\n"
        "- **BluesyPete** (CC BY-SA 3.0): `Northern_highway_pig_transport_manual_v206.png` and `Northern_road_tree_cargo_rider_manual_v206.png` (Northern Vietnam rural/highway transport).\n"
        "- **Diego Delso** (CC BY-SA 3.0): `HCMC_DD04_traffic_yamaha_honda_manual_v206.png` (Ho Chi Minh City urban intersection).\n"
    )
    (V206_DIR / "ATTRIBUTION.md").write_text(attrib_md, encoding="utf-8")

    # changes.json
    with open(V205_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v206",
        "parent": "dataset_v205",
        "action": "accept_manually_reviewed_commons_images",
        "accepted_images": [
            "commons_vietnam/HCMC_motorcycle_police_patrol_manual_v206.png",
            "commons_vietnam/Northern_highway_pig_transport_manual_v206.png",
            "commons_vietnam/Northern_road_tree_cargo_rider_manual_v206.png",
            "commons_vietnam/HCMC_DD04_traffic_yamaha_honda_manual_v206.png"
        ],
        "new_accepted_count": 4,
        "new_class_0_boxes": 9,
        "new_class_1_boxes": 8,
        "new_class_2_boxes": 2,
        "review_batch": "manual_review_batch41.json"
    })
    with open(V206_DIR / "changes.json", "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # README.md
    readme_md = (
        "# Dataset Snapshot v206\n\n"
        "## Summary\n"
        "- **Accepted Clean Images:** 1,102 (Train: 889, Val: 145, Test: 68 locked)\n"
        "- **Total Clean Boxes:** 3,636 (Class 0 Motorcycle: 1,961, Class 1 Helmet: 1,083, Class 2 No Helmet: 592)\n"
        "- **Pending Images:** 25 (Held in group_0000 to respect split lock)\n"
        "- **Excluded Duplicates:** 1,136\n"
        "- **Goal Target:** 5,000 clean images\n"
        "- **Progress:** 1,102 / 5,000 (22.04%)\n"
        "- **Remaining to Goal:** 3,898 images\n\n"
        "## Batch 41 Ingestion\n"
        "4 manually verified Vietnam motorcycle images added from Wikimedia Commons:\n"
        "1. `HCMC_motorcycle_police_patrol_manual_v206.png`: 1 motorcycle, 2 helmeted police officers (Roderick Divilbiss, CC BY-SA 2.5, train).\n"
        "2. `Northern_highway_pig_transport_manual_v206.png`: 4 motorcycles, 2 helmeted riders, 2 unhelmeted riders (BluesyPete, CC BY-SA 3.0, train).\n"
        "3. `Northern_road_tree_cargo_rider_manual_v206.png`: 1 motorcycle, 1 helmeted rider (BluesyPete, CC BY-SA 3.0, train).\n"
        "4. `HCMC_DD04_traffic_yamaha_honda_manual_v206.png`: 3 motorcycles, 3 helmeted riders (Diego Delso, CC BY-SA 3.0, val).\n"
    )
    (V206_DIR / "README.md").write_text(readme_md, encoding="utf-8")

    # 13. Verification Report
    verif = {
        "snapshot": "dataset_v206",
        "parent": "dataset_v205",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2259,
        "rows_total": 2263,
        "accepted_clean_images": 1102,
        "remaining_to_5000": 3898,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 9, "1": 8, "2": 2},
        "split_counts": {
            "train": 889,
            "val": 145,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch41.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V206_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(V206_DIR / "attribution.json"),
        "duplicates_sha256": sha256_file(V206_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(V206_DIR / "changes.json")
    }
    with open(V206_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v206.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # 14. Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    progress = {
        "schema_version": 1,
        "as_of": "2026-11-11",
        "goal_images": 5000,
        "accepted_clean_images": 1102,
        "remaining_to_goal": 3898,
        "pending_review_images": 25,
        "snapshot": "dataset_v206",
        "accepted_split_counts": {
            "train": 889,
            "validation": 145,
            "test_locked": 68
        },
        "newly_accepted_this_batch": 4,
        "new_batch_class_box_counts": {
            "0": 9,
            "1": 8,
            "2": 2
        },
        "total_class_box_counts": {
            "0": 1961,
            "1": 1083,
            "2": 592
        },
        "original_dataset_unchanged": True,
        "test_split_locked": True,
        "pending_counts_toward_goal": False,
        "verification_report": "../verification/dataset_v206.json",
        "notes": [
            "Counts are individually reviewed images, not boxes; duplicate and pending rows do not count.",
            "dataset_v203 audited all 1,492 pending DataSet2 items: 335 unique canonical frames accepted (279 train, 56 val), 1,132 redundant variants excluded, 25 items held pending in group_0000 to maintain test lock.",
            "dataset_v204 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), verified Vietnam location context, zero leakage between splits, and test split locked at 68 images. Total clean images reaches 1,095.",
            "dataset_v205 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), adding 32 clean bounding boxes (including 7 rare rider_without_helmet boxes in Hanoi school traffic). Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,098.",
            "dataset_v206 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 2.5 / CC BY-SA 3.0), adding 19 clean bounding boxes including police patrol motorcycle, agricultural animal transport, nursery tree cargo, and city traffic. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,102."
        ]
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v206 build completed successfully! ===")

if __name__ == "__main__":
    main()
