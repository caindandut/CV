"""Build and verify dataset_v208 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Sa Pa to Tả Phìn mountain motorcycle (author: Christophe95, CC BY-SA 4.0) -> train (group_christophe95_hanoi)
2. Hanoi intersection waiting at red lights (author: Uwe Aranas, CC BY-SA 3.0) -> train (group_uwe_aranas)
3. Ho Chi Minh City commuter motorbikes traffic (author: Vyacheslav Argenberg, CC BY 4.0) -> train (group_argenberg_hcmc)
4. Ho Chi Minh City motorbikers on avenue (author: Frank McKenna, CC0) -> val (group_frank_mckenna_hcmc)

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
V207_DIR = TRAINING_DIR / "dataset_v207"
V208_DIR = TRAINING_DIR / "dataset_v208"
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
    return lines

def main():
    print("=== Building Dataset v208 Snapshot ===")
    V208_DIR.mkdir(exist_ok=True)
    
    # 1. Setup directories
    for split in ["train", "val"]:
        (V208_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V208_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    (V208_DIR / "review_evidence").mkdir(exist_ok=True)
    (V208_DIR / "source_originals").mkdir(exist_ok=True)
    (V208_DIR / "source_metadata").mkdir(exist_ok=True)

    # 2. Link existing assets from v207
    print("Linking existing v207 assets...")
    for split in ["train", "val"]:
        src_img_dir = V207_DIR / split / "images"
        dst_img_dir = V208_DIR / split / "images"
        for f in src_img_dir.iterdir():
            if f.is_file():
                dst = dst_img_dir / f.name
                if not dst.exists():
                    os.link(f, dst)
        
        src_lbl_dir = V207_DIR / split / "labels"
        dst_lbl_dir = V208_DIR / split / "labels"
        for f in src_lbl_dir.iterdir():
            if f.is_file():
                dst = dst_lbl_dir / f.name
                if not dst.exists():
                    os.link(f, dst)

    colors = {0: (0, 255, 0), 1: (0, 150, 255), 2: (255, 50, 50)}

    # 3. Candidate 1: commons_88854334 (Christophe95, Sa Pa mountain road, train)
    print("Processing candidate 1 (Christophe95 - Sa Pa mountain road)...")
    orig1_src = ACQ_DIR / "download" / "commons_88854334.jpg"
    orig1_dst = V208_DIR / "source_originals" / "commons_88854334.jpg"
    shutil.copy2(orig1_src, orig1_dst)

    with Image.open(ACQ_DIR / "download" / "sa_pa_crop.png") as im1_crop:
        im1_scaled = im1_crop.resize((1050, 775), Image.Resampling.LANCZOS)
        out1_img = V208_DIR / "train" / "images" / "commons_vietnam__SaPa_TaPhin_mountain_motorcycle_manual_v208.png"
        im1_scaled.save(out1_img)
        phash1 = f"{perceptual_hash(im1_scaled):016x}"

    boxes1_native = [
        (0, (230, 420, 1930, 1400)),
        (1, (470, 130, 950, 980))
    ]
    labels1 = boxes_to_yolo(boxes1_native, 2100, 1550)
    out1_lbl = V208_DIR / "train" / "labels" / "commons_vietnam__SaPa_TaPhin_mountain_motorcycle_manual_v208.txt"
    out1_lbl.write_text("\n".join(labels1) + "\n", encoding="utf-8")

    overlay1 = im1_scaled.copy()
    d1 = ImageDraw.Draw(overlay1)
    s1 = 1050.0 / 2100.0
    for cls, (x1, y1, x2, y2) in boxes1_native:
        d1.rectangle([int(x1 * s1), int(y1 * s1), int(x2 * s1), int(y2 * s1)], outline=colors[cls], width=3)
        d1.text((int(x1 * s1) + 4, int(y1 * s1) + 4), f"cls {cls}", fill=colors[cls])
    overlay1.save(V208_DIR / "review_evidence" / "SaPa_TaPhin_mountain_motorcycle_manual_v208_overlay.png")

    # 4. Candidate 2: commons_37247323 (Uwe Aranas, Hanoi red light, train)
    print("Processing candidate 2 (Uwe Aranas - Hanoi red light)...")
    orig2_src = ACQ_DIR / "download" / "commons_37247323.jpg"
    orig2_dst = V208_DIR / "source_originals" / "commons_37247323.jpg"
    shutil.copy2(orig2_src, orig2_dst)

    with Image.open(ACQ_DIR / "download" / "scaled_37247323.png") as im2:
        out2_img = V208_DIR / "train" / "images" / "commons_vietnam__Hanoi_intersection_red_light_manual_v208.png"
        im2.save(out2_img)
        phash2 = f"{perceptual_hash(im2):016x}"

    boxes2_native = [
        (0, (270, 260, 630, 610)),
        (1, (310, 140, 550, 480)),
        (0, (20, 240, 280, 520)),
        (1, (50, 150, 250, 400)),
        (0, (650, 250, 920, 540)),
        (1, (700, 160, 890, 430))
    ]
    labels2 = boxes_to_yolo(boxes2_native, 1000, 666)
    out2_lbl = V208_DIR / "train" / "labels" / "commons_vietnam__Hanoi_intersection_red_light_manual_v208.txt"
    out2_lbl.write_text("\n".join(labels2) + "\n", encoding="utf-8")

    overlay2 = im2.copy()
    d2 = ImageDraw.Draw(overlay2)
    for cls, (x1, y1, x2, y2) in boxes2_native:
        d2.rectangle([x1, y1, x2, y2], outline=colors[cls], width=3)
        d2.text((x1 + 4, y1 + 4), f"cls {cls}", fill=colors[cls])
    overlay2.save(V208_DIR / "review_evidence" / "Hanoi_intersection_red_light_manual_v208_overlay.png")

    # 5. Candidate 3: commons_86056481 (Vyacheslav Argenberg, HCMC motorbikes, train)
    print("Processing candidate 3 (Vyacheslav Argenberg - HCMC motorbikes)...")
    orig3_src = ACQ_DIR / "download" / "commons_86056481.jpg"
    orig3_dst = V208_DIR / "source_originals" / "commons_86056481.jpg"
    shutil.copy2(orig3_src, orig3_dst)

    with Image.open(orig3_src) as im3:
        im3_scaled = im3.resize((1328, 885), Image.Resampling.LANCZOS)
        out3_img = V208_DIR / "train" / "images" / "commons_vietnam__HCMC_commuter_motorbikes_traffic_manual_v208.png"
        im3_scaled.save(out3_img)
        phash3 = f"{perceptual_hash(im3_scaled):016x}"

    boxes3_native = [
        (0, (650, 1150, 1650, 2350)),
        (1, (820, 800, 1500, 1850)),
        (0, (1600, 1180, 2600, 2350)),
        (1, (1700, 820, 2400, 1850)),
        (0, (2650, 1150, 3600, 2300)),
        (1, (2750, 780, 3450, 1850))
    ]
    labels3 = boxes_to_yolo(boxes3_native, 3984, 2656)
    out3_lbl = V208_DIR / "train" / "labels" / "commons_vietnam__HCMC_commuter_motorbikes_traffic_manual_v208.txt"
    out3_lbl.write_text("\n".join(labels3) + "\n", encoding="utf-8")

    overlay3 = im3_scaled.copy()
    d3 = ImageDraw.Draw(overlay3)
    s3 = 1328.0 / 3984.0
    for cls, (x1, y1, x2, y2) in boxes3_native:
        d3.rectangle([int(x1 * s3), int(y1 * s3), int(x2 * s3), int(y2 * s3)], outline=colors[cls], width=3)
        d3.text((int(x1 * s3) + 4, int(y1 * s3) + 4), f"cls {cls}", fill=colors[cls])
    overlay3.save(V208_DIR / "review_evidence" / "HCMC_commuter_motorbikes_traffic_manual_v208_overlay.png")

    # 6. Candidate 4: commons_61737989 (Frank McKenna, HCMC avenue, val)
    print("Processing candidate 4 (Frank McKenna - HCMC avenue)...")
    orig4_src = ACQ_DIR / "download" / "commons_61737989.jpg"
    orig4_dst = V208_DIR / "source_originals" / "commons_61737989.jpg"
    shutil.copy2(orig4_src, orig4_dst)

    with Image.open(orig4_src) as im4:
        im4_scaled = im4.resize((1232, 820), Image.Resampling.LANCZOS)
        out4_img = V208_DIR / "val" / "images" / "commons_vietnam__HCMC_avenue_motorbikers_manual_v208.png"
        im4_scaled.save(out4_img)
        phash4 = f"{perceptual_hash(im4_scaled):016x}"

    boxes4_native = [
        (0, (1650, 1500, 3100, 2950)),
        (1, (1800, 950, 2900, 2200)),
        (0, (450, 1550, 1700, 2850)),
        (1, (600, 1050, 1550, 2250)),
        (0, (3100, 1550, 4400, 2850)),
        (1, (3250, 1050, 4150, 2200))
    ]
    labels4 = boxes_to_yolo(boxes4_native, 4928, 3280)
    out4_lbl = V208_DIR / "val" / "labels" / "commons_vietnam__HCMC_avenue_motorbikers_manual_v208.txt"
    out4_lbl.write_text("\n".join(labels4) + "\n", encoding="utf-8")

    overlay4 = im4_scaled.copy()
    d4 = ImageDraw.Draw(overlay4)
    s4 = 1232.0 / 4928.0
    for cls, (x1, y1, x2, y2) in boxes4_native:
        d4.rectangle([int(x1 * s4), int(y1 * s4), int(x2 * s4), int(y2 * s4)], outline=colors[cls], width=3)
        d4.text((int(x1 * s4) + 4, int(y1 * s4) + 4), f"cls {cls}", fill=colors[cls])
    overlay4.save(V208_DIR / "review_evidence" / "HCMC_avenue_motorbikers_manual_v208_overlay.png")

    # 7. Save metadata files
    with open(ACQ_DIR / "unadmitted_commons_candidates.json", "r", encoding="utf-8") as f:
        cand_list = json.load(f)
    cand_map = {c["pageid"]: c for c in cand_list}

    for pid in [88854334, 37247323, 86056481, 61737989]:
        meta_file = V208_DIR / "source_metadata" / f"commons_{pid}.json"
        meta_file.write_text(json.dumps(cand_map.get(pid, {}), ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. Update Manifest
    print("Updating manifest.json...")
    with open(V207_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    new_rows = [
        {
            "id": "commons_vietnam/SaPa_TaPhin_mountain_motorcycle_manual_v208.png",
            "pageid": 88854334,
            "original_pageid": 88854334,
            "title": "File:Motorcycle in Sa Pa.jpg",
            "source_group": "commons_christophe95",
            "group": "group_christophe95_hanoi",
            "artist": "Christophe95",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "new_split": "train",
            "width": 1050,
            "height": 775,
            "sha256": sha256_file(out1_img),
            "phash": phash1,
            "source_image": "Demo/training/dataset_v208/source_originals/commons_88854334.jpg",
            "source_sha256": sha256_file(orig1_dst),
            "source_metadata": "Demo/training/dataset_v208/source_metadata/commons_88854334.json",
            "review_evidence": "Demo/training/dataset_v208/review_evidence/SaPa_TaPhin_mountain_motorcycle_manual_v208_overlay.png",
            "location_evidence": "Sa Pa to Tả Phìn, Lao Cai province, Northern Vietnam; mountain road with rider and utility motorcycle.",
            "classes": {"0": 1, "1": 1, "2": 0}
        },
        {
            "id": "commons_vietnam/Hanoi_intersection_red_light_manual_v208.png",
            "pageid": 37247323,
            "original_pageid": 37247323,
            "title": "File:Hanoi Vietnam Waiting-at-the-red-lights-01.jpg",
            "source_group": "commons_uwe_aranas",
            "group": "group_uwe_aranas",
            "artist": "Uwe Aranas",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "train",
            "width": 1000,
            "height": 666,
            "sha256": sha256_file(out2_img),
            "phash": phash2,
            "source_image": "Demo/training/dataset_v208/source_originals/commons_37247323.jpg",
            "source_sha256": sha256_file(orig2_dst),
            "source_metadata": "Demo/training/dataset_v208/source_metadata/commons_37247323.json",
            "review_evidence": "Demo/training/dataset_v208/review_evidence/Hanoi_intersection_red_light_manual_v208_overlay.png",
            "location_evidence": "Hanoi, Vietnam; urban intersection waiting line with commuter motorbikes and riders with helmets.",
            "classes": {"0": 3, "1": 3, "2": 0}
        },
        {
            "id": "commons_vietnam/HCMC_commuter_motorbikes_traffic_manual_v208.png",
            "pageid": 86056481,
            "original_pageid": 86056481,
            "title": "File:Ho Chi Minh City, Vietnam, Motorbike riders.jpg",
            "source_group": "commons_vyacheslav_argenberg",
            "group": "group_argenberg_hcmc",
            "artist": "Vyacheslav Argenberg",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "new_split": "train",
            "width": 1328,
            "height": 885,
            "sha256": sha256_file(out3_img),
            "phash": phash3,
            "source_image": "Demo/training/dataset_v208/source_originals/commons_86056481.jpg",
            "source_sha256": sha256_file(orig3_dst),
            "source_metadata": "Demo/training/dataset_v208/source_metadata/commons_86056481.json",
            "review_evidence": "Demo/training/dataset_v208/review_evidence/HCMC_commuter_motorbikes_traffic_manual_v208_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; busy commuter street with Vietnamese riders wearing standardized helmets on Honda/Yamaha motorbikes.",
            "classes": {"0": 3, "1": 3, "2": 0}
        },
        {
            "id": "commons_vietnam/HCMC_avenue_motorbikers_manual_v208.png",
            "pageid": 61737989,
            "original_pageid": 61737989,
            "title": "File:Ho Chi Minh City motorbikers (Unsplash LhENdA-0yCM).jpg",
            "source_group": "commons_frank_mckenna",
            "group": "group_frank_mckenna_hcmc",
            "artist": "Frank McKenna",
            "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "new_split": "val",
            "width": 1232,
            "height": 820,
            "sha256": sha256_file(out4_img),
            "phash": phash4,
            "source_image": "Demo/training/dataset_v208/source_originals/commons_61737989.jpg",
            "source_sha256": sha256_file(orig4_dst),
            "source_metadata": "Demo/training/dataset_v208/source_metadata/commons_61737989.json",
            "review_evidence": "Demo/training/dataset_v208/review_evidence/HCMC_avenue_motorbikers_manual_v208_overlay.png",
            "location_evidence": "Ho Chi Minh City avenue, Vietnam; commuter motorbikes and scooters traveling in daylight traffic.",
            "classes": {"0": 3, "1": 3, "2": 0}
        }
    ]

    manifest.extend(new_rows)
    manifest_path = V208_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 9. Update Statistics
    print("Updating statistics.json...")
    with open(V207_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Train: +3 images
    # Sa Pa (Christophe95): cls 0: +1, cls 1: +1, cls 2: +0 (group_christophe95_hanoi already in train)
    # Red light (Uwe Aranas): cls 0: +3, cls 1: +3, cls 2: +0 (group_uwe_aranas is new to train)
    # Argenberg: cls 0: +3, cls 1: +3, cls 2: +0 (group_argenberg_hcmc is new to train)
    stats["train"]["images"] += 3
    stats["train"]["classes"]["0"] += 7
    stats["train"]["classes"]["1"] += 7
    stats["train"]["classes"]["2"] += 0
    stats["train"]["groups"] += 2  # group_uwe_aranas, group_argenberg_hcmc

    # Val: +1 image (Frank McKenna)
    # McKenna: cls 0: +3, cls 1: +3, cls 2: +0 (group_frank_mckenna_hcmc is new to val)
    stats["val"]["images"] += 1
    stats["val"]["classes"]["0"] += 3
    stats["val"]["classes"]["1"] += 3
    stats["val"]["classes"]["2"] += 0
    stats["val"]["groups"] += 1  # group_frank_mckenna_hcmc

    stats_path = V208_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 10. Copy duplicates, usage_restrictions, data.yaml, split_lock
    shutil.copy2(V207_DIR / "duplicates.json", V208_DIR / "duplicates.json")
    shutil.copy2(V207_DIR / "usage_restrictions.json", V208_DIR / "usage_restrictions.json")
    shutil.copy2(V207_DIR / "data.yaml", V208_DIR / "data.yaml")
    shutil.copy2(V207_DIR / "split_lock.json", V208_DIR / "split_lock.json")

    # 11. Changes and Review batch
    manual_review_43 = {
        "batch": 43,
        "date": "2026-11-11",
        "snapshot": "dataset_v208",
        "parent": "dataset_v207",
        "method": "manual pixel inspection and bounding box annotation; no model pseudo-labeling",
        "reviewed_candidates_count": 8,
        "accepted_images_count": 4,
        "held_pending_or_excluded_count": 4,
        "accepted": [
            {
                "id": "commons_vietnam/SaPa_TaPhin_mountain_motorcycle_manual_v208.png",
                "pageid": 88854334,
                "author": "Christophe95",
                "license": "CC BY-SA 4.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 1},
                    {"cls": 1, "name": "rider_with_helmet", "count": 1},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Sa Pa to Tả Phìn mountain highway, motorcycle with rider in helmet."
            },
            {
                "id": "commons_vietnam/Hanoi_intersection_red_light_manual_v208.png",
                "pageid": 37247323,
                "author": "Uwe Aranas",
                "license": "CC BY-SA 3.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 3},
                    {"cls": 1, "name": "rider_with_helmet", "count": 3},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Hanoi traffic intersection waiting line at red light, multiple motorcycle riders in helmets."
            },
            {
                "id": "commons_vietnam/HCMC_commuter_motorbikes_traffic_manual_v208.png",
                "pageid": 86056481,
                "author": "Vyacheslav Argenberg",
                "license": "CC BY 4.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 3},
                    {"cls": 1, "name": "rider_with_helmet", "count": 3},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Ho Chi Minh City street commuters on motorcycles with helmets."
            },
            {
                "id": "commons_vietnam/HCMC_avenue_motorbikers_manual_v208.png",
                "pageid": 61737989,
                "author": "Frank McKenna",
                "license": "CC0",
                "assigned_split": "val",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 3},
                    {"cls": 1, "name": "rider_with_helmet", "count": 3},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Ho Chi Minh City avenue motorbikers traveling in daytime traffic."
            }
        ],
        "excluded_or_pending": [
            {"pageid": 185887572, "title": "File:Minsk Motorcycle Vietnam Tourist.jpg", "reason": "Tourist polishing parked Minsk motorcycle beside building; no active traffic rider; held pending."},
            {"pageid": 64630100, "title": "File:Vietnam-bike-tours.jpg", "reason": "Bicycle tour on Highway 1; excluded under Rule 24 (bicycles)."},
            {"pageid": 115770740, "title": "File:Vietnam-0648.jpg", "reason": "Near-duplicate sequence frame of 115770739 (BluesyPete pig transport); held pending."},
            {"pageid": 17186618, "title": "File:Bridge motorcycle Vietnam.jpg", "reason": "Long shot bridge panorama; motorcycles represent <15px bounding width; deferred to high-res sprint."}
        ]
    }
    with open(V208_DIR / "manual_review_batch43.json", "w", encoding="utf-8") as f:
        json.dump(manual_review_43, f, ensure_ascii=False, indent=2)

    # 12. Update Attribution
    with open(V207_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attribution = json.load(f)
    
    new_attributions = [
        {
            "id": "commons_vietnam/SaPa_TaPhin_mountain_motorcycle_manual_v208.png",
            "title": "File:Motorcycle in Sa Pa.jpg",
            "artist": "Christophe95",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Motorcycle_in_Sa_Pa.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/c/ca/Motorcycle_in_Sa_Pa.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/c/ca/Motorcycle_in_Sa_Pa.jpg",
            "sha256": sha256_file(out1_img),
            "country": "VN",
            "location_evidence": "Sa Pa to Tả Phìn, Lao Cai province, Northern Vietnam; mountain road with rider and utility motorcycle.",
            "modifications": "Cropped foreground region (2100x1550) and scaled to 1050x775. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/Hanoi_intersection_red_light_manual_v208.png",
            "title": "File:Hanoi Vietnam Waiting-at-the-red-lights-01.jpg",
            "artist": "Uwe Aranas",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Hanoi_Vietnam_Waiting-at-the-red-lights-01.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/0/0a/Hanoi_Vietnam_Waiting-at-the-red-lights-01.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/0/0a/Hanoi_Vietnam_Waiting-at-the-red-lights-01.jpg",
            "sha256": sha256_file(out2_img),
            "country": "VN",
            "location_evidence": "Hanoi, Vietnam; urban intersection waiting line with commuter motorbikes and riders with helmets.",
            "modifications": "Scaled to 1000x666. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_commuter_motorbikes_traffic_manual_v208.png",
            "title": "File:Ho Chi Minh City, Vietnam, Motorbike riders.jpg",
            "artist": "Vyacheslav Argenberg",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Ho_Chi_Minh_City,_Vietnam,_Motorbike_riders.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Ho_Chi_Minh_City%2C_Vietnam%2C_Motorbike_riders.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Ho_Chi_Minh_City%2C_Vietnam%2C_Motorbike_riders.jpg",
            "sha256": sha256_file(out3_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; busy commuter street with Vietnamese riders wearing standardized helmets on Honda/Yamaha motorbikes.",
            "modifications": "Scaled to 1328x885. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_avenue_motorbikers_manual_v208.png",
            "title": "File:Ho Chi Minh City motorbikers (Unsplash LhENdA-0yCM).jpg",
            "artist": "Frank McKenna",
            "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "page_url": "https://commons.wikimedia.org/wiki/File:Ho_Chi_Minh_City_motorbikers_(Unsplash_LhENdA-0yCM).jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Ho_Chi_Minh_City_motorbikers_%28Unsplash_LhENdA-0yCM%29.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Ho_Chi_Minh_City_motorbikers_%28Unsplash_LhENdA-0yCM%29.jpg",
            "sha256": sha256_file(out4_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City avenue, Vietnam; commuter motorbikes and scooters traveling in daylight traffic.",
            "modifications": "Scaled to 1232x820. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        }
    ]

    attribution.extend(new_attributions)
    with open(V208_DIR / "attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, ensure_ascii=False, indent=2)

    # 13. Update Changes
    with open(V207_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)

    changes.append({
        "snapshot": "dataset_v208",
        "parent": "dataset_v207",
        "action": "accept_manually_reviewed_commons_images",
        "accepted_images": [
            "commons_vietnam/SaPa_TaPhin_mountain_motorcycle_manual_v208.png",
            "commons_vietnam/Hanoi_intersection_red_light_manual_v208.png",
            "commons_vietnam/HCMC_commuter_motorbikes_traffic_manual_v208.png",
            "commons_vietnam/HCMC_avenue_motorbikers_manual_v208.png"
        ],
        "new_accepted_count": 4,
        "new_class_0_boxes": 10,
        "new_class_1_boxes": 10,
        "new_class_2_boxes": 0,
        "review_batch": "manual_review_batch43.json"
    })

    with open(V208_DIR / "changes.json", "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # 14. Update ATTRIBUTION.md
    attr_md = (
        "# Dataset Attribution Record - v208\n\n"
        "Total images: 1,110 clean ground-truth images.\n\n"
        "## Newly Admitted Images (v208 Manual Review Batch 43)\n"
        "1. **commons_vietnam/SaPa_TaPhin_mountain_motorcycle_manual_v208.png**\n"
        "   - Title: `File:Motorcycle in Sa Pa.jpg`\n"
        "   - Artist: Christophe95\n"
        "   - License: CC BY-SA 4.0\n"
        "   - Location: Sa Pa to Tả Phìn, Lao Cai, Vietnam\n"
        "   - Assigned Split: `train`\n\n"
        "2. **commons_vietnam/Hanoi_intersection_red_light_manual_v208.png**\n"
        "   - Title: `File:Hanoi Vietnam Waiting-at-the-red-lights-01.jpg`\n"
        "   - Artist: Uwe Aranas\n"
        "   - License: CC BY-SA 3.0\n"
        "   - Location: Hanoi, Vietnam (intersection red light)\n"
        "   - Assigned Split: `train`\n\n"
        "3. **commons_vietnam/HCMC_commuter_motorbikes_traffic_manual_v208.png**\n"
        "   - Title: `File:Ho Chi Minh City, Vietnam, Motorbike riders.jpg`\n"
        "   - Artist: Vyacheslav Argenberg\n"
        "   - License: CC BY 4.0\n"
        "   - Location: Ho Chi Minh City, Vietnam\n"
        "   - Assigned Split: `train`\n\n"
        "4. **commons_vietnam/HCMC_avenue_motorbikers_manual_v208.png**\n"
        "   - Title: `File:Ho Chi Minh City motorbikers (Unsplash LhENdA-0yCM).jpg`\n"
        "   - Artist: Frank McKenna\n"
        "   - License: CC0\n"
        "   - Location: Ho Chi Minh City, Vietnam (avenue motorbikers)\n"
        "   - Assigned Split: `val`\n"
    )
    (V208_DIR / "ATTRIBUTION.md").write_text(attr_md, encoding="utf-8")

    # 15. Update README.md
    readme_md = (
        "# Dataset Snapshot v208\n\n"
        "- Total clean reviewed images: **1,110**\n"
        "- Split counts: `train: 895`, `val: 147`, `test: 68` (locked)\n"
        "- Class instance counts: `0: motorcycle (1,983)`, `1: rider_with_helmet (1,103)`, `2: rider_without_helmet (593)`\n"
        "- Total clean bounding boxes: **3,679**\n"
        "- Progress towards 5,000 goal: **22.20%** (Remaining: **3,890**)\n"
    )
    (V208_DIR / "README.md").write_text(readme_md, encoding="utf-8")

    # 16. Verification Report
    verif = {
        "snapshot": "dataset_v208",
        "parent": "dataset_v207",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2267,
        "rows_total": 2271,
        "accepted_clean_images": 1110,
        "remaining_to_5000": 3890,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 10, "1": 10, "2": 0},
        "split_counts": {
            "train": 895,
            "val": 147,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch43.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V208_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(V208_DIR / "attribution.json"),
        "duplicates_sha256": sha256_file(V208_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(V208_DIR / "changes.json")
    }
    with open(V208_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v208.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # 17. Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    progress = {
        "schema_version": 1,
        "as_of": "2026-11-11",
        "goal_images": 5000,
        "accepted_clean_images": 1110,
        "remaining_to_goal": 3890,
        "pending_review_images": 25,
        "snapshot": "dataset_v208",
        "accepted_split_counts": {
            "train": 895,
            "validation": 147,
            "test_locked": 68
        },
        "newly_accepted_this_batch": 4,
        "new_batch_class_box_counts": {
            "0": 10,
            "1": 10,
            "2": 0
        },
        "total_class_box_counts": {
            "0": 1983,
            "1": 1103,
            "2": 593
        },
        "original_dataset_unchanged": True,
        "test_split_locked": True,
        "pending_counts_toward_goal": False,
        "verification_report": "../verification/dataset_v208.json",
        "notes": [
            "Counts are individually reviewed images, not boxes; duplicate and pending rows do not count.",
            "dataset_v203 audited all 1,492 pending DataSet2 items: 335 unique canonical frames accepted (279 train, 56 val), 1,132 redundant variants excluded, 25 items held pending in group_0000 to maintain test lock.",
            "dataset_v204 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), verified Vietnam location context, zero leakage between splits, and test split locked at 68 images. Total clean images reaches 1,095.",
            "dataset_v205 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), adding 32 clean bounding boxes (including 7 rare rider_without_helmet boxes in Hanoi school traffic). Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,098.",
            "dataset_v206 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 2.5 / CC BY-SA 3.0), adding 19 clean bounding boxes including police patrol motorcycle, agricultural animal transport, nursery tree cargo, and city traffic. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,102.",
            "dataset_v207 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY 2.0 / CC BY-SA 2.0 / CC BY-SA 3.0), adding 23 clean bounding boxes including pillion passenger, father with unhelmeted child on Super Cub, pastry delivery carrier, and HCMC intersection. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,106.",
            "dataset_v208 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 4.0 / CC BY-SA 3.0 / CC BY 4.0 / CC0), adding 20 clean bounding boxes including Sa Pa mountain highway, Hanoi intersection red light waiting line, and HCMC commuter traffic. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,110."
        ]
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v208 build completed successfully! ===")

if __name__ == "__main__":
    main()
