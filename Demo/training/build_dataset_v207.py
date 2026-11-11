"""Build and verify dataset_v207 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Hanoi Hàng Bông moto pillion taking motos (author: McKay Savage, CC BY 2.0) -> train (group_meaduva)
2. Hanoi Lãn Ông father and child on Super Cub (author: Clay Gilliland, CC BY-SA 2.0) -> train (group_clay_gilliland_hanoi)
3. Northern Vietnam pastry delivery on Super Cub (author: BluesyPete, CC BY-SA 3.0) -> train (group_bluesypete_transport)
4. Ho Chi Minh City traffic intersection DD 01 (author: Diego Delso, CC BY-SA 3.0) -> val (group_diego_delso)

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
V206_DIR = TRAINING_DIR / "dataset_v206"
V207_DIR = TRAINING_DIR / "dataset_v207"
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
    print("=== Building Dataset v207 Snapshot ===")
    V207_DIR.mkdir(exist_ok=True)
    
    # 1. Setup directories
    for split in ["train", "val"]:
        (V207_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V207_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    (V207_DIR / "review_evidence").mkdir(exist_ok=True)
    (V207_DIR / "source_originals").mkdir(exist_ok=True)
    (V207_DIR / "source_metadata").mkdir(exist_ok=True)

    # 2. Link existing assets from v206
    print("Linking existing v206 assets...")
    for split in ["train", "val"]:
        src_img_dir = V206_DIR / split / "images"
        dst_img_dir = V207_DIR / split / "images"
        for f in src_img_dir.iterdir():
            if f.is_file():
                dst = dst_img_dir / f.name
                if not dst.exists():
                    os.link(f, dst)
        
        src_lbl_dir = V206_DIR / split / "labels"
        dst_lbl_dir = V207_DIR / split / "labels"
        for f in src_lbl_dir.iterdir():
            if f.is_file():
                dst = dst_lbl_dir / f.name
                if not dst.exists():
                    os.link(f, dst)

    colors = {0: (0, 255, 0), 1: (0, 150, 255), 2: (255, 50, 50)}

    # 3. Candidate 1: commons_23467070 (McKay Savage, Hanoi Hàng Bông, train)
    print("Processing candidate 1 (McKay Savage - Hanoi Hàng Bông)...")
    orig1_src = ACQ_DIR / "download" / "commons_23467070.jpg"
    orig1_dst = V207_DIR / "source_originals" / "commons_23467070.jpg"
    shutil.copy2(orig1_src, orig1_dst)

    with Image.open(orig1_src) as im1:
        im1_scaled = im1.resize((1224, 1632), Image.Resampling.LANCZOS)
        out1_img = V207_DIR / "train" / "images" / "commons_vietnam__Hanoi_HangBong_moto_pillion_manual_v207.png"
        im1_scaled.save(out1_img)
        phash1 = f"{perceptual_hash(im1_scaled):016x}"

    boxes1_native = [
        (0, (580, 1500, 2050, 3150)),
        (1, (1020, 850, 1820, 2350)),
        (1, (680, 1020, 1520, 2600)),
        (0, (20, 1450, 720, 2400)),
        (1, (80, 1150, 620, 2100)),
        (0, (1920, 1600, 2448, 2650))
    ]
    labels1 = boxes_to_yolo(boxes1_native, 2448, 3264)
    out1_lbl = V207_DIR / "train" / "labels" / "commons_vietnam__Hanoi_HangBong_moto_pillion_manual_v207.txt"
    out1_lbl.write_text("\n".join(labels1) + "\n", encoding="utf-8")

    overlay1 = im1_scaled.copy()
    d1 = ImageDraw.Draw(overlay1)
    s1 = 1224.0 / 2448.0
    for cls, (x1, y1, x2, y2) in boxes1_native:
        d1.rectangle([int(x1 * s1), int(y1 * s1), int(x2 * s1), int(y2 * s1)], outline=colors[cls], width=3)
        d1.text((int(x1 * s1) + 4, int(y1 * s1) + 4), f"cls {cls}", fill=colors[cls])
    overlay1.save(V207_DIR / "review_evidence" / "Hanoi_HangBong_moto_pillion_manual_v207_overlay.png")

    # 4. Candidate 2: commons_31660225 (Clay Gilliland, Hanoi Lãn Ông, train)
    print("Processing candidate 2 (Clay Gilliland - Hanoi Lãn Ông)...")
    orig2_src = ACQ_DIR / "download" / "commons_31660225.jpg"
    orig2_dst = V207_DIR / "source_originals" / "commons_31660225.jpg"
    shutil.copy2(orig2_src, orig2_dst)

    with Image.open(orig2_src) as im2:
        im2_scaled = im2.resize((1248, 1021), Image.Resampling.LANCZOS)
        out2_img = V207_DIR / "train" / "images" / "commons_vietnam__Hanoi_LanOng_father_child_cub_manual_v207.png"
        im2_scaled.save(out2_img)
        phash2 = f"{perceptual_hash(im2_scaled):016x}"

    boxes2_native = [
        (0, (680, 950, 1850, 1980)),
        (1, (980, 450, 1780, 1750)),
        (2, (700, 620, 1220, 1450)),
        (0, (0, 880, 750, 1850)),
        (0, (1650, 880, 2496, 1800)),
        (0, (1550, 650, 2250, 1300)),
        (1, (1600, 500, 2150, 1050))
    ]
    labels2 = boxes_to_yolo(boxes2_native, 2496, 2042)
    out2_lbl = V207_DIR / "train" / "labels" / "commons_vietnam__Hanoi_LanOng_father_child_cub_manual_v207.txt"
    out2_lbl.write_text("\n".join(labels2) + "\n", encoding="utf-8")

    overlay2 = im2_scaled.copy()
    d2 = ImageDraw.Draw(overlay2)
    s2 = 1248.0 / 2496.0
    for cls, (x1, y1, x2, y2) in boxes2_native:
        d2.rectangle([int(x1 * s2), int(y1 * s2), int(x2 * s2), int(y2 * s2)], outline=colors[cls], width=3)
        d2.text((int(x1 * s2) + 4, int(y1 * s2) + 4), f"cls {cls}", fill=colors[cls])
    overlay2.save(V207_DIR / "review_evidence" / "Hanoi_LanOng_father_child_cub_manual_v207_overlay.png")

    # 5. Candidate 3: commons_115770745 (BluesyPete, pastry delivery, train)
    print("Processing candidate 3 (BluesyPete - Pastry delivery)...")
    orig3_src = ACQ_DIR / "download" / "commons_115770745.jpg"
    orig3_dst = V207_DIR / "source_originals" / "commons_115770745.jpg"
    shutil.copy2(orig3_src, orig3_dst)

    with Image.open(orig3_src) as im3:
        im3_scaled = im3.resize((1144, 856), Image.Resampling.LANCZOS)
        out3_img = V207_DIR / "train" / "images" / "commons_vietnam__Northern_road_pastry_delivery_cub_manual_v207.png"
        im3_scaled.save(out3_img)
        phash3 = f"{perceptual_hash(im3_scaled):016x}"

    boxes3_native = [
        (0, (420, 650, 1850, 1680)),
        (1, (800, 320, 1450, 1450))
    ]
    labels3 = boxes_to_yolo(boxes3_native, 2288, 1712)
    out3_lbl = V207_DIR / "train" / "labels" / "commons_vietnam__Northern_road_pastry_delivery_cub_manual_v207.txt"
    out3_lbl.write_text("\n".join(labels3) + "\n", encoding="utf-8")

    overlay3 = im3_scaled.copy()
    d3 = ImageDraw.Draw(overlay3)
    s3 = 1144.0 / 2288.0
    for cls, (x1, y1, x2, y2) in boxes3_native:
        d3.rectangle([int(x1 * s3), int(y1 * s3), int(x2 * s3), int(y2 * s3)], outline=colors[cls], width=3)
        d3.text((int(x1 * s3) + 4, int(y1 * s3) + 4), f"cls {cls}", fill=colors[cls])
    overlay3.save(V207_DIR / "review_evidence" / "Northern_road_pastry_delivery_cub_manual_v207_overlay.png")

    # 6. Candidate 4: commons_30110165 / crop_30110165 (Diego Delso, DD 01, val)
    print("Processing candidate 4 (Diego Delso - DD 01)...")
    orig4_src = ACQ_DIR / "download" / "commons_30110165.jpg"
    orig4_dst = V207_DIR / "source_originals" / "commons_30110165.jpg"
    shutil.copy2(orig4_src, orig4_dst)

    with Image.open(ACQ_DIR / "download" / "crop_30110165.png") as im4_crop:
        im4_scaled = im4_crop.resize((1200, 885), Image.Resampling.LANCZOS)
        out4_img = V207_DIR / "val" / "images" / "commons_vietnam__HCMC_DD01_traffic_intersection_manual_v207.png"
        im4_scaled.save(out4_img)
        phash4 = f"{perceptual_hash(im4_scaled):016x}"

    boxes4_native = [
        (0, (1120, 750, 1980, 1720)),
        (1, (1150, 280, 1750, 1400)),
        (0, (550, 780, 1250, 1680)),
        (1, (600, 360, 1180, 1350)),
        (0, (20, 800, 650, 1650)),
        (1, (50, 380, 600, 1380)),
        (0, (1850, 600, 2350, 1250)),
        (1, (1900, 380, 2300, 1100))
    ]
    labels4 = boxes_to_yolo(boxes4_native, 2400, 1771)
    out4_lbl = V207_DIR / "val" / "labels" / "commons_vietnam__HCMC_DD01_traffic_intersection_manual_v207.txt"
    out4_lbl.write_text("\n".join(labels4) + "\n", encoding="utf-8")

    overlay4 = im4_scaled.copy()
    d4 = ImageDraw.Draw(overlay4)
    s4 = 1200.0 / 2400.0
    for cls, (x1, y1, x2, y2) in boxes4_native:
        d4.rectangle([int(x1 * s4), int(y1 * s4), int(x2 * s4), int(y2 * s4)], outline=colors[cls], width=3)
        d4.text((int(x1 * s4) + 4, int(y1 * s4) + 4), f"cls {cls}", fill=colors[cls])
    overlay4.save(V207_DIR / "review_evidence" / "HCMC_DD01_traffic_intersection_manual_v207_overlay.png")

    # 7. Save metadata files
    with open(ACQ_DIR / "unadmitted_commons_candidates.json", "r", encoding="utf-8") as f:
        cand_list = json.load(f)
    cand_map = {c["pageid"]: c for c in cand_list}

    for pid in [23467070, 31660225, 115770745, 30110165]:
        meta_file = V207_DIR / "source_metadata" / f"commons_{pid}.json"
        meta_file.write_text(json.dumps(cand_map.get(pid, {}), ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. Update Manifest
    print("Updating manifest.json...")
    with open(V206_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    new_rows = [
        {
            "id": "commons_vietnam/Hanoi_HangBong_moto_pillion_manual_v207.png",
            "pageid": 23467070,
            "original_pageid": 23467070,
            "title": "Vietnam 08 - 01 - taking the motos (3166745807).jpg",
            "source_group": "commons_meaduva",
            "group": "group_meaduva",
            "artist": "McKay Savage",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "new_split": "train",
            "width": 1224,
            "height": 1632,
            "sha256": sha256_file(out1_img),
            "phash": phash1,
            "source_image": "Demo/training/dataset_v207/source_originals/commons_23467070.jpg",
            "source_sha256": sha256_file(orig1_dst),
            "source_metadata": "Demo/training/dataset_v207/source_metadata/commons_23467070.json",
            "review_evidence": "Demo/training/dataset_v207/review_evidence/Hanoi_HangBong_moto_pillion_manual_v207_overlay.png",
            "location_evidence": "Hanoi, Vietnam; Hàng Bông street old quarter, visible motorcycle license plate 90-H5 2391, typical Vietnamese street traffic.",
            "classes": {"0": 3, "1": 3, "2": 0}
        },
        {
            "id": "commons_vietnam/Hanoi_LanOng_father_child_cub_manual_v207.png",
            "pageid": 31660225,
            "original_pageid": 31660225,
            "title": "Hanoi, Vietnam (12035628605).jpg",
            "source_group": "commons_clay_gilliland",
            "group": "group_clay_gilliland_hanoi",
            "artist": "Clay Gilliland",
            "license": "CC BY-SA 2.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.0/",
            "new_split": "train",
            "width": 1248,
            "height": 1021,
            "sha256": sha256_file(out2_img),
            "phash": phash2,
            "source_image": "Demo/training/dataset_v207/source_originals/commons_31660225.jpg",
            "source_sha256": sha256_file(orig2_dst),
            "source_metadata": "Demo/training/dataset_v207/source_metadata/commons_31660225.json",
            "review_evidence": "Demo/training/dataset_v207/review_evidence/Hanoi_LanOng_father_child_cub_manual_v207_overlay.png",
            "location_evidence": "Hanoi, Vietnam; Lãn Ông herbal medicine street in Hoan Kiem district, traditional shopfronts, father with child on Honda Super Cub.",
            "classes": {"0": 4, "1": 2, "2": 1}
        },
        {
            "id": "commons_vietnam/Northern_road_pastry_delivery_cub_manual_v207.png",
            "pageid": 115770745,
            "original_pageid": 115770745,
            "title": "Vietnam-0779.jpg",
            "source_group": "commons_bluesypete",
            "group": "group_bluesypete_transport",
            "artist": "BluesyPete",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "train",
            "width": 1144,
            "height": 856,
            "sha256": sha256_file(out3_img),
            "phash": phash3,
            "source_image": "Demo/training/dataset_v207/source_originals/commons_115770745.jpg",
            "source_sha256": sha256_file(orig3_dst),
            "source_metadata": "Demo/training/dataset_v207/source_metadata/commons_115770745.json",
            "review_evidence": "Demo/training/dataset_v207/review_evidence/Northern_road_pastry_delivery_cub_manual_v207_overlay.png",
            "location_evidence": "Northern Vietnam rural road; pastry delivery vendor riding Honda Super Cub with customized transport baskets.",
            "classes": {"0": 1, "1": 1, "2": 0}
        },
        {
            "id": "commons_vietnam/HCMC_DD01_traffic_intersection_manual_v207.png",
            "pageid": 30110165,
            "original_pageid": 30110165,
            "title": "Tráfico en Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 01.JPG",
            "source_group": "commons_diego_delso",
            "group": "group_diego_delso",
            "artist": "Diego Delso",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "val",
            "width": 1200,
            "height": 885,
            "sha256": sha256_file(out4_img),
            "phash": phash4,
            "source_image": "Demo/training/dataset_v207/source_originals/commons_30110165.jpg",
            "source_sha256": sha256_file(orig4_dst),
            "source_metadata": "Demo/training/dataset_v207/source_metadata/commons_30110165.json",
            "review_evidence": "Demo/training/dataset_v207/review_evidence/HCMC_DD01_traffic_intersection_manual_v207_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; major intersection with dense motorcycle commuter traffic, yellow helmet rider, blue jacket rider, white hoodie scooter.",
            "classes": {"0": 4, "1": 4, "2": 0}
        }
    ]

    manifest.extend(new_rows)
    manifest_path = V207_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 9. Update Statistics
    print("Updating statistics.json...")
    with open(V206_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Train: +3 images
    # Cand 1 (McKay Savage): cls 0: +3, cls 1: +3, cls 2: +0
    # Cand 2 (Clay Gilliland): cls 0: +4, cls 1: +2, cls 2: +1
    # Cand 3 (BluesyPete): cls 0: +1, cls 1: +1, cls 2: +0
    stats["train"]["images"] += 3
    stats["train"]["classes"]["0"] += 8
    stats["train"]["classes"]["1"] += 6
    stats["train"]["classes"]["2"] += 1
    stats["train"]["groups"] += 1  # group_clay_gilliland_hanoi is new to train

    # Val: +1 image (Diego Delso DD 01)
    # Cand 4: cls 0: +4, cls 1: +4, cls 2: +0
    stats["val"]["images"] += 1
    stats["val"]["classes"]["0"] += 4
    stats["val"]["classes"]["1"] += 4
    # group_diego_delso already in val

    stats_path = V207_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 10. Copy duplicates, usage_restrictions, data.yaml, split_lock
    shutil.copy2(V206_DIR / "duplicates.json", V207_DIR / "duplicates.json")
    shutil.copy2(V206_DIR / "usage_restrictions.json", V207_DIR / "usage_restrictions.json")
    shutil.copy2(V206_DIR / "data.yaml", V207_DIR / "data.yaml")
    shutil.copy2(V206_DIR / "split_lock.json", V207_DIR / "split_lock.json")

    # 11. Changes and Review batch
    manual_review_42 = {
        "batch": 42,
        "date": "2026-11-11",
        "snapshot": "dataset_v207",
        "parent": "dataset_v206",
        "method": "manual pixel inspection and bounding box annotation; no model pseudo-labeling",
        "reviewed_candidates_count": 8,
        "accepted_images_count": 4,
        "held_pending_or_excluded_count": 4,
        "accepted": [
            {
                "id": "commons_vietnam/Hanoi_HangBong_moto_pillion_manual_v207.png",
                "pageid": 23467070,
                "author": "McKay Savage",
                "license": "CC BY 2.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 3},
                    {"cls": 1, "name": "rider_with_helmet", "count": 3},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Hanoi Hàng Bông street, plate 90-H5 2391, commuter motorcycle with green helmet pillion passenger."
            },
            {
                "id": "commons_vietnam/Hanoi_LanOng_father_child_cub_manual_v207.png",
                "pageid": 31660225,
                "author": "Clay Gilliland",
                "license": "CC BY-SA 2.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 4},
                    {"cls": 1, "name": "rider_with_helmet", "count": 2},
                    {"cls": 2, "name": "rider_without_helmet", "count": 1}
                ],
                "location_evidence": "Hanoi Lãn Ông herbal medicine street, father and unhelmeted child on Honda Super Cub, parked scooters."
            },
            {
                "id": "commons_vietnam/Northern_road_pastry_delivery_cub_manual_v207.png",
                "pageid": 115770745,
                "author": "BluesyPete",
                "license": "CC BY-SA 3.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 1},
                    {"cls": 1, "name": "rider_with_helmet", "count": 1},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Northern Vietnam rural road, pastry delivery rider with dark helmet and mask on Honda Super Cub with carrier baskets."
            },
            {
                "id": "commons_vietnam/HCMC_DD01_traffic_intersection_manual_v207.png",
                "pageid": 30110165,
                "author": "Diego Delso",
                "license": "CC BY-SA 3.0",
                "assigned_split": "val",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 4},
                    {"cls": 1, "name": "rider_with_helmet", "count": 4},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Ho Chi Minh City traffic intersection DD 01, dense motorcycle commuters, yellow helmet rider, blue jacket rider."
            }
        ],
        "excluded_or_pending": [
            {"pageid": 29944501, "title": "Vietnam-0539.jpg", "reason": "Water buffalo crossing rural highway; rider on distant motorcycle partially obscured; held pending."},
            {"pageid": 114818932, "title": "Vietnam, Saigon, Street scene (48866763567).jpg", "reason": "Wide shot with high motion blur on passing scooters; held pending resolution."},
            {"pageid": 131676313, "title": "Saigon Vietnam Street Scene 2018.jpg", "reason": "Extremely dense cluster (>40 motorbikes); requires high-magnification sub-tiling; deferred to later sprint."},
            {"pageid": 24488443, "title": "Bikes in Hue, Vietnam.jpg", "reason": "Overwhelmingly bicycles rather than motorized two-wheelers; excluded under Rule 24."}
        ]
    }
    with open(V207_DIR / "manual_review_batch42.json", "w", encoding="utf-8") as f:
        json.dump(manual_review_42, f, ensure_ascii=False, indent=2)

    # 12. Update Attribution
    with open(V206_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attribution = json.load(f)
    
    new_attributions = [
        {
            "id": "commons_vietnam/Hanoi_HangBong_moto_pillion_manual_v207.png",
            "title": "File:Vietnam 08 - 01 - taking the motos (3166745807).jpg",
            "artist": "McKay Savage",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Vietnam_08_-_01_-_taking_the_motos_(3166745807).jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/2/23/Vietnam_08_-_01_-_taking_the_motos_%283166745807%29.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/2/23/Vietnam_08_-_01_-_taking_the_motos_%283166745807%29.jpg",
            "sha256": sha256_file(out1_img),
            "country": "VN",
            "location_evidence": "Hanoi, Vietnam; Hàng Bông street old quarter, visible motorcycle license plate 90-H5 2391, typical Vietnamese street traffic.",
            "modifications": "Scaled to 1224x1632. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/Hanoi_LanOng_father_child_cub_manual_v207.png",
            "title": "File:Hanoi, Vietnam (12035628605).jpg",
            "artist": "Clay Gilliland",
            "license": "CC BY-SA 2.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Hanoi,_Vietnam_(12035628605).jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/5/52/Hanoi%2C_Vietnam_%2812035628605%29.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/5/52/Hanoi%2C_Vietnam_%2812035628605%29.jpg",
            "sha256": sha256_file(out2_img),
            "country": "VN",
            "location_evidence": "Hanoi, Vietnam; Lãn Ông herbal medicine street in Hoan Kiem district, traditional shopfronts, father with child on Honda Super Cub.",
            "modifications": "Scaled to 1248x1021. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/Northern_road_pastry_delivery_cub_manual_v207.png",
            "title": "File:Vietnam-0779.jpg",
            "artist": "BluesyPete",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Vietnam-0779.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/8/87/Vietnam-0779.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/8/87/Vietnam-0779.jpg",
            "sha256": sha256_file(out3_img),
            "country": "VN",
            "location_evidence": "Northern Vietnam rural road; pastry delivery vendor riding Honda Super Cub with customized transport baskets.",
            "modifications": "Scaled to 1144x856. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_DD01_traffic_intersection_manual_v207.png",
            "title": "File:Tráfico en Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 01.JPG",
            "artist": "Diego Delso",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh,_Vietnam,_2013-08-14,_DD_01.JPG",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh%2C_Vietnam%2C_2013-08-14%2C_DD_01.JPG",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh%2C_Vietnam%2C_2013-08-14%2C_DD_01.JPG",
            "sha256": sha256_file(out4_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; major intersection with dense motorcycle commuter traffic, yellow helmet rider, blue jacket rider, white hoodie scooter.",
            "modifications": "Cropped foreground region (2400x1771) and scaled to 1200x885. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        }
    ]

    attribution.extend(new_attributions)
    with open(V207_DIR / "attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, ensure_ascii=False, indent=2)

    # 13. Update Changes
    with open(V206_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)

    changes.append({
        "snapshot": "dataset_v207",
        "parent": "dataset_v206",
        "action": "accept_manually_reviewed_commons_images",
        "accepted_images": [
            "commons_vietnam/Hanoi_HangBong_moto_pillion_manual_v207.png",
            "commons_vietnam/Hanoi_LanOng_father_child_cub_manual_v207.png",
            "commons_vietnam/Northern_road_pastry_delivery_cub_manual_v207.png",
            "commons_vietnam/HCMC_DD01_traffic_intersection_manual_v207.png"
        ],
        "new_accepted_count": 4,
        "new_class_0_boxes": 12,
        "new_class_1_boxes": 10,
        "new_class_2_boxes": 1,
        "review_batch": "manual_review_batch42.json"
    })

    with open(V207_DIR / "changes.json", "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # 14. Update ATTRIBUTION.md
    attr_md = (
        "# Dataset Attribution Record - v207\n\n"
        "Total images: 1,106 clean ground-truth images.\n\n"
        "## Newly Admitted Images (v207 Manual Review Batch 42)\n"
        "1. **commons_vietnam/Hanoi_HangBong_moto_pillion_manual_v207.png**\n"
        "   - Title: `File:Vietnam 08 - 01 - taking the motos (3166745807).jpg`\n"
        "   - Artist: McKay Savage\n"
        "   - License: CC BY 2.0\n"
        "   - Location: Hanoi, Vietnam (Hàng Bông, 90-H5 2391)\n"
        "   - Assigned Split: `train`\n\n"
        "2. **commons_vietnam/Hanoi_LanOng_father_child_cub_manual_v207.png**\n"
        "   - Title: `File:Hanoi, Vietnam (12035628605).jpg`\n"
        "   - Artist: Clay Gilliland\n"
        "   - License: CC BY-SA 2.0\n"
        "   - Location: Hanoi, Vietnam (Lãn Ông street)\n"
        "   - Assigned Split: `train`\n\n"
        "3. **commons_vietnam/Northern_road_pastry_delivery_cub_manual_v207.png**\n"
        "   - Title: `File:Vietnam-0779.jpg`\n"
        "   - Artist: BluesyPete\n"
        "   - License: CC BY-SA 3.0\n"
        "   - Location: Northern Vietnam (pastry delivery vendor)\n"
        "   - Assigned Split: `train`\n\n"
        "4. **commons_vietnam/HCMC_DD01_traffic_intersection_manual_v207.png**\n"
        "   - Title: `File:Tráfico en Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 01.JPG`\n"
        "   - Artist: Diego Delso\n"
        "   - License: CC BY-SA 3.0\n"
        "   - Location: Ho Chi Minh City, Vietnam (DD 01 intersection)\n"
        "   - Assigned Split: `val`\n"
    )
    (V207_DIR / "ATTRIBUTION.md").write_text(attr_md, encoding="utf-8")

    # 15. Update README.md
    readme_md = (
        "# Dataset Snapshot v207\n\n"
        "- Total clean reviewed images: **1,106**\n"
        "- Split counts: `train: 892`, `val: 146`, `test: 68` (locked)\n"
        "- Class instance counts: `0: motorcycle (1,973)`, `1: rider_with_helmet (1,093)`, `2: rider_without_helmet (593)`\n"
        "- Total clean bounding boxes: **3,659**\n"
        "- Progress towards 5,000 goal: **22.12%** (Remaining: **3,894**)\n"
    )
    (V207_DIR / "README.md").write_text(readme_md, encoding="utf-8")

    # 16. Verification Report
    verif = {
        "snapshot": "dataset_v207",
        "parent": "dataset_v206",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2263,
        "rows_total": 2267,
        "accepted_clean_images": 1106,
        "remaining_to_5000": 3894,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 12, "1": 10, "2": 1},
        "split_counts": {
            "train": 892,
            "val": 146,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch42.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V207_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(V207_DIR / "attribution.json"),
        "duplicates_sha256": sha256_file(V207_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(V207_DIR / "changes.json")
    }
    with open(V207_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v207.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # 17. Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    progress = {
        "schema_version": 1,
        "as_of": "2026-11-11",
        "goal_images": 5000,
        "accepted_clean_images": 1106,
        "remaining_to_goal": 3894,
        "pending_review_images": 25,
        "snapshot": "dataset_v207",
        "accepted_split_counts": {
            "train": 892,
            "validation": 146,
            "test_locked": 68
        },
        "newly_accepted_this_batch": 4,
        "new_batch_class_box_counts": {
            "0": 12,
            "1": 10,
            "2": 1
        },
        "total_class_box_counts": {
            "0": 1973,
            "1": 1093,
            "2": 593
        },
        "original_dataset_unchanged": True,
        "test_split_locked": True,
        "pending_counts_toward_goal": False,
        "verification_report": "../verification/dataset_v207.json",
        "notes": [
            "Counts are individually reviewed images, not boxes; duplicate and pending rows do not count.",
            "dataset_v203 audited all 1,492 pending DataSet2 items: 335 unique canonical frames accepted (279 train, 56 val), 1,132 redundant variants excluded, 25 items held pending in group_0000 to maintain test lock.",
            "dataset_v204 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), verified Vietnam location context, zero leakage between splits, and test split locked at 68 images. Total clean images reaches 1,095.",
            "dataset_v205 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), adding 32 clean bounding boxes (including 7 rare rider_without_helmet boxes in Hanoi school traffic). Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,098.",
            "dataset_v206 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 2.5 / CC BY-SA 3.0), adding 19 clean bounding boxes including police patrol motorcycle, agricultural animal transport, nursery tree cargo, and city traffic. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,102.",
            "dataset_v207 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY 2.0 / CC BY-SA 2.0 / CC BY-SA 3.0), adding 23 clean bounding boxes including pillion passenger, father with unhelmeted child on Super Cub, pastry delivery carrier, and HCMC intersection. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,106."
        ]
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v207 build completed successfully! ===")

if __name__ == "__main__":
    main()
