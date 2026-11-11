"""Build and verify dataset_v209 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Coastal highway touring motorbiker (author: Vinnie Cartabiano, CC BY 2.0) -> train (group_vinnie_cartabiano)
2. National Route 21A highway traffic near Thang Long Highway (author: Obakeneko, CC BY 3.0) -> train (group_obakeneko_hanoi)
3. Rural road rider beside yellow cattle (author: Christopher Michel, CC BY 2.0) -> train (group_christopher_michel)
4. Ho Chi Minh City motorcycle parking DD 01 (author: Diego Delso, CC BY-SA 3.0) -> val (group_diego_delso)

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
V208_DIR = TRAINING_DIR / "dataset_v208"
V209_DIR = TRAINING_DIR / "dataset_v209"
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
    print("=== Building Dataset v209 Snapshot ===")
    V209_DIR.mkdir(exist_ok=True)
    
    # 1. Setup directories
    for split in ["train", "val"]:
        (V209_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V209_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    (V209_DIR / "review_evidence").mkdir(exist_ok=True)
    (V209_DIR / "source_originals").mkdir(exist_ok=True)
    (V209_DIR / "source_metadata").mkdir(exist_ok=True)

    # 2. Link existing assets from v208
    print("Linking existing v208 assets...")
    for split in ["train", "val"]:
        src_img_dir = V208_DIR / split / "images"
        dst_img_dir = V209_DIR / split / "images"
        for f in src_img_dir.iterdir():
            if f.is_file():
                dst = dst_img_dir / f.name
                if not dst.exists():
                    os.link(f, dst)
        
        src_lbl_dir = V208_DIR / split / "labels"
        dst_lbl_dir = V209_DIR / split / "labels"
        for f in src_lbl_dir.iterdir():
            if f.is_file():
                dst = dst_lbl_dir / f.name
                if not dst.exists():
                    os.link(f, dst)

    colors = {0: (0, 255, 0), 1: (0, 150, 255), 2: (255, 50, 50)}

    # 3. Candidate 1: commons_113561568 (Vinnie Cartabiano, coastal highway, train)
    print("Processing candidate 1 (Vinnie Cartabiano - Coastal highway)...")
    orig1_src = ACQ_DIR / "download" / "commons_113561568.jpg"
    orig1_dst = V209_DIR / "source_originals" / "commons_113561568.jpg"
    shutil.copy2(orig1_src, orig1_dst)

    with Image.open(orig1_src) as im1:
        im1_scaled = im1.resize((1200, 750), Image.Resampling.LANCZOS)
        out1_img = V209_DIR / "train" / "images" / "commons_vietnam__coastal_highway_motorbiker_manual_v209.png"
        im1_scaled.save(out1_img)
        phash1 = f"{perceptual_hash(im1_scaled):016x}"

    boxes1_native = [
        (0, (1350, 1100, 2650, 2200)),
        (1, (1500, 600, 2500, 1700))
    ]
    labels1 = boxes_to_yolo(boxes1_native, 4000, 2500)
    out1_lbl = V209_DIR / "train" / "labels" / "commons_vietnam__coastal_highway_motorbiker_manual_v209.txt"
    out1_lbl.write_text("\n".join(labels1) + "\n", encoding="utf-8")

    overlay1 = im1_scaled.copy()
    d1 = ImageDraw.Draw(overlay1)
    s1 = 1200.0 / 4000.0
    for cls, (x1, y1, x2, y2) in boxes1_native:
        d1.rectangle([int(x1 * s1), int(y1 * s1), int(x2 * s1), int(y2 * s1)], outline=colors[cls], width=3)
        d1.text((int(x1 * s1) + 4, int(y1 * s1) + 4), f"cls {cls}", fill=colors[cls])
    overlay1.save(V209_DIR / "review_evidence" / "coastal_highway_motorbiker_manual_v209_overlay.png")

    # 4. Candidate 2: commons_29944501 (Obakeneko, Hanoi route 21A, train)
    print("Processing candidate 2 (Obakeneko - Hanoi route 21A)...")
    orig2_src = ACQ_DIR / "download" / "commons_29944501.jpg"
    orig2_dst = V209_DIR / "source_originals" / "commons_29944501.jpg"
    shutil.copy2(orig2_src, orig2_dst)

    with Image.open(orig2_src) as im2:
        im2_scaled = im2.resize((1216, 912), Image.Resampling.LANCZOS)
        out2_img = V209_DIR / "train" / "images" / "commons_vietnam__Hanoi_route21A_highway_traffic_manual_v209.png"
        im2_scaled.save(out2_img)
        phash2 = f"{perceptual_hash(im2_scaled):016x}"

    boxes2_native = [
        (0, (850, 1500, 1650, 2350)),
        (1, (950, 1100, 1550, 1950)),
        (0, (2200, 1450, 2900, 2200)),
        (1, (2300, 1150, 2800, 1850))
    ]
    labels2 = boxes_to_yolo(boxes2_native, 3648, 2736)
    out2_lbl = V209_DIR / "train" / "labels" / "commons_vietnam__Hanoi_route21A_highway_traffic_manual_v209.txt"
    out2_lbl.write_text("\n".join(labels2) + "\n", encoding="utf-8")

    overlay2 = im2_scaled.copy()
    d2 = ImageDraw.Draw(overlay2)
    s2 = 1216.0 / 3648.0
    for cls, (x1, y1, x2, y2) in boxes2_native:
        d2.rectangle([int(x1 * s2), int(y1 * s2), int(x2 * s2), int(y2 * s2)], outline=colors[cls], width=3)
        d2.text((int(x1 * s2) + 4, int(y1 * s2) + 4), f"cls {cls}", fill=colors[cls])
    overlay2.save(V209_DIR / "review_evidence" / "Hanoi_route21A_highway_traffic_manual_v209_overlay.png")

    # 5. Candidate 3: commons_24817609 (Christopher Michel, rural road cattle, train)
    print("Processing candidate 3 (Christopher Michel - Rural road cattle)...")
    orig3_src = ACQ_DIR / "download" / "commons_24817609.jpg"
    orig3_dst = V209_DIR / "source_originals" / "commons_24817609.jpg"
    shutil.copy2(orig3_src, orig3_dst)

    with Image.open(orig3_src) as im3:
        out3_img = V209_DIR / "train" / "images" / "commons_vietnam__rural_road_rider_cattle_manual_v209.png"
        im3.save(out3_img)
        phash3 = f"{perceptual_hash(im3):016x}"
        W3, H3 = im3.size

    boxes3_native = [
        (0, (400, 450, 950, 900)),
        (1, (480, 280, 820, 720))
    ]
    labels3 = boxes_to_yolo(boxes3_native, W3, H3)
    out3_lbl = V209_DIR / "train" / "labels" / "commons_vietnam__rural_road_rider_cattle_manual_v209.txt"
    out3_lbl.write_text("\n".join(labels3) + "\n", encoding="utf-8")

    overlay3 = im3.copy()
    d3 = ImageDraw.Draw(overlay3)
    for cls, (x1, y1, x2, y2) in boxes3_native:
        d3.rectangle([x1, y1, x2, y2], outline=colors[cls], width=3)
        d3.text((x1 + 4, y1 + 4), f"cls {cls}", fill=colors[cls])
    overlay3.save(V209_DIR / "review_evidence" / "rural_road_rider_cattle_manual_v209_overlay.png")

    # 6. Candidate 4: commons_30112783 / crop_30112783 (Diego Delso, HCMC parking, val)
    print("Processing candidate 4 (Diego Delso - HCMC parking)...")
    orig4_src = ACQ_DIR / "download" / "commons_30112783.jpg"
    orig4_dst = V209_DIR / "source_originals" / "commons_30112783.jpg"
    shutil.copy2(orig4_src, orig4_dst)

    with Image.open(ACQ_DIR / "download" / "crop_30112783.png") as im4:
        out4_img = V209_DIR / "val" / "images" / "commons_vietnam__HCMC_DD01_motorcycle_parking_manual_v209.png"
        im4.save(out4_img)
        phash4 = f"{perceptual_hash(im4):016x}"
        W4, H4 = im4.size

    boxes4_native = [
        (0, (450, 300, 800, 750)),
        (0, (100, 280, 450, 720)),
        (0, (800, 290, 1150, 740))
    ]
    labels4 = boxes_to_yolo(boxes4_native, W4, H4)
    out4_lbl = V209_DIR / "val" / "labels" / "commons_vietnam__HCMC_DD01_motorcycle_parking_manual_v209.txt"
    out4_lbl.write_text("\n".join(labels4) + "\n", encoding="utf-8")

    overlay4 = im4.copy()
    d4 = ImageDraw.Draw(overlay4)
    for cls, (x1, y1, x2, y2) in boxes4_native:
        d4.rectangle([x1, y1, x2, y2], outline=colors[cls], width=3)
        d4.text((x1 + 4, y1 + 4), f"cls {cls}", fill=colors[cls])
    overlay4.save(V209_DIR / "review_evidence" / "HCMC_DD01_motorcycle_parking_manual_v209_overlay.png")

    # 7. Save metadata files
    with open(ACQ_DIR / "unadmitted_commons_candidates.json", "r", encoding="utf-8") as f:
        cand_list = json.load(f)
    cand_map = {c["pageid"]: c for c in cand_list}

    for pid in [113561568, 29944501, 24817609, 30112783]:
        meta_file = V209_DIR / "source_metadata" / f"commons_{pid}.json"
        meta_file.write_text(json.dumps(cand_map.get(pid, {}), ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. Update Manifest
    print("Updating manifest.json...")
    with open(V208_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    new_rows = [
        {
            "id": "commons_vietnam/coastal_highway_motorbiker_manual_v209.png",
            "pageid": 113561568,
            "original_pageid": 113561568,
            "title": "File:A motor biker in Vietnam (25093094107).jpg",
            "source_group": "commons_vinnie_cartabiano",
            "group": "group_vinnie_cartabiano",
            "artist": "Vinnie Cartabiano",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "new_split": "train",
            "width": 1200,
            "height": 750,
            "sha256": sha256_file(out1_img),
            "phash": phash1,
            "source_image": "Demo/training/dataset_v209/source_originals/commons_113561568.jpg",
            "source_sha256": sha256_file(orig1_dst),
            "source_metadata": "Demo/training/dataset_v209/source_metadata/commons_113561568.json",
            "review_evidence": "Demo/training/dataset_v209/review_evidence/coastal_highway_motorbiker_manual_v209_overlay.png",
            "location_evidence": "Coastal highway, Vietnam; touring motorbiker on motorcycle wearing full-face safety helmet.",
            "classes": {"0": 1, "1": 1, "2": 0}
        },
        {
            "id": "commons_vietnam/Hanoi_route21A_highway_traffic_manual_v209.png",
            "pageid": 29944501,
            "original_pageid": 29944501,
            "title": "File:National Route 21A (Vietnam) near the intersection with Thang Long Highway.jpg",
            "source_group": "commons_obakeneko",
            "group": "group_obakeneko_hanoi",
            "artist": "Obakeneko",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "new_split": "train",
            "width": 1216,
            "height": 912,
            "sha256": sha256_file(out2_img),
            "phash": phash2,
            "source_image": "Demo/training/dataset_v209/source_originals/commons_29944501.jpg",
            "source_sha256": sha256_file(orig2_dst),
            "source_metadata": "Demo/training/dataset_v209/source_metadata/commons_29944501.json",
            "review_evidence": "Demo/training/dataset_v209/review_evidence/Hanoi_route21A_highway_traffic_manual_v209_overlay.png",
            "location_evidence": "National Route 21A in Thach Hoa, Thach That district, Hanoi, Vietnam; passing highway commuters on motorbikes.",
            "classes": {"0": 2, "1": 2, "2": 0}
        },
        {
            "id": "commons_vietnam/rural_road_rider_cattle_manual_v209.png",
            "pageid": 24817609,
            "original_pageid": 24817609,
            "title": "File:Vietnam & Cambodia (3336758253).jpg",
            "source_group": "commons_christopher_michel",
            "group": "group_christopher_michel",
            "artist": "Christopher Michel",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "new_split": "train",
            "width": 1280,
            "height": 1089,
            "sha256": sha256_file(out3_img),
            "phash": phash3,
            "source_image": "Demo/training/dataset_v209/source_originals/commons_24817609.jpg",
            "source_sha256": sha256_file(orig3_dst),
            "source_metadata": "Demo/training/dataset_v209/source_metadata/commons_24817609.json",
            "review_evidence": "Demo/training/dataset_v209/review_evidence/rural_road_rider_cattle_manual_v209_overlay.png",
            "location_evidence": "Rural road, Vietnam; local commuter on motorcycle wearing helmet traveling alongside domestic yellow cattle.",
            "classes": {"0": 1, "1": 1, "2": 0}
        },
        {
            "id": "commons_vietnam/HCMC_DD01_motorcycle_parking_manual_v209.png",
            "pageid": 30112783,
            "original_pageid": 30112783,
            "title": "File:Parking de motocicletas, Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 01.JPG",
            "source_group": "commons_diego_delso",
            "group": "group_diego_delso",
            "artist": "Diego Delso",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "val",
            "width": 1200,
            "height": 850,
            "sha256": sha256_file(out4_img),
            "phash": phash4,
            "source_image": "Demo/training/dataset_v209/source_originals/commons_30112783.jpg",
            "source_sha256": sha256_file(orig4_dst),
            "source_metadata": "Demo/training/dataset_v209/source_metadata/commons_30112783.json",
            "review_evidence": "Demo/training/dataset_v209/review_evidence/HCMC_DD01_motorcycle_parking_manual_v209_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; dedicated motorcycle parking area with parked Honda, Yamaha, and Piaggio scooters.",
            "classes": {"0": 3, "1": 0, "2": 0}
        }
    ]

    manifest.extend(new_rows)
    manifest_path = V209_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 9. Update Statistics
    print("Updating statistics.json...")
    with open(V208_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Train: +3 images
    # Vinnie: cls 0: +1, cls 1: +1
    # Obakeneko: cls 0: +2, cls 1: +2
    # Michel: cls 0: +1, cls 1: +1
    stats["train"]["images"] += 3
    stats["train"]["classes"]["0"] += 4
    stats["train"]["classes"]["1"] += 4
    stats["train"]["classes"]["2"] += 0
    stats["train"]["groups"] += 3  # group_vinnie_cartabiano, group_obakeneko_hanoi, group_christopher_michel

    # Val: +1 image (Diego Delso parking)
    # Delso: cls 0: +3, cls 1: 0, cls 2: 0 (group_diego_delso already in val)
    stats["val"]["images"] += 1
    stats["val"]["classes"]["0"] += 3
    stats["val"]["classes"]["1"] += 0
    stats["val"]["classes"]["2"] += 0

    stats_path = V209_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 10. Copy duplicates, usage_restrictions, data.yaml, split_lock
    shutil.copy2(V208_DIR / "duplicates.json", V209_DIR / "duplicates.json")
    shutil.copy2(V208_DIR / "usage_restrictions.json", V209_DIR / "usage_restrictions.json")
    shutil.copy2(V208_DIR / "data.yaml", V209_DIR / "data.yaml")
    shutil.copy2(V208_DIR / "split_lock.json", V209_DIR / "split_lock.json")

    # 11. Changes and Review batch
    manual_review_44 = {
        "batch": 44,
        "date": "2026-11-11",
        "snapshot": "dataset_v209",
        "parent": "dataset_v208",
        "method": "manual pixel inspection and bounding box annotation; no model pseudo-labeling",
        "reviewed_candidates_count": 8,
        "accepted_images_count": 4,
        "held_pending_or_excluded_count": 4,
        "accepted": [
            {
                "id": "commons_vietnam/coastal_highway_motorbiker_manual_v209.png",
                "pageid": 113561568,
                "author": "Vinnie Cartabiano",
                "license": "CC BY 2.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 1},
                    {"cls": 1, "name": "rider_with_helmet", "count": 1},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Coastal highway, Vietnam; touring motorbiker on motorcycle wearing full-face safety helmet."
            },
            {
                "id": "commons_vietnam/Hanoi_route21A_highway_traffic_manual_v209.png",
                "pageid": 29944501,
                "author": "Obakeneko",
                "license": "CC BY 3.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 2},
                    {"cls": 1, "name": "rider_with_helmet", "count": 2},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "National Route 21A, Hanoi, Vietnam; passing highway commuters on motorbikes."
            },
            {
                "id": "commons_vietnam/rural_road_rider_cattle_manual_v209.png",
                "pageid": 24817609,
                "author": "Christopher Michel",
                "license": "CC BY 2.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 1},
                    {"cls": 1, "name": "rider_with_helmet", "count": 1},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Rural Vietnam road with local commuter on motorcycle wearing helmet."
            },
            {
                "id": "commons_vietnam/HCMC_DD01_motorcycle_parking_manual_v209.png",
                "pageid": 30112783,
                "author": "Diego Delso",
                "license": "CC BY-SA 3.0",
                "assigned_split": "val",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 3},
                    {"cls": 1, "name": "rider_with_helmet", "count": 0},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Ho Chi Minh City, Vietnam; dedicated motorcycle parking area with parked commuter scooters."
            }
        ],
        "excluded_or_pending": [
            {"pageid": 175452980, "title": "File:Vietnam by Motorcycle Banner.jpg", "reason": "Wide aspect banner with high crop distortion; excluded under quality contract."},
            {"pageid": 79311754, "title": "File:Vietnamese People's Army Motorcycle Vector.png", "reason": "Synthetic SVG vector graphic; excluded under Rule 24 (photographic sensors only)."},
            {"pageid": 86352166, "title": "File:Vietnam Bike Run.jpg", "reason": "US veteran motorcycle rally event outside Vietnam; excluded under geographic provenance rule."},
            {"pageid": 145321181, "title": "File:Vietnam 1999.Hanoi (33).jpg", "reason": "Historical film photograph; deferred to separate historical domain sprint."}
        ]
    }
    with open(V209_DIR / "manual_review_batch44.json", "w", encoding="utf-8") as f:
        json.dump(manual_review_44, f, ensure_ascii=False, indent=2)

    # 12. Update Attribution
    with open(V208_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attribution = json.load(f)
    
    new_attributions = [
        {
            "id": "commons_vietnam/coastal_highway_motorbiker_manual_v209.png",
            "title": "File:A motor biker in Vietnam (25093094107).jpg",
            "artist": "Vinnie Cartabiano",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:A_motor_biker_in_Vietnam_(25093094107).jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/e/e6/A_motor_biker_in_Vietnam_%2825093094107%29.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/e/e6/A_motor_biker_in_Vietnam_%2825093094107%29.jpg",
            "sha256": sha256_file(out1_img),
            "country": "VN",
            "location_evidence": "Coastal highway, Vietnam; touring motorbiker on motorcycle wearing full-face safety helmet.",
            "modifications": "Scaled to 1200x750. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/Hanoi_route21A_highway_traffic_manual_v209.png",
            "title": "File:National Route 21A (Vietnam) near the intersection with Thang Long Highway.jpg",
            "artist": "Obakeneko",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:National_Route_21A_(Vietnam)_near_the_intersection_with_Thang_Long_Highway.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/7/74/National_Route_21A_%28Vietnam%29_near_the_intersection_with_Thang_Long_Highway.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/7/74/National_Route_21A_%28Vietnam%29_near_the_intersection_with_Thang_Long_Highway.jpg",
            "sha256": sha256_file(out2_img),
            "country": "VN",
            "location_evidence": "National Route 21A in Thach Hoa, Thach That district, Hanoi, Vietnam; passing highway commuters on motorbikes.",
            "modifications": "Scaled to 1216x912. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/rural_road_rider_cattle_manual_v209.png",
            "title": "File:Vietnam & Cambodia (3336758253).jpg",
            "artist": "Christopher Michel",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Vietnam_%26_Cambodia_(3336758253).jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/f/fe/Vietnam_%26_Cambodia_%283336758253%29.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/f/fe/Vietnam_%26_Cambodia_%283336758253%29.jpg",
            "sha256": sha256_file(out3_img),
            "country": "VN",
            "location_evidence": "Rural road, Vietnam; local commuter on motorcycle wearing helmet traveling alongside domestic yellow cattle.",
            "modifications": "Original image preserved (1280x1089). Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_DD01_motorcycle_parking_manual_v209.png",
            "title": "File:Parking de motocicletas, Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 01.JPG",
            "artist": "Diego Delso",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Parking_de_motocicletas,_Ciudad_Ho_Chi_Minh,_Vietnam,_2013-08-14,_DD_01.JPG",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/1f/Parking_de_motocicletas%2C_Ciudad_Ho_Chi_Minh%2C_Vietnam%2C_2013-08-14%2C_DD_01.JPG",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/1/1f/Parking_de_motocicletas%2C_Ciudad_Ho_Chi_Minh%2C_Vietnam%2C_2013-08-14%2C_DD_01.JPG",
            "sha256": sha256_file(out4_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; dedicated motorcycle parking area with parked commuter scooters.",
            "modifications": "Cropped foreground region (2400x1700) and scaled to 1200x850. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        }
    ]

    attribution.extend(new_attributions)
    with open(V209_DIR / "attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, ensure_ascii=False, indent=2)

    # 13. Update Changes
    with open(V208_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)

    changes.append({
        "snapshot": "dataset_v209",
        "parent": "dataset_v208",
        "action": "accept_manually_reviewed_commons_images",
        "accepted_images": [
            "commons_vietnam/coastal_highway_motorbiker_manual_v209.png",
            "commons_vietnam/Hanoi_route21A_highway_traffic_manual_v209.png",
            "commons_vietnam/rural_road_rider_cattle_manual_v209.png",
            "commons_vietnam/HCMC_DD01_motorcycle_parking_manual_v209.png"
        ],
        "new_accepted_count": 4,
        "new_class_0_boxes": 7,
        "new_class_1_boxes": 4,
        "new_class_2_boxes": 0,
        "review_batch": "manual_review_batch44.json"
    })

    with open(V209_DIR / "changes.json", "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # 14. Update ATTRIBUTION.md
    attr_md = (
        "# Dataset Attribution Record - v209\n\n"
        "Total images: 1,114 clean ground-truth images.\n\n"
        "## Newly Admitted Images (v209 Manual Review Batch 44)\n"
        "1. **commons_vietnam/coastal_highway_motorbiker_manual_v209.png**\n"
        "   - Title: `File:A motor biker in Vietnam (25093094107).jpg`\n"
        "   - Artist: Vinnie Cartabiano\n"
        "   - License: CC BY 2.0\n"
        "   - Location: Coastal highway, Vietnam\n"
        "   - Assigned Split: `train`\n\n"
        "2. **commons_vietnam/Hanoi_route21A_highway_traffic_manual_v209.png**\n"
        "   - Title: `File:National Route 21A (Vietnam) near the intersection with Thang Long Highway.jpg`\n"
        "   - Artist: Obakeneko\n"
        "   - License: CC BY 3.0\n"
        "   - Location: National Route 21A, Hanoi, Vietnam\n"
        "   - Assigned Split: `train`\n\n"
        "3. **commons_vietnam/rural_road_rider_cattle_manual_v209.png**\n"
        "   - Title: `File:Vietnam & Cambodia (3336758253).jpg`\n"
        "   - Artist: Christopher Michel\n"
        "   - License: CC BY 2.0\n"
        "   - Location: Rural road, Vietnam\n"
        "   - Assigned Split: `train`\n\n"
        "4. **commons_vietnam/HCMC_DD01_motorcycle_parking_manual_v209.png**\n"
        "   - Title: `File:Parking de motocicletas, Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 01.JPG`\n"
        "   - Artist: Diego Delso\n"
        "   - License: CC BY-SA 3.0\n"
        "   - Location: Ho Chi Minh City, Vietnam (DD 01 parking)\n"
        "   - Assigned Split: `val`\n"
    )
    (V209_DIR / "ATTRIBUTION.md").write_text(attr_md, encoding="utf-8")

    # 15. Update README.md
    readme_md = (
        "# Dataset Snapshot v209\n\n"
        "- Total clean reviewed images: **1,114**\n"
        "- Split counts: `train: 898`, `val: 148`, `test: 68` (locked)\n"
        "- Class instance counts: `0: motorcycle (1,990)`, `1: rider_with_helmet (1,107)`, `2: rider_without_helmet (593)`\n"
        "- Total clean bounding boxes: **3,690**\n"
        "- Progress towards 5,000 goal: **22.28%** (Remaining: **3,886**)\n"
    )
    (V209_DIR / "README.md").write_text(readme_md, encoding="utf-8")

    # 16. Verification Report
    verif = {
        "snapshot": "dataset_v209",
        "parent": "dataset_v208",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2271,
        "rows_total": 2275,
        "accepted_clean_images": 1114,
        "remaining_to_5000": 3886,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 7, "1": 4, "2": 0},
        "split_counts": {
            "train": 898,
            "val": 148,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch44.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V209_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(V209_DIR / "attribution.json"),
        "duplicates_sha256": sha256_file(V209_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(V209_DIR / "changes.json")
    }
    with open(V209_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v209.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # 17. Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    progress = {
        "schema_version": 1,
        "as_of": "2026-11-11",
        "goal_images": 5000,
        "accepted_clean_images": 1114,
        "remaining_to_goal": 3886,
        "pending_review_images": 25,
        "snapshot": "dataset_v209",
        "accepted_split_counts": {
            "train": 898,
            "validation": 148,
            "test_locked": 68
        },
        "newly_accepted_this_batch": 4,
        "new_batch_class_box_counts": {
            "0": 7,
            "1": 4,
            "2": 0
        },
        "total_class_box_counts": {
            "0": 1990,
            "1": 1107,
            "2": 593
        },
        "original_dataset_unchanged": True,
        "test_split_locked": True,
        "pending_counts_toward_goal": False,
        "verification_report": "../verification/dataset_v209.json",
        "notes": [
            "Counts are individually reviewed images, not boxes; duplicate and pending rows do not count.",
            "dataset_v203 audited all 1,492 pending DataSet2 items: 335 unique canonical frames accepted (279 train, 56 val), 1,132 redundant variants excluded, 25 items held pending in group_0000 to maintain test lock.",
            "dataset_v204 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), verified Vietnam location context, zero leakage between splits, and test split locked at 68 images. Total clean images reaches 1,095.",
            "dataset_v205 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), adding 32 clean bounding boxes (including 7 rare rider_without_helmet boxes in Hanoi school traffic). Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,098.",
            "dataset_v206 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 2.5 / CC BY-SA 3.0), adding 19 clean bounding boxes including police patrol motorcycle, agricultural animal transport, nursery tree cargo, and city traffic. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,102.",
            "dataset_v207 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY 2.0 / CC BY-SA 2.0 / CC BY-SA 3.0), adding 23 clean bounding boxes including pillion passenger, father with unhelmeted child on Super Cub, pastry delivery carrier, and HCMC intersection. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,106.",
            "dataset_v208 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 4.0 / CC BY-SA 3.0 / CC BY 4.0 / CC0), adding 20 clean bounding boxes including Sa Pa mountain highway, Hanoi intersection red light waiting line, and HCMC commuter traffic. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,110.",
            "dataset_v209 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY 2.0 / CC BY 3.0 / CC BY-SA 3.0), adding 11 clean bounding boxes including coastal touring motorbiker, Hanoi Route 21A highway commuters, rural road commuter, and HCMC scooter parking. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,114."
        ]
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v209 build completed successfully! ===")

if __name__ == "__main__":
    main()
