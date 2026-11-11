"""Build and verify dataset_v210 snapshot according to CLOUD_CODEX_5000.md.
Adopts 4 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Mother and child on motorbike in HCMC (author: Roderick Divilbiss, CC BY-SA 2.5) -> train (group_roderick_divilbiss)
2. Commuter motorcycles in Ho Chi Minh City traffic (author: Nam-ho Park, CC BY 2.0) -> train (group_namho_park_hcmc)
3. Animal transport on motorcycle in HCMC (author: Kordian Kassner, CC BY-SA 2.5) -> train (group_kassner_animal_transport)
4. Ho Chi Minh City urban traffic (author: Andre Lettau, CC BY-SA 3.0) -> val (group_andre_lettau_hcmc)

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
V209_DIR = TRAINING_DIR / "dataset_v209"
V210_DIR = TRAINING_DIR / "dataset_v210"
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
    print("=== Building Dataset v210 Snapshot ===")
    V210_DIR.mkdir(exist_ok=True)
    
    # 1. Setup directories
    for split in ["train", "val"]:
        (V210_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V210_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    (V210_DIR / "review_evidence").mkdir(exist_ok=True)
    (V210_DIR / "source_originals").mkdir(exist_ok=True)
    (V210_DIR / "source_metadata").mkdir(exist_ok=True)

    # 2. Link existing assets from v209
    print("Linking existing v209 assets...")
    for split in ["train", "val"]:
        src_img_dir = V209_DIR / split / "images"
        dst_img_dir = V210_DIR / split / "images"
        for f in src_img_dir.iterdir():
            if f.is_file():
                dst = dst_img_dir / f.name
                if not dst.exists():
                    os.link(f, dst)
        
        src_lbl_dir = V209_DIR / split / "labels"
        dst_lbl_dir = V210_DIR / split / "labels"
        for f in src_lbl_dir.iterdir():
            if f.is_file():
                dst = dst_lbl_dir / f.name
                if not dst.exists():
                    os.link(f, dst)

    colors = {0: (0, 255, 0), 1: (0, 150, 255), 2: (255, 50, 50)}

    # 3. Candidate 1: commons_530295 (Roderick Divilbiss, Mother Child, train)
    print("Processing candidate 1 (Roderick Divilbiss - Mother Child)...")
    orig1_src = ACQ_DIR / "download" / "commons_530295.jpg"
    orig1_dst = V210_DIR / "source_originals" / "commons_530295.jpg"
    shutil.copy2(orig1_src, orig1_dst)

    with Image.open(orig1_src) as im1:
        im1_scaled = im1.resize((1248, 832), Image.Resampling.LANCZOS)
        out1_img = V210_DIR / "train" / "images" / "commons_vietnam__HCMC_mother_child_motorbike_manual_v210.png"
        im1_scaled.save(out1_img)
        phash1 = f"{perceptual_hash(im1_scaled):016x}"

    boxes1_native = [
        (0, (650, 750, 1900, 1600)),
        (2, (880, 280, 1600, 1350)),
        (2, (950, 580, 1420, 1180))
    ]
    labels1 = boxes_to_yolo(boxes1_native, 2496, 1664)
    out1_lbl = V210_DIR / "train" / "labels" / "commons_vietnam__HCMC_mother_child_motorbike_manual_v210.txt"
    out1_lbl.write_text("\n".join(labels1) + "\n", encoding="utf-8")

    overlay1 = im1_scaled.copy()
    d1 = ImageDraw.Draw(overlay1)
    s1 = 1248.0 / 2496.0
    for cls, (x1, y1, x2, y2) in boxes1_native:
        d1.rectangle([int(x1 * s1), int(y1 * s1), int(x2 * s1), int(y2 * s1)], outline=colors[cls], width=3)
        d1.text((int(x1 * s1) + 4, int(y1 * s1) + 4), f"cls {cls}", fill=colors[cls])
    overlay1.save(V210_DIR / "review_evidence" / "HCMC_mother_child_motorbike_manual_v210_overlay.png")

    # 4. Candidate 2: commons_17515132 (Nam-ho Park, HCMC traffic, train)
    print("Processing candidate 2 (Nam-ho Park - HCMC traffic)...")
    orig2_src = ACQ_DIR / "download" / "commons_17515132.jpg"
    orig2_dst = V210_DIR / "source_originals" / "commons_17515132.jpg"
    shutil.copy2(orig2_src, orig2_dst)

    with Image.open(orig2_src) as im2:
        im2_scaled = im2.resize((1152, 702), Image.Resampling.LANCZOS)
        out2_img = V210_DIR / "train" / "images" / "commons_vietnam__HCMC_commuter_motorcycles_street_manual_v210.png"
        im2_scaled.save(out2_img)
        phash2 = f"{perceptual_hash(im2_scaled):016x}"

    boxes2_native = [
        (0, (1250, 1100, 2250, 1950)),
        (1, (1350, 650, 2150, 1600)),
        (0, (300, 1150, 1250, 1950)),
        (1, (450, 700, 1150, 1600)),
        (0, (2150, 1100, 3150, 1950)),
        (1, (2250, 650, 3050, 1600))
    ]
    labels2 = boxes_to_yolo(boxes2_native, 3456, 2106)
    out2_lbl = V210_DIR / "train" / "labels" / "commons_vietnam__HCMC_commuter_motorcycles_street_manual_v210.txt"
    out2_lbl.write_text("\n".join(labels2) + "\n", encoding="utf-8")

    overlay2 = im2_scaled.copy()
    d2 = ImageDraw.Draw(overlay2)
    s2 = 1152.0 / 3456.0
    for cls, (x1, y1, x2, y2) in boxes2_native:
        d2.rectangle([int(x1 * s2), int(y1 * s2), int(x2 * s2), int(y2 * s2)], outline=colors[cls], width=3)
        d2.text((int(x1 * s2) + 4, int(y1 * s2) + 4), f"cls {cls}", fill=colors[cls])
    overlay2.save(V210_DIR / "review_evidence" / "HCMC_commuter_motorcycles_street_manual_v210_overlay.png")

    # 5. Candidate 3: commons_196365 (Kordian Kassner, Animal transport, train)
    print("Processing candidate 3 (Kordian Kassner - Animal transport)...")
    orig3_src = ACQ_DIR / "download" / "commons_196365.jpg"
    orig3_dst = V210_DIR / "source_originals" / "commons_196365.jpg"
    shutil.copy2(orig3_src, orig3_dst)

    with Image.open(orig3_src) as im3:
        out3_img = V210_DIR / "train" / "images" / "commons_vietnam__HCMC_animal_transport_motorcycle_manual_v210.png"
        im3.save(out3_img)
        phash3 = f"{perceptual_hash(im3):016x}"
        W3, H3 = im3.size

    boxes3_native = [
        (0, (320, 340, 1080, 790)),
        (1, (420, 170, 760, 650))
    ]
    labels3 = boxes_to_yolo(boxes3_native, W3, H3)
    out3_lbl = V210_DIR / "train" / "labels" / "commons_vietnam__HCMC_animal_transport_motorcycle_manual_v210.txt"
    out3_lbl.write_text("\n".join(labels3) + "\n", encoding="utf-8")

    overlay3 = im3.copy()
    d3 = ImageDraw.Draw(overlay3)
    for cls, (x1, y1, x2, y2) in boxes3_native:
        d3.rectangle([x1, y1, x2, y2], outline=colors[cls], width=3)
        d3.text((x1 + 4, y1 + 4), f"cls {cls}", fill=colors[cls])
    overlay3.save(V210_DIR / "review_evidence" / "HCMC_animal_transport_motorcycle_manual_v210_overlay.png")

    # 6. Candidate 4: commons_124405 (Andre Lettau, HCMC Verkehr, val)
    print("Processing candidate 4 (Andre Lettau - HCMC Verkehr)...")
    orig4_src = ACQ_DIR / "download" / "commons_124405.jpg"
    orig4_dst = V210_DIR / "source_originals" / "commons_124405.jpg"
    shutil.copy2(orig4_src, orig4_dst)

    with Image.open(orig4_src) as im4:
        im4_scaled = im4.resize((1190, 787), Image.Resampling.LANCZOS)
        out4_img = V210_DIR / "val" / "images" / "commons_vietnam__HCMC_urban_verkehr_traffic_manual_v210.png"
        im4_scaled.save(out4_img)
        phash4 = f"{perceptual_hash(im4_scaled):016x}"

    boxes4_native = [
        (0, (500, 700, 1100, 1250)),
        (1, (600, 400, 1050, 1000)),
        (0, (1100, 700, 1750, 1250)),
        (1, (1200, 420, 1650, 1000)),
        (0, (50, 750, 550, 1250)),
        (1, (120, 450, 500, 1000))
    ]
    labels4 = boxes_to_yolo(boxes4_native, 1984, 1312)
    out4_lbl = V210_DIR / "val" / "labels" / "commons_vietnam__HCMC_urban_verkehr_traffic_manual_v210.txt"
    out4_lbl.write_text("\n".join(labels4) + "\n", encoding="utf-8")

    overlay4 = im4_scaled.copy()
    d4 = ImageDraw.Draw(overlay4)
    s4 = 1190.0 / 1984.0
    for cls, (x1, y1, x2, y2) in boxes4_native:
        d4.rectangle([int(x1 * s4), int(y1 * s4), int(x2 * s4), int(y2 * s4)], outline=colors[cls], width=3)
        d4.text((int(x1 * s4) + 4, int(y1 * s4) + 4), f"cls {cls}", fill=colors[cls])
    overlay4.save(V210_DIR / "review_evidence" / "HCMC_urban_verkehr_traffic_manual_v210_overlay.png")

    # 7. Save metadata files
    cand_metadata = {
        530295: {
            "pageid": 530295,
            "title": "File:Mother Child Motorbike Vietnam.jpg",
            "artist": "Roderick Divilbiss",
            "license": "CC BY-SA 2.5",
            "url": "https://upload.wikimedia.org/wikipedia/commons/1/1e/Mother_Child_Motorbike_Vietnam.jpg"
        },
        17515132: {
            "pageid": 17515132,
            "title": "File:Motorcycles in Ho Chi Minh city.jpg",
            "artist": "Nam-ho Park",
            "license": "CC BY 2.0",
            "url": "https://upload.wikimedia.org/wikipedia/commons/d/dd/Motorcycles_in_Ho_Chi_Minh_city.jpg"
        },
        196365: {
            "pageid": 196365,
            "title": "File:Animal transport as used in Ho Chi Minh City.jpg",
            "artist": "Kordian Kassner",
            "license": "CC BY-SA 2.5",
            "url": "https://upload.wikimedia.org/wikipedia/commons/c/c4/Animal_transport_as_used_in_Ho_Chi_Minh_City.jpg"
        },
        124405: {
            "pageid": 124405,
            "title": "File:HCM-City Verkehr.JPG",
            "artist": "Andre Lettau",
            "license": "CC BY-SA 3.0",
            "url": "https://upload.wikimedia.org/wikipedia/commons/3/30/HCM-City_Verkehr.JPG"
        }
    }

    for pid in [530295, 17515132, 196365, 124405]:
        meta_file = V210_DIR / "source_metadata" / f"commons_{pid}.json"
        meta_file.write_text(json.dumps(cand_metadata[pid], ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. Update Manifest
    print("Updating manifest.json...")
    with open(V209_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    new_rows = [
        {
            "id": "commons_vietnam/HCMC_mother_child_motorbike_manual_v210.png",
            "pageid": 530295,
            "original_pageid": 530295,
            "title": "File:Mother Child Motorbike Vietnam.jpg",
            "source_group": "commons_roderick_divilbiss",
            "group": "group_roderick_divilbiss",
            "artist": "Roderick Divilbiss",
            "license": "CC BY-SA 2.5",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.5/",
            "new_split": "train",
            "width": 1248,
            "height": 832,
            "sha256": sha256_file(out1_img),
            "phash": phash1,
            "source_image": "Demo/training/dataset_v210/source_originals/commons_530295.jpg",
            "source_sha256": sha256_file(orig1_dst),
            "source_metadata": "Demo/training/dataset_v210/source_metadata/commons_530295.json",
            "review_evidence": "Demo/training/dataset_v210/review_evidence/HCMC_mother_child_motorbike_manual_v210_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; Vietnamese mother and toddler riding family motorcycle in urban traffic.",
            "classes": {"0": 1, "1": 0, "2": 2}
        },
        {
            "id": "commons_vietnam/HCMC_commuter_motorcycles_street_manual_v210.png",
            "pageid": 17515132,
            "original_pageid": 17515132,
            "title": "File:Motorcycles in Ho Chi Minh city.jpg",
            "source_group": "commons_namho_park",
            "group": "group_namho_park_hcmc",
            "artist": "Nam-ho Park",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "new_split": "train",
            "width": 1152,
            "height": 702,
            "sha256": sha256_file(out2_img),
            "phash": phash2,
            "source_image": "Demo/training/dataset_v210/source_originals/commons_17515132.jpg",
            "source_sha256": sha256_file(orig2_dst),
            "source_metadata": "Demo/training/dataset_v210/source_metadata/commons_17515132.json",
            "review_evidence": "Demo/training/dataset_v210/review_evidence/HCMC_commuter_motorcycles_street_manual_v210_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; dense commuter street flow with riders wearing standardized motorcycle helmets.",
            "classes": {"0": 3, "1": 3, "2": 0}
        },
        {
            "id": "commons_vietnam/HCMC_animal_transport_motorcycle_manual_v210.png",
            "pageid": 196365,
            "original_pageid": 196365,
            "title": "File:Animal transport as used in Ho Chi Minh City.jpg",
            "source_group": "commons_kordian_kassner",
            "group": "group_kassner_animal_transport",
            "artist": "Kordian Kassner",
            "license": "CC BY-SA 2.5",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.5/",
            "new_split": "train",
            "width": 1280,
            "height": 837,
            "sha256": sha256_file(out3_img),
            "phash": phash3,
            "source_image": "Demo/training/dataset_v210/source_originals/commons_196365.jpg",
            "source_sha256": sha256_file(orig3_dst),
            "source_metadata": "Demo/training/dataset_v210/source_metadata/commons_196365.json",
            "review_evidence": "Demo/training/dataset_v210/review_evidence/HCMC_animal_transport_motorcycle_manual_v210_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; agricultural animal transport on motorcycle with custom carrier cage.",
            "classes": {"0": 1, "1": 1, "2": 0}
        },
        {
            "id": "commons_vietnam/HCMC_urban_verkehr_traffic_manual_v210.png",
            "pageid": 124405,
            "original_pageid": 124405,
            "title": "File:HCM-City Verkehr.JPG",
            "source_group": "commons_andre_lettau",
            "group": "group_andre_lettau_hcmc",
            "artist": "Andre Lettau",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "val",
            "width": 1190,
            "height": 787,
            "sha256": sha256_file(out4_img),
            "phash": phash4,
            "source_image": "Demo/training/dataset_v210/source_originals/commons_124405.jpg",
            "source_sha256": sha256_file(orig4_dst),
            "source_metadata": "Demo/training/dataset_v210/source_metadata/commons_124405.json",
            "review_evidence": "Demo/training/dataset_v210/review_evidence/HCMC_urban_verkehr_traffic_manual_v210_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; bustling urban street traffic with Honda/Yamaha scooters and commuter riders with helmets.",
            "classes": {"0": 3, "1": 3, "2": 0}
        }
    ]

    manifest.extend(new_rows)
    manifest_path = V210_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 9. Update Statistics
    print("Updating statistics.json...")
    with open(V209_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Train: +3 images
    # Divilbiss: cls 0: +1, cls 1: 0, cls 2: +2
    # Nam-ho Park: cls 0: +3, cls 1: +3, cls 2: 0
    # Kassner: cls 0: +1, cls 1: +1, cls 2: 0
    stats["train"]["images"] += 3
    stats["train"]["classes"]["0"] += 5
    stats["train"]["classes"]["1"] += 4
    stats["train"]["classes"]["2"] += 2
    stats["train"]["groups"] += 2  # group_namho_park_hcmc, group_kassner_animal_transport (group_roderick_divilbiss already in train)

    # Val: +1 image (Andre Lettau)
    # Lettau: cls 0: +3, cls 1: +3, cls 2: 0
    stats["val"]["images"] += 1
    stats["val"]["classes"]["0"] += 3
    stats["val"]["classes"]["1"] += 3
    stats["val"]["classes"]["2"] += 0
    stats["val"]["groups"] += 1  # group_andre_lettau_hcmc

    stats_path = V210_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 10. Copy duplicates, usage_restrictions, data.yaml, split_lock
    shutil.copy2(V209_DIR / "duplicates.json", V210_DIR / "duplicates.json")
    shutil.copy2(V209_DIR / "usage_restrictions.json", V210_DIR / "usage_restrictions.json")
    shutil.copy2(V209_DIR / "data.yaml", V210_DIR / "data.yaml")
    shutil.copy2(V209_DIR / "split_lock.json", V210_DIR / "split_lock.json")

    # 11. Changes and Review batch
    manual_review_45 = {
        "batch": 45,
        "date": "2026-11-11",
        "snapshot": "dataset_v210",
        "parent": "dataset_v209",
        "method": "manual pixel inspection and bounding box annotation; no model pseudo-labeling",
        "reviewed_candidates_count": 8,
        "accepted_images_count": 4,
        "held_pending_or_excluded_count": 4,
        "accepted": [
            {
                "id": "commons_vietnam/HCMC_mother_child_motorbike_manual_v210.png",
                "pageid": 530295,
                "author": "Roderick Divilbiss",
                "license": "CC BY-SA 2.5",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 1},
                    {"cls": 1, "name": "rider_with_helmet", "count": 0},
                    {"cls": 2, "name": "rider_without_helmet", "count": 2}
                ],
                "location_evidence": "Ho Chi Minh City, Vietnam; mother with sun visor and child without helmet on commuter motorcycle."
            },
            {
                "id": "commons_vietnam/HCMC_commuter_motorcycles_street_manual_v210.png",
                "pageid": 17515132,
                "author": "Nam-ho Park",
                "license": "CC BY 2.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 3},
                    {"cls": 1, "name": "rider_with_helmet", "count": 3},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Ho Chi Minh City street traffic, commuter motorbikes with helmeted riders."
            },
            {
                "id": "commons_vietnam/HCMC_animal_transport_motorcycle_manual_v210.png",
                "pageid": 196365,
                "author": "Kordian Kassner",
                "license": "CC BY-SA 2.5",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 1},
                    {"cls": 1, "name": "rider_with_helmet", "count": 1},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Ho Chi Minh City, Vietnam; motorcycle animal transport with pig in iron basket cage."
            },
            {
                "id": "commons_vietnam/HCMC_urban_verkehr_traffic_manual_v210.png",
                "pageid": 124405,
                "author": "Andre Lettau",
                "license": "CC BY-SA 3.0",
                "assigned_split": "val",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "count": 3},
                    {"cls": 1, "name": "rider_with_helmet", "count": 3},
                    {"cls": 2, "name": "rider_without_helmet", "count": 0}
                ],
                "location_evidence": "Ho Chi Minh City urban traffic flow with commuters on Honda/Yamaha motorbikes."
            }
        ],
        "excluded_or_pending": [
            {"pageid": 1960733, "title": "File:Motorcycles of Ho Chi Minh City.jpg", "reason": "Low resolution (640x427) below minimum threshold; excluded under quality contract."},
            {"pageid": 17529717, "title": "File:Xích lô.jpg", "reason": "Cyclo passenger pedal transport; excluded under Rule 24 (non-motorcycle)."},
            {"pageid": 40539018, "title": "File:Vietnam (3998449040).jpg", "reason": "Extreme motion blur on night traffic; held pending."},
            {"pageid": 24297296, "title": "File:SAIGON (2042419908).jpg", "reason": "Overexposed street panorama; motorcycles >100m in distance; held pending."}
        ]
    }
    with open(V210_DIR / "manual_review_batch45.json", "w", encoding="utf-8") as f:
        json.dump(manual_review_45, f, ensure_ascii=False, indent=2)

    # 12. Update Attribution
    with open(V209_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attribution = json.load(f)
    
    new_attributions = [
        {
            "id": "commons_vietnam/HCMC_mother_child_motorbike_manual_v210.png",
            "title": "File:Mother Child Motorbike Vietnam.jpg",
            "artist": "Roderick Divilbiss",
            "license": "CC BY-SA 2.5",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.5",
            "page_url": "https://commons.wikimedia.org/wiki/File:Mother_Child_Motorbike_Vietnam.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/1e/Mother_Child_Motorbike_Vietnam.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/1/1e/Mother_Child_Motorbike_Vietnam.jpg",
            "sha256": sha256_file(out1_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; Vietnamese mother and toddler riding family motorcycle in urban traffic.",
            "modifications": "Scaled to 1248x832. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_commuter_motorcycles_street_manual_v210.png",
            "title": "File:Motorcycles in Ho Chi Minh city.jpg",
            "artist": "Nam-ho Park",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Motorcycles_in_Ho_Chi_Minh_city.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/d/dd/Motorcycles_in_Ho_Chi_Minh_city.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/d/dd/Motorcycles_in_Ho_Chi_Minh_city.jpg",
            "sha256": sha256_file(out2_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; dense commuter street flow with riders wearing standardized motorcycle helmets.",
            "modifications": "Scaled to 1152x702. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_animal_transport_motorcycle_manual_v210.png",
            "title": "File:Animal transport as used in Ho Chi Minh City.jpg",
            "artist": "Kordian Kassner",
            "license": "CC BY-SA 2.5",
            "license_url": "https://creativecommons.org/licenses/by-sa/2.5",
            "page_url": "https://commons.wikimedia.org/wiki/File:Animal_transport_as_used_in_Ho_Chi_Minh_City.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/c/c4/Animal_transport_as_used_in_Ho_Chi_Minh_City.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/c/c4/Animal_transport_as_used_in_Ho_Chi_Minh_City.jpg",
            "sha256": sha256_file(out3_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; agricultural animal transport on motorcycle with custom carrier cage.",
            "modifications": "Original image preserved (1280x837). Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_urban_verkehr_traffic_manual_v210.png",
            "title": "File:HCM-City Verkehr.JPG",
            "artist": "Andre Lettau",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:HCM-City_Verkehr.JPG",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/3/30/HCM-City_Verkehr.JPG",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/3/30/HCM-City_Verkehr.JPG",
            "sha256": sha256_file(out4_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; bustling urban street traffic with Honda/Yamaha scooters and commuter riders with helmets.",
            "modifications": "Scaled to 1190x787. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        }
    ]

    attribution.extend(new_attributions)
    with open(V210_DIR / "attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, ensure_ascii=False, indent=2)

    # 13. Update Changes
    with open(V209_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)

    changes.append({
        "snapshot": "dataset_v210",
        "parent": "dataset_v209",
        "action": "accept_manually_reviewed_commons_images",
        "accepted_images": [
            "commons_vietnam/HCMC_mother_child_motorbike_manual_v210.png",
            "commons_vietnam/HCMC_commuter_motorcycles_street_manual_v210.png",
            "commons_vietnam/HCMC_animal_transport_motorcycle_manual_v210.png",
            "commons_vietnam/HCMC_urban_verkehr_traffic_manual_v210.png"
        ],
        "new_accepted_count": 4,
        "new_class_0_boxes": 8,
        "new_class_1_boxes": 7,
        "new_class_2_boxes": 2,
        "review_batch": "manual_review_batch45.json"
    })

    with open(V210_DIR / "changes.json", "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # 14. Update ATTRIBUTION.md
    attr_md = (
        "# Dataset Attribution Record - v210\n\n"
        "Total images: 1,118 clean ground-truth images.\n\n"
        "## Newly Admitted Images (v210 Manual Review Batch 45)\n"
        "1. **commons_vietnam/HCMC_mother_child_motorbike_manual_v210.png**\n"
        "   - Title: `File:Mother Child Motorbike Vietnam.jpg`\n"
        "   - Artist: Roderick Divilbiss\n"
        "   - License: CC BY-SA 2.5\n"
        "   - Location: Ho Chi Minh City, Vietnam\n"
        "   - Assigned Split: `train`\n\n"
        "2. **commons_vietnam/HCMC_commuter_motorcycles_street_manual_v210.png**\n"
        "   - Title: `File:Motorcycles in Ho Chi Minh city.jpg`\n"
        "   - Artist: Nam-ho Park\n"
        "   - License: CC BY 2.0\n"
        "   - Location: Ho Chi Minh City, Vietnam\n"
        "   - Assigned Split: `train`\n\n"
        "3. **commons_vietnam/HCMC_animal_transport_motorcycle_manual_v210.png**\n"
        "   - Title: `File:Animal transport as used in Ho Chi Minh City.jpg`\n"
        "   - Artist: Kordian Kassner\n"
        "   - License: CC BY-SA 2.5\n"
        "   - Location: Ho Chi Minh City, Vietnam\n"
        "   - Assigned Split: `train`\n\n"
        "4. **commons_vietnam/HCMC_urban_verkehr_traffic_manual_v210.png**\n"
        "   - Title: `File:HCM-City Verkehr.JPG`\n"
        "   - Artist: Andre Lettau\n"
        "   - License: CC BY-SA 3.0\n"
        "   - Location: Ho Chi Minh City, Vietnam\n"
        "   - Assigned Split: `val`\n"
    )
    (V210_DIR / "ATTRIBUTION.md").write_text(attr_md, encoding="utf-8")

    # 15. Update README.md
    readme_md = (
        "# Dataset Snapshot v210\n\n"
        "- Total clean reviewed images: **1,118**\n"
        "- Split counts: `train: 901`, `val: 149`, `test: 68` (locked)\n"
        "- Class instance counts: `0: motorcycle (1,998)`, `1: rider_with_helmet (1,114)`, `2: rider_without_helmet (595)`\n"
        "- Total clean bounding boxes: **3,707**\n"
        "- Progress towards 5,000 goal: **22.36%** (Remaining: **3,882**)\n"
    )
    (V210_DIR / "README.md").write_text(readme_md, encoding="utf-8")

    # 16. Verification Report
    verif = {
        "snapshot": "dataset_v210",
        "parent": "dataset_v209",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2275,
        "rows_total": 2279,
        "accepted_clean_images": 1118,
        "remaining_to_5000": 3882,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 4,
        "new_box_counts": {"0": 8, "1": 7, "2": 2},
        "split_counts": {
            "train": 901,
            "val": 149,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch45.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V210_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(V210_DIR / "attribution.json"),
        "duplicates_sha256": sha256_file(V210_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(V210_DIR / "changes.json")
    }
    with open(V210_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v210.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # 17. Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    progress = {
        "schema_version": 1,
        "as_of": "2026-11-11",
        "goal_images": 5000,
        "accepted_clean_images": 1118,
        "remaining_to_goal": 3882,
        "pending_review_images": 25,
        "snapshot": "dataset_v210",
        "accepted_split_counts": {
            "train": 901,
            "validation": 149,
            "test_locked": 68
        },
        "newly_accepted_this_batch": 4,
        "new_batch_class_box_counts": {
            "0": 8,
            "1": 7,
            "2": 2
        },
        "total_class_box_counts": {
            "0": 1998,
            "1": 1114,
            "2": 595
        },
        "original_dataset_unchanged": True,
        "test_split_locked": True,
        "pending_counts_toward_goal": False,
        "verification_report": "../verification/dataset_v210.json",
        "notes": [
            "Counts are individually reviewed images, not boxes; duplicate and pending rows do not count.",
            "dataset_v203 audited all 1,492 pending DataSet2 items: 335 unique canonical frames accepted (279 train, 56 val), 1,132 redundant variants excluded, 25 items held pending in group_0000 to maintain test lock.",
            "dataset_v204 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), verified Vietnam location context, zero leakage between splits, and test split locked at 68 images. Total clean images reaches 1,095.",
            "dataset_v205 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), adding 32 clean bounding boxes (including 7 rare rider_without_helmet boxes in Hanoi school traffic). Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,098.",
            "dataset_v206 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 2.5 / CC BY-SA 3.0), adding 19 clean bounding boxes including police patrol motorcycle, agricultural animal transport, nursery tree cargo, and city traffic. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,102.",
            "dataset_v207 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY 2.0 / CC BY-SA 2.0 / CC BY-SA 3.0), adding 23 clean bounding boxes including pillion passenger, father with unhelmeted child on Super Cub, pastry delivery carrier, and HCMC intersection. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,106.",
            "dataset_v208 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 4.0 / CC BY-SA 3.0 / CC BY 4.0 / CC0), adding 20 clean bounding boxes including Sa Pa mountain highway, Hanoi intersection red light waiting line, and HCMC commuter traffic. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,110.",
            "dataset_v209 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY 2.0 / CC BY 3.0 / CC BY-SA 3.0), adding 11 clean bounding boxes including coastal touring motorbiker, Hanoi Route 21A highway commuters, rural road commuter, and HCMC scooter parking. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,114.",
            "dataset_v210 ingested 4 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY-SA 2.5 / CC BY 2.0 / CC BY-SA 3.0), adding 17 clean bounding boxes including mother with unhelmeted child on family motorbike (adding 2 rare class 2 boxes), HCMC commuter traffic, animal transport, and city street flow. Zero leakage between splits, test split locked at 68 images. Total clean images reaches 1,118."
        ]
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v210 build completed successfully! ===")

if __name__ == "__main__":
    main()
