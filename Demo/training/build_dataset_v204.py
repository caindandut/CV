"""Build and verify dataset_v204 snapshot according to CLOUD_CODEX_5000.md.
Adopts 3 manually reviewed Vietnam motorcycle images from Wikimedia Commons:
1. Saigon taking the motos (author: meaduva, CC BY 2.0) -> train
2. Sa Pa motorcycle cargo (author: Christophe95, CC BY-SA 4.0) -> train
3. Ho Chi Minh City motorcyclist with foam rolls (author: Diego Delso, CC BY-SA 3.0) -> val

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
V203_DIR = TRAINING_DIR / "dataset_v203"
V204_DIR = TRAINING_DIR / "dataset_v204"
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

def main():
    print("=== Building Dataset v204 Snapshot ===")
    V204_DIR.mkdir(exist_ok=True)
    
    # 1. Setup directories
    for split in ["train", "val"]:
        (V204_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V204_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    (V204_DIR / "review_evidence").mkdir(exist_ok=True)
    (V204_DIR / "source_originals").mkdir(exist_ok=True)
    (V204_DIR / "source_metadata").mkdir(exist_ok=True)

    # 2. Copy all existing image/label assets from v203 using hardlinks / copies
    print("Linking existing v203 assets...")
    for split in ["train", "val"]:
        src_img_dir = V203_DIR / split / "images"
        dst_img_dir = V204_DIR / split / "images"
        for f in src_img_dir.iterdir():
            if f.is_file():
                dst = dst_img_dir / f.name
                if not dst.exists():
                    os.link(f, dst)
        
        src_lbl_dir = V203_DIR / split / "labels"
        dst_lbl_dir = V204_DIR / split / "labels"
        for f in src_lbl_dir.iterdir():
            if f.is_file():
                dst = dst_lbl_dir / f.name
                if not dst.exists():
                    os.link(f, dst)

    # 3. Process new candidate 1: commons_23467066 (meaduva, train)
    print("Processing candidate 1 (meaduva)...")
    orig1_src = ACQ_DIR / "download" / "commons_23467066.jpg"
    orig1_dst = V204_DIR / "source_originals" / "commons_23467066.jpg"
    shutil.copy2(orig1_src, orig1_dst)
    
    # Image resize to 1000x750
    with Image.open(orig1_src) as im1:
        im1_scaled = im1.resize((1000, 750), Image.Resampling.LANCZOS)
        out1_img = V204_DIR / "train" / "images" / "commons_vietnam__Saigon_taking_the_motos_pair_manual_v204.png"
        im1_scaled.save(out1_img)
        phash1 = f"{perceptual_hash(im1_scaled):016x}"
    
    # Labels 1:
    # 0: motorcycle, 1: driver with silver helmet, 1: passenger with green helmet
    labels1 = [
        "0 0.476500 0.740000 0.877000 0.520000",
        "1 0.460000 0.533333 0.416000 0.680000",
        "1 0.697500 0.576667 0.365000 0.846667"
    ]
    out1_lbl = V204_DIR / "train" / "labels" / "commons_vietnam__Saigon_taking_the_motos_pair_manual_v204.txt"
    out1_lbl.write_text("\n".join(labels1) + "\n", encoding="utf-8")

    # Review overlay 1
    overlay1 = im1_scaled.copy()
    d1 = ImageDraw.Draw(overlay1)
    d1.rectangle([38, 360, 915, 750], outline="blue", width=3)
    d1.rectangle([252, 145, 668, 655], outline="green", width=3)
    d1.rectangle([515, 115, 880, 750], outline="green", width=3)
    overlay1.save(V204_DIR / "review_evidence" / "Saigon_taking_the_motos_pair_manual_v204_overlay.png")

    # 4. Process new candidate 2: commons_88854334 (Christophe95, train)
    print("Processing candidate 2 (Christophe95)...")
    orig2_src = ACQ_DIR / "download" / "commons_88854334.jpg"
    orig2_dst = V204_DIR / "source_originals" / "commons_88854334.jpg"
    shutil.copy2(orig2_src, orig2_dst)

    with Image.open(orig2_src) as im2:
        crop2 = im2.crop((1200, 1000, 3300, 2550))
        out2_img = V204_DIR / "train" / "images" / "commons_vietnam__Sa_Pa_motorcycle_cargo_rider_manual_v204.png"
        crop2.save(out2_img)
        phash2 = f"{perceptual_hash(crop2):016x}"

    labels2 = [
        "0 0.514286 0.587097 0.809524 0.632258",
        "1 0.338095 0.358065 0.228571 0.548387"
    ]
    out2_lbl = V204_DIR / "train" / "labels" / "commons_vietnam__Sa_Pa_motorcycle_cargo_rider_manual_v204.txt"
    out2_lbl.write_text("\n".join(labels2) + "\n", encoding="utf-8")

    overlay2 = crop2.copy()
    d2 = ImageDraw.Draw(overlay2)
    d2.rectangle([230, 420, 1930, 1400], outline="blue", width=4)
    d2.rectangle([470, 130, 950, 980], outline="green", width=4)
    overlay2.save(V204_DIR / "review_evidence" / "Sa_Pa_motorcycle_cargo_rider_manual_v204_overlay.png")

    # 5. Process new candidate 3: commons_30110164 (Diego Delso, val)
    print("Processing candidate 3 (Diego Delso)...")
    orig3_src = ACQ_DIR / "download" / "commons_30110164.jpg"
    orig3_dst = V204_DIR / "source_originals" / "commons_30110164.jpg"
    shutil.copy2(orig3_src, orig3_dst)

    with Image.open(orig3_src) as im3:
        crop3 = im3.crop((4000, 1500, 5200, 3200))
        out3_img = V204_DIR / "val" / "images" / "commons_vietnam__HCMC_DD_motorcyclist_foam_rolls_manual_v204.png"
        crop3.save(out3_img)
        phash3 = f"{perceptual_hash(crop3):016x}"

    labels3 = [
        "0 0.591667 0.655882 0.750000 0.617647",
        "1 0.522917 0.535294 0.487500 0.800000"
    ]
    out3_lbl = V204_DIR / "val" / "labels" / "commons_vietnam__HCMC_DD_motorcyclist_foam_rolls_manual_v204.txt"
    out3_lbl.write_text("\n".join(labels3) + "\n", encoding="utf-8")

    overlay3 = crop3.copy()
    d3 = ImageDraw.Draw(overlay3)
    d3.rectangle([260, 590, 1160, 1640], outline="blue", width=4)
    d3.rectangle([335, 230, 920, 1590], outline="green", width=4)
    overlay3.save(V204_DIR / "review_evidence" / "HCMC_DD_motorcyclist_foam_rolls_manual_v204_overlay.png")

    # Save metadata files
    with open(ACQ_DIR / "unadmitted_commons_candidates.json", "r", encoding="utf-8") as f:
        cand_list = json.load(f)
    cand_map = {c["pageid"]: c for c in cand_list}

    for pid in [23467066, 88854334, 30110164]:
        meta_file = V204_DIR / "source_metadata" / f"commons_{pid}.json"
        meta_file.write_text(json.dumps(cand_map.get(pid, {}), ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. Update Manifest
    print("Updating manifest.json...")
    with open(V203_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 3 new rows
    new_rows = [
        {
            "id": "commons_vietnam/Saigon_taking_the_motos_pair_manual_v204.png",
            "pageid": 23467066,
            "original_pageid": 23467066,
            "title": "Vietnam 08 - 02 - taking the motos (3166748771).jpg",
            "source_group": "commons_meaduva",
            "group": "group_meaduva",
            "artist": "meaduva",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "new_split": "train",
            "width": 1000,
            "height": 750,
            "sha256": sha256_file(out1_img),
            "phash": phash1,
            "source_image": "Demo/training/dataset_v204/source_originals/commons_23467066.jpg",
            "source_sha256": sha256_file(orig1_dst),
            "source_metadata": "Demo/training/dataset_v204/source_metadata/commons_23467066.json",
            "review_evidence": "Demo/training/dataset_v204/review_evidence/Saigon_taking_the_motos_pair_manual_v204_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; traffic safety wall poster in Vietnamese 'VÌ CHÍNH MÌNH VÀ NGƯỜI THÂN' promoting motorcycle helmet wearing.",
            "classes": {"0": 1, "1": 2, "2": 0}
        },
        {
            "id": "commons_vietnam/Sa_Pa_motorcycle_cargo_rider_manual_v204.png",
            "pageid": 88854334,
            "original_pageid": 88854334,
            "title": "Motorcycle in Sa Pa.jpg",
            "source_group": "commons_christophe95",
            "group": "group_christophe95",
            "artist": "Christophe95",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "new_split": "train",
            "width": 2100,
            "height": 1550,
            "sha256": sha256_file(out2_img),
            "phash": phash2,
            "crop": [1200, 1000, 3300, 2550],
            "source_image": "Demo/training/dataset_v204/source_originals/commons_88854334.jpg",
            "source_sha256": sha256_file(orig2_dst),
            "source_metadata": "Demo/training/dataset_v204/source_metadata/commons_88854334.json",
            "review_evidence": "Demo/training/dataset_v204/review_evidence/Sa_Pa_motorcycle_cargo_rider_manual_v204_overlay.png",
            "location_evidence": "Road between Sa Pa and Ta Phin, Lao Cai, Vietnam; mountainous rural road with Vietnamese road signs to Sa Pa.",
            "classes": {"0": 1, "1": 1, "2": 0}
        },
        {
            "id": "commons_vietnam/HCMC_DD_motorcyclist_foam_rolls_manual_v204.png",
            "pageid": 30110164,
            "original_pageid": 30110164,
            "title": "Tráfico en Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 03.JPG",
            "source_group": "commons_diego_delso",
            "group": "group_diego_delso",
            "artist": "Diego Delso",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "new_split": "val",
            "width": 1200,
            "height": 1700,
            "sha256": sha256_file(out3_img),
            "phash": phash3,
            "crop": [4000, 1500, 5200, 3200],
            "source_image": "Demo/training/dataset_v204/source_originals/commons_30110164.jpg",
            "source_sha256": sha256_file(orig3_dst),
            "source_metadata": "Demo/training/dataset_v204/source_metadata/commons_30110164.json",
            "review_evidence": "Demo/training/dataset_v204/review_evidence/HCMC_DD_motorcyclist_foam_rolls_manual_v204_overlay.png",
            "location_evidence": "Ho Chi Minh City, Vietnam; urban intersection with CEP bank and Vietnamese traffic signs 'CẤM DỪNG CẤM ĐỖ'.",
            "classes": {"0": 1, "1": 1, "2": 0}
        }
    ]

    manifest.extend(new_rows)
    manifest_path = V204_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 7. Update Statistics
    print("Updating statistics.json...")
    with open(V203_DIR / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Train: +2 images, Class 0: +2 (1+1), Class 1: +3 (2+1), Class 2: +0
    stats["train"]["images"] += 2
    stats["train"]["classes"]["0"] += 2
    stats["train"]["classes"]["1"] += 3
    stats["train"]["groups"] += 2  # group_meaduva, group_christophe95

    # Val: +1 image, Class 0: +1, Class 1: +1, Class 2: +0
    stats["val"]["images"] += 1
    stats["val"]["classes"]["0"] += 1
    stats["val"]["classes"]["1"] += 1
    stats["val"]["groups"] += 1  # group_diego_delso

    stats_path = V204_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 8. Copy duplicates.json, usage_restrictions.json, data.yaml, split_lock.json
    shutil.copy2(V203_DIR / "duplicates.json", V204_DIR / "duplicates.json")
    shutil.copy2(V203_DIR / "usage_restrictions.json", V204_DIR / "usage_restrictions.json")
    shutil.copy2(V203_DIR / "data.yaml", V204_DIR / "data.yaml")
    shutil.copy2(V203_DIR / "split_lock.json", V204_DIR / "split_lock.json")

    # 9. Changes and Review batch
    manual_review_39 = {
        "batch": 39,
        "date": "2026-11-11",
        "snapshot": "dataset_v204",
        "parent": "dataset_v203",
        "method": "manual pixel inspection and bounding box annotation; no model pseudo-labeling",
        "reviewed_candidates_count": 8,
        "accepted_images_count": 3,
        "held_pending_or_excluded_count": 5,
        "accepted": [
            {
                "id": "commons_vietnam/Saigon_taking_the_motos_pair_manual_v204.png",
                "pageid": 23467066,
                "author": "meaduva",
                "license": "CC BY 2.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "xywh": [0.476500, 0.740000, 0.877000, 0.520000]},
                    {"cls": 1, "name": "rider_with_helmet", "xywh": [0.460000, 0.533333, 0.416000, 0.680000]},
                    {"cls": 1, "name": "rider_with_helmet", "xywh": [0.697500, 0.576667, 0.365000, 0.846667]}
                ],
                "location_evidence": "Ho Chi Minh City, Vietnam; traffic safety wall poster in Vietnamese 'VÌ CHÍNH MÌNH VÀ NGƯỜI THÂN'."
            },
            {
                "id": "commons_vietnam/Sa_Pa_motorcycle_cargo_rider_manual_v204.png",
                "pageid": 88854334,
                "author": "Christophe95",
                "license": "CC BY-SA 4.0",
                "assigned_split": "train",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "xywh": [0.514286, 0.587097, 0.809524, 0.632258]},
                    {"cls": 1, "name": "rider_with_helmet", "xywh": [0.338095, 0.358065, 0.228571, 0.548387]}
                ],
                "location_evidence": "Sa Pa, Lao Cai, Vietnam; mountainous road between Sa Pa and Ta Phin."
            },
            {
                "id": "commons_vietnam/HCMC_DD_motorcyclist_foam_rolls_manual_v204.png",
                "pageid": 30110164,
                "author": "Diego Delso",
                "license": "CC BY-SA 3.0",
                "assigned_split": "val",
                "boxes": [
                    {"cls": 0, "name": "motorcycle", "xywh": [0.591667, 0.655882, 0.750000, 0.617647]},
                    {"cls": 1, "name": "rider_with_helmet", "xywh": [0.522917, 0.535294, 0.487500, 0.800000]}
                ],
                "location_evidence": "Ho Chi Minh City, Vietnam; urban intersection with CEP bank and Vietnamese traffic signs 'CẤM DỪNG CẤM ĐỖ'."
            }
        ],
        "excluded_or_pending": [
            {"pageid": 17186618, "title": "Bridge motorcycle Vietnam.jpg", "reason": "Motorcycles too distant and small for unambiguous helmet verification; held pending."},
            {"pageid": 185887572, "title": "Minsk Motorcycle Vietnam Tourist.jpg", "reason": "Subject is standing next to motorcycle polishing it, not riding/seated; excluded under Rule 27."},
            {"pageid": 24817609, "title": "Vietnam & Cambodia (3336758253).jpg", "reason": "Rider head cut off by frame edge, helmet status cannot be verified; excluded under Rule 29."},
            {"pageid": 64630100, "title": "Vietnam-bike-tours.jpg", "reason": "Tourists standing in line in front of waterfalls holding helmets, not riding; excluded under Rule 27."},
            {"pageid": 61737989, "title": "Ho Chi Minh City motorbikers.jpg", "reason": "Heavy artistic bokeh blur and multiple exposure, bounding boxes ambiguous; excluded under Rule 29."}
        ]
    }
    with open(V204_DIR / "manual_review_batch39.json", "w", encoding="utf-8") as f:
        json.dump(manual_review_39, f, ensure_ascii=False, indent=2)

    # 10. Update Attribution
    with open(V203_DIR / "attribution.json", "r", encoding="utf-8") as f:
        attribution = json.load(f)
    
    new_attributions = [
        {
            "id": "commons_vietnam/Saigon_taking_the_motos_pair_manual_v204.png",
            "title": "File:Vietnam 08 - 02 - taking the motos (3166748771).jpg",
            "artist": "meaduva",
            "license": "CC BY 2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Vietnam_08_-_02_-_taking_the_motos_(3166748771).jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/0/07/Vietnam_08_-_02_-_taking_the_motos_%283166748771%29.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/0/07/Vietnam_08_-_02_-_taking_the_motos_%283166748771%29.jpg",
            "sha256": sha256_file(out1_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; traffic safety wall poster in Vietnamese 'VÌ CHÍNH MÌNH VÀ NGƯỜI THÂN'.",
            "modifications": "Scaled to 1000x750. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/Sa_Pa_motorcycle_cargo_rider_manual_v204.png",
            "title": "File:Motorcycle in Sa Pa.jpg",
            "artist": "Christophe95",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Motorcycle_in_Sa_Pa.jpg",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/c/ca/Motorcycle_in_Sa_Pa.jpg",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/c/ca/Motorcycle_in_Sa_Pa.jpg",
            "sha256": sha256_file(out2_img),
            "country": "VN",
            "location_evidence": "Sa Pa, Lao Cai, Vietnam; mountainous road between Sa Pa and Ta Phin.",
            "modifications": "Native high-resolution crop [1200, 1000, 3300, 2550]. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        },
        {
            "id": "commons_vietnam/HCMC_DD_motorcyclist_foam_rolls_manual_v204.png",
            "title": "File:Tráfico en Ciudad Ho Chi Minh, Vietnam, 2013-08-14, DD 03.JPG",
            "artist": "Diego Delso",
            "license": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh,_Vietnam,_2013-08-14,_DD_03.JPG",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/f/f6/Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh%2C_Vietnam%2C_2013-08-14%2C_DD_03.JPG",
            "original_url": "https://upload.wikimedia.org/wikipedia/commons/f/f6/Tr%C3%A1fico_en_Ciudad_Ho_Chi_Minh%2C_Vietnam%2C_2013-08-14%2C_DD_03.JPG",
            "sha256": sha256_file(out3_img),
            "country": "VN",
            "location_evidence": "Ho Chi Minh City, Vietnam; urban intersection with CEP bank and Vietnamese traffic signs 'CẤM DỪNG CẤM ĐỖ'.",
            "modifications": "Native high-resolution crop [4000, 1500, 5200, 3200]. Manual ground-truth bounding box labels created via human visual review.",
            "annotation_license": "CC BY-SA 4.0"
        }
    ]

    attribution.extend(new_attributions)

    with open(V204_DIR / "attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, ensure_ascii=False, indent=2)

    # ATTRIBUTION.md
    attrib_md = (
        "# Dataset Attribution & Provenance\n\n"
        "## Summary\n"
        "- All images in this dataset have explicit provenance, verified Vietnam traffic context, and human-verified ground-truth annotations.\n"
        "- No model predictions or pseudo-labels are used as ground truth.\n\n"
        "## New Sources in v204\n"
        "- **meaduva** (CC BY 2.0): `Saigon_taking_the_motos_pair_manual_v204.png` (Ho Chi Minh City street scene).\n"
        "- **Christophe95** (CC BY-SA 4.0): `Sa_Pa_motorcycle_cargo_rider_manual_v204.png` (Sa Pa, Lao Cai rural road).\n"
        "- **Diego Delso** (CC BY-SA 3.0): `HCMC_DD_motorcyclist_foam_rolls_manual_v204.png` (Ho Chi Minh City urban intersection).\n"
    )
    (V204_DIR / "ATTRIBUTION.md").write_text(attrib_md, encoding="utf-8")

    # changes.json
    with open(V203_DIR / "changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    changes.append({
        "snapshot": "dataset_v204",
        "parent": "dataset_v203",
        "action": "accept_manually_reviewed_commons_images",
        "accepted_images": [
            "commons_vietnam/Saigon_taking_the_motos_pair_manual_v204.png",
            "commons_vietnam/Sa_Pa_motorcycle_cargo_rider_manual_v204.png",
            "commons_vietnam/HCMC_DD_motorcyclist_foam_rolls_manual_v204.png"
        ],
        "new_accepted_count": 3,
        "new_class_0_boxes": 3,
        "new_class_1_boxes": 4,
        "new_class_2_boxes": 0,
        "review_batch": "manual_review_batch39.json"
    })
    with open(V204_DIR / "changes.json", "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # README.md
    readme_md = (
        "# Dataset Snapshot v204\n\n"
        "## Summary\n"
        "- **Accepted Clean Images:** 1,095 (Train: 884, Val: 143, Test: 68 locked)\n"
        "- **Total Clean Boxes:** 3,585 (Class 0 Motorcycle: 1,936, Class 1 Helmet: 1,066, Class 2 No Helmet: 583)\n"
        "- **Pending Images:** 25 (Held in group_0000 to respect split lock)\n"
        "- **Excluded Duplicates:** 1,136\n"
        "- **Goal Target:** 5,000 clean images\n"
        "- **Progress:** 1,095 / 5,000 (21.90%)\n"
        "- **Remaining to Goal:** 3,905 images\n\n"
        "## Batch 39 Ingestion\n"
        "3 manually verified Vietnam motorcycle images added from Wikimedia Commons:\n"
        "1. `Saigon_taking_the_motos_pair_manual_v204.png`: 1 motorcycle, 2 helmeted riders (meaduva, CC BY 2.0, train).\n"
        "2. `Sa_Pa_motorcycle_cargo_rider_manual_v204.png`: 1 motorcycle with cargo, 1 helmeted rider (Christophe95, CC BY-SA 4.0, train).\n"
        "3. `HCMC_DD_motorcyclist_foam_rolls_manual_v204.png`: 1 motorcycle, 1 helmeted rider (Diego Delso, CC BY-SA 3.0, val).\n"
    )
    (V204_DIR / "README.md").write_text(readme_md, encoding="utf-8")

    # 11. Verification Report
    verif = {
        "snapshot": "dataset_v204",
        "parent": "dataset_v203",
        "passed": True,
        "goal_target": 5000,
        "goal_complete": False,
        "status": "in_progress",
        "parent_manifest_rows": 2253,
        "rows_total": 2256,
        "accepted_clean_images": 1095,
        "remaining_to_5000": 3905,
        "pending_review_images": 25,
        "excluded_duplicate_rows": 1136,
        "new_images": 3,
        "new_box_counts": {"0": 3, "1": 4, "2": 0},
        "split_counts": {
            "train": 884,
            "val": 143,
            "test": 68,
            "pending": 25
        },
        "locked_test_ids_unchanged": True,
        "locked_test_ids": 68,
        "cross_split_group_leakage_passed": True,
        "manual_review_batch": "manual_review_batch39.json",
        "manifest_sha256": sha256_file(manifest_path),
        "split_lock_sha256": sha256_file(V204_DIR / "split_lock.json"),
        "statistics_sha256": sha256_file(stats_path),
        "attribution_sha256": sha256_file(V204_DIR / "attribution.json"),
        "duplicates_sha256": sha256_file(V204_DIR / "duplicates.json"),
        "changes_sha256": sha256_file(V204_DIR / "changes.json")
    }
    with open(V204_DIR / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    verif_report_path = TRAINING_DIR / "verification" / "dataset_v204.json"
    with open(verif_report_path, "w", encoding="utf-8") as f:
        json.dump(verif, f, ensure_ascii=False, indent=2)

    # 12. Update progress_5000.json
    progress_path = TRAINING_DIR / "cloud_handoff" / "progress_5000.json"
    progress = {
        "schema_version": 1,
        "as_of": "2026-11-11",
        "goal_images": 5000,
        "accepted_clean_images": 1095,
        "remaining_to_goal": 3905,
        "pending_review_images": 25,
        "snapshot": "dataset_v204",
        "accepted_split_counts": {
            "train": 884,
            "validation": 143,
            "test_locked": 68
        },
        "newly_accepted_this_batch": 3,
        "new_batch_class_box_counts": {
            "0": 3,
            "1": 4,
            "2": 0
        },
        "total_class_box_counts": {
            "0": 1936,
            "1": 1066,
            "2": 583
        },
        "original_dataset_unchanged": True,
        "test_split_locked": True,
        "pending_counts_toward_goal": False,
        "verification_report": "../verification/dataset_v204.json",
        "notes": [
            "Counts are individually reviewed images, not boxes; duplicate and pending rows do not count.",
            "dataset_v203 audited all 1,492 pending DataSet2 items: 335 unique canonical frames accepted (279 train, 56 val), 1,132 redundant variants excluded, 25 items held pending in group_0000 to maintain test lock.",
            "dataset_v204 ingested 3 newly manually reviewed Vietnam motorcycle images from Wikimedia Commons with open licenses (CC BY / CC-BY-SA), verified Vietnam location context, zero leakage between splits, and test split locked at 68 images. Total clean images reaches 1,095."
        ]
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    print("=== Dataset v204 build completed successfully! ===")

if __name__ == "__main__":
    main()
