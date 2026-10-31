import glob
import json
from pathlib import Path

ROOT = Path("Demo/training")
batches = sorted((ROOT / "acquisition_2000").glob("kaggle_reviews_batch*.json"))

used_huan = {}
for b in batches:
    data = json.loads(b.read_text(encoding="utf-8"))
    for item in data:
        if "huan" in item.get("dataset_ref", ""):
            img_id = item["image_id"]
            cam = item.get("camera_prefix", "day_cam_24")
            used_huan[f"{cam}_{img_id}"] = b.name

print(f"Total used Huan camera frames: {len(used_huan)}")
for k in sorted(used_huan.keys()):
    print(f"  {k} -> {used_huan[k]}")
