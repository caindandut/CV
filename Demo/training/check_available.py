import zipfile
import json
from pathlib import Path

archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")

# Load used frames
used = set()
for b in Path("Demo/training/acquisition_2000").glob("kaggle_reviews_batch*.json"):
    for item in json.loads(b.read_text(encoding="utf-8")):
        if "huan" in item.get("dataset_ref", ""):
            used.add(item["image_id"])

print(f"Used {len(used)} huan frames.")

# Let's check which frames exist in the archive for day_cam_24
with zipfile.ZipFile(archive_path, 'r') as zf:
    names = [n for n in zf.namelist() if n.startswith("images/train/day_cam_24_") and n.endswith(".jpg")]

print(f"Total day_cam_24 frames in train: {len(names)}")
available = [n for n in names if n.split("_")[-1].split(".")[0] not in used]
print(f"Available day_cam_24 frames: {len(available)}")
