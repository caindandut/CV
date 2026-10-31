import zipfile
import io
from pathlib import Path
from PIL import Image

archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")
out_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")

candidates = {
    "00010": [745, 390, 985, 630],
    "00275": [594, 413, 834, 653],
    "00824": [704, 401, 944, 641],
    "00588": [752, 413, 992, 653],
    "00816": [514, 292, 754, 532],
}

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid, crop in candidates.items():
        member_name = f"images/train/day_cam_24_{cid}.jpg"
        print(f"Reading {member_name}...")
        img_bytes = zf.read(member_name)
        img = Image.open(io.BytesIO(img_bytes))
        cropped = img.crop(crop)
        cropped_path = out_dir / f"{cid}_cand.png"
        cropped.save(cropped_path)
        print(f"Saved {cropped_path} (size {cropped.size})")

print("All candidate crops generated!")
