import zipfile
import hashlib
from pathlib import Path

review_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")
archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid in ["00010", "00817"]:
        member = f"images/train/day_cam_24_{cid}.jpg"
        out_path = review_dir / f"day_cam_24_{cid}.jpg"
        data = zf.read(member)
        out_path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        print(f"{cid} source_sha256: {sha}")
