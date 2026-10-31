import zipfile
import io
from pathlib import Path
from PIL import Image

archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")
out_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")

candidates = {
    "00636": [592, 372, 832, 612],
    "00246": [683, 362, 923, 602],
    "00879": [575, 282, 815, 522],
}

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid, crop in candidates.items():
        member_name = f"images/train/day_cam_24_{cid}.jpg"
        img_bytes = zf.read(member_name)
        img = Image.open(io.BytesIO(img_bytes))
        cropped = img.crop(crop)
        cropped_path = out_dir / f"{cid}_cand.png"
        cropped.save(cropped_path)
        print(f"Saved {cropped_path}")

print("Done.")
