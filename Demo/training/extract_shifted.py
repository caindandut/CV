import zipfile
import io
from pathlib import Path
from PIL import Image

archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")
out_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/batch_cand")

cands = {
    "00688": [564, 410, 804, 650],
    "00696": [732, 410, 972, 650],
    "00759": [695, 410, 935, 650],
    "00882": [707, 410, 947, 650],
    "00704": [685, 410, 925, 650],
    "00868": [770, 410, 1010, 650],
    "00714": [627, 410, 867, 650],
    "00161": [742, 410, 982, 650],
    "00041": [672, 410, 912, 650],
    "00239": [562, 410, 802, 650],
    "00878": [722, 410, 962, 650],
}

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid, crop in cands.items():
        member_name = f"images/train/day_cam_24_{cid}.jpg"
        img_bytes = zf.read(member_name)
        img = Image.open(io.BytesIO(img_bytes))
        cropped = img.crop(crop)
        cropped_path = out_dir / f"{cid}_shifted.png"
        cropped.save(cropped_path)
        print(f"Saved {cid}_shifted.png")

print("Done shifting batch candidates.")
