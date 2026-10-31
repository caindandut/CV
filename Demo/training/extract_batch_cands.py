import zipfile
import io
from pathlib import Path
from PIL import Image

archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")
out_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/batch_cand")
out_dir.mkdir(exist_ok=True)

# Propose crops centered on xc, yc with y shifted if needed so y2 <= 640
candidates = {
    "00777": [706 - 120, 500, 706 + 120, 740],  # wait, if yc=600, let's check native image
    "00688": [684 - 120, 595 - 120, 684 + 120, 595 + 120],
    "00696": [852 - 120, 595 - 120, 852 + 120, 595 + 120],
    "00759": [815 - 120, 480, 815 + 120, 720],
    "00882": [827 - 120, 480, 827 + 120, 720],
    "00704": [805 - 120, 480, 805 + 120, 720],
    "00868": [890 - 120, 480, 890 + 120, 720],
    "00714": [747 - 120, 480, 747 + 120, 720],
    "00161": [862 - 120, 480, 862 + 120, 720],
    "00041": [792 - 120, 480, 792 + 120, 720],
    "00239": [682 - 120, 430, 682 + 120, 670],
    "00505": [885 - 120, 480, 885 + 120, 720],
    "00878": [842 - 120, 440, 842 + 120, 680],
}

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid, crop in candidates.items():
        member_name = f"images/train/day_cam_24_{cid}.jpg"
        img_bytes = zf.read(member_name)
        img = Image.open(io.BytesIO(img_bytes))
        # Clamp crop to image size (1280x720)
        x1 = max(0, min(crop[0], 1280 - 240))
        y1 = max(0, min(crop[1], 720 - 240))
        x2, y2 = x1 + 240, y1 + 240
        cropped = img.crop((x1, y1, x2, y2))
        cropped_path = out_dir / f"{cid}_cand.png"
        cropped.save(cropped_path)
        print(f"Saved {cid}_cand.png with crop [{x1}, {y1}, {x2}, {y2}]")

print("Done extracting batch candidates.")
