import zipfile
import hashlib
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path("d:/Nam 4/CV/Demo/training")
review_dir = ROOT / "acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review"
archive_path = ROOT / "acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip"
inspect_dir = review_dir / "batch27_inspect"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid in ["00759", "00041"]:
        data = zf.read(f"images/train/day_cam_24_{cid}.jpg")
        (review_dir / f"day_cam_24_{cid}.jpg").write_bytes(data)

# 1. 00759
img_00759 = Image.open(inspect_dir / "00759_crop.png")
clean_00759 = review_dir / "00759_clean.png"
pixels_00759 = review_dir / "00759_review_pixels_final.png"
overlay_00759 = review_dir / "00759_overlay_final.png"

img_00759.save(clean_00759)
img_00759.save(pixels_00759)

boxes_00759 = [
    (0, 82, 104, 160, 238),   # motorcycle
    (1, 100, 88, 160, 212),   # rider with white helmet
]
im_overlay_00759 = img_00759.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00759)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00759:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00759.save(overlay_00759)

# 2. 00041
img_00041 = Image.open(inspect_dir / "00041_shifted.png")
clean_00041 = review_dir / "00041_clean.png"
pixels_00041 = review_dir / "00041_review_pixels_final.png"
overlay_00041 = review_dir / "00041_overlay_final.png"

img_00041.save(clean_00041)
img_00041.save(pixels_00041)

boxes_00041 = [
    (0, 62, 104, 142, 236),   # motorcycle
    (1, 78, 84, 140, 202),    # rider with green helmet
]
im_overlay_00041 = img_00041.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00041)
for cls_id, x1, y1, x2, y2 in boxes_00041:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00041.save(overlay_00041)

print("00759 source_sha256:", digest(review_dir / "day_cam_24_00759.jpg"))
print("00759 clean/pixels sha256:", digest(pixels_00759))
print("00759 overlay sha256:", digest(overlay_00759))

print("00041 source_sha256:", digest(review_dir / "day_cam_24_00041.jpg"))
print("00041 clean/pixels sha256:", digest(pixels_00041))
print("00041 overlay sha256:", digest(overlay_00041))
