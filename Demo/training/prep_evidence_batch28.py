import zipfile
import hashlib
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path("d:/Nam 4/CV/Demo/training")
review_dir = ROOT / "acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review"
archive_path = ROOT / "acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip"
inspect_dir = review_dir / "batch28_inspect"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid in ["00113", "00192"]:
        data = zf.read(f"images/train/day_cam_24_{cid}.jpg")
        (review_dir / f"day_cam_24_{cid}.jpg").write_bytes(data)

# 1. 00113
img_00113 = Image.open(inspect_dir / "00113_crop.png")
clean_00113 = review_dir / "00113_clean.png"
pixels_00113 = review_dir / "00113_review_pixels_final.png"
overlay_00113 = review_dir / "00113_overlay_final.png"

img_00113.save(clean_00113)
img_00113.save(pixels_00113)

boxes_00113 = [
    (0, 80, 138, 154, 240),   # motorcycle
    (1, 90, 120, 154, 224),   # rider with silver/grey helmet
]
im_overlay_00113 = img_00113.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00113)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00113:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00113.save(overlay_00113)

# 2. 00192
img_00192 = Image.open(inspect_dir / "00192_crop.png")
clean_00192 = review_dir / "00192_clean.png"
pixels_00192 = review_dir / "00192_review_pixels_final.png"
overlay_00192 = review_dir / "00192_overlay_final.png"

img_00192.save(clean_00192)
img_00192.save(pixels_00192)

boxes_00192 = [
    (0, 86, 124, 166, 240),   # motorcycle
    (1, 108, 94, 166, 214),   # rider with dark helmet
]
im_overlay_00192 = img_00192.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00192)
for cls_id, x1, y1, x2, y2 in boxes_00192:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00192.save(overlay_00192)

print("00113 source_sha256:", digest(review_dir / "day_cam_24_00113.jpg"))
print("00113 clean/pixels sha256:", digest(pixels_00113))
print("00113 overlay sha256:", digest(overlay_00113))

print("00192 source_sha256:", digest(review_dir / "day_cam_24_00192.jpg"))
print("00192 clean/pixels sha256:", digest(pixels_00192))
print("00192 overlay sha256:", digest(overlay_00192))
