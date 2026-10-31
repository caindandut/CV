import zipfile
import hashlib
from PIL import Image, ImageDraw
from pathlib import Path

review_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")
archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")
cand_dir = review_dir / "batch_cand"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid in ["00777", "00882"]:
        data = zf.read(f"images/train/day_cam_24_{cid}.jpg")
        (review_dir / f"day_cam_24_{cid}.jpg").write_bytes(data)

# 1. 00777
img_00777 = Image.open(cand_dir / "00777_shifted.png")
clean_00777 = review_dir / "00777_clean.png"
pixels_00777 = review_dir / "00777_review_pixels_final.png"
overlay_00777 = review_dir / "00777_overlay_final.png"

img_00777.save(clean_00777)
img_00777.save(pixels_00777)

boxes_00777 = [
    (0, 64, 130, 174, 238),   # motorcycle
    (1, 96, 100, 172, 218),   # rider with helmet
]
im_overlay_00777 = img_00777.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00777)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00777:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00777.save(overlay_00777)

# 2. 00882
img_00882 = Image.open(cand_dir / "00882_shifted.png")
clean_00882 = review_dir / "00882_clean.png"
pixels_00882 = review_dir / "00882_review_pixels_final.png"
overlay_00882 = review_dir / "00882_overlay_final.png"

img_00882.save(clean_00882)
img_00882.save(pixels_00882)

boxes_00882 = [
    (0, 72, 134, 176, 238),   # motorcycle
    (2, 112, 105, 172, 224),  # driver without helmet (class 2)
    (2, 84, 118, 128, 232),   # passenger without helmet (class 2)
]
im_overlay_00882 = img_00882.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00882)
for cls_id, x1, y1, x2, y2 in boxes_00882:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00882.save(overlay_00882)

print("00777 source_sha256:", digest(review_dir / "day_cam_24_00777.jpg"))
print("00777 clean/pixels sha256:", digest(pixels_00777))
print("00777 overlay sha256:", digest(overlay_00777))

print("00882 source_sha256:", digest(review_dir / "day_cam_24_00882.jpg"))
print("00882 clean/pixels sha256:", digest(pixels_00882))
print("00882 overlay sha256:", digest(overlay_00882))
