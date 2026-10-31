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
    for cid in ["00688", "00696"]:
        data = zf.read(f"images/train/day_cam_24_{cid}.jpg")
        (review_dir / f"day_cam_24_{cid}.jpg").write_bytes(data)

# 1. 00688
img_00688 = Image.open(cand_dir / "00688_shifted.png")
clean_00688 = review_dir / "00688_clean.png"
pixels_00688 = review_dir / "00688_review_pixels_final.png"
overlay_00688 = review_dir / "00688_overlay_final.png"

img_00688.save(clean_00688)
img_00688.save(pixels_00688)

boxes_00688 = [
    (0, 80, 134, 188, 238),   # motorcycle
    (1, 108, 110, 186, 232),  # rider with helmet
]
im_overlay_00688 = img_00688.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00688)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00688:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00688.save(overlay_00688)

# 2. 00696
img_00696 = Image.open(cand_dir / "00696_shifted.png")
clean_00696 = review_dir / "00696_clean.png"
pixels_00696 = review_dir / "00696_review_pixels_final.png"
overlay_00696 = review_dir / "00696_overlay_final.png"

img_00696.save(clean_00696)
img_00696.save(pixels_00696)

boxes_00696 = [
    (0, 86, 124, 182, 238),   # motorcycle
    (1, 106, 106, 176, 230),  # rider with helmet
]
im_overlay_00696 = img_00696.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00696)
for cls_id, x1, y1, x2, y2 in boxes_00696:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00696.save(overlay_00696)

print("00688 source_sha256:", digest(review_dir / "day_cam_24_00688.jpg"))
print("00688 clean/pixels sha256:", digest(pixels_00688))
print("00688 overlay sha256:", digest(overlay_00688))

print("00696 source_sha256:", digest(review_dir / "day_cam_24_00696.jpg"))
print("00696 clean/pixels sha256:", digest(pixels_00696))
print("00696 overlay sha256:", digest(overlay_00696))
