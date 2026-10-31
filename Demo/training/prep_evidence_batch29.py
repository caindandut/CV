import zipfile
import hashlib
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path("d:/Nam 4/CV/Demo/training")
review_dir = ROOT / "acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review"
archive_path = ROOT / "acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip"
inspect_dir = review_dir / "batch29_inspect"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

with zipfile.ZipFile(archive_path, 'r') as zf:
    for cid in ["00212", "00285"]:
        data = zf.read(f"images/train/day_cam_24_{cid}.jpg")
        (review_dir / f"day_cam_24_{cid}.jpg").write_bytes(data)

# 1. 00212
img_00212 = Image.open(inspect_dir / "00212_crop.png")
clean_00212 = review_dir / "00212_clean.png"
pixels_00212 = review_dir / "00212_review_pixels_final.png"
overlay_00212 = review_dir / "00212_overlay_final.png"

img_00212.save(clean_00212)
img_00212.save(pixels_00212)

boxes_00212 = [
    (0, 82, 104, 178, 222),   # motorcycle
    (1, 108, 62, 174, 186),   # rider with orange/brown helmet
]
im_overlay_00212 = img_00212.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00212)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00212:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00212.save(overlay_00212)

# 2. 00285
img_00285 = Image.open(inspect_dir / "00285_shifted.png")
clean_00285 = review_dir / "00285_clean.png"
pixels_00285 = review_dir / "00285_review_pixels_final.png"
overlay_00285 = review_dir / "00285_overlay_final.png"

img_00285.save(clean_00285)
img_00285.save(pixels_00285)

boxes_00285 = [
    (0, 126, 110, 222, 238),  # motorcycle
    (1, 146, 74, 218, 198),   # rider with dark helmet
]
im_overlay_00285 = img_00285.copy().convert("RGB")
draw = ImageDraw.Draw(im_overlay_00285)
for cls_id, x1, y1, x2, y2 in boxes_00285:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_overlay_00285.save(overlay_00285)

print("00212 source_sha256:", digest(review_dir / "day_cam_24_00212.jpg"))
print("00212 clean/pixels sha256:", digest(pixels_00212))
print("00212 overlay sha256:", digest(overlay_00212))

print("00285 source_sha256:", digest(review_dir / "day_cam_24_00285.jpg"))
print("00285 clean/pixels sha256:", digest(pixels_00285))
print("00285 overlay sha256:", digest(overlay_00285))
