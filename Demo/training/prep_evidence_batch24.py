import hashlib
from PIL import Image, ImageDraw
from pathlib import Path

review_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# 1. 00010
img_00010 = Image.open(review_dir / "00010_s_cand.png")
clean_00010_path = review_dir / "00010_clean.png"
pixels_00010_path = review_dir / "00010_review_pixels_final.png"
overlay_00010_path = review_dir / "00010_overlay_final.png"

img_00010.save(clean_00010_path)
img_00010.save(pixels_00010_path)

# Draw overlay
boxes_00010 = [
    (0, 122, 56, 192, 168),   # motorcycle
    (1, 138, 48, 190, 142),   # rider with helmet
]
overlay_00010 = img_00010.copy().convert("RGB")
draw_00010 = ImageDraw.Draw(overlay_00010)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00010:
    draw_00010.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
overlay_00010.save(overlay_00010_path)

# 2. 00817
img_00817 = Image.open(review_dir / "00817_cand.png")
clean_00817_path = review_dir / "00817_clean.png"
pixels_00817_path = review_dir / "00817_review_pixels_final.png"
overlay_00817_path = review_dir / "00817_overlay_final.png"

img_00817.save(clean_00817_path)
img_00817.save(pixels_00817_path)

# Draw overlay
boxes_00817 = [
    (0, 90, 74, 156, 186),    # motorcycle
    (1, 112, 62, 156, 164),   # driver with helmet (class 1)
    (2, 98, 72, 142, 168),    # passenger without helmet (class 2)
]
overlay_00817 = img_00817.copy().convert("RGB")
draw_00817 = ImageDraw.Draw(overlay_00817)
for cls_id, x1, y1, x2, y2 in boxes_00817:
    draw_00817.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
overlay_00817.save(overlay_00817_path)

print("00010:")
print("  clean/pixels sha256:", digest(pixels_00010_path))
print("  overlay sha256:", digest(overlay_00010_path))

print("00817:")
print("  clean/pixels sha256:", digest(pixels_00817_path))
print("  overlay sha256:", digest(overlay_00817_path))
