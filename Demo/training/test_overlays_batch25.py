from PIL import Image, ImageDraw
from pathlib import Path

review_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")
cand_dir = review_dir / "batch_cand"

# 1. 00777
img_00777 = Image.open(cand_dir / "00777_shifted.png")
boxes_00777 = [
    (0, 64, 130, 174, 238),   # motorcycle
    (1, 96, 100, 172, 218),   # rider with helmet
]

im_draw_00777 = img_00777.copy().convert("RGB")
draw = ImageDraw.Draw(im_draw_00777)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00777:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_draw_00777.save(review_dir / "00777_overlay_test.png")

# 2. 00882
img_00882 = Image.open(cand_dir / "00882_shifted.png")
boxes_00882 = [
    (0, 72, 134, 176, 238),   # motorcycle
    (2, 114, 105, 174, 224),  # driver without helmet (class 2)
    (2, 84, 118, 142, 232),   # passenger without helmet (class 2)
]

im_draw_00882 = img_00882.copy().convert("RGB")
draw = ImageDraw.Draw(im_draw_00882)
for cls_id, x1, y1, x2, y2 in boxes_00882:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_draw_00882.save(review_dir / "00882_overlay_test.png")

print("Saved test overlays for 00777 and 00882.")
