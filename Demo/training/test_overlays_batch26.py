from PIL import Image, ImageDraw
from pathlib import Path

review_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")
cand_dir = review_dir / "batch_cand"

# 1. 00688
img_00688 = Image.open(cand_dir / "00688_shifted.png")
boxes_00688 = [
    (0, 86, 134, 188, 238),   # motorcycle
    (1, 108, 110, 186, 232),  # rider with helmet
]

im_draw_00688 = img_00688.copy().convert("RGB")
draw = ImageDraw.Draw(im_draw_00688)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00688:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_draw_00688.save(review_dir / "00688_overlay_test.png")

# 2. 00696
img_00696 = Image.open(cand_dir / "00696_shifted.png")
boxes_00696 = [
    (0, 86, 124, 182, 238),   # motorcycle
    (1, 106, 106, 176, 230),  # rider with helmet
]

im_draw_00696 = img_00696.copy().convert("RGB")
draw = ImageDraw.Draw(im_draw_00696)
for cls_id, x1, y1, x2, y2 in boxes_00696:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_draw_00696.save(review_dir / "00696_overlay_test.png")

print("Saved test overlays for 00688 and 00696.")
