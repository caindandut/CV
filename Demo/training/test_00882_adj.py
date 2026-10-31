from PIL import Image, ImageDraw
from pathlib import Path

review_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")
cand_dir = review_dir / "batch_cand"

img_00882 = Image.open(cand_dir / "00882_shifted.png")
boxes_00882 = [
    (0, 72, 134, 176, 238),   # motorcycle
    (2, 112, 105, 172, 224),  # driver without helmet (class 2)
    (2, 84, 118, 128, 232),   # passenger without helmet (class 2)
]

im_draw_00882 = img_00882.copy().convert("RGB")
draw = ImageDraw.Draw(im_draw_00882)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in boxes_00882:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)
im_draw_00882.save(review_dir / "00882_overlay_adj.png")

print("Saved 00882_overlay_adj.png")
