from PIL import Image, ImageDraw
import numpy as np

img = Image.open("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/00010_s_cand.png")

# Let's test proposed bounding boxes:
# Bike: x1=124, y1=84, x2=194, y2=166
# Rider: x1=126, y1=48, x2=192, y2=142
# Let's draw and save overlay to inspect!

test_boxes = [
    (0, 124, 84, 194, 166),   # motorcycle
    (1, 134, 48, 192, 142),   # rider (helmet + torso + legs)
]

im_draw = img.copy().convert("RGB")
draw = ImageDraw.Draw(im_draw)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in test_boxes:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)

im_draw.save("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/00010_overlay_test.png")
print("Saved 00010_overlay_test.png")
