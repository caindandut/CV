from PIL import Image, ImageDraw

img = Image.open("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/00010_s_cand.png")

test_boxes = [
    (0, 122, 56, 192, 168),   # motorcycle
    (1, 138, 48, 190, 142),   # rider (class 1)
]

im_draw = img.copy().convert("RGB")
draw = ImageDraw.Draw(im_draw)
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}
for cls_id, x1, y1, x2, y2 in test_boxes:
    draw.rectangle([x1, y1, x2, y2], outline=colors[cls_id], width=1)

im_draw.save("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/00010_overlay_adj.png")
print("Saved 00010_overlay_adj.png")
