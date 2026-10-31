from PIL import Image, ImageDraw
import numpy as np

img = Image.open("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/00010_s_cand.png")
w, h = img.size
print(f"Image size: {w}x{h}")

# Let's draw a grid every 10 pixels to allow exact pixel coordinate measurement
grid_img = img.copy().convert("RGB")
draw = ImageDraw.Draw(grid_img)
for x in range(0, w, 20):
    draw.line([(x, 0), (x, h)], fill=(100, 100, 100) if x % 50 != 0 else (255, 255, 0))
for y in range(0, h, 20):
    draw.line([(0, y), (w, y)], fill=(100, 100, 100) if y % 50 != 0 else (255, 255, 0))

grid_img.save("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/00010_grid.png")
print("Saved 00010_grid.png")
