from PIL import Image, ImageDraw
from pathlib import Path

review_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review")
cand_dir = review_dir / "batch_cand"

for cid, cand_file in [("00688", cand_dir / "00688_shifted.png"), ("00696", cand_dir / "00696_shifted.png")]:
    img = Image.open(cand_file)
    w, h = img.size
    grid_img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(grid_img)
    for x in range(0, w, 20):
        draw.line([(x, 0), (x, h)], fill=(100, 100, 100) if x % 50 != 0 else (255, 255, 0))
    for y in range(0, h, 20):
        draw.line([(0, y), (w, y)], fill=(100, 100, 100) if y % 50 != 0 else (255, 255, 0))
    grid_path = review_dir / f"{cid}_grid.png"
    grid_img.save(grid_path)
    print(f"Saved {grid_path}")
