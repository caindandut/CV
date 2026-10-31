from PIL import Image
from pathlib import Path

img_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/00817_cand.png")
im = Image.open(img_path)
# In 00817_cand.png, let's find where the bike and riders are
# Crop a 160x160 box centered on the bike
cropped = im.crop((40, 40, 200, 200))
zoom_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/00817_zoom.png")
cropped.resize((320, 320), Image.Resampling.NEAREST).save(zoom_path)
print(f"Saved {zoom_path}")
