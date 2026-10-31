import zipfile
import io
from pathlib import Path
from PIL import Image

archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")
out_dir = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/batch_cand")

with zipfile.ZipFile(archive_path, 'r') as zf:
    img_bytes = zf.read("images/train/day_cam_24_00777.jpg")
    img = Image.open(io.BytesIO(img_bytes))
    # Crop [586, 420, 826, 660]
    cropped = img.crop((586, 420, 826, 660))
    cropped.save(out_dir / "00777_shifted.png")
    print("Saved 00777_shifted.png")
