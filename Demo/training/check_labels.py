import zipfile
from pathlib import Path

archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")
with zipfile.ZipFile(archive_path, 'r') as zf:
    sample_lbl = [n for n in zf.namelist() if n.startswith("labels/train/day_cam_24_")][:10]
    print("Sample label files:", sample_lbl)
    if sample_lbl:
        content = zf.read(sample_lbl[0]).decode('utf-8')
        print(f"Content of {sample_lbl[0]}:\n{content[:300]}")
