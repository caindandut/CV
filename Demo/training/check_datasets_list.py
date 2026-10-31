import os
from pathlib import Path

root = Path("d:/Nam 4/CV/Demo/training")
ds_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("dataset_v")], key=lambda d: int(d.name[9:]))

print(f"Total dataset_v directories: {len(ds_dirs)}")
total_size = 0
for d in ds_dirs:
    sz = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
    total_size += sz
    print(f"{d.name}: {sz / (1024**2):.1f} MB")

print(f"Total size of all dataset_v*: {total_size / (1024**3):.2f} GB")
