import shutil
import os
from pathlib import Path

ROOT = Path("d:/Nam 4/CV/Demo/training")

bytes_freed = 0

def remove_dir(path):
    global bytes_freed
    if not path.exists():
        return
    sz = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    shutil.rmtree(path)
    bytes_freed += sz
    print(f"Removed dir: {path.name} ({sz / (1024**2):.1f} MB)")

def remove_file(path):
    global bytes_freed
    if not path.is_file():
        return
    sz = path.stat().st_size
    path.unlink()
    bytes_freed += sz
    print(f"Removed file: {path.name} ({sz / (1024**2):.1f} MB)")

print("=== 1. Cleaning verification/colab_bundle_v*_extract (keeping v90) ===")
ver_dir = ROOT / "verification"
for entry in ver_dir.iterdir():
    if entry.is_dir() and entry.name.startswith("colab_bundle_v") and entry.name.endswith("_extract"):
        if entry.name != "colab_bundle_v90_extract":
            remove_dir(entry)

print("\n=== 2. Cleaning unused large downloads in acquisition_2000 ===")
unused_hf_zip = ROOT / "acquisition_2000/research_huggingface/traffic-sign-detection-vietnam.zip"
remove_file(unused_hf_zip)

print("\n=== 3. Cleaning older colab_bundles (keeping v90 and v74) ===")
bundles_dir = ROOT / "colab_bundles"
for entry in bundles_dir.iterdir():
    if entry.is_dir() and entry.name.startswith("dataset_v"):
        if entry.name not in {"dataset_v90", "dataset_v74"}:
            remove_dir(entry)

print("\n=== 4. Cleaning image media in older dataset snapshots v4..v88 (keeping manifests, json, yaml) ===")
for v in range(4, 89):
    ds_dir = ROOT / f"dataset_v{v}"
    if not ds_dir.is_dir():
        continue
    for split in ["pending", "train", "val", "test"]:
        img_dir = ds_dir / split / "images"
        lbl_dir = ds_dir / split / "labels"
        if img_dir.is_dir():
            remove_dir(img_dir)
        if lbl_dir.is_dir():
            remove_dir(lbl_dir)

print("\n=== 5. Cleaning pytest tmp cache ===")
pytest_tmp = ROOT / ".cache/pytest_tmp"
if pytest_tmp.exists():
    remove_dir(pytest_tmp)

print(f"\n==========================================")
print(f"TOTAL SPACE FREED: {bytes_freed / (1024**3):.2f} GB ({bytes_freed / (1024**2):.1f} MB)")
print(f"==========================================")
