"""Safe cleanup script to free disk space in Demo directory.
Removes:
- Intermediate dataset images/labels from dataset_v92 (keeping manifest, split_lock, data.yaml, etc.)
- Visual HTML review galleries and asset folders (review_v92*, review_v93*)
- Legacy training_colab_70.zip at training root (superseded by colab_bundles/dataset_v93)
- Large obsolete uploaded test videos in storage/uploads and storage/temp
Preserves:
- Active dataset_v93 (all images and labels)
- All manifest.json, split_lock.json, data.yaml, attribution.json across all datasets v1..v93
- All active colab bundles (dataset_v74 baseline and dataset_v93 active)
- All raw archives, evidence images, and review ledgers required for cryptographic integrity
"""

import shutil
from pathlib import Path

ROOT = Path("d:/Nam 4/CV/Demo/training")
DEMO = Path("d:/Nam 4/CV/Demo")

bytes_freed = 0


def remove_dir(path: Path):
    global bytes_freed
    if not path.is_dir():
        return
    sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    shutil.rmtree(path)
    bytes_freed += sz
    print(f"Removed dir: {path} ({sz / (1024**2):.1f} MB)")


def remove_file(path: Path):
    global bytes_freed
    if not path.is_file():
        return
    sz = path.stat().st_size
    path.unlink()
    bytes_freed += sz
    print(f"Removed file: {path.name} ({sz / (1024**2):.1f} MB)")


def main():
    print("=== 1. Cleaning intermediate dataset_v92 images & labels ===")
    v92 = ROOT / "dataset_v92"
    if v92.is_dir():
        for split in ["train", "val", "test", "pending"]:
            for kind in ["images", "labels"]:
                p = v92 / split / kind
                if p.is_dir():
                    remove_dir(p)

    print("\n=== 2. Cleaning HTML review galleries and assets ===")
    for p in list(ROOT.glob("review_v*.html")):
        remove_file(p)
    for p in list(ROOT.glob("review_v*_assets")):
        remove_dir(p)

    print("\n=== 3. Cleaning legacy training_colab_70.zip in training root ===")
    legacy_zip = ROOT / "training_colab_70.zip"
    if legacy_zip.is_file():
        remove_file(legacy_zip)

    print("\n=== 4. Cleaning large test videos in storage/uploads and storage/temp ===")
    large_upload = DEMO / "storage/uploads/7d8ab370-f27c-4cbf-8517-6c1c2db00961.mp4"
    if large_upload.is_file():
        remove_file(large_upload)

    temp_dir = DEMO / "storage/temp"
    if temp_dir.is_dir():
        for f in temp_dir.glob("*.mp4"):
            remove_file(f)

    print("\n=== 5. Cleaning pytest cache ===")
    pytest_tmp = ROOT / ".cache/pytest_tmp"
    if pytest_tmp.exists():
        remove_dir(pytest_tmp)
    pytest_temp = ROOT / ".cache/pytest_temp"
    if pytest_temp.exists():
        remove_dir(pytest_temp)

    print("\n==========================================")
    print(f"TOTAL SPACE FREED: {bytes_freed / (1024**3):.2f} GB ({bytes_freed / (1024**2):.1f} MB)")
    print("==========================================")


if __name__ == "__main__":
    main()
