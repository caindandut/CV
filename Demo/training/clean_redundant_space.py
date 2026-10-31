import shutil
import os
from pathlib import Path

ROOT = Path("d:/Nam 4/CV/Demo/training")
bytes_freed = 0

def remove_dir(path: Path):
    global bytes_freed
    if not path.is_dir():
        return
    sz = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    shutil.rmtree(path)
    bytes_freed += sz
    print(f"Removed dir: {path.name} ({sz / (1024**2):.1f} MB)")

def remove_file(path: Path):
    global bytes_freed
    if not path.is_file():
        return
    sz = path.stat().st_size
    path.unlink()
    bytes_freed += sz
    print(f"Removed file: {path.name} ({sz / (1024**2):.1f} MB)")

def main():
    print("=== 1. Cleaning 90 review_v*_assets directories ===")
    for p in list(ROOT.glob("review_v*_assets")):
        remove_dir(p)

    print("\n=== 2. Cleaning 90 review_v*.html files ===")
    for p in list(ROOT.glob("review_v*.html")):
        remove_file(p)

    print("\n=== 3. Cleaning verification smoke and bundle extract directories ===")
    ver_dir = ROOT / "verification"
    for name in [
        "smoke_final", "smoke_reviewed", "smoke_v3", "colab_v54_smoke",
        "smoke_portable", "smoke", "colab_v56_relocated", "colab_v54_relocated",
        "colab_bundle_v91_extract", "colab_bundle_v90_extract"
    ]:
        p = ver_dir / name
        if p.exists():
            remove_dir(p)

    print("\n=== 4. Cleaning intermediate dataset images (v1..v3, v89) ===")
    for v in [1, 2, 3, 89]:
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

    print("\n=== 5. Cleaning intermediate colab bundle v90 ===")
    b90 = ROOT / "colab_bundles/dataset_v90"
    if b90.is_dir():
        remove_dir(b90)

    print("\n=== 6. Cleaning pytest cache ===")
    pytest_tmp = ROOT / ".cache/pytest_tmp"
    if pytest_tmp.exists():
        remove_dir(pytest_tmp)

    print("\n==========================================")
    print(f"TOTAL SPACE FREED: {bytes_freed / (1024**3):.2f} GB ({bytes_freed / (1024**2):.1f} MB)")
    print("==========================================")

if __name__ == "__main__":
    main()
