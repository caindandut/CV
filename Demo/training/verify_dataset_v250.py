"""Verification assertion suite for dataset_v250."""
import json
import hashlib
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
V250 = ROOT / "dataset_v250"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Running Verification Assertions for Dataset v250 ===")

    # 1. Manifest
    with open(V250 / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == 2439, f"Expected 2439 rows, got {len(manifest)}"

    train_rows = [r for r in manifest if r.get("new_split") == "train"]
    val_rows = [r for r in manifest if r.get("new_split") == "val"]
    test_rows = [r for r in manifest if r.get("new_split") == "test"]
    pending_rows = [r for r in manifest if r.get("new_split") == "pending"]
    dup_rows = [r for r in manifest if r.get("new_split") == "excluded"]

    assert len(train_rows) == 1000, f"Expected 1000 train, got {len(train_rows)}"
    assert len(val_rows) == 210, f"Expected 210 val, got {len(val_rows)}"
    assert len(test_rows) == 68, f"Expected 68 test, got {len(test_rows)}"
    assert len(pending_rows) == 25, f"Expected 25 pending, got {len(pending_rows)}"
    assert len(dup_rows) == 1136, f"Expected 1136 duplicates, got {len(dup_rows)}"

    total_accepted = len(train_rows) + len(val_rows) + len(test_rows)
    assert total_accepted == 1278, f"Expected 1278 accepted, got {total_accepted}"

    # 2. Test split lock
    with open(V250 / "split_lock.json", "r", encoding="utf-8") as f:
        lock = json.load(f)
    test_ids_locked = set(lock["test_ids"])
    assert len(test_ids_locked) == 68
    manifest_test_ids = {r["id"] for r in test_rows}
    assert test_ids_locked == manifest_test_ids, "Test split lock violated!"

    # 3. Group isolation
    train_groups = {r.get("group") for r in train_rows if r.get("group")}
    val_groups = {r.get("group") for r in val_rows if r.get("group")}
    test_groups = {r.get("group") for r in test_rows if r.get("group")}

    assert not (train_groups & val_groups), f"Train/Val group leak: {train_groups & val_groups}"
    assert not (train_groups & test_groups), f"Train/Test group leak: {train_groups & test_groups}"
    assert not (val_groups & test_groups), f"Val/Test group leak: {val_groups & test_groups}"

    # 4. New files exist and match sha256
    new_items = [
        ("train/images/commons_vietnam__Hanoi_OldQuarter_street_traffic_unhelmeted_commuters_manual_v250.png",
         "train/labels/commons_vietnam__Hanoi_OldQuarter_street_traffic_unhelmeted_commuters_manual_v250.txt",
         "commons_vietnam/Hanoi_OldQuarter_street_traffic_unhelmeted_commuters_manual_v250.png"),
        ("train/images/commons_vietnam__Vietnam_tire_express_cargo_delivery_manual_v250.png",
         "train/labels/commons_vietnam__Vietnam_tire_express_cargo_delivery_manual_v250.txt",
         "commons_vietnam/Vietnam_tire_express_cargo_delivery_manual_v250.png"),
        ("val/images/commons_vietnam__MuiNe_beach_promenade_parked_touring_motorbikes_manual_v250.png",
         "val/labels/commons_vietnam__MuiNe_beach_promenade_parked_touring_motorbikes_manual_v250.txt",
         "commons_vietnam/MuiNe_beach_promenade_parked_touring_motorbikes_manual_v250.png"),
        ("train/images/commons_vietnam__NhaTrang_cultural_center_roundabout_commuters_manual_v250.png",
         "train/labels/commons_vietnam__NhaTrang_cultural_center_roundabout_commuters_manual_v250.txt",
         "commons_vietnam/NhaTrang_cultural_center_roundabout_commuters_manual_v250.png")
    ]

    manifest_map = {r["id"]: r for r in manifest}

    for img_rel, lbl_rel, rid in new_items:
        img_p = V250 / img_rel
        lbl_p = V250 / lbl_rel
        assert img_p.is_file(), f"Missing image: {img_p}"
        assert lbl_p.is_file(), f"Missing label: {lbl_p}"

        # SHA256 check
        actual_sha = sha(img_p)
        assert actual_sha == manifest_map[rid]["sha256"], f"SHA mismatch for {rid}"

        # PIL open
        with Image.open(img_p) as im:
            w, h = im.size
            assert w == manifest_map[rid]["width"]
            assert h == manifest_map[rid]["height"]

        # Parse labels
        for line in lbl_p.read_text("utf-8").strip().splitlines():
            parts = line.split()
            assert len(parts) == 5, f"Bad label format: {line}"
            cls_id = int(parts[0])
            assert cls_id in (0, 1, 2), f"Bad class id: {cls_id}"
            cx, cy, bw, bh = map(float, parts[1:])
            assert 0 < cx < 1 and 0 < cy < 1, f"Center out of bounds: {line}"
            assert 0 < bw <= 1 and 0 < bh <= 1, f"Dims out of bounds: {line}"
            assert cx - bw/2 >= -1e-5 and cx + bw/2 <= 1.00001
            assert cy - bh/2 >= -1e-5 and cy + bh/2 <= 1.00001

    print("ALL VERIFICATION CHECKS PASSED FOR DATASET v250!")

if __name__ == "__main__":
    main()
