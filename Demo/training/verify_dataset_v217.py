"""Verification assertion suite for dataset_v217."""
import json
import hashlib
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
V217 = ROOT / "dataset_v217"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Running Verification Assertions for Dataset v217 ===")

    # 1. Manifest
    with open(V217 / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == 2307, f"Expected 2307 rows, got {len(manifest)}"

    train_rows = [r for r in manifest if r.get("new_split") == "train"]
    val_rows = [r for r in manifest if r.get("new_split") == "val"]
    test_rows = [r for r in manifest if r.get("new_split") == "test"]
    pending_rows = [r for r in manifest if r.get("new_split") == "pending"]
    dup_rows = [r for r in manifest if r.get("new_split") == "excluded"]

    assert len(train_rows) == 922, f"Expected 922 train, got {len(train_rows)}"
    assert len(val_rows) == 156, f"Expected 156 val, got {len(val_rows)}"
    assert len(test_rows) == 68, f"Expected 68 test, got {len(test_rows)}"
    assert len(pending_rows) == 25, f"Expected 25 pending, got {len(pending_rows)}"
    assert len(dup_rows) == 1136, f"Expected 1136 duplicates, got {len(dup_rows)}"

    total_accepted = len(train_rows) + len(val_rows) + len(test_rows)
    assert total_accepted == 1146, f"Expected 1146 accepted, got {total_accepted}"

    # 2. Test split lock
    with open(V217 / "split_lock.json", "r", encoding="utf-8") as f:
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
        ("train/images/commons_vietnam__HCMC_DienBienPhu_rain_traffic_manual_v217.png",
         "train/labels/commons_vietnam__HCMC_DienBienPhu_rain_traffic_manual_v217.txt",
         "commons_vietnam/HCMC_DienBienPhu_rain_traffic_manual_v217.png"),
        ("train/images/commons_vietnam__HCMC_LyTuTrong_traffic_stop_manual_v217.png",
         "train/labels/commons_vietnam__HCMC_LyTuTrong_traffic_stop_manual_v217.txt",
         "commons_vietnam/HCMC_LyTuTrong_traffic_stop_manual_v217.png"),
        ("train/images/commons_vietnam__Hue_ImperialBridge_citadel_commuters_manual_v217.png",
         "train/labels/commons_vietnam__Hue_ImperialBridge_citadel_commuters_manual_v217.txt",
         "commons_vietnam/Hue_ImperialBridge_citadel_commuters_manual_v217.png"),
        ("val/images/commons_vietnam__DaLat_Market_Lead_scooter_manual_v217.png",
         "val/labels/commons_vietnam__DaLat_Market_Lead_scooter_manual_v217.txt",
         "commons_vietnam/DaLat_Market_Lead_scooter_manual_v217.png")
    ]

    manifest_map = {r["id"]: r for r in manifest}

    for img_rel, lbl_rel, rid in new_items:
        img_p = V217 / img_rel
        lbl_p = V217 / lbl_rel
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

    # 5. Check statistics
    with open(V217 / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    assert stats["train"]["images"] == 922
    assert stats["val"]["images"] == 156
    assert stats["test"]["images"] == 68
    assert stats["train"]["classes"]["0"] == 1630 + 12
    assert stats["train"]["classes"]["1"] == 816 + 13
    assert stats["train"]["classes"]["2"] == 506
    assert stats["val"]["classes"]["0"] == 317 + 1
    assert stats["val"]["classes"]["1"] == 265 + 1
    assert stats["val"]["classes"]["2"] == 85

    # 6. Progress and Verification JSONs
    with open(ROOT / "cloud_handoff" / "progress_5000.json", "r", encoding="utf-8") as f:
        prog = json.load(f)
    assert prog["accepted_clean_images"] == 1146
    assert prog["remaining_to_goal"] == 3854
    assert prog["snapshot"] == "dataset_v217"

    with open(ROOT / "verification" / "dataset_v217.json", "r", encoding="utf-8") as f:
        ver = json.load(f)
    assert ver["passed"] is True
    assert ver["accepted_clean_images"] == 1146
    assert ver["manifest_sha256"] == sha(V217 / "manifest.json")
    assert ver["statistics_sha256"] == sha(V217 / "statistics.json")
    assert ver["split_lock_sha256"] == sha(V217 / "split_lock.json")

    print("ALL 6 VERIFICATION TEST SUITES PASSED 100% FOR V217!")

if __name__ == "__main__":
    main()
