"""Verification assertion suite for dataset_v224."""
import json
import hashlib
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
V224 = ROOT / "dataset_v224"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Running Verification Assertions for Dataset v224 ===")

    # 1. Manifest
    with open(V224 / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == 2335, f"Expected 2335 rows, got {len(manifest)}"

    train_rows = [r for r in manifest if r.get("new_split") == "train"]
    val_rows = [r for r in manifest if r.get("new_split") == "val"]
    test_rows = [r for r in manifest if r.get("new_split") == "test"]
    pending_rows = [r for r in manifest if r.get("new_split") == "pending"]
    dup_rows = [r for r in manifest if r.get("new_split") == "excluded"]

    assert len(train_rows) == 942, f"Expected 942 train, got {len(train_rows)}"
    assert len(val_rows) == 164, f"Expected 164 val, got {len(val_rows)}"
    assert len(test_rows) == 68, f"Expected 68 test, got {len(test_rows)}"
    assert len(pending_rows) == 25, f"Expected 25 pending, got {len(pending_rows)}"
    assert len(dup_rows) == 1136, f"Expected 1136 duplicates, got {len(dup_rows)}"

    total_accepted = len(train_rows) + len(val_rows) + len(test_rows)
    assert total_accepted == 1174, f"Expected 1174 accepted, got {total_accepted}"

    # 2. Test split lock
    with open(V224 / "split_lock.json", "r", encoding="utf-8") as f:
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
        ("train/images/commons_vietnam__Northern_highway_pig_transport_02_manual_v224.png",
         "train/labels/commons_vietnam__Northern_highway_pig_transport_02_manual_v224.txt",
         "commons_vietnam/Northern_highway_pig_transport_02_manual_v224.png"),
        ("val/images/commons_vietnam__Hanoi_woman_scooter_sunhood_manual_v224.png",
         "val/labels/commons_vietnam__Hanoi_woman_scooter_sunhood_manual_v224.txt",
         "commons_vietnam/Hanoi_woman_scooter_sunhood_manual_v224.png"),
        ("train/images/commons_vietnam__Hanoi_waiting_red_light_father_son_manual_v224.png",
         "train/labels/commons_vietnam__Hanoi_waiting_red_light_father_son_manual_v224.txt",
         "commons_vietnam/Hanoi_waiting_red_light_father_son_manual_v224.png"),
        ("train/images/commons_vietnam__Saigon_intersection_traffic_commuters_manual_v224.png",
         "train/labels/commons_vietnam__Saigon_intersection_traffic_commuters_manual_v224.txt",
         "commons_vietnam/Saigon_intersection_traffic_commuters_manual_v224.png")
    ]

    manifest_map = {r["id"]: r for r in manifest}

    for img_rel, lbl_rel, rid in new_items:
        img_p = V224 / img_rel
        lbl_p = V224 / lbl_rel
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
    with open(V224 / "statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    assert stats["train"]["images"] == 942
    assert stats["val"]["images"] == 164
    assert stats["test"]["images"] == 68
    assert stats["train"]["classes"]["0"] == 1682 + 6
    assert stats["train"]["classes"]["1"] == 882 + 6
    assert stats["train"]["classes"]["2"] == 507 + 2
    assert stats["val"]["classes"]["0"] == 331 + 1
    assert stats["val"]["classes"]["1"] == 273
    assert stats["val"]["classes"]["2"] == 88 + 1

    # 6. Progress and Verification JSONs
    with open(ROOT / "cloud_handoff" / "progress_5000.json", "r", encoding="utf-8") as f:
        prog = json.load(f)
    assert prog["accepted_clean_images"] == 1174
    assert prog["remaining_to_goal"] == 3826
    assert prog["snapshot"] == "dataset_v224"

    with open(ROOT / "verification" / "dataset_v224.json", "r", encoding="utf-8") as f:
        ver = json.load(f)
    assert ver["passed"] is True
    assert ver["accepted_clean_images"] == 1174
    assert ver["manifest_sha256"] == sha(V224 / "manifest.json")
    assert ver["statistics_sha256"] == sha(V224 / "statistics.json")
    assert ver["split_lock_sha256"] == sha(V224 / "split_lock.json")

    print("ALL 6 VERIFICATION TEST SUITES PASSED 100% FOR V224!")

if __name__ == "__main__":
    main()
