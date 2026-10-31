"""Verify hashes, CRC, clean split counts and notebook syntax for a Colab bundle."""

import argparse
import ast
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify(dataset_name, expected_target=None):
    match = re.fullmatch(r"dataset_v(\d+)", dataset_name)
    if not match:
        raise ValueError("Dataset must use the frozen dataset_vN name")
    version = match.group(1)
    bundle_dir = ROOT / "colab_bundles" / dataset_name
    info = json.loads((bundle_dir / "bundle_info.json").read_text(encoding="utf-8"))
    bundle = ROOT / info["archive"]
    receipt_path = ROOT / "verification" / f"colab_bundle_v{version}.json"
    assert info["dataset"] == dataset_name
    assert bundle.is_file(), f"Missing bundle: {bundle}"

    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None, "ZIP CRC failed"
        names = set(archive.namelist())
        metadata = json.loads(archive.read("bundle_metadata.json"))
        target = metadata["target"]
        assert type(target) is int and target > 0
        if expected_target is not None:
            assert target == expected_target, "Bundle target differs from requested target"
        target_text = f"{target:,}".replace(",", ".")
        assert not any(
            name.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
            for name in names
        ), "Raw video research media must stay outside the Colab bundle"
        assert not any("research_pexels/" in name for name in names)
        hashes = json.loads(archive.read("files_sha256.json"))
        assert set(hashes) <= names
        for name, expected in hashes.items():
            assert sha256(archive.read(name)) == expected, name

        config = json.loads(archive.read("code/train_config.json"))
        assert config["epochs"] == 70
        assert config["imgsz"] == 768 and config["patience"] == 20

        data_yaml = yaml.safe_load(archive.read(f"{dataset_name}/data.yaml"))
        assert data_yaml["nc"] == 3
        assert data_yaml["names"] == [
            "motorcycle",
            "rider_with_helmet",
            "rider_without_helmet",
        ]

        manifest = json.loads(archive.read(f"{dataset_name}/manifest.json"))
        active = [row for row in manifest if row["new_split"] != "pending"]
        split_counts = Counter(row["new_split"] for row in active)
        assert len(active) == info["accepted_images"]
        assert metadata["accepted_images"] == len(active)
        assert metadata["target_achieved"] == (len(active) >= target)
        for split in ("train", "val", "test"):
            count = sum(
                name.startswith(f"{dataset_name}/{split}/images/") for name in names
            )
            assert count == split_counts[split]
        assert not any("/pending/images/" in name for name in names)

        statistics = json.loads(archive.read(f"{dataset_name}/statistics.json"))
        class_counts = {
            split: statistics[split]["classes"] for split in ("train", "val", "test")
        }
        notebook_name = info["notebook"]
        notebook = json.loads(archive.read(notebook_name))
        intro = "".join(notebook["cells"][0]["source"])
        assert f"{len(active)}/{target_text}" in intro
        readme = archive.read("README.md").decode("utf-8")
        assert f"{len(active)}/{target_text}" in readme
        if len(active) < target:
            missing_text = f"{target-len(active):,}".replace(",", ".")
            assert f"Mục tiêu {target_text} ảnh chưa hoàn thành" in intro
            assert missing_text in readme
        else:
            assert f"Mục tiêu {target_text} ảnh chưa hoàn thành" not in intro
        assert "giấy phép" in readme

        python_cells = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                python_cells += 1

        dataset_receipt = json.loads(archive.read("verification/dataset.json"))
        assert dataset_receipt["passed"] is True
        assert dataset_receipt["dataset"] == dataset_name

        with bundle.open("rb") as stream:
            archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = {
            "passed": True,
            "dataset": dataset_name,
            "archive": bundle.relative_to(ROOT).as_posix(),
            "archive_sha256": archive_sha256,
            "archive_crc": "passed",
            "member_hashes_checked": len(hashes),
            "zip_members": len(names),
            "manifest_sha256": sha256(archive.read(f"{dataset_name}/manifest.json")),
            "accepted_images": len(active),
            "target_clean_images": target,
            "target_achieved": len(active) >= target,
            "split_images": dict(split_counts),
            "box_counts_by_split": class_counts,
            "epochs_total": config["epochs"],
            "imgsz": config["imgsz"],
            "patience": config["patience"],
            "notebook_python_cells_parsed": python_cells,
            "pending_media_included": False,
            "dataset_verifier_passed": True,
            "scope": "package integrity only; not training or model accuracy",
        }

    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-target", type=int)
    args = parser.parse_args()
    verify(args.dataset, args.expected_target)
