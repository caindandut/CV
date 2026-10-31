"""Verify hashes, CRC, dataset inventory and notebook syntax for the v126 bundle."""

import ast
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = "dataset_v126"
BUNDLE = ROOT / "colab_bundles/dataset_v126/training_colab_70_v126.zip"
RECEIPT = ROOT / "verification/colab_bundle_v126.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert BUNDLE.is_file(), f"Missing bundle: {BUNDLE}"
    with zipfile.ZipFile(BUNDLE) as archive:
        assert archive.testzip() is None, "ZIP CRC failed"
        names = set(archive.namelist())
        hashes = json.loads(archive.read("files_sha256.json"))
        assert set(hashes) <= names
        for name, expected in hashes.items():
            assert sha256(archive.read(name)) == expected, name

        config = json.loads(archive.read("code/train_config.json"))
        assert config["epochs"] == 70
        assert config["imgsz"] == 768 and config["patience"] == 20

        manifest = json.loads(archive.read(f"{DATASET}/manifest.json"))
        active = [row for row in manifest if row["new_split"] != "pending"]
        split_counts = Counter(row["new_split"] for row in active)
        assert len(active) == 581
        for split in ("train", "val", "test"):
            count = sum(
                name.startswith(f"{DATASET}/{split}/images/") for name in names
            )
            assert count == split_counts[split]
        assert not any("/pending/images/" in name for name in names)

        statistics = json.loads(archive.read(f"{DATASET}/statistics.json"))
        class_counts = {
            split: statistics[split]["classes"] for split in ("train", "val", "test")
        }
        notebook_name = "Train_SCConv_Clean_70_v126_Colab.ipynb"
        notebook = json.loads(archive.read(notebook_name))
        python_cells = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                python_cells += 1

        dataset_receipt = json.loads(archive.read("verification/dataset.json"))
        assert dataset_receipt["passed"] is True

        receipt = {
            "passed": True,
            "dataset": DATASET,
            "archive": BUNDLE.relative_to(ROOT).as_posix(),
            "archive_sha256": hashlib.file_digest(BUNDLE.open("rb"), "sha256").hexdigest(),
            "archive_crc": "passed",
            "member_hashes_checked": len(hashes),
            "zip_members": len(names),
            "manifest_sha256": sha256(archive.read(f"{DATASET}/manifest.json")),
            "accepted_images": len(active),
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

    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()


