import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_current_bundle as bundler


@pytest.fixture
def fixture_root(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    for name in bundler.CODE_FILES + ["Train_SCConv_Clean_70_Colab.ipynb"]:
        (tmp_path / name).write_bytes((source / name).read_bytes())
    (tmp_path / ".cache").mkdir()
    weights = b"known fixture weights"
    (tmp_path / ".cache/yolo26s.pt").write_bytes(weights)
    monkeypatch.setattr(
        bundler, "PRETRAINED_SHA256", hashlib.sha256(weights).hexdigest()
    )
    dataset = tmp_path / "dataset_v99"
    dataset.mkdir()
    (dataset / "data.yaml").write_text(
        "names: [motorcycle, rider_with_helmet, rider_without_helmet]\nnc: 3\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
    )
    rows = []
    for i, split in enumerate(["train", "val", "test", "pending"]):
        for kind in ["images", "labels"]:
            (dataset / split / kind).mkdir(parents=True)
        image = dataset / split / "images" / f"source__{i}.png"
        Image.new("RGB", (32, 32), (i * 60, 0, 0)).save(image)
        labels = [[c, 0.5, 0.5, 0.2, 0.2] for c in range(3)]
        (dataset / split / "labels" / f"source__{i}.txt").write_text(
            "\n".join(" ".join(map(str, box)) for box in labels)
        )
        rows.append(
            {
                "id": f"source/{i}.png",
                "new_split": split,
                "labels": labels,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "group": f"group{i}",
                "source_verified": True,
                "review_status": "pending" if split == "pending" else "accepted",
            }
        )
    manifest = json.dumps(rows).encode()
    (dataset / "manifest.json").write_bytes(manifest)
    (dataset / "split_lock.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "test_ids": ["source/2.png"],
            }
        )
    )
    (dataset / "attribution.json").write_text("[]")
    (dataset / "usage_restrictions.json").write_text("{}")
    return tmp_path


def test_bundle_is_portable_excludes_pending_and_reports_actual_counts(fixture_root):
    result = bundler.build_bundle(fixture_root, "dataset_v99")
    assert result["accepted_images"] == 3 and not result["target_achieved"]
    archive = fixture_root / result["archive"]
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert not any("/pending/" in name for name in names)
        assert "dataset_v99/usage_restrictions.json" in names
        hashes = json.loads(z.read("files_sha256.json"))
        assert set(hashes) == set(names) - {"files_sha256.json"}
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
        nb = json.loads(z.read(result["notebook"]))
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "dataset_v99" in text and "dataset_v1'" not in text
        assert "scconv_clean70_v99" in text
        assert "validate_resume_checkpoint(payload, 70)" in text
        assert "RUN_FINAL_TEST = False" in text
        assert "test hiện không có GT lớp 2" not in text.lower()
    extracted = fixture_root / "relocated"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    bundler.verify_dataset(extracted / "dataset_v99/data.yaml")


def test_existing_bundle_is_not_overwritten(fixture_root):
    bundler.build_bundle(fixture_root, "dataset_v99")
    with pytest.raises(FileExistsError):
        bundler.build_bundle(fixture_root, "dataset_v99")


def test_corrupt_dataset_is_rejected_before_output(fixture_root):
    (fixture_root / "dataset_v99/train/labels/source__0.txt").write_text("")
    with pytest.raises(AssertionError):
        bundler.build_bundle(fixture_root, "dataset_v99")
    assert not (fixture_root / "colab_bundles/dataset_v99").exists()


@pytest.mark.parametrize(
    "name", ["../dataset_v99", "dataset_v99/../dataset_v99", "dataset_v99\\bad"]
)
def test_dataset_path_cannot_escape_workspace(fixture_root, name):
    with pytest.raises(ValueError):
        bundler.build_bundle(fixture_root, name)


def test_wrong_epoch_budget_is_rejected(fixture_root):
    path = fixture_root / "train_config.json"
    config = json.loads(path.read_text())
    config["epochs"] = 150
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="70"):
        bundler.build_bundle(fixture_root, "dataset_v99")
