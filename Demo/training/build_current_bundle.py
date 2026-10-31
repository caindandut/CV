"""Package one immutable reviewed snapshot for Colab without replacing v1 artifacts."""

import argparse
import ast
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from train_colab import PRETRAINED_SHA256, verify_dataset

ROOT = Path(__file__).resolve().parent
CODE_FILES = [
    "train_colab.py",
    "scconv.py",
    "train_config.json",
    "smoke_test.py",
    "evaluate_model.py",
    "evaluation_metrics.py",
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render_notebook(root, dataset_name, archive_name, notebook_name, rows, target=2000):
    nb = json.loads((root / "Train_SCConv_Clean_70_Colab.ipynb").read_text("utf-8"))
    active = [r for r in rows if r["new_split"] != "pending"]
    counts = Counter(r["new_split"] for r in active)
    class2 = sum(
        b[0] == 2 for r in active if r["new_split"] == "test" for b in r["labels"]
    )
    version = dataset_name.removeprefix("dataset_")
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        text = text.replace("dataset_v1", dataset_name)
        text = text.replace("training_colab_70.zip", archive_name)
        text = text.replace("scconv_clean70_v1", f"scconv_clean70_{version}")
        cell["source"] = text.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])

    intro = next(cell for cell in nb["cells"] if cell["cell_type"] == "markdown")
    intro_text = (
        f"# SCConv đúng 3 lớp — tối đa 70 epoch — {dataset_name}\n\n"
        f"Bản dữ liệu được chấp nhận hiện có **{len(active)}/2.000 ảnh**; "
        f"train {counts['train']}, validation {counts['val']}, test {counts['test']}. "
        "Đây là tiến độ số lượng, không phải độ chính xác model. "
        "Ảnh đang chờ duyệt không được đóng gói để train. Mục tiêu 2.000 ảnh chưa hoàn thành.\n\n"
        f"Chọn Runtime → Change runtime type → GPU. Tải `{archive_name}` "
        "lên `MyDrive/HelmetSCConv70`. Đọc README.md và giấy phép theo từng nguồn.\n\n"
        "Chạy SCConv trước; baseline là lựa chọn đối chứng sau. 70 là tổng số epoch, "
        "kể cả resume; patience=20, imgsz=768, batch thử 8→4→2. "
        "Colab không bảo đảm hoàn tất trong một phiên. Không thay checkpoint demo.\n"
    )
    if len(active) >= target:
        intro_text = intro_text.replace("Mục tiêu 2.000 ảnh chưa hoàn thành.", "Đã đạt mục tiêu số lượng ảnh đã duyệt.")
    intro_text = intro_text.replace("2.000", f"{target:,}".replace(",", "."))
    intro["source"] = intro_text.splitlines(keepends=True)

    test_note = next(
        cell
        for cell in nb["cells"]
        if cell["cell_type"] == "markdown" and "Test cuối" in "".join(cell["source"])
    )
    test_text = (
        "## Test cuối — chỉ sau khi khóa cấu hình và ngưỡng\n"
        f"Test có {counts['test']} ảnh và {class2} box người không đội mũ. "
        "Giữ test khóa; chỉ chọn ngưỡng và epoch từ validation. "
        "Ô test tắt mặc định và có marker chống chạy lặp.\n"
    )
    test_note["source"] = test_text.splitlines(keepends=True)
    nb["metadata"]["colab"]["name"] = notebook_name
    return json.dumps(nb, ensure_ascii=False, indent=2).encode("utf-8")


def render_readme(dataset_name, rows, target=2000):
    active = [r for r in rows if r["new_split"] != "pending"]
    counts = Counter(r["new_split"] for r in active)
    pending = sum(r["new_split"] == "pending" for r in rows)
    missing = max(0, target - len(active))
    missing_text = f"{missing:,}".replace(",", ".")
    version = dataset_name.removeprefix("dataset_")
    return (
        f"# Gói Colab SCConv tối đa 70 epoch — {dataset_name}\n\n"
        f"Có **{len(active)}/2.000 ảnh sạch** đã được chấp nhận; còn thiếu **{missing_text} ảnh**. "
        f"Train {counts['train']}, validation {counts['val']}, test {counts['test']}; "
        f"{pending} ảnh vẫn chờ duyệt và không có media chờ duyệt trong ZIP. "
        "Mục tiêu dữ liệu chưa hoàn thành; chưa chạy train chính hoặc đo mAP.\n\n"
        "## Chạy trên Colab\n\n"
        f"Tải `training_colab_70_{version}.zip` lên `MyDrive/HelmetSCConv70`, "
        f"mở `Train_SCConv_Clean_70_{version}_Colab.ipynb`, chọn GPU và chạy từng ô. "
        "Notebook đọc ảnh từ `/content`, lưu optimizer, learning rate, epoch và nhật ký "
        "lên Drive sau mỗi epoch, đồng thời giữ checkpoint dự phòng. Khi resume, dùng cùng "
        "dataset và tên run; 70 là tổng epoch, không cộng thêm 70.\n\n"
        "## Cấu hình và đánh giá\n\n"
        "Ultralytics 8.4.155; `epochs=70`, `patience=20`, `imgsz=768`, `seed=0`, `close_mosaic=10`. "
        "Notebook dựng head 3 lớp trước hai khối SCConv, thử batch 8 rồi 4 rồi 2 và không tự giảm "
        "độ phân giải. Chạy kiểm tra ngắn về train, lưu/nạp và resume trước run chính. Colab không "
        "bảo đảm 70 epoch hoàn tất trong một phiên: "
        "https://research.google.com/colaboratory/faq.html\n\n"
        "Chọn ngưỡng bằng validation; chỉ đánh giá test sau khi khóa mô hình và ngưỡng. "
        "Gói này không chứa kết quả accuracy hay mAP và không thay checkpoint demo.\n\n"
        "## Nguồn và giấy phép\n\n"
        f"Xem `{dataset_name}/ATTRIBUTION.md` và `usage_restrictions.json` để biết ghi công và "
        "điều kiện sử dụng của từng ảnh. Một số ảnh chỉ được dùng phi thương mại; điều kiện đó "
        "chỉ áp dụng cho các mục được liệt kê, không áp dụng lên toàn bộ dữ liệu gốc. Manifest "
        "giữ nguyên byte để bảo toàn split đã khóa; đường dẫn nguồn Windows phục vụ truy nguyên, "
        "còn train dùng thư mục split trong ZIP.\n"
    )


def build_bundle(root, dataset_name):
    root = Path(root).resolve()
    progress = json.loads((root / "acquisition_2000/progress.json").read_text("utf-8"))
    target = progress.get("target_clean_images", 2000)
    if type(target) is not int or target <= 0:
        raise ValueError("target_clean_images must be a positive integer")
    if not re.fullmatch(r"dataset_v[1-9][0-9]*", dataset_name):
        raise ValueError("Expected a frozen dataset_v<number> directory")
    dataset = (root / dataset_name).resolve()
    if not dataset.is_relative_to(root):
        raise ValueError("Dataset escapes workspace")
    destination = root / "colab_bundles" / dataset_name
    if destination.exists():
        raise FileExistsError(f"Immutable bundle already exists: {destination}")
    verify_dataset(dataset / "data.yaml")
    config = json.loads((root / "train_config.json").read_text("utf-8"))
    if config["epochs"] != 70 or config["imgsz"] != 768 or config["patience"] != 20:
        raise ValueError("Expected total epochs=70, imgsz=768, patience=20")
    weights = root / ".cache/yolo26s.pt"
    if digest(weights) != PRETRAINED_SHA256:
        raise ValueError("Pretrained weights checksum mismatch")
    rows = json.loads((dataset / "manifest.json").read_text("utf-8"))
    active = [r for r in rows if r["new_split"] != "pending"]
    version = dataset_name.removeprefix("dataset_")
    archive_name = f"training_colab_70_{version}.zip"
    notebook_name = f"Train_SCConv_Clean_70_{version}_Colab.ipynb"
    files = {"code/" + name: root / name for name in CODE_FILES}
    files["pretrained/yolo26s.pt"] = weights
    # Frozen metadata retains its original bytes and manifest hash, including the
    # pending ledger. Only accepted split images/labels are included as media.
    for path in dataset.iterdir():
        if path.is_file() and path.suffix in {".json", ".yaml", ".md"}:
            files[f"{dataset_name}/{path.name}"] = path
    for split in ["train", "val", "test"]:
        for kind in ["images", "labels"]:
            for path in (dataset / split / kind).iterdir():
                if path.is_file():
                    files[path.relative_to(root).as_posix()] = path
    for name, path in files.items():
        if not path.resolve().is_relative_to(root) or ".." in Path(name).parts:
            raise ValueError("Bundle member escapes workspace")
    verification = root / "verification" / f"{dataset_name}.json"
    if not verification.is_file():
        raise FileNotFoundError(
            f"Dataset verification receipt missing: {verification}. "
            "Run verify_dataset_2000.py before building the Colab bundle."
        )
    files["verification/dataset.json"] = verification
    notebook = render_notebook(root, dataset_name, archive_name, notebook_name, rows, target)
    pending = [
        {"id": r["id"], "reason": r.get("reason", "")}
        for r in rows
        if r["new_split"] == "pending"
    ]
    stats = dict(Counter(r["new_split"] for r in active))
    metadata = {
        "dataset": dataset_name,
        "accepted_images": len(active),
        "target": target,
        "target_achieved": len(active) >= target,
        "split_images": stats,
        "manifest_sha256": digest(dataset / "manifest.json"),
        "epochs_total": 70,
        "pending_images_included": False,
        "verification_scope": "packaging_and_frozen_training_integrity; not model accuracy",
    }
    readme = render_readme(dataset_name, rows, target).replace("2.000", f"{target:,}".replace(",", ".")).encode("utf-8")
    if len(active) >= target:
        readme = readme.replace("Mục tiêu dữ liệu chưa hoàn thành".encode("utf-8"), "Đã đạt mục tiêu số lượng ảnh đã duyệt".encode("utf-8"))
    generated = {
        notebook_name: notebook,
        "README.md": readme,
        "bundle_metadata.json": json.dumps(
            metadata, ensure_ascii=False, indent=2
        ).encode("utf-8"),
        "pending_review_list.json": json.dumps(
            pending, ensure_ascii=False, indent=2
        ).encode("utf-8"),
    }
    hashes = {name: digest(path) for name, path in files.items()}
    hashes.update(
        {name: hashlib.sha256(data).hexdigest() for name, data in generated.items()}
    )
    destination.mkdir(parents=True)
    archive_path = destination / archive_name
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, path in files.items():
            z.write(path, name)
        for name, data in generated.items():
            z.writestr(name, data)
        z.writestr("files_sha256.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(archive_path) as z:
        for name, expected in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError(f"Archive hash mismatch: {name}")
    (destination / notebook_name).write_bytes(notebook)
    (destination / "README.md").write_bytes(readme)
    result = dict(
        metadata,
        archive=archive_path.relative_to(root).as_posix(),
        notebook=notebook_name,
        bytes=archive_path.stat().st_size,
        sha256=digest(archive_path),
        members=len(hashes) + 1,
    )
    (destination / "bundle_info.json").write_text(json.dumps(result, indent=2), "utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(ROOT, args.dataset)))
