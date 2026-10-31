"""Prepare a Vietnam-only Stage 2 helmet classifier dataset.

This module intentionally performs extraction, audit, and reporting only.  It
does not import a detector/classifier framework and it never trains a model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised only without Pillow
    raise RuntimeError(
        "Pillow is required to create native-resolution crops and contact sheets."
    ) from exc


REQUIRED_WORKSPACE = "nckh-2023"
REQUIRED_PROJECT = "helmet-detection-project"
REQUIRED_VERSION = 19
REQUIRED_NAMES = ("motorcycle", "rider_with_helmet", "rider_without_helmet")
REQUIRED_NC = 3
CLASS_TO_NAME = {1: "helmet", 2: "no_helmet"}
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANIFEST_FIELDS = (
    "split",
    "class_name",
    "class_id",
    "source_image",
    "source_label",
    "object_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "source_width",
    "source_height",
    "crop_width",
    "crop_height",
    "bbox_area_normalized",
    "output_crop",
)


class DatasetError(RuntimeError):
    """Raised when the source dataset is missing or fails a safety check."""


class LeakageError(DatasetError):
    """Raised when exact data leakage is found across source splits."""


@dataclass(frozen=True)
class DatasetMetadata:
    yaml_path: Path
    train: str | None
    val: str | None
    test: str | None
    nc: int | None
    names: tuple[str, ...]
    workspace: str | None
    project: str | None
    version: int | None

    @property
    def is_required_dataset(self) -> bool:
        return (
            self.workspace == REQUIRED_WORKSPACE
            and self.project == REQUIRED_PROJECT
            and self.version == REQUIRED_VERSION
            and self.nc == REQUIRED_NC
            and self.names == REQUIRED_NAMES
        )


@dataclass(frozen=True)
class YoloRow:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line_number: int
    raw: str


@dataclass(frozen=True)
class ParsedLabels:
    valid_rows: tuple[YoloRow, ...]
    invalid_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class CropRecord:
    split: str
    class_name: str
    class_id: int
    source_image: Path
    source_label: Path
    object_index: int
    bbox: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    source_width: int
    source_height: int
    output_crop: Path

    @property
    def crop_width(self) -> int:
        return self.crop_box[2] - self.crop_box[0]

    @property
    def crop_height(self) -> int:
        return self.crop_box[3] - self.crop_box[1]

    @property
    def bbox_area_normalized(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return ((x2 - x1) * (y2 - y1)) / (self.source_width * self.source_height)


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("'\"")


def _remove_yaml_comment(value: str) -> str:
    in_quote: str | None = None
    for index, char in enumerate(value):
        if char in "'\"":
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
        elif char == "#" and in_quote is None:
            return value[:index].rstrip()
    return value.strip()


def load_dataset_metadata(yaml_path: Path) -> DatasetMetadata:
    """Read the small metadata subset needed to identify a Roboflow export."""

    values: dict[str, object] = {}
    roboflow: dict[str, object] = {}
    in_roboflow = False
    in_names_mapping = False
    for raw_line in yaml_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip())
        line = _remove_yaml_comment(raw_line.strip())
        if not line:
            continue
        if indentation == 0:
            in_roboflow = False
            in_names_mapping = False
        if line == "roboflow:":
            in_roboflow = True
            continue
        if line == "names:" and indentation == 0:
            values["names"] = {}
            in_names_mapping = True
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", line)
        numeric_match = re.match(r"^(\d+):\s*(.*)$", line)
        if in_names_mapping and numeric_match:
            name_index, raw_name = numeric_match.groups()
            names_mapping = values.setdefault("names", {})
            if isinstance(names_mapping, dict):
                names_mapping[int(name_index)] = _parse_scalar(raw_name)
            continue
        if not match:
            continue
        key, raw_value = match.groups()
        parsed = _parse_scalar(raw_value)
        if in_roboflow:
            roboflow[key] = parsed
        else:
            values[key] = parsed

    raw_names = values.get("names", ())
    if isinstance(raw_names, dict):
        names = tuple(str(raw_names[index]) for index in sorted(raw_names))
    elif isinstance(raw_names, (list, tuple)):
        names = tuple(str(item) for item in raw_names)
    elif isinstance(raw_names, str):
        names = (raw_names,)
    else:
        names = ()
    raw_nc = values.get("nc")
    raw_version = roboflow.get("version")
    try:
        nc = int(raw_nc) if raw_nc is not None else None
    except (TypeError, ValueError):
        nc = None
    try:
        version = int(raw_version) if raw_version is not None else None
    except (TypeError, ValueError):
        version = None
    return DatasetMetadata(
        yaml_path=yaml_path,
        train=str(values["train"]) if values.get("train") is not None else None,
        val=str(values["val"]) if values.get("val") is not None else None,
        test=str(values["test"]) if values.get("test") is not None else None,
        nc=nc,
        names=names,
        workspace=str(roboflow["workspace"]) if roboflow.get("workspace") is not None else None,
        project=str(roboflow["project"]) if roboflow.get("project") is not None else None,
        version=version,
    )


def _validate_yolo_row(tokens: Sequence[str], line_number: int, raw: str) -> YoloRow | str:
    if len(tokens) != 5:
        return f"EXPECTED_5_VALUES_GOT_{len(tokens)}"
    try:
        class_id = int(tokens[0])
        x_center, y_center, width, height = (float(token) for token in tokens[1:])
    except ValueError:
        return "NON_NUMERIC_VALUE"
    if class_id not in {0, 1, 2}:
        return f"INVALID_CLASS_ID_{class_id}"
    if not 0 <= x_center <= 1:
        return "x_center_OUT_OF_RANGE"
    if not 0 <= y_center <= 1:
        return "y_center_OUT_OF_RANGE"
    if not 0 < width <= 1:
        return "width_OUT_OF_RANGE"
    if not 0 < height <= 1:
        return "height_OUT_OF_RANGE"
    return YoloRow(class_id, x_center, y_center, width, height, line_number, raw)


def parse_yolo_label_file(label_path: Path) -> ParsedLabels:
    valid_rows: list[YoloRow] = []
    invalid_rows: list[dict[str, object]] = []
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        return ParsedLabels((), ({"line_number": 0, "raw": "", "reason": f"DECODE_ERROR: {exc}"},))
    for line_number, raw_line in enumerate(lines, start=1):
        raw = raw_line.strip()
        if not raw:
            continue
        parsed = _validate_yolo_row(raw.split(), line_number, raw)
        if isinstance(parsed, str):
            invalid_rows.append({"line_number": line_number, "raw": raw, "reason": parsed})
        else:
            valid_rows.append(parsed)
    return ParsedLabels(tuple(valid_rows), tuple(invalid_rows))


def crop_rider_region(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    crop_height_ratio: float = 0.50,
    pad_x_ratio: float = 0.15,
    pad_top_ratio: float = 0.10,
    pad_bottom_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    """Return a clipped upper-body crop box as (left, top, right, bottom)."""

    x1, y1, x2, y2 = bbox
    rider_width = x2 - x1
    rider_height = y2 - y1
    left = math.floor(x1 - pad_x_ratio * rider_width)
    right = math.ceil(x2 + pad_x_ratio * rider_width)
    top = math.floor(y1 - pad_top_ratio * rider_height)
    bottom = math.ceil(y1 + crop_height_ratio * rider_height + pad_bottom_ratio * rider_height)
    return (
        max(0, min(image_width, left)),
        max(0, min(image_height, top)),
        max(0, min(image_width, right)),
        max(0, min(image_height, bottom)),
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_source_image_leakage(
    source_files_by_split: Mapping[str, Mapping[str, Path]]
) -> list[dict[str, str]]:
    by_digest: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for split, files in source_files_by_split.items():
        for source_key, path in files.items():
            by_digest[_sha256(path)].append((split, source_key, path))
    leakage: list[dict[str, str]] = []
    for digest, entries in by_digest.items():
        splits = sorted({entry[0] for entry in entries}, key=SPLITS.index)
        if len(splits) < 2:
            continue
        for index, split_a in enumerate(splits[:-1]):
            for split_b in splits[index + 1 :]:
                first = next(entry for entry in entries if entry[0] == split_a)
                second = next(entry for entry in entries if entry[0] == split_b)
                leakage.append(
                    {
                        "sha256": digest,
                        "split_a": split_a,
                        "source_a": str(first[2]),
                        "split_b": split_b,
                        "source_b": str(second[2]),
                    }
                )
    return leakage


def _split_alias(split: str) -> str:
    return "valid" if split == "valid" else split


def resolve_split_images_dir(yaml_path: Path, configured_path: str, split: str) -> Path:
    """Resolve imperfect relative paths commonly emitted by dataset exports."""

    dataset_root = yaml_path.parent
    raw = Path(configured_path)
    candidates = [dataset_root / raw]
    if raw.is_absolute():
        candidates.insert(0, raw)

    # Some exports put ../train/images in data.yaml even though train is next
    # to data.yaml.  Keep this fallback narrow and split-specific.
    trimmed_parts = [part for part in raw.parts if part not in {".", ".."}]
    if trimmed_parts:
        candidates.append(dataset_root.joinpath(*trimmed_parts))
    candidates.extend(
        [
            dataset_root / _split_alias(split) / "images",
            dataset_root / _split_alias(split),
        ]
    )

    checked: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_dir() and resolved.name.lower() == "images":
            return resolved
        if resolved.is_dir() and (resolved / "images").is_dir():
            return (resolved / "images").resolve()
    raise DatasetError(
        f"Could not resolve {split} images directory from {yaml_path}: {configured_path}"
    )


def find_vietnam_dataset(project_root: Path) -> tuple[DatasetMetadata, dict[str, Path]]:
    """Find exactly one matching Dataset2 Vietnam v19 export under project_root."""

    candidates: list[DatasetMetadata] = []
    for yaml_path in sorted(project_root.rglob("data.yaml")):
        if any(part in {".git", ".venv", "venv", "Stage2_Vietnam"} for part in yaml_path.parts):
            continue
        try:
            metadata = load_dataset_metadata(yaml_path)
        except (OSError, UnicodeError, ValueError):
            continue
        if metadata.is_required_dataset:
            candidates.append(metadata)
    if not candidates:
        raise DatasetError(
            "STOP: Dataset2 Vietnam was not found. Expected data.yaml with "
            "workspace=nckh-2023, project=helmet-detection-project, version=19, "
            "nc=3, and the exact three class names."
        )
    if len(candidates) > 1:
        paths = "\n".join(f"- {candidate.yaml_path}" for candidate in candidates)
        raise DatasetError(
            "STOP: More than one matching Dataset2 Vietnam v19 data.yaml was found; "
            f"source selection is ambiguous:\n{paths}"
        )
    metadata = candidates[0]
    configured = {"train": metadata.train, "valid": metadata.val, "test": metadata.test}
    split_dirs: dict[str, Path] = {}
    for split, configured_path in configured.items():
        if not configured_path:
            raise DatasetError(f"STOP: {metadata.yaml_path} has no path for {split}.")
        split_dirs[split] = resolve_split_images_dir(metadata.yaml_path, configured_path, split)
        if not (split_dirs[split].parent / "labels").is_dir():
            raise DatasetError(f"STOP: Labels directory is missing for {split}: {split_dirs[split]}")
    return metadata, split_dirs


def _source_label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _list_source_images(images_dir: Path) -> list[Path]:
    return sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def _pixel_bbox(row: YoloRow, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(image_width, math.floor((row.x_center - row.width / 2) * image_width)))
    y1 = max(0, min(image_height, math.floor((row.y_center - row.height / 2) * image_height)))
    x2 = max(0, min(image_width, math.ceil((row.x_center + row.width / 2) * image_width)))
    y2 = max(0, min(image_height, math.ceil((row.y_center + row.height / 2) * image_height)))
    return x1, y1, x2, y2


def _output_filename(source_image: Path, object_index: int, class_name: str, output_dir: Path) -> Path:
    base = f"{source_image.stem}__obj{object_index:03d}__{class_name}"
    candidate = output_dir / f"{base}.jpg"
    duplicate_index = 1
    while candidate.exists():
        candidate = output_dir / f"{base}__file{duplicate_index:03d}.jpg"
        duplicate_index += 1
    return candidate


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _prepare_output_layout(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise DatasetError(
            f"Output already exists and is non-empty: {output_root}. "
            "Choose a new output path so stale crops cannot mix with this run."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for class_name in CLASS_TO_NAME.values():
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)
    (output_root / "reports" / "contact_sheets").mkdir(parents=True, exist_ok=True)
    (output_root / "reports" / "rejected_crops").mkdir(parents=True, exist_ok=True)


def _manifest_row(record: CropRecord, output_root: Path) -> dict[str, object]:
    x1, y1, x2, y2 = record.bbox
    cx1, cy1, cx2, cy2 = record.crop_box
    return {
        "split": record.split,
        "class_name": record.class_name,
        "class_id": record.class_id,
        "source_image": str(record.source_image.resolve()),
        "source_label": str(record.source_label.resolve()),
        "object_index": record.object_index,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "crop_x1": cx1,
        "crop_y1": cy1,
        "crop_x2": cx2,
        "crop_y2": cy2,
        "source_width": record.source_width,
        "source_height": record.source_height,
        "crop_width": record.crop_width,
        "crop_height": record.crop_height,
        "bbox_area_normalized": f"{record.bbox_area_normalized:.8f}",
        "output_crop": str(record.output_crop.relative_to(output_root).as_posix()),
    }


def _write_manifest(records: Sequence[CropRecord], output_root: Path) -> None:
    _write_csv(
        output_root / "manifest.csv",
        MANIFEST_FIELDS,
        (_manifest_row(record, output_root) for record in records),
    )


def extract_crops(
    split_dirs: Mapping[str, Path],
    output_root: Path,
) -> tuple[list[CropRecord], list[dict[str, object]], dict[str, dict[str, Path]]]:
    records: list[CropRecord] = []
    rejected: list[dict[str, object]] = []
    source_files_by_split: dict[str, dict[str, Path]] = {split: {} for split in SPLITS}
    invalid_rows: list[dict[str, object]] = []

    for split in SPLITS:
        for source_image in _list_source_images(split_dirs[split]):
            source_key = str(source_image.resolve())
            source_files_by_split[split][source_key] = source_image
            source_label = _source_label_path(source_image)
            if not source_label.exists():
                invalid_rows.append(
                    {
                        "split": split,
                        "source_image": str(source_image),
                        "source_label": str(source_label),
                        "line_number": "",
                        "raw": "",
                        "reason": "MISSING_LABEL",
                    }
                )
                continue
            parsed = parse_yolo_label_file(source_label)
            for invalid in parsed.invalid_rows:
                invalid_rows.append(
                    {
                        "split": split,
                        "source_image": str(source_image),
                        "source_label": str(source_label),
                        **invalid,
                    }
                )
            try:
                with Image.open(source_image) as image:
                    image_width, image_height = image.size
                    rgb_image = image.convert("RGB")
                    for row in parsed.valid_rows:
                        if row.class_id == 0:
                            continue
                        class_name = CLASS_TO_NAME[row.class_id]
                        bbox = _pixel_bbox(row, image_width, image_height)
                        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                            invalid_rows.append(
                                {
                                    "split": split,
                                    "source_image": str(source_image),
                                    "source_label": str(source_label),
                                    "line_number": row.line_number,
                                    "raw": row.raw,
                                    "reason": "INVALID_PIXEL_BBOX_AFTER_CLIP",
                                }
                            )
                            continue
                        crop_box = crop_rider_region(bbox, image_width, image_height)
                        crop_width = crop_box[2] - crop_box[0]
                        crop_height = crop_box[3] - crop_box[1]
                        if crop_width < 24 or crop_height < 24:
                            rejected.append(
                                {
                                    "split": split,
                                    "source_image": str(source_image),
                                    "class_id": row.class_id,
                                    "bbox": " ".join(str(value) for value in bbox),
                                    "crop_width": crop_width,
                                    "crop_height": crop_height,
                                    "reason": "TOO_SMALL",
                                }
                            )
                            continue
                        output_dir = output_root / split / class_name
                        output_crop = _output_filename(source_image, row.line_number, class_name, output_dir)
                        rgb_image.crop(crop_box).save(output_crop, format="JPEG", quality=95)
                        records.append(
                            CropRecord(
                                split=split,
                                class_name=class_name,
                                class_id=row.class_id,
                                source_image=source_image,
                                source_label=source_label,
                                object_index=row.line_number,
                                bbox=bbox,
                                crop_box=crop_box,
                                source_width=image_width,
                                source_height=image_height,
                                output_crop=output_crop,
                            )
                        )
            except (OSError, Image.DecompressionBombError) as exc:
                invalid_rows.append(
                    {
                        "split": split,
                        "source_image": str(source_image),
                        "source_label": str(source_label),
                        "line_number": "",
                        "raw": "",
                        "reason": f"IMAGE_READ_ERROR: {exc}",
                    }
                )
    _write_csv(
        output_root / "reports" / "invalid_labels.csv",
        ("split", "source_image", "source_label", "line_number", "raw", "reason"),
        invalid_rows,
    )
    _write_csv(
        output_root / "reports" / "rejected_crops.csv",
        ("split", "source_image", "class_id", "bbox", "crop_width", "crop_height", "reason"),
        rejected,
    )
    return records, invalid_rows, source_files_by_split


def find_crop_duplicates(records: Sequence[CropRecord]) -> list[dict[str, str]]:
    by_digest: dict[str, list[CropRecord]] = defaultdict(list)
    for record in records:
        by_digest[_sha256(record.output_crop)].append(record)
    duplicates: list[dict[str, str]] = []
    for digest, duplicate_records in by_digest.items():
        if len(duplicate_records) < 2:
            continue
        for index, first in enumerate(duplicate_records[:-1]):
            for second in duplicate_records[index + 1 :]:
                kind = (
                    "INTRA_SPLIT_DUPLICATE"
                    if first.split == second.split
                    else "CROSS_SPLIT_LEAKAGE"
                )
                duplicates.append(
                    {
                        "kind": kind,
                        "sha256": digest,
                        "split_a": first.split,
                        "output_crop_a": str(first.output_crop),
                        "split_b": second.split,
                        "output_crop_b": str(second.output_crop),
                    }
                )
    return duplicates


def _size_category(area: int) -> str:
    if area < 32 * 32:
        return "tiny"
    if area < 64 * 64:
        return "small"
    if area < 128 * 128:
        return "medium"
    return "large"


def write_analysis_reports(records: Sequence[CropRecord], output_root: Path) -> None:
    size_counter: Counter[tuple[str, str, str]] = Counter()
    aspect_bins = (
        (0.0, 0.50, "<0.50"),
        (0.50, 0.75, "0.50-0.75"),
        (0.75, 1.00, "0.75-1.00"),
        (1.00, 1.50, "1.00-1.50"),
        (1.50, 2.00, "1.50-2.00"),
        (2.00, 3.00, "2.00-3.00"),
        (3.00, float("inf"), ">=3.00"),
    )
    aspect_counter: Counter[tuple[str, str, str, float, str]] = Counter()
    for record in records:
        area = record.crop_width * record.crop_height
        size_counter[(record.split, record.class_name, _size_category(area))] += 1
        aspect_ratio = record.crop_width / record.crop_height
        for lower, upper, label in aspect_bins:
            if lower <= aspect_ratio < upper:
                aspect_counter[(record.split, record.class_name, label, lower, str(upper))] += 1
                break
    _write_csv(
        output_root / "reports" / "crop_size_distribution.csv",
        ("split", "class_name", "size_category", "count"),
        (
            {
                "split": split,
                "class_name": class_name,
                "size_category": category,
                "count": size_counter[(split, class_name, category)],
            }
            for split in SPLITS
            for class_name in CLASS_TO_NAME.values()
            for category in ("tiny", "small", "medium", "large")
        ),
    )
    _write_csv(
        output_root / "reports" / "crop_aspect_ratio.csv",
        ("split", "class_name", "aspect_bin", "lower_bound", "upper_bound", "count"),
        (
            {
                "split": split,
                "class_name": class_name,
                "aspect_bin": label,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": aspect_counter[(split, class_name, label, lower, str(upper))],
            }
            for split in SPLITS
            for class_name in CLASS_TO_NAME.values()
            for lower, upper, label in aspect_bins
        ),
    )
    counts = Counter((record.split, record.class_name) for record in records)
    _write_csv(
        output_root / "reports" / "class_balance.csv",
        ("split", "helmet_count", "no_helmet_count", "total", "helmet_percentage", "no_helmet_percentage"),
        (
            {
                "split": split,
                "helmet_count": counts[(split, "helmet")],
                "no_helmet_count": counts[(split, "no_helmet")],
                "total": counts[(split, "helmet")] + counts[(split, "no_helmet")],
                "helmet_percentage": f"{100 * counts[(split, 'helmet')] / max(1, counts[(split, 'helmet')] + counts[(split, 'no_helmet')]):.2f}",
                "no_helmet_percentage": f"{100 * counts[(split, 'no_helmet')] / max(1, counts[(split, 'helmet')] + counts[(split, 'no_helmet')]):.2f}",
            }
            for split in SPLITS
        ),
    )


def _contact_sheet(
    records: Sequence[CropRecord],
    output_path: Path,
    title: str,
    columns: int = 5,
    tile_width: int = 220,
    tile_height: int = 280,
) -> None:
    selected = list(records)
    rows = max(1, math.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for index, record in enumerate(selected):
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        try:
            with Image.open(record.output_crop) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((tile_width - 10, 178))
                image_left = left + (tile_width - thumbnail.width) // 2
                sheet.paste(thumbnail, (image_left, top + 3))
        except OSError:
            draw.text((left + 5, top + 5), "IMAGE READ ERROR", fill="red", font=font)
        source_name = "\n".join(
            textwrap.wrap(
                record.source_image.name,
                width=28,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
        label = f"{record.class_name}\n{source_name}\n{record.crop_width}x{record.crop_height}"
        draw.multiline_text((left + 5, top + 185), label, fill="black", font=font, spacing=1)
    if not selected:
        draw.text((10, 10), f"{title}\nNO SAMPLES", fill="black", font=font)
    sheet.save(output_path, format="JPEG", quality=95)


def write_contact_sheets(records: Sequence[CropRecord], output_root: Path, seed: int) -> None:
    rng = random.Random(seed)
    contact_dir = output_root / "reports" / "contact_sheets"

    def random_sample(items: Sequence[CropRecord], amount: int) -> list[CropRecord]:
        return rng.sample(list(items), min(amount, len(items)))

    train_helmet = [record for record in records if record.split == "train" and record.class_name == "helmet"]
    train_no_helmet = [record for record in records if record.split == "train" and record.class_name == "no_helmet"]
    valid_helmet = [record for record in records if record.split == "valid" and record.class_name == "helmet"]
    valid_no_helmet = [record for record in records if record.split == "valid" and record.class_name == "no_helmet"]
    accepted_no_helmet = [record for record in records if record.class_name == "no_helmet"]
    accepted_no_helmet.sort(key=lambda record: record.crop_width * record.crop_height)

    def hard_case_score(record: CropRecord) -> tuple[int, float, int]:
        aspect = record.crop_width / record.crop_height
        extreme_aspect = int(aspect < 0.50 or aspect > 1.80)
        area = record.crop_width * record.crop_height
        return extreme_aspect, -area, record.object_index

    hard_cases = sorted(records, key=hard_case_score, reverse=True)
    sheets = (
        ("train_helmet_50.jpg", random_sample(train_helmet, 50), "50 random train helmet"),
        ("train_no_helmet_50.jpg", random_sample(train_no_helmet, 50), "50 random train no_helmet"),
        ("valid_helmet_30.jpg", random_sample(valid_helmet, 30), "30 valid helmet"),
        ("valid_no_helmet_30.jpg", random_sample(valid_no_helmet, 30), "30 valid no_helmet"),
        ("smallest_accepted_no_helmet_30.jpg", accepted_no_helmet[:30], "30 smallest accepted no_helmet"),
        ("hard_cases_heuristic_30.jpg", hard_cases[:30], "30 rear/side/occlusion heuristics"),
    )
    for filename, selected, title in sheets:
        _contact_sheet(selected, contact_dir / filename, title)


def run_visual_audit(records: Sequence[CropRecord], output_root: Path, seed: int) -> None:
    rng = random.Random(seed + 1)
    sample = rng.sample(list(records), min(100, len(records)))
    audit_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for record in sample:
        status = "PASS"
        reason = ""
        try:
            crop_x1, crop_y1, crop_x2, crop_y2 = record.crop_box
            if not (
                0 <= crop_x1 < crop_x2 <= record.source_width
                and 0 <= crop_y1 < crop_y2 <= record.source_height
            ):
                status, reason = "FAIL", "CROP_OUTSIDE_SOURCE_BOUNDS"
            with Image.open(record.output_crop) as image:
                width, height = image.size
                if status == "PASS" and (width <= 0 or height <= 0):
                    status, reason = "FAIL", "EMPTY_CROP"
                elif status == "PASS" and (width != record.crop_width or height != record.crop_height):
                    status, reason = "FAIL", "NATIVE_SIZE_MISMATCH"
        except OSError as exc:
            status, reason = "FAIL", f"IMAGE_READ_ERROR: {exc}"
        if status == "FAIL":
            failures.append(f"{record.output_crop}: {reason}")
        audit_rows.append(
            {
                "split": record.split,
                "class_name": record.class_name,
                "source_image": str(record.source_image),
                "output_crop": str(record.output_crop),
                "crop_width": record.crop_width,
                "crop_height": record.crop_height,
                "status": status,
                "reason": reason,
            }
        )
    _write_csv(
        output_root / "reports" / "visual_audit.csv",
        (
            "split",
            "class_name",
            "source_image",
            "output_crop",
            "crop_width",
            "crop_height",
            "status",
            "reason",
        ),
        audit_rows,
    )
    if failures:
        raise DatasetError("Visual audit failed:\n" + "\n".join(failures[:10]))


def _write_summary(
    output_root: Path,
    metadata: DatasetMetadata,
    records: Sequence[CropRecord],
    rejected: Sequence[Mapping[str, object]],
    invalid_rows: Sequence[Mapping[str, object]],
    train_valid_leakage: bool,
    train_test_leakage: bool,
    valid_test_leakage: bool,
) -> None:
    counts = Counter((record.split, record.class_name) for record in records)
    lines = [
        "Dataset source: Vietnam Dataset2",
        f"Workspace: {metadata.workspace}",
        f"Project: {metadata.project}",
        f"Version: {metadata.version}",
        "Crop policy: upper 50% rider bbox +15% horizontal padding +10% top padding +10% bottom padding",
        "Minimum crop: 24x24",
        "",
        f"Train helmet: {counts[('train', 'helmet')]}",
        f"Train no_helmet: {counts[('train', 'no_helmet')]}",
        f"Valid helmet: {counts[('valid', 'helmet')]}",
        f"Valid no_helmet: {counts[('valid', 'no_helmet')]}",
        f"Test helmet: {counts[('test', 'helmet')]}",
        f"Test no_helmet: {counts[('test', 'no_helmet')]}",
        f"Rejected too small: {sum(row.get('reason') == 'TOO_SMALL' for row in rejected)}",
        f"Invalid labels: {len(invalid_rows)}",
        f"Train/valid leakage: {'FAIL' if train_valid_leakage else 'PASS'}",
        f"Train/test leakage: {'FAIL' if train_test_leakage else 'PASS'}",
        f"Valid/test leakage: {'FAIL' if valid_test_leakage else 'PASS'}",
        "Test used for tuning: NO",
        "Training performed: NO",
        "No source files modified: YES",
        "",
        "Source split policy: Dataset2 train -> train, Dataset2 valid -> valid, Dataset2 test -> test",
        "Augmentation during extraction: NONE",
        "Classifier resize during extraction: NONE",
    ]
    (output_root / "stage2_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assert_final_layout(
    output_root: Path,
    records: Sequence[CropRecord],
    train_valid_leakage: bool,
    train_test_leakage: bool,
    valid_test_leakage: bool,
) -> None:
    required_paths = [
        *(output_root / split / class_name for split in SPLITS for class_name in CLASS_TO_NAME.values()),
        output_root / "manifest.csv",
        output_root / "stage2_summary.txt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise DatasetError("Final layout assertion failed; missing:\n" + "\n".join(missing))
    if not records:
        raise DatasetError("Final layout assertion failed: no accepted rider crops were produced.")
    if train_valid_leakage or train_test_leakage or valid_test_leakage:
        raise LeakageError("STOP: exact source-image leakage was detected across splits.")


def run_preparation(
    project_root: Path,
    output_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    project_root = project_root.resolve()
    output_root = (output_dir or project_root / "Stage2_Vietnam").resolve()
    metadata, split_dirs = find_vietnam_dataset(project_root)
    _prepare_output_layout(output_root)

    records, invalid_rows, source_files_by_split = extract_crops(split_dirs, output_root)
    rejected_rows = []
    rejected_path = output_root / "reports" / "rejected_crops.csv"
    with rejected_path.open(encoding="utf-8", newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))

    _write_manifest(records, output_root)
    write_analysis_reports(records, output_root)
    write_contact_sheets(records, output_root, seed)
    run_visual_audit(records, output_root, seed)

    source_leakage = find_source_image_leakage(source_files_by_split)
    _write_csv(
        output_root / "reports" / "source_image_leakage.csv",
        ("sha256", "split_a", "source_a", "split_b", "source_b"),
        source_leakage,
    )
    crop_duplicates = find_crop_duplicates(records)
    _write_csv(
        output_root / "reports" / "duplicate_crops.csv",
        ("kind", "sha256", "split_a", "output_crop_a", "split_b", "output_crop_b"),
        crop_duplicates,
    )
    train_valid_leakage = any(
        {row["split_a"], row["split_b"]} == {"train", "valid"} for row in source_leakage
    ) or any(
        row["kind"] == "CROSS_SPLIT_LEAKAGE"
        and {row["split_a"], row["split_b"]} == {"train", "valid"}
        for row in crop_duplicates
    )
    train_test_leakage = any(
        {row["split_a"], row["split_b"]} == {"train", "test"} for row in source_leakage
    ) or any(
        row["kind"] == "CROSS_SPLIT_LEAKAGE"
        and {row["split_a"], row["split_b"]} == {"train", "test"}
        for row in crop_duplicates
    )
    valid_test_leakage = any(
        {row["split_a"], row["split_b"]} == {"valid", "test"} for row in source_leakage
    ) or any(
        row["kind"] == "CROSS_SPLIT_LEAKAGE"
        and {row["split_a"], row["split_b"]} == {"valid", "test"}
        for row in crop_duplicates
    )
    _write_summary(
        output_root,
        metadata,
        records,
        rejected_rows,
        invalid_rows,
        train_valid_leakage,
        train_test_leakage,
        valid_test_leakage,
    )
    _assert_final_layout(
        output_root,
        records,
        train_valid_leakage,
        train_test_leakage,
        valid_test_leakage,
    )
    print("========================================")
    print("STAGE2 VIETNAM DATASET READY")
    print("========================================")
    print("Source: Dataset2 Vietnam v19")
    print("Crop strategy: Top 50% of rider + head/shoulder padding")
    counts = Counter((record.split, record.class_name) for record in records)
    for split in SPLITS:
        print(split.upper())
        print(f"helmet: {counts[(split, 'helmet')]}")
        print(f"no_helmet: {counts[(split, 'no_helmet')]}")
    print(f"Rejected: {len(rejected_rows)}")
    print(f"Train/Valid leakage: {'FAIL' if train_valid_leakage else 'PASS'}")
    print(f"Train/Test leakage: {'FAIL' if train_test_leakage else 'PASS'}")
    print(f"Valid/Test leakage: {'FAIL' if valid_test_leakage else 'PASS'}")
    print("Test used for tuning: NO")
    print("MODEL TRAINING: NOT PERFORMED")
    print(f"Output: {output_root}")
    print("========================================")
    return output_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root to search recursively for Dataset2 Vietnam v19 data.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <project-root>/Stage2_Vietnam.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic contact sheets.")
    args = parser.parse_args(argv)
    try:
        run_preparation(args.project_root, args.output_dir, args.seed)
    except LeakageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except DatasetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
