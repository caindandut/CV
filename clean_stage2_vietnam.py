"""Clean an existing Stage 2 crop dataset without recropping or training.

The input dataset is treated as read-only.  The cleaner audits exact crops,
source images, source video/sequence groups, and perceptual duplicates before
copying accepted files to a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import shutil
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required to clean Stage 2 crops.") from exc

try:  # Optional.  A compatible Pillow-only fallback is provided below.
    import imagehash  # type: ignore
except ImportError:  # pragma: no cover - depends on the runtime environment
    imagehash = None


SPLITS = ("train", "valid", "test")
PRIORITY = {"train": 0, "valid": 1, "test": 2}
CLASS_NAMES = ("helmet", "no_helmet")
PHASH_THRESHOLD = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CleaningError(RuntimeError):
    """Raised when the input/output contract cannot be safely fulfilled."""


class LabelConflictError(CleaningError):
    """Raised after output creation when manual label-conflict review is required."""


@dataclass
class CropRecord:
    uid: int
    split: str
    class_name: str
    class_id: str
    source_image: str
    source_label: str
    object_index: str
    bbox: tuple[str, str, str, str]
    crop_box: tuple[str, str, str, str]
    source_width: str
    source_height: str
    crop_width: int
    crop_height: int
    original_crop: Path
    original_crop_relative: str
    canonical_source_name: str
    source_group: str
    sha256: str
    source_sha256: str
    phash: str
    clean_crop: Path | None = None
    drop_reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def priority(self) -> int:
        return PRIORITY[self.split]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_source_name(source_filename: str) -> str:
    """Remove the Roboflow hash and trailing image-format marker."""

    stem = Path(source_filename).name
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    stem = re.sub(r"\.rf\.[^.]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    return stem


def source_group(source_filename: str) -> str:
    """Return a normalized MOV/MP4 sequence group or the canonical source name."""

    canonical = canonical_source_name(source_filename)
    match = re.search(
        r"(?<![A-Z0-9])(MOV|MP4)[_-]?(\d+)(?![A-Z0-9])",
        canonical,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1).upper()}-{int(match.group(2))}"
    return canonical


def hamming_distance(phash_a: str, phash_b: str) -> int:
    if len(phash_a) != len(phash_b):
        raise ValueError("pHash values must have the same hexadecimal length")
    return sum((int(char_a, 16) ^ int(char_b, 16)).bit_count() for char_a, char_b in zip(phash_a, phash_b))


@lru_cache(maxsize=None)
def _dct_basis(sample_size: int, hash_size: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            (2 / sample_size)
            * (1 / math.sqrt(2) if frequency == 0 else 1.0)
            * math.cos((2 * position + 1) * frequency * math.pi / (2 * sample_size))
            for position in range(sample_size)
        )
        for frequency in range(hash_size)
    )


def _dct_phash(image: Image.Image, hash_size: int = 8) -> str:
    """Pillow-only equivalent of imagehash.phash for environments without ImageHash."""

    sample_size = hash_size * 4
    gray = ImageOps.exif_transpose(image).convert("L").resize(
        (sample_size, sample_size), Image.Resampling.LANCZOS
    )
    pixels = list(gray.tobytes())
    basis = _dct_basis(sample_size, hash_size)
    intermediate = [
        [
            sum(basis[u][x] * pixels[x * sample_size + y] for x in range(sample_size))
            for y in range(sample_size)
        ]
        for u in range(hash_size)
    ]
    coefficients = [
        sum(intermediate[u][y] * basis[v][y] for y in range(sample_size))
        for u in range(hash_size)
        for v in range(hash_size)
    ]
    median = sorted(coefficients)[len(coefficients) // 2]
    bit_string = "".join("1" if coefficient > median else "0" for coefficient in coefficients)
    return f"{int(bit_string, 2):016x}"


def phash_image(image_path: Path) -> str:
    with Image.open(image_path) as image:
        transposed = ImageOps.exif_transpose(image)
        if imagehash is not None:
            return str(imagehash.phash(transposed, hash_size=8))
        return _dct_phash(transposed, hash_size=8)


def phash_backend() -> str:
    return "imagehash.phash" if imagehash is not None else "Pillow DCT-compatible fallback"


def _parse_int_tuple(row: Mapping[str, str], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(row[field] for field in fields)


def load_manifest(input_root: Path) -> list[CropRecord]:
    manifest_path = input_root / "manifest.csv"
    if not manifest_path.exists():
        raise CleaningError(f"STOP: manifest.csv was not found: {manifest_path}")
    required = {
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
        "output_crop",
    }
    records: list[CropRecord] = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise CleaningError(f"STOP: manifest is missing columns: {sorted(missing)}")
        for uid, row in enumerate(reader):
            split = row["split"]
            class_name = row["class_name"]
            if split not in SPLITS or class_name not in CLASS_NAMES:
                raise CleaningError(f"STOP: unsupported split/class in manifest row {uid}: {row}")
            output_crop = Path(row["output_crop"])
            original_crop = output_crop if output_crop.is_absolute() else input_root / output_crop
            if not original_crop.is_file():
                raise CleaningError(f"STOP: crop listed in manifest does not exist: {original_crop}")
            try:
                crop_width = int(row["crop_width"])
                crop_height = int(row["crop_height"])
            except ValueError as exc:
                raise CleaningError(f"STOP: invalid crop dimensions in manifest row {uid}: {row}") from exc
            source_path = Path(row["source_image"])
            source_hash = _sha256(source_path) if source_path.is_file() else ""
            records.append(
                CropRecord(
                    uid=uid,
                    split=split,
                    class_name=class_name,
                    class_id=row["class_id"],
                    source_image=row["source_image"],
                    source_label=row["source_label"],
                    object_index=row["object_index"],
                    bbox=_parse_int_tuple(row, ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")),
                    crop_box=_parse_int_tuple(row, ("crop_x1", "crop_y1", "crop_x2", "crop_y2")),
                    source_width=row["source_width"],
                    source_height=row["source_height"],
                    crop_width=crop_width,
                    crop_height=crop_height,
                    original_crop=original_crop,
                    original_crop_relative=output_crop.as_posix(),
                    canonical_source_name=canonical_source_name(source_path.name),
                    source_group=source_group(source_path.name),
                    sha256=_sha256(original_crop),
                    source_sha256=source_hash,
                    phash=phash_image(original_crop),
                )
            )
    if not records:
        raise CleaningError("STOP: manifest.csv contains no samples")
    return records


def highest_priority_split(splits: Iterable[str]) -> str:
    split_values = list(splits)
    if not split_values:
        raise ValueError("At least one split is required")
    return max(split_values, key=PRIORITY.__getitem__)


def _cross_split_pairs(records: Sequence[CropRecord]) -> Iterable[tuple[CropRecord, CropRecord]]:
    ordered = sorted(records, key=lambda record: (record.priority, record.uid))
    for index, first in enumerate(ordered[:-1]):
        for second in ordered[index + 1 :]:
            if first.split != second.split:
                yield first, second


def _pair_report_base(first: CropRecord, second: CropRecord) -> dict[str, object]:
    return {
        "crop_a": first.original_crop_relative,
        "split_a": first.split,
        "class_a": first.class_name,
        "crop_b": second.original_crop_relative,
        "split_b": second.split,
        "class_b": second.class_name,
    }


def _mark_drop(record: CropRecord, reason: str, evidence: str) -> None:
    if reason not in record.drop_reasons:
        record.drop_reasons.append(reason)
    tagged_evidence = f"{reason}:{evidence}"
    if tagged_evidence not in record.evidence:
        record.evidence.append(tagged_evidence)


def _priority_action(first: CropRecord, second: CropRecord) -> tuple[CropRecord, CropRecord, str]:
    kept = max((first, second), key=lambda record: (record.priority, -record.uid))
    dropped = second if kept is first else first
    return kept, dropped, f"KEEP_{kept.split}_DROP_{dropped.split}"


def find_exact_crop_duplicates(
    records: Sequence[CropRecord],
) -> tuple[list[dict[str, object]], list[tuple[int, int]]]:
    grouped: dict[str, list[CropRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sha256].append(record)
    rows: list[dict[str, object]] = []
    exact_pairs: list[tuple[int, int]] = []
    for digest, group in sorted(grouped.items()):
        for first, second in _cross_split_pairs(group):
            exact_pairs.append((min(first.uid, second.uid), max(first.uid, second.uid)))
            row = _pair_report_base(first, second)
            row["sha256"] = digest
            if first.class_name != second.class_name:
                row["action"] = "LABEL_CONFLICT"
            else:
                kept, dropped, action = _priority_action(first, second)
                _mark_drop(
                    dropped,
                    "EXACT_CROP_DUPLICATE",
                    f"{digest}:kept_{kept.split}:{kept.original_crop_relative}",
                )
                row["action"] = action
            rows.append(row)
    return rows, exact_pairs


def find_source_image_duplicates(records: Sequence[CropRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[CropRecord]] = defaultdict(list)
    for record in records:
        if record.source_sha256:
            grouped[record.source_sha256].append(record)
    rows: list[dict[str, object]] = []
    for digest, group in sorted(grouped.items()):
        by_source: dict[tuple[str, str], CropRecord] = {}
        for record in group:
            by_source.setdefault((record.split, record.source_image), record)
        source_records = list(by_source.values())
        if len({record.split for record in source_records}) < 2:
            continue
        kept_split = highest_priority_split(record.split for record in source_records)
        for first, second in _cross_split_pairs(source_records):
            rows.append(
                {
                    "source_image_a": first.source_image,
                    "split_a": first.split,
                    "source_image_b": second.source_image,
                    "split_b": second.split,
                    "sha256": digest,
                    "action": f"KEEP_{kept_split}_DROP_LOWER_PRIORITY",
                }
            )
        for record in group:
            if record.split != kept_split:
                _mark_drop(record, "SOURCE_IMAGE_DUPLICATE", f"{digest}:kept_{kept_split}")
    return rows


def find_source_group_conflicts(records: Sequence[CropRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[CropRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_group].append(record)
    rows: list[dict[str, object]] = []
    for group_name, group in sorted(grouped.items()):
        splits_found = sorted({record.split for record in group}, key=PRIORITY.__getitem__)
        if len(splits_found) < 2:
            continue
        kept_split = highest_priority_split(splits_found)
        counts = Counter(record.split for record in group)
        for record in group:
            if record.split != kept_split:
                _mark_drop(record, "SOURCE_SEQUENCE_LEAKAGE", f"{group_name}:kept_{kept_split}")
        rows.append(
            {
                "source_group": group_name,
                "splits_found": "|".join(splits_found),
                "train_samples": counts["train"],
                "valid_samples": counts["valid"],
                "test_samples": counts["test"],
                "kept_split": kept_split,
                "dropped_train": counts["train"] if kept_split != "train" else 0,
                "dropped_valid": counts["valid"] if kept_split == "test" else 0,
                "reason": "SOURCE_SEQUENCE_LEAKAGE",
            }
        )
    return rows


@dataclass
class _BKNode:
    value: int
    records: list[CropRecord] = field(default_factory=list)
    children: dict[int, "_BKNode"] = field(default_factory=dict)


class BKTree:
    """Small BK-tree for Hamming-distance lookup over 64-bit pHashes."""

    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, value: int, record: CropRecord) -> None:
        if self.root is None:
            self.root = _BKNode(value=value, records=[record])
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                node.records.append(record)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value=value, records=[record])
                return
            node = child

    def query(self, value: int, threshold: int) -> list[CropRecord]:
        if self.root is None:
            return []
        matches: list[CropRecord] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= threshold:
                matches.extend(node.records)
            lower = max(0, distance - threshold)
            upper = distance + threshold
            pending.extend(child for edge, child in node.children.items() if lower <= edge <= upper)
        return matches


def find_near_crop_duplicates(
    records: Sequence[CropRecord],
    exact_pairs: Iterable[tuple[int, int]] = (),
    mark_drops: bool = True,
) -> tuple[list[dict[str, object]], list[tuple[int, int]]]:
    exact_pair_set = set(exact_pairs)
    tree = BKTree()
    rows: list[dict[str, object]] = []
    near_pairs: list[tuple[int, int]] = []
    for record in sorted(records, key=lambda item: item.uid):
        value = int(record.phash, 16)
        for other in tree.query(value, PHASH_THRESHOLD):
            if record.split == other.split:
                continue
            pair = (min(record.uid, other.uid), max(record.uid, other.uid))
            if pair in exact_pair_set or pair in near_pairs:
                continue
            distance = hamming_distance(record.phash, other.phash)
            if distance > PHASH_THRESHOLD:
                continue
            near_pairs.append(pair)
            first, second = sorted((record, other), key=lambda item: item.uid)
            row = _pair_report_base(first, second)
            row.update(
                {
                    "phash_a": first.phash,
                    "phash_b": second.phash,
                    "distance": distance,
                    "source_group_a": first.source_group,
                    "source_group_b": second.source_group,
                }
            )
            if first.class_name != second.class_name:
                row["action"] = "LABEL_CONFLICT"
            else:
                kept, dropped, action = _priority_action(first, second)
                if mark_drops:
                    _mark_drop(
                        dropped,
                        "NEAR_DUPLICATE",
                        f"distance_{distance}:kept_{kept.split}:{kept.original_crop_relative}",
                    )
                row["action"] = action
            rows.append(row)
        tree.add(value, record)
    return rows, near_pairs


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _prepare_output_layout(output_root: Path) -> None:
    if output_root.exists():
        raise CleaningError(
            f"STOP: output folder already exists: {output_root}. "
            "Delete or rename it manually before running the cleaner."
        )
    output_root.mkdir(parents=True)
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (output_root / split / class_name).mkdir(parents=True)
    (output_root / "reports" / "contact_sheets").mkdir(parents=True)
    (output_root / "reports" / "duplicate_examples").mkdir(parents=True)


def _run_duplicate_audit(
    records: list[CropRecord],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    exact_rows, exact_pairs = find_exact_crop_duplicates(records)
    source_rows = find_source_image_duplicates(records)
    group_rows = find_source_group_conflicts(records)
    near_candidates = [record for record in records if not record.drop_reasons]
    near_rows, _ = find_near_crop_duplicates(near_candidates, exact_pairs)
    label_conflicts = [
        {"duplicate_type": "EXACT", **row}
        for row in exact_rows
        if row["action"] == "LABEL_CONFLICT"
    ] + [
        {"duplicate_type": "NEAR", **row}
        for row in near_rows
        if row["action"] == "LABEL_CONFLICT"
    ]
    return exact_rows, source_rows, group_rows, near_rows, label_conflicts


def _kept_split_from_evidence(record: CropRecord, reason: str) -> str:
    for evidence in record.evidence:
        if not evidence.startswith(f"{reason}:"):
            continue
        match = re.search(r"kept_(train|valid|test)", evidence)
        if match:
            return match.group(1)
    return ""


def _removed_rows(records: Sequence[CropRecord]) -> list[dict[str, object]]:
    reason_priority = (
        "SOURCE_SEQUENCE_LEAKAGE",
        "SOURCE_IMAGE_DUPLICATE",
        "EXACT_CROP_DUPLICATE",
        "NEAR_DUPLICATE",
    )
    rows: list[dict[str, object]] = []
    for record in records:
        if not record.drop_reasons:
            continue
        reason = next(reason for reason in reason_priority if reason in record.drop_reasons)
        rows.append(
            {
                "original_crop": record.original_crop_relative,
                "class_name": record.class_name,
                "original_split": record.split,
                "source_image": record.source_image,
                "canonical_source_name": record.canonical_source_name,
                "source_group": record.source_group,
                "reason": reason,
                "conflict_with": " | ".join(record.evidence),
                "kept_split": _kept_split_from_evidence(record, reason),
            }
        )
    return rows


def _clean_manifest_row(record: CropRecord, output_root: Path) -> dict[str, object]:
    clean_relative = record.clean_crop.relative_to(output_root).as_posix() if record.clean_crop else ""
    return {
        "split": record.split,
        "class_name": record.class_name,
        "source_image": record.source_image,
        "source_label": record.source_label,
        "object_index": record.object_index,
        "canonical_source_name": record.canonical_source_name,
        "source_group": record.source_group,
        "bbox_x1": record.bbox[0],
        "bbox_y1": record.bbox[1],
        "bbox_x2": record.bbox[2],
        "bbox_y2": record.bbox[3],
        "crop_x1": record.crop_box[0],
        "crop_y1": record.crop_box[1],
        "crop_x2": record.crop_box[2],
        "crop_y2": record.crop_box[3],
        "crop_width": record.crop_width,
        "crop_height": record.crop_height,
        "sha256": record.sha256,
        "phash": record.phash,
        "original_crop": record.original_crop_relative,
        "clean_crop": clean_relative,
    }


def _copy_accepted_records(records: Sequence[CropRecord], output_root: Path) -> None:
    for record in records:
        destination = output_root / record.split / record.class_name / record.original_crop.name
        if destination.exists():
            raise CleaningError(f"Duplicate clean output filename would overwrite: {destination}")
        shutil.copy2(record.original_crop, destination)
        record.clean_crop = destination


def _write_duplicate_examples(
    label_conflicts: Sequence[Mapping[str, object]],
    records_by_relative_path: Mapping[str, CropRecord],
    output_root: Path,
) -> None:
    for index, row in enumerate(label_conflicts, start=1):
        for side in ("a", "b"):
            relative = str(row.get(f"crop_{side}", ""))
            record = records_by_relative_path.get(relative)
            if record is None:
                continue
            destination = (
                output_root
                / "reports"
                / "duplicate_examples"
                / f"pair{index:04d}__{side}__{record.original_crop.name}"
            )
            shutil.copy2(record.original_crop, destination)


def _size_category(area: int) -> str:
    if area < 32 * 32:
        return "tiny"
    if area < 64 * 64:
        return "small"
    if area < 128 * 128:
        return "medium"
    return "large"


def _write_analysis_reports(
    records: Sequence[CropRecord],
    removed_rows: Sequence[Mapping[str, object]],
    output_root: Path,
) -> None:
    before_counts = Counter((record.split, record.class_name) for record in records)
    removed_counts = Counter(
        (str(row["original_split"]), str(row["class_name"])) for row in removed_rows
    )
    _write_csv(
        output_root / "reports" / "class_counts_before_after.csv",
        ("split", "class_name", "before", "removed", "after"),
        (
            {
                "split": split,
                "class_name": class_name,
                "before": before_counts[(split, class_name)],
                "removed": removed_counts[(split, class_name)],
                "after": before_counts[(split, class_name)] - removed_counts[(split, class_name)],
            }
            for split in SPLITS
            for class_name in CLASS_NAMES
        ),
    )
    accepted = [record for record in records if not record.drop_reasons]
    size_counts = Counter(
        (record.split, record.class_name, _size_category(record.crop_width * record.crop_height))
        for record in accepted
    )
    _write_csv(
        output_root / "reports" / "crop_size_distribution.csv",
        ("split", "class_name", "size_category", "count"),
        (
            {
                "split": split,
                "class_name": class_name,
                "size_category": category,
                "count": size_counts[(split, class_name, category)],
            }
            for split in SPLITS
            for class_name in CLASS_NAMES
            for category in ("tiny", "small", "medium", "large")
        ),
    )
    clean_counts = Counter((record.split, record.class_name) for record in accepted)
    _write_csv(
        output_root / "reports" / "class_balance.csv",
        (
            "split",
            "helmet_count",
            "no_helmet_count",
            "total",
            "helmet_percentage",
            "no_helmet_percentage",
        ),
        (
            {
                "split": split,
                "helmet_count": clean_counts[(split, "helmet")],
                "no_helmet_count": clean_counts[(split, "no_helmet")],
                "total": clean_counts[(split, "helmet")] + clean_counts[(split, "no_helmet")],
                "helmet_percentage": f"{100 * clean_counts[(split, 'helmet')] / max(1, clean_counts[(split, 'helmet')] + clean_counts[(split, 'no_helmet')]):.2f}",
                "no_helmet_percentage": f"{100 * clean_counts[(split, 'no_helmet')] / max(1, clean_counts[(split, 'helmet')] + clean_counts[(split, 'no_helmet')]):.2f}",
            }
            for split in SPLITS
        ),
    )


@dataclass(frozen=True)
class _ContactItem:
    record: CropRecord
    image_path: Path
    note: str = ""


def _contact_sheet(items: Sequence[_ContactItem], output_path: Path) -> None:
    columns = 5
    tile_width = 220
    tile_height = 280
    rows = max(1, math.ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for index, item in enumerate(items):
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        try:
            with Image.open(item.image_path) as image:
                thumbnail = ImageOps.exif_transpose(image).convert("RGB")
                thumbnail.thumbnail((tile_width - 10, 178))
                sheet.paste(thumbnail, (left + (tile_width - thumbnail.width) // 2, top + 3))
        except OSError:
            draw.text((left + 5, top + 5), "IMAGE READ ERROR", fill="red", font=font)
        source_name = "\n".join(
            textwrap.wrap(
                item.record.source_image.rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
                width=28,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
        label = (
            f"{item.record.split}/{item.record.class_name}\n"
            f"{source_name}\n"
            f"{item.record.crop_width}x{item.record.crop_height}"
        )
        if item.note:
            label += f"\n{item.note}"
        draw.multiline_text((left + 5, top + 185), label, fill="black", font=font, spacing=1)
    if not items:
        draw.text((10, 10), "NO SAMPLES", fill="black", font=font)
    sheet.save(output_path, format="JPEG", quality=95)


def _write_contact_sheets(records: Sequence[CropRecord], output_root: Path, seed: int) -> None:
    rng = random.Random(seed)
    accepted = [record for record in records if not record.drop_reasons]
    removed_sequence = [
        record for record in records if "SOURCE_SEQUENCE_LEAKAGE" in record.drop_reasons
    ]

    def sample(items: Sequence[CropRecord], amount: int) -> list[_ContactItem]:
        return [_ContactItem(record, record.clean_crop or record.original_crop) for record in rng.sample(list(items), min(amount, len(items)))]

    def class_items(split: str, class_name: str) -> list[CropRecord]:
        return [record for record in accepted if record.split == split and record.class_name == class_name]

    no_helmet = [record for record in accepted if record.class_name == "no_helmet"]
    no_helmet.sort(key=lambda record: record.crop_width * record.crop_height)
    contact_dir = output_root / "reports" / "contact_sheets"
    sheets = (
        ("train_helmet_50.jpg", sample(class_items("train", "helmet"), 50)),
        ("train_no_helmet_50.jpg", sample(class_items("train", "no_helmet"), 50)),
        ("valid_helmet_30.jpg", sample(class_items("valid", "helmet"), 30)),
        ("valid_no_helmet_30.jpg", sample(class_items("valid", "no_helmet"), 30)),
        ("test_helmet_30.jpg", sample(class_items("test", "helmet"), 30)),
        ("test_no_helmet_30.jpg", sample(class_items("test", "no_helmet"), 30)),
        ("smallest_accepted_no_helmet_30.jpg", sample(no_helmet[:30], 30)),
        (
            "removed_source_sequence_leakage_30.jpg",
            [
                _ContactItem(record, record.original_crop, "SOURCE_SEQUENCE_LEAKAGE")
                for record in rng.sample(removed_sequence, min(30, len(removed_sequence)))
            ],
        ),
    )
    for filename, items in sheets:
        _contact_sheet(items, contact_dir / filename)


def _cross_split_values(records: Sequence[CropRecord], attribute: str) -> list[dict[str, object]]:
    grouped: dict[str, list[CropRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, attribute)
        if value:
            grouped[value].append(record)
    rows: list[dict[str, object]] = []
    for value, group in grouped.items():
        splits = sorted({record.split for record in group}, key=PRIORITY.__getitem__)
        if len(splits) > 1:
            rows.append({"value": value, "splits": "|".join(splits)})
    return rows


def _final_checks(records: Sequence[CropRecord]) -> dict[str, object]:
    accepted = [record for record in records if not record.drop_reasons]
    exact_crop_leakage = _cross_split_values(accepted, "sha256")
    source_image_leakage = _cross_split_values(accepted, "source_sha256")
    source_group_leakage = _cross_split_values(accepted, "source_group")
    near_rows, _ = find_near_crop_duplicates(accepted, mark_drops=False)
    near_leakage = [row for row in near_rows if row["action"] != "LABEL_CONFLICT"]
    label_conflicts = [row for row in near_rows if row["action"] == "LABEL_CONFLICT"]
    for record in accepted:
        if record.clean_crop is None or _sha256(record.clean_crop) != record.sha256:
            raise CleaningError(f"Copied crop bytes changed: {record.original_crop}")
    return {
        "accepted": accepted,
        "exact_crop_leakage": exact_crop_leakage,
        "source_image_leakage": source_image_leakage,
        "source_group_leakage": source_group_leakage,
        "near_rows": near_rows,
        "near_leakage": near_leakage,
        "label_conflicts": label_conflicts,
    }


def _write_summary(
    input_root: Path,
    output_root: Path,
    records: Sequence[CropRecord],
    removed_rows: Sequence[Mapping[str, object]],
    exact_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
    near_rows: Sequence[Mapping[str, object]],
    label_conflicts: Sequence[Mapping[str, object]],
    checks: Mapping[str, object],
) -> None:
    before = Counter((record.split, record.class_name) for record in records)
    after = Counter((record.split, record.class_name) for record in checks["accepted"])
    removed = Counter(str(row["reason"]) for row in removed_rows)
    exact_pass = not checks["exact_crop_leakage"]
    source_image_pass = not checks["source_image_leakage"]
    source_group_pass = not checks["source_group_leakage"]
    near_pass = not checks["near_leakage"]
    label_pass = not label_conflicts
    lines = [
        "Original dataset: Stage2_Vietnam",
        "Clean dataset: Stage2_Vietnam_Clean",
        "Priority: TEST > VALID > TRAIN",
        f"pHash backend: {phash_backend()}",
        "",
        "BEFORE",
        f"Train helmet: {before[('train', 'helmet')]}",
        f"Train no_helmet: {before[('train', 'no_helmet')]}",
        f"Valid helmet: {before[('valid', 'helmet')]}",
        f"Valid no_helmet: {before[('valid', 'no_helmet')]}",
        f"Test helmet: {before[('test', 'helmet')]}",
        f"Test no_helmet: {before[('test', 'no_helmet')]}",
        "",
        "REMOVED",
        f"Exact crop duplicate: {removed['EXACT_CROP_DUPLICATE']}",
        f"Source image duplicate: {removed['SOURCE_IMAGE_DUPLICATE']}",
        f"Source sequence leakage: {removed['SOURCE_SEQUENCE_LEAKAGE']}",
        f"Near duplicate: {removed['NEAR_DUPLICATE']}",
        f"Label conflicts: {len(label_conflicts)}",
        "",
        "AFTER",
        f"Train helmet: {after[('train', 'helmet')]}",
        f"Train no_helmet: {after[('train', 'no_helmet')]}",
        f"Valid helmet: {after[('valid', 'helmet')]}",
        f"Valid no_helmet: {after[('valid', 'no_helmet')]}",
        f"Test helmet: {after[('test', 'helmet')]}",
        f"Test no_helmet: {after[('test', 'no_helmet')]}",
        "",
        "FINAL CHECK",
        f"Train/Valid exact leakage: {'PASS' if exact_pass else 'FAIL'}",
        f"Train/Test exact leakage: {'PASS' if exact_pass else 'FAIL'}",
        f"Valid/Test exact leakage: {'PASS' if exact_pass else 'FAIL'}",
        f"Train/Valid source-group leakage: {'PASS' if source_group_pass else 'FAIL'}",
        f"Train/Test source-group leakage: {'PASS' if source_group_pass else 'FAIL'}",
        f"Valid/Test source-group leakage: {'PASS' if source_group_pass else 'FAIL'}",
        f"Near-duplicate leakage: {'PASS' if near_pass else 'FAIL'}",
        f"Label conflict: {'PASS' if label_pass else 'FAIL'}",
        "Training performed: NO",
        "Test used for training: NO",
        f"Input unchanged/read-only: YES ({input_root})",
    ]
    (output_root / "stage2_clean_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reports(
    output_root: Path,
    records: Sequence[CropRecord],
    removed_rows: Sequence[Mapping[str, object]],
    exact_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
    near_rows: Sequence[Mapping[str, object]],
    label_conflicts: Sequence[Mapping[str, object]],
) -> None:
    _write_csv(
        output_root / "reports" / "exact_crop_duplicates.csv",
        ("crop_a", "split_a", "class_a", "crop_b", "split_b", "class_b", "sha256", "action"),
        exact_rows,
    )
    _write_csv(
        output_root / "reports" / "source_image_duplicates.csv",
        ("source_image_a", "split_a", "source_image_b", "split_b", "sha256", "action"),
        source_rows,
    )
    _write_csv(
        output_root / "reports" / "source_group_conflicts.csv",
        (
            "source_group",
            "splits_found",
            "train_samples",
            "valid_samples",
            "test_samples",
            "kept_split",
            "dropped_train",
            "dropped_valid",
            "reason",
        ),
        group_rows,
    )
    _write_csv(
        output_root / "reports" / "near_crop_duplicates.csv",
        (
            "crop_a",
            "split_a",
            "class_a",
            "crop_b",
            "split_b",
            "class_b",
            "phash_a",
            "phash_b",
            "distance",
            "source_group_a",
            "source_group_b",
            "action",
        ),
        near_rows,
    )
    _write_csv(
        output_root / "reports" / "label_conflict_duplicates.csv",
        (
            "duplicate_type",
            "crop_a",
            "split_a",
            "class_a",
            "crop_b",
            "split_b",
            "class_b",
            "sha256",
            "phash_a",
            "phash_b",
            "distance",
            "source_group_a",
            "source_group_b",
            "action",
        ),
        label_conflicts,
    )
    _write_csv(
        output_root / "reports" / "removed_samples.csv",
        (
            "original_crop",
            "class_name",
            "original_split",
            "source_image",
            "canonical_source_name",
            "source_group",
            "reason",
            "conflict_with",
            "kept_split",
        ),
        removed_rows,
    )
    _write_analysis_reports(records, removed_rows, output_root)


def _write_clean_manifest(records: Sequence[CropRecord], output_root: Path) -> None:
    fields = (
        "split",
        "class_name",
        "source_image",
        "source_label",
        "object_index",
        "canonical_source_name",
        "source_group",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "crop_width",
        "crop_height",
        "sha256",
        "phash",
        "original_crop",
        "clean_crop",
    )
    _write_csv(
        output_root / "manifest.csv",
        fields,
        (_clean_manifest_row(record, output_root) for record in records if not record.drop_reasons),
    )


def _write_final_check_report(output_root: Path, checks: Mapping[str, object]) -> None:
    rows = [
        ("exact_crop_leakage", "PASS" if not checks["exact_crop_leakage"] else "FAIL"),
        ("source_image_leakage", "PASS" if not checks["source_image_leakage"] else "FAIL"),
        ("source_group_leakage", "PASS" if not checks["source_group_leakage"] else "FAIL"),
        ("near_duplicate_leakage", "PASS" if not checks["near_leakage"] else "FAIL"),
        ("label_conflict", "PASS" if not checks["label_conflicts"] else "FAIL"),
    ]
    _write_csv(
        output_root / "reports" / "final_leakage_check.csv",
        ("check", "status"),
        ({"check": check, "status": status} for check, status in rows),
    )


def run_cleaning(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    input_root = (input_dir or Path.cwd() / "Stage2_Vietnam").resolve()
    output_root = (output_dir or Path.cwd() / "Stage2_Vietnam_Clean").resolve()
    if not (input_root / "manifest.csv").exists():
        raise CleaningError(f"STOP: manifest.csv was not found: {input_root / 'manifest.csv'}")
    _prepare_output_layout(output_root)
    records = load_manifest(input_root)
    exact_rows, source_rows, group_rows, near_rows, label_conflicts = _run_duplicate_audit(records)
    removed_rows = _removed_rows(records)
    accepted = [record for record in records if not record.drop_reasons]
    _copy_accepted_records(accepted, output_root)
    _write_clean_manifest(accepted, output_root)
    records_by_relative = {record.original_crop_relative: record for record in records}
    _write_duplicate_examples(label_conflicts, records_by_relative, output_root)
    _write_reports(
        output_root,
        records,
        removed_rows,
        exact_rows,
        source_rows,
        group_rows,
        near_rows,
        label_conflicts,
    )
    _write_contact_sheets(records, output_root, seed)
    checks = _final_checks(records)
    _write_final_check_report(output_root, checks)
    _write_summary(
        input_root,
        output_root,
        records,
        removed_rows,
        exact_rows,
        source_rows,
        group_rows,
        near_rows,
        label_conflicts,
        checks,
    )
    required_paths = [
        *(output_root / split / class_name for split in SPLITS for class_name in CLASS_NAMES),
        output_root / "manifest.csv",
        output_root / "stage2_clean_summary.txt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise CleaningError("Final clean layout assertion failed:\n" + "\n".join(missing))

    exact_pass = not checks["exact_crop_leakage"]
    source_group_pass = not checks["source_group_leakage"]
    near_pass = not checks["near_leakage"]
    label_pass = not checks["label_conflicts"]
    ready = exact_pass and not checks["source_image_leakage"] and source_group_pass and near_pass and label_pass
    counts = Counter((record.split, record.class_name) for record in checks["accepted"])
    removed_counts = Counter(row["reason"] for row in removed_rows)
    print("========================================")
    print("STAGE2 CLEANING COMPLETE")
    print("========================================")
    print(f"Original: {input_root}")
    print(f"Clean: {output_root}")
    print(f"Removed exact: {removed_counts['EXACT_CROP_DUPLICATE']}")
    print(f"Removed source-image duplicates: {removed_counts['SOURCE_IMAGE_DUPLICATE']}")
    print(f"Removed source-sequence leakage: {removed_counts['SOURCE_SEQUENCE_LEAKAGE']}")
    print(f"Removed near duplicates: {removed_counts['NEAR_DUPLICATE']}")
    print(f"Label conflicts: {len(label_conflicts)}")
    for split in SPLITS:
        print(split.upper())
        print(f"helmet: {counts[(split, 'helmet')]}")
        print(f"no_helmet: {counts[(split, 'no_helmet')]}")
    print(f"Exact leakage: {'PASS' if exact_pass else 'FAIL'}")
    print(f"Source group leakage: {'PASS' if source_group_pass else 'FAIL'}")
    print(f"Near duplicate leakage: {'PASS' if near_pass else 'FAIL'}")
    print(f"Label conflicts: {'PASS' if label_pass else 'FAIL'}")
    print("TEST PRIORITY: PRESERVED")
    print("TEST USED FOR TRAINING: NO")
    print("MODEL TRAINING: NO")
    print(f"READY FOR CLASSIFIER TRAINING: {'YES' if ready else 'NO'}")
    print("========================================")
    if not ready:
        if label_conflicts:
            raise LabelConflictError(
                "STOP: label conflicts remain; manual review is required before READY."
            )
        raise CleaningError("STOP: final clean leakage checks did not pass.")
    return output_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd() / "Stage2_Vietnam",
        help="Existing Stage2_Vietnam directory; it is read-only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "Stage2_Vietnam_Clean",
        help="New clean output directory; it must not already exist.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        run_cleaning(args.input_dir, args.output_dir, args.seed)
    except LabelConflictError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except CleaningError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
