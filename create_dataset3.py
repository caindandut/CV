"""Create Dataset3 from the approved Vietnam and Luna YOLO datasets.

This script is intentionally dataset-building only. It never imports a model
package, performs inference, or trains a model. Source directories are read
only and all output is written below Dataset3/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal environments
    yaml = None

try:
    from PIL import Image, ImageDraw, ImageOps
except ImportError as exc:  # pragma: no cover - required runtime dependency
    raise SystemExit("Pillow is required. Install it with: pip install Pillow") from exc

try:
    import imagehash as imagehash_lib  # type: ignore
except ImportError:  # pragma: no cover - fallback is covered instead
    imagehash_lib = None

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - fallback is covered instead
    tqdm = None

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - pure-Python fallback remains available
    np = None


LOGGER = logging.getLogger("dataset3")
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "Dataset3"
VIETNAM_DIR_NAME = "DataSet2-20260927T023713Z-1-001"
LUNA_DIR_NAME = "Helmet-Detection-Project.v5-version-2.1.yolo26"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["motorcycle", "rider_with_helmet", "rider_without_helmet"]
CLASS_KEYS = {0: "num_motorcycle", 1: "num_with_helmet", 2: "num_without_helmet"}
PHASH_THRESHOLD = 4
RANDOM_SEED = 20260927


class DatasetError(RuntimeError):
    """Raised when a source or output invariant is not safe to continue."""


@dataclass(frozen=True)
class LabelParseResult:
    num_objects: int
    class_counts: dict[int, int]
    invalid_rows: list[dict[str, Any]]
    bbox_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class SourceConfig:
    name: str
    directory_name: str
    workspace: str
    project: str
    version: int | None
    prefix: str


@dataclass(frozen=True)
class SourceRoot:
    config: SourceConfig
    yaml_path: Path
    root: Path
    metadata: dict[str, Any]
    splits: dict[str, tuple[Path, Path]]


@dataclass
class ImageRecord:
    source_dataset: str
    source_split: str
    image_path: Path
    label_path: Path
    filename: str
    width: int
    height: int
    sha256: str
    phash: str
    num_objects: int
    class_counts: dict[int, int]
    bbox_rows: list[dict[str, Any]] = field(default_factory=list)
    invalid: bool = False
    invalid_reasons: list[str] = field(default_factory=list)

    @property
    def is_vietnam(self) -> bool:
        return self.source_dataset == "Vietnam"

    @property
    def is_luna(self) -> bool:
        return self.source_dataset == "Luna"


@dataclass(frozen=True)
class DuplicateDecision:
    luna: ImageRecord
    exact_vietnam_splits: frozenset[str]
    exact_luna: bool
    near_vietnam_splits: frozenset[str]
    near_luna: bool
    action: str


@dataclass
class SplitScan:
    source_dataset: str
    source_split: str
    records: list[ImageRecord] = field(default_factory=list)
    missing_labels: list[dict[str, Any]] = field(default_factory=list)
    orphan_labels: list[dict[str, Any]] = field(default_factory=list)
    invalid_labels: list[dict[str, Any]] = field(default_factory=list)
    total_images: int = 0
    total_labels: int = 0


@dataclass
class SourceInventory:
    source_root: SourceRoot
    scans: dict[str, SplitScan]

    def records(self, split: str) -> list[ImageRecord]:
        return self.scans[split].records


def progress(items: Iterable[Any], desc: str, total: int | None = None) -> Iterator[Any]:
    """Use tqdm when available, with a deterministic logging fallback."""
    if tqdm is not None:
        yield from tqdm(items, desc=desc, total=total, unit="item")
        return
    for index, item in enumerate(items, start=1):
        if index == 1 or index % 250 == 0 or (total is not None and index == total):
            LOGGER.info("%s: %d%s", desc, index, f"/{total}" if total else "")
        yield item


def _strip_yaml_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:index].rstrip()
    return value.strip()


def _fallback_scalar(value: str) -> Any:
    value = _strip_yaml_comment(value).strip()
    if not value:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    if value.lower() in {"null", "none"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        values = []
        for part in value[1:-1].split(","):
            part = part.strip()
            if part:
                values.append(_fallback_scalar(part))
        return values
    return value


def _fallback_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by Roboflow exports."""
    result: dict[str, Any] = {}
    current_map: dict[str, Any] | None = None
    current_indent = -1
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = _strip_yaml_comment(raw_line.strip())
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        parsed_key = _fallback_scalar(key)
        if not isinstance(parsed_key, (str, int)):
            parsed_key = key
        raw_value = raw_value.strip()
        if indent == 0:
            if raw_value:
                result[parsed_key] = _fallback_scalar(raw_value)
                current_map = None
            else:
                result[parsed_key] = {}
                current_map = result[parsed_key]
                current_indent = indent
        elif current_map is not None and indent > current_indent:
            current_map[parsed_key] = _fallback_scalar(raw_value)
    return result


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if yaml is not None:
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise DatasetError(f"YAML root is not a mapping: {path}")
        return dict(loaded)
    return _fallback_yaml_load(text)


def _roboflow_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = metadata.get("roboflow")
    return nested if isinstance(nested, Mapping) else metadata


def validate_source_metadata(path: Path, config: SourceConfig) -> dict[str, Any]:
    metadata = load_yaml_mapping(path)
    roboflow = _roboflow_metadata(metadata)
    workspace = roboflow.get("workspace")
    project = roboflow.get("project")
    version = roboflow.get("version")
    if workspace != config.workspace or project != config.project:
        raise DatasetError(
            f"Wrong {config.name} metadata in {path}: "
            f"workspace={workspace!r}, project={project!r}"
        )
    if config.version is not None and version != config.version:
        raise DatasetError(f"Wrong {config.name} version in {path}: {version!r}")
    nc = metadata.get("nc")
    names = metadata.get("names")
    if nc != 3 or list(names or []) != CLASS_NAMES:
        raise DatasetError(f"Wrong class schema in {path}: nc={nc!r}, names={names!r}")
    return metadata


def resolve_split_path(yaml_path: Path, configured_path: str) -> Path:
    candidate = Path(configured_path).expanduser()
    if not candidate.is_absolute():
        candidate = yaml_path.parent / candidate
    resolved = candidate.resolve()
    if not resolved.is_dir():
        # Some downloaded exports keep data.yaml one level below the actual
        # dataset root while retaining paths such as ../train/images. In that
        # case, resolve the same relative leaf only within the approved YAML
        # subtree; never search outside the source dataset folder.
        relative_parts = [part for part in Path(configured_path).parts if part not in {".", ".."}]
        if relative_parts:
            fallback_pattern = "/".join(relative_parts)
            fallback_matches = sorted(
                path.resolve()
                for path in yaml_path.parent.rglob(fallback_pattern)
                if path.is_dir()
            )
            if len(fallback_matches) == 1:
                LOGGER.warning(
                    "Resolved %s from nested dataset root: %s -> %s",
                    configured_path,
                    yaml_path,
                    fallback_matches[0],
                )
                return fallback_matches[0]
            if len(fallback_matches) > 1:
                raise DatasetError(
                    f"Ambiguous nested split path for {yaml_path}: {fallback_matches}"
                )
        raise DatasetError(f"Resolved split path does not exist: {resolved}")
    return resolved


def resolve_split_directories(yaml_path: Path, metadata: Mapping[str, Any]) -> dict[str, tuple[Path, Path]]:
    split_paths: dict[str, tuple[Path, Path]] = {}
    for output_name, yaml_key in (("train", "train"), ("valid", "val"), ("test", "test")):
        configured = metadata.get(yaml_key)
        if not isinstance(configured, str):
            raise DatasetError(f"Missing {yaml_key!r} path in {yaml_path}")
        images_dir = resolve_split_path(yaml_path, configured)
        if images_dir.name.lower() != "images":
            LOGGER.warning("%s split path is not named images: %s", output_name, images_dir)
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            raise DatasetError(f"Missing labels directory for {images_dir}: {labels_dir}")
        split_paths[output_name] = (images_dir, labels_dir)
    return split_paths


def find_source_root(config: SourceConfig, project_root: Path) -> SourceRoot:
    base = project_root / config.directory_name
    if not base.is_dir():
        raise DatasetError(f"Approved source folder not found: {base}")
    matches = sorted(base.rglob("data.yaml"))
    valid: list[tuple[Path, dict[str, Any]]] = []
    metadata_matches: list[tuple[Path, str]] = []
    for yaml_path in matches:
        try:
            metadata = validate_source_metadata(yaml_path, config)
        except DatasetError as exc:
            LOGGER.debug("Ignoring non-matching YAML %s: %s", yaml_path, exc)
            continue
        try:
            resolve_split_directories(yaml_path, metadata)
        except DatasetError as exc:
            metadata_matches.append((yaml_path, str(exc)))
            continue
        valid.append((yaml_path, metadata))
    if len(valid) != 1:
        if not valid and metadata_matches:
            details = "; ".join(f"{path}: {reason}" for path, reason in metadata_matches)
            raise DatasetError(f"{config.name} metadata matched, but split resolution failed: {details}")
        candidates = ", ".join(str(path) for path, _ in valid) or "none"
        raise DatasetError(f"Expected exactly one valid {config.name} data.yaml under {base}; found {candidates}")
    yaml_path, metadata = valid[0]
    return SourceRoot(
        config=config,
        yaml_path=yaml_path,
        root=yaml_path.parent,
        metadata=metadata,
        splits=resolve_split_directories(yaml_path, metadata),
    )


def parse_label_file(label_path: Path) -> LabelParseResult:
    class_counts = {0: 0, 1: 0, 2: 0}
    invalid_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    with label_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            reason = None
            if len(fields) != 5:
                reason = f"expected 5 fields, found {len(fields)}"
            else:
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError:
                    reason = "non-numeric field"
                else:
                    x_center, y_center, width, height = values
                    if class_id not in class_counts:
                        reason = f"class_id {class_id} is outside 0..2"
                    elif not 0 <= x_center <= 1:
                        reason = f"x_center {x_center} outside [0, 1]"
                    elif not 0 <= y_center <= 1:
                        reason = f"y_center {y_center} outside [0, 1]"
                    elif not 0 < width <= 1:
                        reason = f"width {width} outside (0, 1]"
                    elif not 0 < height <= 1:
                        reason = f"height {height} outside (0, 1]"
                    else:
                        class_counts[class_id] += 1
                        bbox_rows.append(
                            {
                                "class_id": class_id,
                                "x_center": x_center,
                                "y_center": y_center,
                                "width": width,
                                "height": height,
                            }
                        )
            if reason is not None:
                invalid_rows.append(
                    {"label_path": str(label_path), "line_number": line_number, "reason": reason, "raw": stripped}
                )
    return LabelParseResult(
        num_objects=sum(class_counts.values()),
        class_counts=class_counts,
        invalid_rows=invalid_rows,
        bbox_rows=bbox_rows,
    )


def destination_filename(prefix: str, source_path: Path) -> str:
    return f"{prefix}__{source_path.name}"


def select_luna_action(
    *,
    exact_vietnam_splits: set[str] | frozenset[str],
    exact_luna: bool,
    near_vietnam_splits: set[str] | frozenset[str],
    near_luna: bool,
) -> str:
    if exact_vietnam_splits:
        return "drop_vietnam_leakage"
    if exact_luna:
        return "drop_luna_duplicate"
    if near_vietnam_splits:
        return "drop_vietnam_leakage"
    if near_luna:
        return "drop_luna_duplicate"
    return "keep"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fallback_phash(path: Path) -> str:
    """Small pure-Python pHash fallback matching ImageHash's 64-bit shape."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = [[image.getpixel((x, y)) for x in range(32)] for y in range(32)]
    if np is not None:
        matrix = np.asarray(pixels, dtype=float)
        coords = np.arange(32, dtype=float)
        basis = np.cos((2 * coords[:, None] + 1) * np.arange(32, dtype=float)[None, :] * math.pi / 64)
        basis[:, 0] *= 1 / math.sqrt(2)
        low_frequency_basis = basis[:, :8]
        dct = 0.25 * low_frequency_basis.T @ matrix @ low_frequency_basis
        coefficients = dct.reshape(-1).tolist()
        median = float(np.median(np.asarray(coefficients[1:])))
        bits = ["1" if value > median else "0" for value in coefficients]
        return f"{int(''.join(bits), 2):016x}"
    coefficients: list[float] = []
    for u in range(8):
        for v in range(8):
            total = 0.0
            for x in range(32):
                for y in range(32):
                    total += pixels[x][y] * math.cos((2 * x + 1) * u * math.pi / 64) * math.cos(
                        (2 * y + 1) * v * math.pi / 64
                    )
            alpha_u = 1 / math.sqrt(2) if u == 0 else 1.0
            alpha_v = 1 / math.sqrt(2) if v == 0 else 1.0
            coefficients.append(0.25 * alpha_u * alpha_v * total)
    median = sorted(coefficients[1:])[len(coefficients[1:]) // 2]
    bits = ["1" if value > median else "0" for value in coefficients]
    return f"{int(''.join(bits), 2):016x}"


def phash_file(path: Path) -> str:
    if imagehash_lib is not None:
        with Image.open(path) as opened:
            value = imagehash_lib.phash(ImageOps.exif_transpose(opened))
        return str(value)
    return fallback_phash(path)


def phash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _relative_image_key(path: Path, base: Path) -> str:
    relative = path.relative_to(base)
    return relative.with_suffix("").as_posix().lower()


def _iter_image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        (path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(images_dir).as_posix().lower(),
    )


def _iter_label_paths(labels_dir: Path) -> list[Path]:
    return sorted(
        (path for path in labels_dir.rglob("*.txt") if path.is_file()),
        key=lambda path: path.relative_to(labels_dir).as_posix().lower(),
    )


def scan_split(
    source_dataset: str,
    source_split: str,
    images_dir: Path,
    labels_dir: Path,
    *,
    compute_hashes: bool = True,
) -> SplitScan:
    image_paths = _iter_image_paths(images_dir)
    label_paths = _iter_label_paths(labels_dir)
    scan = SplitScan(
        source_dataset=source_dataset,
        source_split=source_split,
        total_images=len(image_paths),
        total_labels=len(label_paths),
    )
    image_keys = {_relative_image_key(path, images_dir): path for path in image_paths}
    label_keys = {_relative_image_key(path, labels_dir): path for path in label_paths}

    for image_path in progress(image_paths, f"{source_dataset} {source_split}", total=len(image_paths)):
        image_key = _relative_image_key(image_path, images_dir)
        relative_image = image_path.relative_to(images_dir)
        expected_label = labels_dir / relative_image.with_suffix(".txt")
        label_path = label_keys.get(image_key)
        if label_path is None:
            scan.missing_labels.append(
                {
                    "source_dataset": source_dataset,
                    "source_split": source_split,
                    "image_path": str(image_path),
                    "expected_label_path": str(expected_label),
                }
            )
            continue
        try:
            parsed = parse_label_file(label_path)
        except (OSError, UnicodeError) as exc:
            parsed = LabelParseResult(
                num_objects=0,
                class_counts={0: 0, 1: 0, 2: 0},
                invalid_rows=[
                    {
                        "label_path": str(label_path),
                        "line_number": "",
                        "reason": f"could not read label: {exc}",
                        "raw": "",
                    }
                ],
                bbox_rows=[],
            )
        if parsed.invalid_rows:
            for row in parsed.invalid_rows:
                scan.invalid_labels.append(
                    {
                        "source_dataset": source_dataset,
                        "source_split": source_split,
                        "image_path": str(image_path),
                        **row,
                    }
                )
            continue
        try:
            with Image.open(image_path) as opened:
                corrected = ImageOps.exif_transpose(opened)
                width, height = corrected.size
            image_sha256 = sha256_file(image_path) if compute_hashes else ""
            image_phash = phash_file(image_path) if compute_hashes else ""
        except (OSError, ValueError) as exc:
            scan.invalid_labels.append(
                {
                    "source_dataset": source_dataset,
                    "source_split": source_split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "line_number": "",
                    "reason": f"could not read image: {exc}",
                    "raw": "",
                }
            )
            continue
        scan.records.append(
            ImageRecord(
                source_dataset=source_dataset,
                source_split=source_split,
                image_path=image_path,
                label_path=label_path,
                filename=image_path.name,
                width=width,
                height=height,
                sha256=image_sha256,
                phash=image_phash,
                num_objects=parsed.num_objects,
                class_counts=parsed.class_counts,
                bbox_rows=parsed.bbox_rows,
            )
        )

    for label_path in label_paths:
        label_key = _relative_image_key(label_path, labels_dir)
        if label_key not in image_keys:
            scan.orphan_labels.append(
                {
                    "source_dataset": source_dataset,
                    "source_split": source_split,
                    "label_path": str(label_path),
                    "expected_image_stem": label_key,
                }
            )
    LOGGER.info(
        "%s %s: images=%d labels=%d valid_pairs=%d missing=%d orphan=%d invalid=%d",
        source_dataset,
        source_split,
        scan.total_images,
        scan.total_labels,
        len(scan.records),
        len(scan.missing_labels),
        len(scan.orphan_labels),
        len(scan.invalid_labels),
    )
    return scan


def scan_inventory(source_root: SourceRoot, source_dataset: str) -> SourceInventory:
    scans = {}
    for split in ("train", "valid", "test"):
        images_dir, labels_dir = source_root.splits[split]
        scans[split] = scan_split(
            source_dataset,
            split,
            images_dir,
            labels_dir,
            compute_hashes=(source_dataset == "Vietnam" or split == "train"),
        )
    return SourceInventory(source_root=source_root, scans=scans)


class BKTree:
    """A small Hamming-distance BK-tree for 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self._root: tuple[str, list[ImageRecord], dict[int, tuple[str, list[ImageRecord], dict]]] | None = None

    def insert(self, record: ImageRecord) -> None:
        if self._root is None:
            self._root = (record.phash, [record], {})
            return
        node = self._root
        while True:
            distance = phash_distance(record.phash, node[0])
            if distance == 0:
                node[1].append(record)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (record.phash, [record], {})
                return
            node = child

    def query(self, phash: str, threshold: int) -> list[tuple[ImageRecord, int]]:
        if self._root is None:
            return []
        found: list[tuple[ImageRecord, int]] = []
        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = phash_distance(phash, node[0])
            if distance <= threshold:
                found.extend((record, distance) for record in node[1])
            lower = max(0, distance - threshold)
            upper = distance + threshold
            pending.extend(child for edge, child in node[2].items() if lower <= edge <= upper)
        return sorted(found, key=lambda item: (item[1], item[0].source_split, str(item[0].image_path)))


@dataclass
class DuplicateAnalysis:
    decisions: list[DuplicateDecision]
    exact_rows: list[dict[str, Any]]
    near_rows: list[dict[str, Any]]
    accepted_luna: list[ImageRecord]

    @property
    def exact_duplicates_removed(self) -> int:
        return sum(1 for decision in self.decisions if decision.action == "drop_luna_duplicate" and decision.exact_luna)

    @property
    def near_duplicates_removed(self) -> int:
        return sum(
            1
            for decision in self.decisions
            if decision.action == "drop_luna_duplicate" and not decision.exact_luna and decision.near_luna
        )

    @property
    def leakage_removed(self) -> int:
        return sum(
            1
            for decision in self.decisions
            if decision.action == "drop_vietnam_leakage"
            and (decision.exact_vietnam_splits or decision.near_vietnam_splits)
        )


def _duplicate_row(
    luna: ImageRecord,
    other: ImageRecord,
    distance: int,
    action: str,
) -> dict[str, Any]:
    return {
        "image_a": str(luna.image_path),
        "dataset_a": luna.source_dataset,
        "split_a": luna.source_split,
        "image_b": str(other.image_path),
        "dataset_b": other.source_dataset,
        "split_b": other.source_split,
        "sha256_equal": luna.sha256 == other.sha256,
        "phash_a": luna.phash,
        "phash_b": other.phash,
        "phash_distance": distance,
        "action": action,
    }


def analyse_duplicates(vietnam: SourceInventory, luna_train: Sequence[ImageRecord]) -> DuplicateAnalysis:
    vietnam_records = {
        split: sorted(vietnam.records(split), key=lambda record: str(record.image_path))
        for split in ("train", "valid", "test")
    }
    vietnam_by_sha: dict[str, list[ImageRecord]] = defaultdict(list)
    vietnam_trees: dict[str, BKTree] = {split: BKTree() for split in ("train", "valid", "test")}
    for split, records in vietnam_records.items():
        for record in records:
            vietnam_by_sha[record.sha256].append(record)
            vietnam_trees[split].insert(record)

    luna_tree = BKTree()
    luna_sha: dict[str, list[ImageRecord]] = defaultdict(list)
    decisions: list[DuplicateDecision] = []
    exact_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    accepted_luna: list[ImageRecord] = []

    for luna in progress(sorted(luna_train, key=lambda record: str(record.image_path)), "deduplicate Luna train", total=len(luna_train)):
        exact_vietnam = vietnam_by_sha.get(luna.sha256, [])
        exact_vietnam_splits = frozenset(record.source_split for record in exact_vietnam)
        exact_luna_matches = luna_sha.get(luna.sha256, [])
        exact_luna = bool(exact_luna_matches)
        for other in exact_vietnam:
            exact_rows.append(
                {
                    "image_a": str(luna.image_path),
                    "dataset_a": luna.source_dataset,
                    "split_a": luna.source_split,
                    "image_b": str(other.image_path),
                    "dataset_b": other.source_dataset,
                    "split_b": other.source_split,
                    "sha256_a": luna.sha256,
                    "sha256_b": other.sha256,
                    "reason": "exact_vietnam_match",
                    "action": "drop_luna",
                }
            )
        for other in exact_luna_matches:
            exact_rows.append(
                {
                    "image_a": str(luna.image_path),
                    "dataset_a": luna.source_dataset,
                    "split_a": luna.source_split,
                    "image_b": str(other.image_path),
                    "dataset_b": other.source_dataset,
                    "split_b": other.source_split,
                    "sha256_a": luna.sha256,
                    "sha256_b": other.sha256,
                    "reason": "exact_luna_match",
                    "action": "drop_luna_duplicate",
                }
            )

        near_matches: dict[tuple[str, str], tuple[ImageRecord, int]] = {}
        for split in ("valid", "test", "train"):
            for other, distance in vietnam_trees[split].query(luna.phash, PHASH_THRESHOLD):
                near_matches[(split, str(other.image_path))] = (other, distance)
        for other, distance in luna_tree.query(luna.phash, PHASH_THRESHOLD):
            near_matches[("luna", str(other.image_path))] = (other, distance)
        near_vietnam_splits = frozenset(key[0] for key in near_matches if key[0] in {"train", "valid", "test"})
        near_luna = any(key[0] == "luna" for key in near_matches)
        action = select_luna_action(
            exact_vietnam_splits=exact_vietnam_splits,
            exact_luna=exact_luna,
            near_vietnam_splits=set(near_vietnam_splits),
            near_luna=near_luna,
        )
        for other, distance in near_matches.values():
            near_rows.append(_duplicate_row(luna, other, distance, action))
        decision = DuplicateDecision(
            luna=luna,
            exact_vietnam_splits=exact_vietnam_splits,
            exact_luna=exact_luna,
            near_vietnam_splits=near_vietnam_splits,
            near_luna=near_luna,
            action=action,
        )
        decisions.append(decision)
        if action == "keep":
            accepted_luna.append(luna)
            luna_tree.insert(luna)
            luna_sha[luna.sha256].append(luna)

    return DuplicateAnalysis(
        decisions=decisions,
        exact_rows=exact_rows,
        near_rows=near_rows,
        accepted_luna=accepted_luna,
    )


def source_file_state(roots: Sequence[SourceRoot]) -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    for source in roots:
        for path in source.root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                state[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
    return state


def ensure_output_layout(output_root: Path) -> None:
    if output_root.exists():
        raise DatasetError("Dataset3 already exists. Delete or rename it manually before rebuilding.")
    for split in ("train", "valid", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=False)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=False)
    (output_root / "reports" / "contact_sheets").mkdir(parents=True, exist_ok=False)
    (output_root / "reports" / "duplicate_examples").mkdir(parents=True, exist_ok=False)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def copy_records(
    records: Sequence[ImageRecord],
    dataset3_split: str,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    image_destination_dir = output_root / dataset3_split / "images"
    label_destination_dir = output_root / dataset3_split / "labels"
    manifest_rows: list[dict[str, Any]] = []
    destination_by_source: dict[str, Path] = {}
    used_image_names: set[str] = set()
    for record in progress(sorted(records, key=lambda item: (item.source_dataset, str(item.image_path))), f"copy {dataset3_split}", total=len(records)):
        prefix = "vn" if record.source_dataset == "Vietnam" else "luna"
        image_name = destination_filename(prefix, Path(record.filename))
        if image_name in used_image_names:
            raise DatasetError(f"Destination filename collision in {dataset3_split}: {image_name}")
        used_image_names.add(image_name)
        destination_image = image_destination_dir / image_name
        destination_label = label_destination_dir / f"{destination_image.stem}.txt"
        shutil.copy2(record.image_path, destination_image)
        shutil.copy2(record.label_path, destination_label)
        destination_by_source[str(record.image_path)] = destination_image
        manifest_rows.append(
            {
                "dataset3_split": dataset3_split,
                "dataset3_image": destination_image.relative_to(output_root).as_posix(),
                "dataset3_label": destination_label.relative_to(output_root).as_posix(),
                "source_dataset": record.source_dataset,
                "source_split": record.source_split,
                "source_image": str(record.image_path),
                "source_label": str(record.label_path),
                "sha256": record.sha256,
                "phash": record.phash,
                "width": record.width,
                "height": record.height,
                "num_motorcycle": record.class_counts[0],
                "num_with_helmet": record.class_counts[1],
                "num_without_helmet": record.class_counts[2],
            }
        )
    return manifest_rows, destination_by_source


def counts_for_records(records: Sequence[ImageRecord]) -> dict[str, int]:
    counts = {"images": len(records), "motorcycle": 0, "rider_with_helmet": 0, "rider_without_helmet": 0, "total_objects": 0}
    for record in records:
        counts["motorcycle"] += record.class_counts[0]
        counts["rider_with_helmet"] += record.class_counts[1]
        counts["rider_without_helmet"] += record.class_counts[2]
        counts["total_objects"] += record.num_objects
    return counts


def distribution_row(name: str, records: Sequence[ImageRecord]) -> dict[str, Any]:
    counts = counts_for_records(records)
    return {
        "dataset_group": name,
        "images": counts["images"],
        "motorcycle": counts["motorcycle"],
        "rider_with_helmet": counts["rider_with_helmet"],
        "rider_without_helmet": counts["rider_without_helmet"],
        "total_objects": counts["total_objects"],
    }


def bbox_size_category(area: float) -> str:
    if area < 0.0025:
        return "tiny"
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def bbox_distribution_rows(source_groups: Mapping[str, Sequence[ImageRecord]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for group_name, records in source_groups.items():
        for record in records:
            for bbox in record.bbox_rows:
                class_name = CLASS_NAMES[bbox["class_id"]]
                category = bbox_size_category(bbox["width"] * bbox["height"])
                counts[(group_name, class_name, category)] += 1
    rows = []
    for (source_dataset, class_name, size_category), count in sorted(counts.items()):
        rows.append(
            {
                "source_dataset": source_dataset,
                "class": class_name,
                "size_category": size_category,
                "count": count,
            }
        )
    return rows


def load_for_contact(record: ImageRecord, dataset3_image: Path | None = None) -> tuple[Image.Image, ImageRecord]:
    image_path = dataset3_image or record.image_path
    with Image.open(image_path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB"), record


BOX_COLORS = {0: (225, 45, 45), 1: (30, 170, 70), 2: (40, 90, 220)}


def make_contact_sheet(
    entries: Sequence[tuple[ImageRecord, Path]],
    output_path: Path,
    title: str,
    *,
    columns: int = 5,
    tile_size: int = 256,
) -> None:
    selected = list(entries)
    rows = max(1, math.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    draw_sheet = ImageDraw.Draw(sheet)
    for index, (record, image_path) in enumerate(selected):
        try:
            image, _ = load_for_contact(record, image_path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Skipping contact-sheet image %s: %s", image_path, exc)
            continue
        tile_x = (index % columns) * tile_size
        tile_y = (index // columns) * tile_size
        image.thumbnail((tile_size - 12, tile_size - 58), Image.Resampling.LANCZOS)
        paste_x = tile_x + (tile_size - image.width) // 2
        paste_y = tile_y + 22
        sheet.paste(image, (paste_x, paste_y))
        draw = ImageDraw.Draw(sheet)
        scale_x = image.width / max(record.width, 1)
        scale_y = image.height / max(record.height, 1)
        for bbox in record.bbox_rows:
            x_center = bbox["x_center"] * image.width
            y_center = bbox["y_center"] * image.height
            box_width = bbox["width"] * image.width
            box_height = bbox["height"] * image.height
            left = paste_x + x_center - box_width / 2
            top = paste_y + y_center - box_height / 2
            right = paste_x + x_center + box_width / 2
            bottom = paste_y + y_center + box_height / 2
            draw.rectangle((left, top, right, bottom), outline=BOX_COLORS[bbox["class_id"]], width=2)
        display_name = Path(record.filename).name
        draw.text((tile_x + 4, tile_y + 4), display_name[:38], fill="black")
    legend_y = max(0, rows * tile_size - 18)
    legend = " | ".join(f"{CLASS_NAMES[class_id]} ({BOX_COLORS[class_id]})" for class_id in range(3))
    draw_sheet.text((6, legend_y), f"{title} — {legend}", fill="black")
    sheet.save(output_path, quality=92)


def choose_contact_entries(
    vietnam_train: Sequence[ImageRecord],
    luna_accepted: Sequence[ImageRecord],
    dataset3_train: Sequence[ImageRecord],
    destination_by_source: Mapping[str, Path],
) -> dict[str, list[tuple[ImageRecord, Path]]]:
    rng = random.Random(RANDOM_SEED)

    def dataset3_entries(items: Sequence[ImageRecord]) -> list[tuple[ImageRecord, Path]]:
        return [(record, destination_by_source[str(record.image_path)]) for record in items]

    vietnam_train = sorted(vietnam_train, key=lambda record: str(record.image_path))
    luna_records = sorted(luna_accepted, key=lambda record: str(record.image_path))
    train_entries = dataset3_entries(dataset3_train)
    smallest_without_helmet = sorted(
        (record for record in dataset3_train if record.class_counts[2] > 0),
        key=lambda record: (
            min(bbox["width"] * bbox["height"] for bbox in record.bbox_rows if bbox["class_id"] == 2),
            str(record.image_path),
        ),
    )
    most_without_helmet = sorted(
        (record for record in dataset3_train if record.class_counts[2] > 0),
        key=lambda record: (-record.class_counts[2], str(record.image_path)),
    )

    def sample(items: Sequence[ImageRecord], count: int) -> list[ImageRecord]:
        if len(items) <= count:
            return list(items)
        return rng.sample(list(items), count)

    return {
        "A_vietnam_train": [(record, record.image_path) for record in sample(vietnam_train, 50)],
        "B_luna_accepted_train": [(record, record.image_path) for record in sample(luna_records, 50)],
        "C_dataset3_train": sample(train_entries, 50),
        "D_smallest_without_helmet": dataset3_entries(smallest_without_helmet[:30]),
        "E_most_without_helmet": dataset3_entries(most_without_helmet[:30]),
    }


MANIFEST_FIELDS = [
    "dataset3_split",
    "dataset3_image",
    "dataset3_label",
    "source_dataset",
    "source_split",
    "source_image",
    "source_label",
    "sha256",
    "phash",
    "width",
    "height",
    "num_motorcycle",
    "num_with_helmet",
    "num_without_helmet",
]


def write_dataset_yaml(output_root: Path) -> None:
    absolute_path = output_root.resolve().as_posix()
    content = (
        f"path: {json.dumps(absolute_path)}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: motorcycle\n"
        "  1: rider_with_helmet\n"
        "  2: rider_without_helmet\n"
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def source_inventory_rows(records: Sequence[ImageRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(records, key=lambda item: (item.source_dataset, item.source_split, str(item.image_path))):
        rows.append(
            {
                "source_dataset": record.source_dataset,
                "source_split": record.source_split,
                "image_path": str(record.image_path),
                "label_path": str(record.label_path),
                "filename": record.filename,
                "width": record.width,
                "height": record.height,
                "sha256": record.sha256,
                "phash": record.phash,
                "num_objects": record.num_objects,
                "num_motorcycle": record.class_counts[0],
                "num_with_helmet": record.class_counts[1],
                "num_without_helmet": record.class_counts[2],
            }
        )
    return rows


def write_reports(
    output_root: Path,
    vietnam: SourceInventory,
    luna: SourceInventory,
    duplicate_analysis: DuplicateAnalysis,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> None:
    reports = output_root / "reports"
    inventory_records = vietnam.records("train") + vietnam.records("valid") + vietnam.records("test") + luna.records("train")
    write_csv(
        reports / "source_inventory.csv",
        [
            "source_dataset",
            "source_split",
            "image_path",
            "label_path",
            "filename",
            "width",
            "height",
            "sha256",
            "phash",
            "num_objects",
            "num_motorcycle",
            "num_with_helmet",
            "num_without_helmet",
        ],
        source_inventory_rows(inventory_records),
    )
    write_csv(
        reports / "invalid_labels.csv",
        ["source_dataset", "source_split", "image_path", "label_path", "line_number", "reason", "raw"],
        [row for inventory in (vietnam, luna) for scan in inventory.scans.values() for row in scan.invalid_labels],
    )
    write_csv(
        reports / "missing_labels.csv",
        ["source_dataset", "source_split", "image_path", "expected_label_path"],
        [row for inventory in (vietnam, luna) for scan in inventory.scans.values() for row in scan.missing_labels],
    )
    write_csv(
        reports / "orphan_labels.csv",
        ["source_dataset", "source_split", "label_path", "expected_image_stem"],
        [row for inventory in (vietnam, luna) for scan in inventory.scans.values() for row in scan.orphan_labels],
    )
    write_csv(
        reports / "exact_duplicates.csv",
        [
            "image_a",
            "dataset_a",
            "split_a",
            "image_b",
            "dataset_b",
            "split_b",
            "sha256_a",
            "sha256_b",
            "reason",
            "action",
        ],
        duplicate_analysis.exact_rows,
    )
    write_csv(
        reports / "near_duplicates.csv",
        [
            "image_a",
            "dataset_a",
            "split_a",
            "image_b",
            "dataset_b",
            "split_b",
            "sha256_equal",
            "phash_a",
            "phash_b",
            "phash_distance",
            "action",
        ],
        duplicate_analysis.near_rows,
    )

    vietnam_train = vietnam.records("train")
    vietnam_valid = vietnam.records("valid")
    vietnam_test = vietnam.records("test")
    luna_original = luna.records("train")
    luna_accepted = duplicate_analysis.accepted_luna
    luna_dropped = [decision.luna for decision in duplicate_analysis.decisions if decision.action != "keep"]
    dataset3_train = vietnam_train + luna_accepted
    class_rows = [
        distribution_row("Vietnam train", vietnam_train),
        distribution_row("Luna original train", luna_original),
        distribution_row("Luna accepted", luna_accepted),
        distribution_row("Luna dropped", luna_dropped),
        distribution_row("Dataset3 train", dataset3_train),
        distribution_row("Dataset3 valid", vietnam_valid),
        distribution_row("Dataset3 test", vietnam_test),
    ]
    write_csv(
        reports / "class_distribution.csv",
        ["dataset_group", "images", "motorcycle", "rider_with_helmet", "rider_without_helmet", "total_objects"],
        class_rows,
    )
    write_csv(
        reports / "bbox_size_distribution.csv",
        ["source_dataset", "class", "size_category", "count"],
        bbox_distribution_rows(
            {
                "Vietnam": vietnam_train + vietnam_valid + vietnam_test,
                "Luna original": luna_original,
                "Luna accepted": luna_accepted,
            }
        ),
    )
    write_csv(output_root / "dataset3_manifest.csv", MANIFEST_FIELDS, manifest_rows)


def _output_record(row: Mapping[str, Any], output_root: Path) -> ImageRecord:
    image_path = output_root / str(row["dataset3_image"])
    with Image.open(image_path) as opened:
        width, height = ImageOps.exif_transpose(opened).size
    return ImageRecord(
        source_dataset=str(row["source_dataset"]),
        source_split=str(row["source_split"]),
        image_path=image_path,
        label_path=output_root / str(row["dataset3_label"]),
        filename=image_path.name,
        width=width,
        height=height,
        sha256=sha256_file(image_path),
        phash=phash_file(image_path),
        num_objects=int(row["num_motorcycle"]) + int(row["num_with_helmet"]) + int(row["num_without_helmet"]),
        class_counts={
            0: int(row["num_motorcycle"]),
            1: int(row["num_with_helmet"]),
            2: int(row["num_without_helmet"]),
        },
    )


def validate_output(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    vietnam: SourceInventory,
    source_state_before: Mapping[str, tuple[int, int]],
    source_roots: Sequence[SourceRoot],
) -> None:
    expected_paths = [
        output_root / "data.yaml",
        output_root / "train" / "images",
        output_root / "train" / "labels",
        output_root / "valid" / "images",
        output_root / "valid" / "labels",
        output_root / "test" / "images",
        output_root / "test" / "labels",
    ]
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise DatasetError(f"Dataset3 structure is incomplete: {missing}")
    yaml_metadata = load_yaml_mapping(output_root / "data.yaml")
    if yaml_metadata.get("nc") != 3 or yaml_metadata.get("names") != {0: "motorcycle", 1: "rider_with_helmet", 2: "rider_without_helmet"}:
        names = yaml_metadata.get("names")
        if names != CLASS_NAMES:
            raise DatasetError(f"Dataset3 class schema is wrong: {names!r}")

    output_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        output_by_split[str(row["dataset3_split"])].append(row)
        image_path = output_root / str(row["dataset3_image"])
        label_path = output_root / str(row["dataset3_label"])
        if not image_path.is_file() or not label_path.is_file():
            raise DatasetError(f"Manifest points to missing output pair: {image_path}, {label_path}")
    for split in ("valid", "test"):
        rows = output_by_split[split]
        if any(row["source_dataset"] != "Vietnam" for row in rows):
            raise DatasetError(f"Dataset3 {split} contains non-Vietnam data")
        source_records = vietnam.records(split)
        if len(rows) != len(source_records):
            raise DatasetError(f"Dataset3 {split} image count changed: {len(rows)} != {len(source_records)}")
        output_counts = Counter()
        for row in rows:
            output_counts["motorcycle"] += int(row["num_motorcycle"])
            output_counts["rider_with_helmet"] += int(row["num_with_helmet"])
            output_counts["rider_without_helmet"] += int(row["num_without_helmet"])
        source_counts = counts_for_records(source_records)
        if output_counts["motorcycle"] != source_counts["motorcycle"] or output_counts["rider_with_helmet"] != source_counts["rider_with_helmet"] or output_counts["rider_without_helmet"] != source_counts["rider_without_helmet"]:
            raise DatasetError(f"Dataset3 {split} object counts changed")

    valid_records = [_output_record(row, output_root) for row in output_by_split["valid"]]
    test_records = [_output_record(row, output_root) for row in output_by_split["test"]]
    luna_records = [_output_record(row, output_root) for row in output_by_split["train"] if row["source_dataset"] == "Luna"]
    valid_tree = BKTree()
    test_tree = BKTree()
    for record in valid_records:
        valid_tree.insert(record)
    for record in test_records:
        test_tree.insert(record)
    for luna_record in luna_records:
        if any(luna_record.sha256 == record.sha256 for record in valid_records + test_records):
            raise DatasetError(f"Post-merge exact leakage detected: {luna_record.image_path}")
        if valid_tree.query(luna_record.phash, PHASH_THRESHOLD) or test_tree.query(luna_record.phash, PHASH_THRESHOLD):
            raise DatasetError(f"Post-merge near leakage detected: {luna_record.image_path}")

    source_state_after = source_file_state(source_roots)
    if dict(source_state_before) != source_state_after:
        raise DatasetError("An approved source file changed during Dataset3 creation")
    if "ultralytics" in sys.modules:
        raise DatasetError("Model package was imported; training/inference is forbidden in this task")


def write_summary(
    output_root: Path,
    vietnam: SourceInventory,
    luna: SourceInventory,
    duplicate_analysis: DuplicateAnalysis,
    source_roots: Sequence[SourceRoot],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> None:
    vietnam_train = vietnam.records("train")
    vietnam_valid = vietnam.records("valid")
    vietnam_test = vietnam.records("test")
    luna_original = luna.records("train")
    luna_accepted = duplicate_analysis.accepted_luna
    dataset3_train = vietnam_train + luna_accepted
    vietnam_percentage = (len(vietnam_train) / len(dataset3_train) * 100) if dataset3_train else 0.0
    luna_percentage = (len(luna_accepted) / len(dataset3_train) * 100) if dataset3_train else 0.0
    summary = [
        "DATASET3 SUMMARY",
        "================",
        "",
        "SOURCE ROOTS",
        f"Vietnam: {source_roots[0].root}",
        f"Vietnam YAML: {source_roots[0].yaml_path}",
        f"Luna: {source_roots[1].root}",
        f"Luna YAML: {source_roots[1].yaml_path}",
        "",
        "SOURCE YAML METADATA",
        json.dumps({"Vietnam": source_roots[0].metadata, "Luna": source_roots[1].metadata}, ensure_ascii=False, indent=2),
        "",
        "IMAGE COUNTS",
        f"Vietnam train source: {len(vietnam_train)} valid pairs / {vietnam.scans['train'].total_images} images",
        f"Vietnam valid source: {len(vietnam_valid)} valid pairs / {vietnam.scans['valid'].total_images} images",
        f"Vietnam test source: {len(vietnam_test)} valid pairs / {vietnam.scans['test'].total_images} images",
        f"Luna train source: {len(luna_original)} valid pairs / {luna.scans['train'].total_images} images",
        "",
        "OBJECT COUNTS",
        json.dumps(
            {
                "Vietnam train": counts_for_records(vietnam_train),
                "Vietnam valid": counts_for_records(vietnam_valid),
                "Vietnam test": counts_for_records(vietnam_test),
                "Luna train": counts_for_records(luna_original),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "DUPLICATE COUNTS",
        f"Luna invalid removed: {sum(len(scan.invalid_labels) for scan in luna.scans.values())}",
        f"Luna missing label removed: {sum(len(scan.missing_labels) for scan in luna.scans.values())}",
        f"Luna exact duplicates removed: {duplicate_analysis.exact_duplicates_removed}",
        f"Luna near duplicates removed: {duplicate_analysis.near_duplicates_removed}",
        f"Luna leakage removed: {duplicate_analysis.leakage_removed}",
        f"Luna accepted: {len(luna_accepted)}",
        "",
        "CLASS DISTRIBUTION",
    ]
    for row in [
        distribution_row("Vietnam train", vietnam_train),
        distribution_row("Luna original train", luna_original),
        distribution_row("Luna accepted", luna_accepted),
        distribution_row("Luna dropped", [decision.luna for decision in duplicate_analysis.decisions if decision.action != "keep"]),
        distribution_row("Dataset3 train", dataset3_train),
        distribution_row("Dataset3 valid", vietnam_valid),
        distribution_row("Dataset3 test", vietnam_test),
    ]:
        summary.append(json.dumps(row, ensure_ascii=False))
    summary.extend(
        [
            "",
            "BBOX SIZE DISTRIBUTION",
            "See reports/bbox_size_distribution.csv; rider_without_helmet is included for Vietnam and Luna separately.",
            "",
            "DOMAIN RATIO",
            f"Vietnam contribution: {vietnam_percentage:.2f}% ({len(vietnam_train)} images)",
            f"Luna contribution: {luna_percentage:.2f}% ({len(luna_accepted)} images)",
            "",
            "DATASET3 FINAL COUNTS",
            f"Dataset3 train: {len(dataset3_train)}",
            f"Dataset3 valid: {len(vietnam_valid)}",
            f"Dataset3 test: {len(vietnam_test)}",
            "",
            "Dataset3 validation and test contain only the Vietnam Dataset2 target domain.",
            "",
            "LEAKAGE CHECK: PASS",
            "MODEL TRAINING: NOT PERFORMED",
            "",
            f"Manifest rows: {len(manifest_rows)}",
        ]
    )
    (output_root / "reports" / "dataset3_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def validate_vietnam_target_splits(vietnam: SourceInventory) -> None:
    problems = []
    for split in ("valid", "test"):
        scan = vietnam.scans[split]
        if scan.missing_labels or scan.orphan_labels or scan.invalid_labels:
            problems.append(
                f"{split}: missing={len(scan.missing_labels)}, orphan={len(scan.orphan_labels)}, invalid={len(scan.invalid_labels)}"
            )
    if problems:
        raise DatasetError(
            "Vietnam valid/test cannot be preserved because source pairs are not clean: " + "; ".join(problems)
        )


def make_contact_sheets(
    output_root: Path,
    vietnam_train: Sequence[ImageRecord],
    luna_accepted: Sequence[ImageRecord],
    dataset3_train: Sequence[ImageRecord],
    destination_by_source: Mapping[str, Path],
) -> None:
    entries_by_name = choose_contact_entries(
        vietnam_train,
        luna_accepted,
        dataset3_train,
        destination_by_source,
    )
    titles = {
        "A_vietnam_train": "A: 50 random Vietnam train",
        "B_luna_accepted_train": "B: 50 random Luna accepted train",
        "C_dataset3_train": "C: 50 random Dataset3 train",
        "D_smallest_without_helmet": "D: smallest rider_without_helmet boxes",
        "E_most_without_helmet": "E: most rider_without_helmet boxes",
    }
    for name, entries in entries_by_name.items():
        make_contact_sheet(
            entries,
            output_root / "reports" / "contact_sheets" / f"{name}.jpg",
            titles[name],
        )


def print_final_summary(
    output_root: Path,
    vietnam_train: Sequence[ImageRecord],
    vietnam_valid: Sequence[ImageRecord],
    vietnam_test: Sequence[ImageRecord],
    luna_original: Sequence[ImageRecord],
    duplicate_analysis: DuplicateAnalysis,
) -> None:
    dataset3_train = list(vietnam_train) + list(duplicate_analysis.accepted_luna)
    train_counts = counts_for_records(dataset3_train)
    valid_counts = counts_for_records(vietnam_valid)
    test_counts = counts_for_records(vietnam_test)
    vietnam_percentage = len(vietnam_train) / len(dataset3_train) * 100 if dataset3_train else 0.0
    luna_percentage = len(duplicate_analysis.accepted_luna) / len(dataset3_train) * 100 if dataset3_train else 0.0
    print("=" * 50)
    print("DATASET3 CREATED SUCCESSFULLY")
    print("=" * 50)
    print(f"Location: {output_root.resolve()}")
    print(f"Vietnam train accepted: {len(vietnam_train)}")
    print(f"Luna train original: {len(luna_original)}")
    print(f"Luna accepted: {len(duplicate_analysis.accepted_luna)}")
    print(f"Exact duplicates removed: {duplicate_analysis.exact_duplicates_removed}")
    print(f"Near duplicates removed: {duplicate_analysis.near_duplicates_removed}")
    print(f"Leakage removed: {duplicate_analysis.leakage_removed}")
    print(f"FINAL TRAIN: {len(dataset3_train)}")
    print(f"FINAL VALID: {len(vietnam_valid)}")
    print(f"FINAL TEST: {len(vietnam_test)}")
    print("TRAIN CLASS COUNTS:")
    print(f"motorcycle: {train_counts['motorcycle']}")
    print(f"rider_with_helmet: {train_counts['rider_with_helmet']}")
    print(f"rider_without_helmet: {train_counts['rider_without_helmet']}")
    print("VALID CLASS COUNTS:")
    print(f"motorcycle: {valid_counts['motorcycle']}")
    print(f"rider_with_helmet: {valid_counts['rider_with_helmet']}")
    print(f"rider_without_helmet: {valid_counts['rider_without_helmet']}")
    print("TEST CLASS COUNTS:")
    print(f"motorcycle: {test_counts['motorcycle']}")
    print(f"rider_with_helmet: {test_counts['rider_with_helmet']}")
    print(f"rider_without_helmet: {test_counts['rider_without_helmet']}")
    print(f"Vietnam contribution: {vietnam_percentage:.2f}%")
    print(f"Luna contribution: {luna_percentage:.2f}%")
    print("Validation/Test domain: VIETNAM ONLY")
    print("Leakage check: PASS")
    print("MODEL TRAINING: NOT PERFORMED")
    print("=" * 50)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if OUTPUT_ROOT.exists():
        LOGGER.error("Dataset3 already exists. Delete or rename it manually before rebuilding.")
        return 1
    vietnam_config = SourceConfig(
        name="Vietnam",
        directory_name=VIETNAM_DIR_NAME,
        workspace="nckh-2023",
        project="helmet-detection-project",
        version=None,
        prefix="vn",
    )
    luna_config = SourceConfig(
        name="Luna",
        directory_name=LUNA_DIR_NAME,
        workspace="luna-ns92w",
        project="helmet-detection-project-o4igi",
        version=5,
        prefix="luna",
    )
    try:
        vietnam_root = find_source_root(vietnam_config, PROJECT_ROOT)
        luna_root = find_source_root(luna_config, PROJECT_ROOT)
        source_roots = [vietnam_root, luna_root]
        LOGGER.info("Vietnam source root: %s", vietnam_root.root)
        LOGGER.info("Luna source root: %s", luna_root.root)
        source_state_before = source_file_state(source_roots)

        vietnam = scan_inventory(vietnam_root, "Vietnam")
        luna = scan_inventory(luna_root, "Luna")
        validate_vietnam_target_splits(vietnam)
        duplicate_analysis = analyse_duplicates(vietnam, luna.records("train"))

        dataset3_train = vietnam.records("train") + duplicate_analysis.accepted_luna
        ensure_output_layout(OUTPUT_ROOT)
        write_dataset_yaml(OUTPUT_ROOT)
        train_manifest, train_destinations = copy_records(dataset3_train, "train", OUTPUT_ROOT)
        valid_manifest, valid_destinations = copy_records(vietnam.records("valid"), "valid", OUTPUT_ROOT)
        test_manifest, test_destinations = copy_records(vietnam.records("test"), "test", OUTPUT_ROOT)
        manifest_rows = train_manifest + valid_manifest + test_manifest
        write_reports(OUTPUT_ROOT, vietnam, luna, duplicate_analysis, manifest_rows)
        all_destinations = {**train_destinations, **valid_destinations, **test_destinations}
        make_contact_sheets(
            OUTPUT_ROOT,
            vietnam.records("train"),
            duplicate_analysis.accepted_luna,
            dataset3_train,
            all_destinations,
        )
        validate_output(OUTPUT_ROOT, manifest_rows, vietnam, source_state_before, source_roots)
        write_summary(OUTPUT_ROOT, vietnam, luna, duplicate_analysis, source_roots, manifest_rows)
        print_final_summary(
            OUTPUT_ROOT,
            vietnam.records("train"),
            vietnam.records("valid"),
            vietnam.records("test"),
            luna.records("train"),
            duplicate_analysis,
        )
        return 0
    except DatasetError as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
