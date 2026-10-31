#!/usr/bin/env python3
"""Create a relative-path manifest for the committed dataset_v165 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_name(image_id: str, source_image: str) -> str:
    normalized_id = image_id.replace("\\", "/")
    name = normalized_id.replace("/", "__")
    if not Path(name).suffix:
        name += Path(source_image).suffix
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset_v165",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset_v165_manifest_portable.json",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    source_manifest = dataset / "manifest.json"
    records = json.loads(source_manifest.read_text(encoding="utf-8"))
    portable_records = []
    seen = set()

    for record in records:
        split = record.get("new_split")
        if split not in {"train", "val", "test", "pending"}:
            raise ValueError(f"Unexpected split for {record.get('id')}: {split}")

        image_name = portable_name(record["id"], record["image"])
        label_name = Path(image_name).stem + ".txt"
        image_rel = Path(split) / "images" / image_name
        label_rel = Path(split) / "labels" / label_name
        image_path = dataset / image_rel
        label_path = dataset / label_rel

        if image_rel.as_posix() in seen:
            raise ValueError(f"Duplicate image path: {image_rel}")
        seen.add(image_rel.as_posix())
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image/label pair: {image_rel}, {label_rel}")
        if record.get("sha256") and sha256(image_path) != record["sha256"]:
            raise ValueError(f"Image hash mismatch: {image_rel}")
        snapshot_label_sha256 = sha256(label_path)

        portable = dict(record)
        portable["image"] = image_rel.as_posix()
        portable["label"] = label_rel.as_posix()
        portable["source_label_sha256"] = record.get("label_sha256")
        portable["label_sha256"] = snapshot_label_sha256
        portable["snapshot_label_sha256"] = snapshot_label_sha256
        portable_records.append(portable)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(portable_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(portable_records)} records to {args.output}")
    print(f"Dataset root: {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
