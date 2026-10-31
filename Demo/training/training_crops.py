"""Deterministic, lossless training crops with original-image lineage."""

import copy
from pathlib import Path

from curate_v2 import normalize_boxes
from PIL import Image, ImageChops
from prepare_dataset import digest, perceptual_hash

# Native 240+ px crops can exclude neighboring partial objects without resizing
# or removing any pixels of the reviewed main riders. Manual class QA still applies.
MIN_REVIEW_EDGE = 240


def prepare_review_image(
    source, decision, destination, pageid, *, original=False, provider="commons"
):
    if provider not in {"commons", "ncpedia", "kaggle", "zenodo"}:
        raise ValueError("Unknown reviewed-image provider")
    source = Path(source)
    with Image.open(source) as raw:
        if raw.getexif().get(274, 1) != 1:
            raise ValueError("Normalize and review EXIF orientation explicitly first")
        lineage = {
            "source_image": str(source.resolve()),
            "source_sha256": digest(source),
            "source_size": list(raw.size),
        }
        rect = decision.get("crop")
        if rect is None:
            if list(raw.size) != decision["canvas"]:
                raise ValueError("Review canvas differs from source dimensions")
            return source, lineage
        if (
            not isinstance(rect, list)
            or len(rect) != 4
            or any(type(value) is not int for value in rect)
        ):
            raise ValueError("Crop needs four integer source coordinates")
        left, top, right, bottom = rect
        if not (0 <= left < right <= raw.width and 0 <= top < bottom <= raw.height):
            raise ValueError("Crop lies outside source image")
        size = [right - left, bottom - top]
        if size != decision["canvas"] or min(size) < MIN_REVIEW_EDGE:
            raise ValueError(
                f"Crop canvas must match and have both dimensions at least {MIN_REVIEW_EDGE}"
            )
        cropped = raw.convert("RGB").crop(rect)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        prefix = "original" if original else provider
        result = destination / f"{prefix}_{int(pageid)}.png"
        if result.exists():
            with Image.open(result) as existing:
                if (
                    existing.size != cropped.size
                    or ImageChops.difference(existing.convert("RGB"), cropped).getbbox()
                ):
                    raise ValueError(
                        "Existing crop differs; do not replace reviewed pixels silently"
                    )
        else:
            cropped.save(result, format="PNG")
        lineage["crop"] = rect
        return result, lineage


def prepare_original_crop(row, decision, destination):
    """Prepare pixels and clipped OLD boxes for comparison, never new ground truth."""
    path, lineage = prepare_review_image(
        row["image"], decision, destination, decision["proposal"], original=True
    )
    result = copy.deepcopy(row)
    result.update(
        lineage,
        source_id=row["id"],
        source_labels=copy.deepcopy(row["labels"]),
        id=f"original_crops/{Path(row['id']).stem}.png",
        image=str(path.resolve()),
        sha256=digest(path),
    )
    left, top, right, bottom = decision["crop"]
    clipped = []
    for cls, x, y, w, h in row["labels"]:
        x1 = max(left, (x - w / 2) * row["width"])
        y1 = max(top, (y - h / 2) * row["height"])
        x2 = min(right, (x + w / 2) * row["width"])
        y2 = min(bottom, (y + h / 2) * row["height"])
        if x1 < x2 and y1 < y2:
            clipped.append([cls, x1 - left, y1 - top, x2 - left, y2 - top])
    result.update(
        width=right - left,
        height=bottom - top,
        labels=normalize_boxes(clipped, decision["canvas"]),
    )
    with Image.open(path) as image:
        result["phash"] = f"{perceptual_hash(image):016x}"
    return result
