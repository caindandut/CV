"""Record one manually reviewed original-resolution Commons photo from HCMC."""

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from commons_sources import resolve_commons_source
from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image


PAGE_ID = 530296
ACQ_REL = Path("acquisition_2000/commons_original_resolution_review/manual_batch59")
SOURCE_REL = ACQ_REL / "source_original.jpg"
RECEIPT_REL = ACQ_REL / "source_acquisition.json"
REVIEW_FILE = ACQ_REL / "manual_batch59_review.json"
OVERLAY_REL = ACQ_REL / "530296_crop_overlay.png"
HEAD_REL = ACQ_REL / "530296_head_detail.png"
LEDGER_FILE = Path("acquisition_2000/external_reviews_batch42.json")
SOURCE_FILE_IN_ACQ = "commons_original_resolution_review/manual_batch59/source_original.jpg"
RECEIPT_FILE_IN_ACQ = "commons_original_resolution_review/manual_batch59/source_acquisition.json"
CROP = [710, 210, 1270, 900]
CANVAS = [560, 690]
# XYXY coordinates relative to the crop: one motorcycle and all three riders.
BOXES = [
    [0, 0, 252, 503, 645],
    [2, 0, 65, 192, 580],
    [2, 120, 72, 297, 590],
    [2, 200, 165, 329, 478],
]
ARTIST_ATTRIBUTION = "Roderick Divilbiss (named as photographer in Commons description; Commons user Rdivilbiss)"


def main():
    if (ROOT / LEDGER_FILE).exists() or (ROOT / REVIEW_FILE).exists():
        raise FileExistsError("Batch 42 evidence already exists; refusing to overwrite")

    sources = json.loads(
        (ROOT / "acquisition_2000/sources.json").read_text(encoding="utf-8")
    )
    source = next(row for row in sources if row["pageid"] == PAGE_ID)
    meta = source["extmetadata"]

    def meta_value(name):
        return (meta.get(name) or {}).get("value", "")

    license_name = meta_value("LicenseShortName")
    license_url = meta_value("LicenseUrl")
    if license_name != "CC BY-SA 2.5" or not license_url.endswith("by-sa/2.5"):
        raise ValueError(f"Unexpected source license: {license_name} / {license_url}")
    if "Ho Chi Minh City, Vietnam" not in meta_value("ImageDescription"):
        raise ValueError("Commons description no longer verifies the Vietnam location")

    source_path = ROOT / SOURCE_REL
    receipt_path = ROOT / RECEIPT_REL
    decision = {
        "pageid": PAGE_ID,
        "source_variant": "original",
        "source_file": SOURCE_FILE_IN_ACQ,
        "original_acquisition_file": RECEIPT_FILE_IN_ACQ,
        "original_acquisition_sha256": digest(receipt_path),
        "image_sha256": digest(source_path),
        "crop": CROP,
        "canvas": CANVAS,
        "boxes": BOXES,
    }
    resolved, _ = resolve_commons_source(source, decision, ROOT)
    if resolved != source_path.resolve():
        raise ValueError("Resolved source differs from the manually reviewed original")

    crop_path, lineage = prepare_review_image(
        source_path, decision, ROOT / "acquisition_2000/curated_crops", PAGE_ID
    )
    review_dir = ROOT / ACQ_REL
    with Image.open(crop_path) as image:
        if list(image.size) != CANVAS:
            raise ValueError(f"Unexpected crop canvas: {image.size}")
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 15)
    colors = {0: "#22c55e", 1: "#3b82f6", 2: "#ef4444"}
    for cls, x1, y1, x2, y2 in BOXES:
        draw.rectangle((x1, y1, x2, y2), outline=colors[cls], width=4)
    evidence = Image.new("RGB", (CANVAS[0], CANVAS[1] + 34), "white")
    evidence.paste(overlay, (0, 0))
    legend = ImageDraw.Draw(evidence)
    legend.rectangle((8, CANVAS[1] + 9, 23, CANVAS[1] + 24), fill=colors[0])
    legend.text((29, CANVAS[1] + 7), "0 = motorcycle", font=font, fill="#111827")
    legend.rectangle((178, CANVAS[1] + 9, 193, CANVAS[1] + 24), fill=colors[2])
    legend.text((199, CANVAS[1] + 7), "2 = rider without helmet", font=font, fill="#111827")
    evidence.save(ROOT / OVERLAY_REL)

    with Image.open(source_path) as raw:
        raw.convert("RGB").crop((710, 205, 1060, 435)).save(ROOT / HEAD_REL)

    location = (
        f"Commons source page {source['descriptionurl']}; its description states Ho Chi Minh City, "
        "Vietnam, and June 2005; the file is categorized under Motorcycles in Ho Chi Minh City."
    )
    reason = (
        "Manually reviewed the full 1381x1013 original, the lossless 560x690 crop, a head-detail "
        "crop, and the annotation overlay. The crop retains one clearly visible motorcycle and "
        "three people seated on it: the driver wears a baseball cap, the rear passenger wears a "
        "fabric sun hat, and the child wears a cloth cap; none is wearing a motorcycle helmet. "
        "Each rider has an individual class-2 box. The motorcycle box covers its body and both "
        "wheels. The crop excludes the adjacent, partly occluded motorcycle and surrounding "
        "traffic visible in the full photo, avoiding uncertain labels. The background car and "
        "pedestrians are not target classes. Boxes were drawn from pixels; no model predictions "
        "were used. The original bytes are preserved and the crop is lossless PNG."
    )
    review = {
        "batch": "manual_batch59",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pageid": PAGE_ID,
        "source_file": SOURCE_REL.as_posix(),
        "source_sha256": digest(source_path),
        "source_url": source["url"],
        "source_page_url": source["descriptionurl"],
        "source_width_height_original": [source["width"], source["height"]],
        "source_variant": "Wikimedia Commons original-resolution JPEG; original bytes preserved",
        "original_acquisition_file": RECEIPT_REL.as_posix(),
        "original_acquisition_sha256": digest(receipt_path),
        "crop_xyxy": CROP,
        "crop_canvas": CANVAS,
        "country": "VN",
        "location_evidence": location,
        "photographer_named_in_description": "Roderick Divilbiss",
        "commons_artist_metadata": meta_value("Artist"),
        "artist_attribution": ARTIST_ATTRIBUTION,
        "license": license_name,
        "license_url": license_url,
        "before": "No usable object annotations were supplied for this Commons photograph.",
        "after_boxes_xyxy": BOXES,
        "class_map": {"0": "motorcycle", "1": "helmeted rider", "2": "rider without helmet"},
        "review_method": "manual visual review of full original, crop, head detail, and overlay",
        "decision": "accepted",
        "reason": reason,
        "crop_file": crop_path.relative_to(ROOT).as_posix(),
        "crop_sha256": digest(crop_path),
        "evidence": OVERLAY_REL.as_posix(),
        "evidence_sha256": digest(ROOT / OVERLAY_REL),
        "head_detail": HEAD_REL.as_posix(),
        "head_detail_sha256": digest(ROOT / HEAD_REL),
        "lineage": lineage,
    }
    write_json(ROOT / REVIEW_FILE, review)
    ledger = [{
        "pageid": PAGE_ID,
        "source_variant": "original",
        "source_file": SOURCE_FILE_IN_ACQ,
        "original_acquisition_file": RECEIPT_FILE_IN_ACQ,
        "original_acquisition_sha256": digest(receipt_path),
        "crop": CROP,
        "canvas": CANVAS,
        "boxes": BOXES,
        "country": "VN",
        "location_evidence": location,
        "visual_reviewed": True,
        "image_sha256": digest(source_path),
        "evidence": OVERLAY_REL.as_posix(),
        "evidence_sha256": digest(ROOT / OVERLAY_REL),
        "review_log": REVIEW_FILE.as_posix(),
        "source_group": "Roderick Divilbiss",
        "artist_attribution": ARTIST_ATTRIBUTION,
        "reason": reason,
    }]
    write_json(ROOT / LEDGER_FILE, ledger)
    print(json.dumps({
        "accepted": 1,
        "pageid": PAGE_ID,
        "crop": CROP,
        "canvas": CANVAS,
        "boxes": len(BOXES),
        "source_sha256": digest(source_path),
        "review_log": REVIEW_FILE.as_posix(),
        "ledger": LEDGER_FILE.as_posix(),
    }))


if __name__ == "__main__":
    main()
