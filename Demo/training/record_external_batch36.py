"""Record a manually reviewed isolated parked motorcycle crop from Hanoi."""

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image


PAGE_ID = 58081349
SOURCE_REL = Path("acquisition_2000/images/commons_58081349.jpg")
REVIEW_DIR = Path("acquisition_2000/commons_original_resolution_review/manual_batch47")
REVIEW_FILE = REVIEW_DIR / "manual_batch47_review.json"
LEDGER_FILE = Path("acquisition_2000/external_reviews_batch36.json")
CROP = [675, 480, 1000, 730]
CANVAS = [325, 250]
BOXES = [[0, 20, 18, 294, 228]]


def main():
    if LEDGER_FILE.exists() or REVIEW_FILE.exists():
        raise FileExistsError("Batch 36 evidence already exists; refusing to overwrite")

    source_path = ROOT / SOURCE_REL
    sources = json.loads((ROOT / "acquisition_2000/sources.json").read_text(encoding="utf-8"))
    source = next(row for row in sources if row["pageid"] == PAGE_ID)
    meta = source["extmetadata"]

    def meta_value(name):
        return (meta.get(name) or {}).get("value", "")

    license_name = meta_value("LicenseShortName")
    artist = "Phan Minh Tuấn"
    gps = [float(meta_value("GPSLatitude")), float(meta_value("GPSLongitude"))]
    categories = meta_value("Categories")
    if not license_name.startswith("CC BY-SA"):
        raise ValueError(f"Unexpected source license: {license_name}")
    if not (8 <= gps[0] <= 24 and 102 <= gps[1] <= 110):
        raise ValueError(f"Coordinates do not establish Vietnam: {gps}")
    if digest(source_path) != source["download_sha256"]:
        raise ValueError("Cached Commons source bytes changed")

    decision = {"pageid": PAGE_ID, "crop": CROP, "canvas": CANVAS, "boxes": BOXES}
    crop_path, lineage = prepare_review_image(
        source_path, decision, ROOT / "acquisition_2000/curated_crops", PAGE_ID
    )
    REVIEW_DIR_ABS = ROOT / REVIEW_DIR
    REVIEW_DIR_ABS.mkdir(parents=True, exist_ok=True)
    overlay = Image.open(crop_path).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 18)
    cls, x1, y1, x2, y2 = BOXES[0]
    color = "#22c55e"
    draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
    label = "class 0 motorcycle"
    bbox = draw.textbbox((0, 0), label, font=font)
    label_w, label_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    label_y = max(0, y1 - label_h - 4)
    draw.rectangle((x1, label_y, x1 + label_w + 8, label_y + label_h + 4), fill=color)
    draw.text((x1 + 4, label_y + 2), label, font=font, fill="white")

    overlay_rel = REVIEW_DIR / "58081349_overlay.png"
    overlay.save(ROOT / overlay_rel)
    reason = (
        "Manual review of the full Hanoi street source and the lossless crop. The crop isolates "
        "one clearly visible red motorcycle parked at the curb; both wheels, body, handlebar, "
        "and mirror are visible. No rider or other motorcycle remains inside the crop, so the "
        "only annotation is class 0. Distant traffic and people at the bus stop are outside the "
        "crop. The source thumbnail pixels were not resized or edited; no model prediction was used."
    )
    location = (
        f"Commons file {source['descriptionurl']}; metadata describes Yên Hòa Ward, urban Hà Nội, "
        f"Việt Nam; GPS {gps[0]:.6f},{gps[1]:.6f}; categories: {categories}."
    )
    review = {
        "batch": "manual_batch47",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pageid": PAGE_ID,
        "source_file": SOURCE_REL.as_posix(),
        "source_sha256": digest(source_path),
        "source_url": source["url"],
        "source_page_url": source["descriptionurl"],
        "source_width_height_original": [source["width"], source["height"]],
        "reviewed_source_canvas": [source["thumbwidth"], source["thumbheight"]],
        "source_variant": "Commons 1280px thumbnail, unchanged bytes before lossless crop",
        "crop_xyxy": CROP,
        "crop_canvas": CANVAS,
        "country": "VN",
        "location_evidence": location,
        "artist": artist,
        "license": license_name,
        "before": "No usable object annotations were supplied for this image.",
        "after_boxes_xyxy": BOXES,
        "class_map": {"0": "motorcycle", "1": "helmeted rider", "2": "rider without helmet"},
        "review_method": "manual visual review of complete source frame, crop, and overlay",
        "decision": "accepted",
        "reason": reason,
        "crop_file": crop_path.relative_to(ROOT).as_posix(),
        "crop_sha256": digest(crop_path),
        "evidence": overlay_rel.as_posix(),
        "evidence_sha256": digest(ROOT / overlay_rel),
        "lineage": lineage,
    }
    write_json(ROOT / REVIEW_FILE, review)
    ledger_entry = {
        "pageid": PAGE_ID,
        "crop": CROP,
        "canvas": CANVAS,
        "boxes": BOXES,
        "country": "VN",
        "location_evidence": location,
        "visual_reviewed": True,
        "image_sha256": digest(source_path),
        "evidence": overlay_rel.as_posix(),
        "evidence_sha256": digest(ROOT / overlay_rel),
        "review_log": REVIEW_FILE.as_posix(),
        "source_group": artist,
        "reason": reason,
    }
    write_json(ROOT / LEDGER_FILE, [ledger_entry])
    print(json.dumps({"accepted": 1, "pageid": PAGE_ID, "crop": CROP, "canvas": CANVAS, "source_sha256": digest(source_path)}))


if __name__ == "__main__":
    main()
