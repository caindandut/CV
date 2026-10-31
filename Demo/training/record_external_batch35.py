"""Record one manually reviewed Commons image from Hanoi, without changing its source."""

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_dataset import ROOT, digest, write_json


PAGE_ID = 58090598
SOURCE_REL = Path("acquisition_2000/images/commons_58090598.jpg")
REVIEW_DIR = Path("acquisition_2000/commons_original_resolution_review/manual_batch46")
REVIEW_FILE = REVIEW_DIR / "manual_batch46_review.json"
LEDGER_FILE = Path("acquisition_2000/external_reviews_batch35.json")
BOXES = [
    [0, 125, 465, 684, 905],  # motorcycle, including mirror and both wheels
    [1, 346, 350, 535, 905],  # seated rider wearing the visible blue helmet
]
CANVAS = [1280, 960]


def main():
    if LEDGER_FILE.exists() or REVIEW_FILE.exists():
        raise FileExistsError("Batch 35 evidence already exists; refusing to overwrite")

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

    REVIEW_DIR_ABS = ROOT / REVIEW_DIR
    REVIEW_DIR_ABS.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as raw:
        if list(raw.size) != CANVAS or raw.getexif().get(274, 1) != 1:
            raise ValueError(f"Unexpected source canvas/orientation: {raw.size}")
        image = raw.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 22)
    colors = {0: "#22c55e", 1: "#3b82f6", 2: "#ef4444"}
    names = {0: "class 0 motorcycle", 1: "class 1 helmeted rider", 2: "class 2 no helmet"}
    for cls, x1, y1, x2, y2 in BOXES:
        color = colors[cls]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        label = names[cls]
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w, label_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - label_h - 10)
        draw.rectangle((x1, label_y, x1 + label_w + 12, label_y + label_h + 8), fill=color)
        draw.text((x1 + 6, label_y + 2), label, font=font, fill="white")

    overlay_rel = REVIEW_DIR / "58090598_overlay.png"
    head_rel = REVIEW_DIR / "58090598_head_detail.png"
    image.save(ROOT / overlay_rel)
    with Image.open(source_path) as raw:
        raw.convert("RGB").crop((320, 310, 570, 500)).save(ROOT / head_rel)

    reason = (
        "Manual review of the complete cached Commons frame (1280x960). One clearly visible "
        "red motorcycle is stopped at the roadside and one person is seated astride it. The "
        "rider wears a blue motorcycle helmet. The motorcycle box covers the visible mirrors, "
        "body, and both wheels; the rider box covers the helmet, body, and planted foot. "
        "People sitting inside the shop are pedestrians/vendors, not riders, and no other "
        "motorcycle or rider is visible in the frame. Labels were drawn from the pixels; no "
        "model prediction was used. The local file is the unchanged 1280px Commons thumbnail."
    )
    location = (
        f"Commons file {source['descriptionurl']}; metadata describes Hà Nội, Việt Nam; "
        f"GPS {gps[0]:.6f},{gps[1]:.6f}; categories: {categories}. The Vietnamese street "
        "sign in the image reads Phố Nguyễn Thiện Thuật."
    )
    review = {
        "batch": "manual_batch46",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pageid": PAGE_ID,
        "source_file": SOURCE_REL.as_posix(),
        "source_sha256": digest(source_path),
        "source_url": source["url"],
        "source_page_url": source["descriptionurl"],
        "source_width_height_original": [source["width"], source["height"]],
        "reviewed_canvas": CANVAS,
        "source_variant": "Commons 1280px thumbnail, unchanged bytes",
        "country": "VN",
        "location_evidence": location,
        "artist": artist,
        "license": license_name,
        "before": "No usable object annotations were supplied for this image.",
        "after_boxes_xyxy": BOXES,
        "class_map": {"0": "motorcycle", "1": "helmeted rider", "2": "rider without helmet"},
        "review_method": "manual visual review of full frame and head detail",
        "decision": "accepted",
        "reason": reason,
        "evidence": overlay_rel.as_posix(),
        "evidence_sha256": digest(ROOT / overlay_rel),
        "head_detail": head_rel.as_posix(),
        "head_detail_sha256": digest(ROOT / head_rel),
    }
    write_json(ROOT / REVIEW_FILE, review)
    ledger_entry = {
        "pageid": PAGE_ID,
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
    print(json.dumps({"accepted": 1, "pageid": PAGE_ID, "ledger": LEDGER_FILE.as_posix(), "source_sha256": digest(source_path)}))


if __name__ == "__main__":
    main()
