"""Acquire and annotate a Commons candidate, refusing duplicate source pages."""

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image


PAGE_ID = 113561565
ACQ_REL = Path("acquisition_2000/commons_original_resolution_review/manual_batch60")
SOURCE_REL = ACQ_REL / "source_original.jpg"
RECEIPT_REL = ACQ_REL / "source_acquisition.json"
REVIEW_REL = ACQ_REL / "manual_batch60_review.json"
OVERLAY_REL = ACQ_REL / "113561565_crop_overlay.png"
HEAD_REL = ACQ_REL / "113561565_head_detail.png"
LEDGER_REL = Path("acquisition_2000/external_reviews_batch43.json")
CROP = [2700, 1200, 4900, 3200]
CANVAS = [2200, 2000]
# XYXY in crop coordinates: motorcycle and each of the two seated helmeted riders.
BOXES = [
    [0, 100, 430, 1900, 1910],
    [1, 925, 200, 1540, 1990],
    [1, 1400, 190, 1850, 1580],
]
ARTIST = "Vinnie Cartabiano"


def main():
    progress_path = ROOT / "acquisition_2000/progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        frozen_name = progress.get("frozen_dataset")
        if frozen_name:
            manifest_path = ROOT / frozen_name / "manifest.json"
            if manifest_path.is_file():
                frozen_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
                already_used = any(
                    row.get("original_pageid") == PAGE_ID
                    or row.get("id", "").endswith(f"_{PAGE_ID}.jpg")
                    or row.get("id", "").endswith(f"_{PAGE_ID}.png")
                    for row in frozen_rows
                )
                if already_used:
                    raise ValueError(
                        f"Commons page {PAGE_ID} is already present in {frozen_name}; "
                        "do not admit another crop from this source photo"
                    )
    for rel in (SOURCE_REL, RECEIPT_REL, REVIEW_REL, OVERLAY_REL, HEAD_REL, LEDGER_REL):
        if (ROOT / rel).exists():
            raise FileExistsError(f"Refusing to overwrite existing evidence: {rel}")

    sources_path = ROOT / "acquisition_2000/sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    source = next(row for row in sources if row["pageid"] == PAGE_ID)
    metadata = source["extmetadata"]

    def meta_value(name):
        return (metadata.get(name) or {}).get("value", "")

    license_name = meta_value("LicenseShortName")
    license_url = meta_value("LicenseUrl")
    if license_name != "CC BY 2.0" or license_url != "https://creativecommons.org/licenses/by/2.0":
        raise ValueError(f"Unexpected source license: {license_name} / {license_url}")
    if "Motorcycle in Vietnam" not in source["title"]:
        raise ValueError("Commons title no longer identifies this as a Vietnam motorcycle image")
    if "Motorcycles in Vietnam" not in meta_value("Categories"):
        raise ValueError("Commons categories no longer verify Vietnam context")
    if ARTIST not in meta_value("Artist"):
        raise ValueError(f"Unexpected source artist: {meta_value('Artist')!r}")

    source_path = ROOT / SOURCE_REL
    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "VietnamHelmetDatasetCuration/1.0"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise RuntimeError(f"Commons original returned HTTP {response.status}")
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get_content_type()
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(data)
    with Image.open(source_path) as original:
        original.load()
        canvas = list(original.size)
        if canvas != [source["width"], source["height"]]:
            raise ValueError(f"Original canvas mismatch: {canvas}")
        if content_type != "image/jpeg":
            raise ValueError(f"Unexpected content type: {content_type}")

    receipt = {
        "pageid": PAGE_ID,
        "requested_url": source["url"],
        "final_url": final_url,
        "status": "downloaded_candidate_only",
        "http_status": 200,
        "content_type": content_type,
        "canvas": canvas,
        "bytes": len(data),
        "sha256": digest(source_path),
        "file": SOURCE_REL.name,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = ROOT / RECEIPT_REL
    write_json(receipt_path, receipt)

    crop_path, lineage = prepare_review_image(
        source_path, {"crop": CROP, "canvas": CANVAS},
        ROOT / "acquisition_2000/curated_crops", PAGE_ID,
    )
    with Image.open(crop_path) as image:
        if list(image.size) != CANVAS:
            raise ValueError(f"Unexpected crop canvas: {image.size}")
        overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 22)
    colors = {0: "#22c55e", 1: "#3b82f6", 2: "#ef4444"}
    for class_id, x1, y1, x2, y2 in BOXES:
        draw.rectangle((x1, y1, x2, y2), outline=colors[class_id], width=6)
    evidence = Image.new("RGB", (CANVAS[0], CANVAS[1] + 46), "white")
    evidence.paste(overlay, (0, 0))
    legend = ImageDraw.Draw(evidence)
    legend.rectangle((12, CANVAS[1] + 12, 32, CANVAS[1] + 32), fill=colors[0])
    legend.text((42, CANVAS[1] + 10), "0 = motorcycle", font=font, fill="#111827")
    legend.rectangle((270, CANVAS[1] + 12, 290, CANVAS[1] + 32), fill=colors[1])
    legend.text((300, CANVAS[1] + 10), "1 = helmeted rider", font=font, fill="#111827")
    overlay_path = ROOT / OVERLAY_REL
    evidence.save(overlay_path)

    with Image.open(source_path) as original:
        original.crop((3600, 1320, 4650, 1770)).save(ROOT / HEAD_REL)

    location = (
        f"Commons source page {source['descriptionurl']} names the file 'Motorcycle in Vietnam' "
        "and categorizes it under Motorcycles in Vietnam. The original shows a motorcycle with "
        "two riders on a coastal Vietnamese road; no specific city is claimed."
    )
    reason = (
        "Manually inspected the full 6000x3375 original, the lossless 2200x2000 crop, the "
        "helmet-detail crop, and the final overlay. The crop contains one clearly visible "
        "motorcycle and two people seated on it; both wear motorcycle helmets. Each rider has "
        "an individual class-1 box, and the motorcycle has one class-0 box covering its visible "
        "body and both wheels. No other motorcycle or rider is inside the crop. The original "
        "pixels were used without model-generated labels or upscaling; the untouched downloaded "
        "original and its acquisition receipt are preserved."
    )
    review = {
        "batch": "manual_batch60",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pageid": PAGE_ID,
        "source_file": SOURCE_REL.as_posix(),
        "source_sha256": digest(source_path),
        "source_url": source["url"],
        "source_page_url": source["descriptionurl"],
        "source_width_height_original": canvas,
        "source_variant": "Wikimedia Commons original-resolution JPEG; original bytes preserved",
        "original_acquisition_file": RECEIPT_REL.as_posix(),
        "original_acquisition_sha256": digest(receipt_path),
        "crop_xyxy": CROP,
        "crop_canvas": CANVAS,
        "country": "VN",
        "location_evidence": location,
        "photographer": ARTIST,
        "commons_artist_metadata": meta_value("Artist"),
        "license": license_name,
        "license_url": license_url,
        "before": "No verified object annotations were supplied for this Commons photograph.",
        "after_boxes_xyxy": BOXES,
        "class_map": {"0": "motorcycle", "1": "helmeted rider", "2": "rider without helmet"},
        "review_method": "manual visual review of full original, crop, head detail, and overlay",
        "decision": "accepted",
        "reason": reason,
        "crop_file": crop_path.relative_to(ROOT).as_posix(),
        "crop_sha256": digest(crop_path),
        "evidence": OVERLAY_REL.as_posix(),
        "evidence_sha256": digest(overlay_path),
        "head_detail": HEAD_REL.as_posix(),
        "head_detail_sha256": digest(ROOT / HEAD_REL),
        "lineage": lineage,
    }
    write_json(ROOT / REVIEW_REL, review)

    ledger = [{
        "pageid": PAGE_ID,
        "source_variant": "original",
        "source_file": SOURCE_REL.as_posix().replace("acquisition_2000/", "", 1),
        "original_acquisition_file": RECEIPT_REL.as_posix().replace("acquisition_2000/", "", 1),
        "original_acquisition_sha256": digest(receipt_path),
        "crop": CROP,
        "canvas": CANVAS,
        "boxes": BOXES,
        "country": "VN",
        "location_evidence": location,
        "visual_reviewed": True,
        "image_sha256": digest(source_path),
        "evidence": OVERLAY_REL.as_posix(),
        "evidence_sha256": digest(overlay_path),
        "review_log": REVIEW_REL.as_posix(),
        "source_group": ARTIST,
        "artist_attribution": ARTIST,
        "reason": reason,
    }]
    write_json(ROOT / LEDGER_REL, ledger)
    print(json.dumps({
        "accepted": 1,
        "pageid": PAGE_ID,
        "license": license_name,
        "source_sha256": digest(source_path),
        "crop": CROP,
        "canvas": CANVAS,
        "boxes": len(BOXES),
        "review_log": REVIEW_REL.as_posix(),
        "ledger": LEDGER_REL.as_posix(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
