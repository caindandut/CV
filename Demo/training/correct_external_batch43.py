"""Record a final box-boundary correction found during overlay QA."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_dataset import ROOT, digest, write_json


LEDGER = Path("acquisition_2000/external_reviews_batch43.json")
REVIEW = Path(
    "acquisition_2000/commons_original_resolution_review/manual_batch60/manual_batch60_review.json"
)
EXPECTED_OLD = [
    [0, 100, 430, 1900, 1910],
    [1, 925, 200, 1540, 1990],
    [1, 1320, 190, 1960, 1580],
]
CORRECTED = [
    [0, 100, 430, 1900, 1910],
    [1, 925, 200, 1540, 1990],
    [1, 1400, 190, 1850, 1580],
]


def main():
    ledger_path = ROOT / LEDGER
    review_path = ROOT / REVIEW
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if len(ledger) != 1 or ledger[0]["boxes"] != EXPECTED_OLD:
        raise ValueError("Review ledger changed; refusing an untracked correction")
    if review.get("after_boxes_xyxy") != EXPECTED_OLD:
        raise ValueError("Review log differs from the expected pre-correction boxes")

    crop_path = ROOT / review["crop_file"]
    with Image.open(crop_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colors = {0: "#22c55e", 1: "#3b82f6", 2: "#ef4444"}
    for class_id, x1, y1, x2, y2 in CORRECTED:
        draw.rectangle((x1, y1, x2, y2), outline=colors[class_id], width=6)
    evidence = Image.new("RGB", (canvas.width, canvas.height + 46), "white")
    evidence.paste(canvas, (0, 0))
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 22)
    legend = ImageDraw.Draw(evidence)
    legend.rectangle((12, canvas.height + 12, 32, canvas.height + 32), fill=colors[0])
    legend.text((42, canvas.height + 10), "0 = motorcycle", font=font, fill="#111827")
    legend.rectangle((270, canvas.height + 12, 290, canvas.height + 32), fill=colors[1])
    legend.text((300, canvas.height + 10), "1 = helmeted rider", font=font, fill="#111827")
    evidence_path = ROOT / review["evidence"]
    evidence.save(evidence_path)

    correction = {
        "before_boxes_xyxy": EXPECTED_OLD,
        "after_boxes_xyxy": CORRECTED,
        "reason": (
            "Overlay QA showed that the passenger box extended into the rear cargo bags. "
            "The right and left edges were tightened to cover the visible passenger, helmet, "
            "raised hand, and seated body while excluding the adjacent bags."
        ),
        "evidence_sha256_after": digest(evidence_path),
    }
    review["box_corrections"] = [correction]
    review["after_boxes_xyxy"] = CORRECTED
    review["evidence_sha256"] = digest(evidence_path)
    review["reason"] += " Final overlay QA tightened the passenger box to exclude adjacent cargo bags."
    ledger[0]["boxes"] = CORRECTED
    ledger[0]["evidence_sha256"] = digest(evidence_path)
    ledger[0]["reason"] = review["reason"]
    write_json(review_path, review)
    write_json(ledger_path, ledger)
    print(json.dumps({
        "pageid": review["pageid"],
        "before": EXPECTED_OLD[-1],
        "after": CORRECTED[-1],
        "evidence_sha256": digest(evidence_path),
    }))


if __name__ == "__main__":
    main()
