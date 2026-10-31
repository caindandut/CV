"""Record manually resolved Commons annotations with immutable original provenance."""

import argparse
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from commons_sources import resolve_commons_source
from curate_v2 import normalize_boxes, plain
from sprint_proposals import ROOT, ACQ, save_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    spec = json.loads(args.decisions.read_text(encoding="utf-8"))
    number = spec["ledger_batch"]
    if type(number) is not int or not 44 <= number <= 999:
        raise ValueError("Invalid new Commons ledger batch")
    ledger = ACQ / f"external_reviews_batch{number}.json"
    if ledger.exists():
        raise FileExistsError(ledger)
    proposal_dir = (ACQ / "sprint_5000_5h" / spec["proposal_batch"]).resolve()
    if not proposal_dir.is_relative_to((ACQ / "sprint_5000_5h").resolve()):
        raise ValueError("Proposal directory escapes acquisition")
    proposals = {r["pageid"]: r for r in json.loads((proposal_dir / "proposals.json").read_text(encoding="utf-8"))}
    sources = {r["pageid"]: r for r in json.loads((ACQ / "sources.json").read_text(encoding="utf-8"))}
    used = {r["pageid"] for p in ACQ.glob("external_reviews_batch*.json") for r in json.loads(p.read_text(encoding="utf-8"))}
    folder = ACQ / "commons_original_resolution_review/sprint_20261030" / f"review_batch{number}"
    folder.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 15)
    entries = []
    for decision in spec["reviewed"]:
        pid = decision["pageid"]
        if pid in used:
            raise ValueError("Source photograph already has an accepted Commons review")
        used.add(pid)
        proposal, source = proposals[pid], sources[pid]
        meta = source["extmetadata"]
        license_name = plain(meta.get("LicenseShortName", {}).get("value", ""))
        artist = plain(meta.get("Artist", {}).get("value", ""))
        if not artist or not license_name.startswith(("CC BY", "CC0", "Public domain")):
            raise ValueError("Missing supported license or attribution")
        entry = {"pageid": pid, "source_variant": proposal.get("source_variant", "thumbnail"), "image_sha256": proposal["source_sha256"], "country": "VN", "location_evidence": decision["location_evidence"], "source_group": artist}
        for key in ("source_file", "original_acquisition_file", "original_acquisition_sha256"):
            if key in proposal:
                entry[key] = proposal[key]
        source_path, _ = resolve_commons_source(source, entry, ROOT)
        full = Image.open(source_path).convert("RGB")
        rect = decision.get("crop", proposal["crop"])
        if not (0 <= rect[0] < rect[2] <= full.width and 0 <= rect[1] < rect[3] <= full.height):
            raise ValueError("Invalid native crop bounds")
        crop = full.crop(rect)
        if min(crop.size) < 240:
            raise ValueError("Native crop is smaller than 240 pixels")
        normalize_boxes(decision["boxes"], crop.size)
        overlay = crop.copy()
        draw = ImageDraw.Draw(overlay)
        for cls, x1, y1, x2, y2 in decision["boxes"]:
            color = ["#1689ff", "#18b84b", "#fa3030"][cls]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1, max(0, y1 - 19)), str(cls), font=font, fill=color)
        crop_path = folder / f"{pid}_crop.png"
        overlay_path = folder / f"{pid}_manual_overlay.png"
        head_path = folder / f"{pid}_head_native.png"
        crop.save(crop_path)
        overlay.save(overlay_path)
        crop.crop(decision["head_detail"]).save(head_path)
        rel = lambda p: p.relative_to(ROOT).as_posix()
        entry.update(crop=rect, canvas=list(crop.size), boxes=decision["boxes"], visual_reviewed=args.commit, evidence=rel(overlay_path), evidence_sha256=sha(overlay_path), head_detail=rel(head_path), head_detail_sha256=sha(head_path), review_log=rel(folder / "manual_review.json"), reason=decision["reason"], before_model_proposals_NOT_GT=proposal["proposal_boxes_NOT_GT"], after_manual_boxes_xyxy=decision["boxes"], model_assisted_review=True, model_predictions_used_as_ground_truth=False, annotator="Codex native-pixel visual review", proposal_run=rel(proposal_dir / "run.json"), reviewed_at_utc=datetime.now(UTC).isoformat())
        entry.update(before_proposal_crop=proposal["crop"], before_proposal_canvas=proposal["canvas"], after_manual_crop=rect)
        entry.update(geometry_origin=decision.get("geometry_origin", "manual_pixel_measurement"), before_coco_geometry_NOT_GT=decision.get("coco_geometry_NOT_GT"))
        if args.commit and decision.get("overlay_checked") is not True:
            raise ValueError("Explicit corrected-overlay review required")
        entries.append(entry)
    log = {"accepted_count": len(entries) if args.commit else 0, "preview_count": len(entries), "reviewed": entries, "pending": spec.get("pending", []), "annotations_source": "Per-image visual decisions; model proposals are retained only as before-history"}
    if args.commit:
        save_json(folder / "manual_review.json", log)
        save_json(ledger, entries)
    else:
        save_json(folder / "preview_review.json", log)
    print(json.dumps({"committed": args.commit, "entries": len(entries), "evidence": str(folder), "ledger": str(ledger) if args.commit else None}))


if __name__ == "__main__":
    main()
