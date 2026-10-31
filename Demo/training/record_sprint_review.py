"""Export an explicitly reviewed VNTS batch, retaining model hints as non-GT history."""

import argparse
from datetime import datetime, UTC
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ACQ = ROOT / "acquisition_2000"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    spec = json.loads(args.decisions.read_text(encoding="utf-8"))
    number = spec["ledger_batch"]
    if type(number) is not int or not 54 <= number <= 999:
        raise ValueError("Invalid new ledger batch")
    ledger = ACQ / f"kaggle_reviews_batch{number}.json"
    if ledger.exists():
        raise FileExistsError(ledger)
    proposal_dir = (ACQ / "sprint_5000_5h" / spec["proposal_batch"]).resolve()
    if not proposal_dir.is_relative_to((ACQ / "sprint_5000_5h").resolve()):
        raise ValueError("Proposal directory escapes workspace")
    proposals = {r["image_id"]: r for r in json.loads((proposal_dir / "proposals.json").read_text(encoding="utf-8"))}
    evidence = ACQ / "research_kaggle/maitam_v8_sprint_review" / f"batch{number}"
    evidence.mkdir(parents=True, exist_ok=True)
    metadata_rel = "research_kaggle/maitam_vietnamese_traffic_signs_metadata01.json"
    archive_rel = "research_kaggle/maitam_vnts_v8.zip"
    metadata = json.loads((ACQ / metadata_rel).read_text(encoding="utf-8"))
    run_info = json.loads((proposal_dir / "run.json").read_text(encoding="utf-8"))
    archive_hash = digest(ACQ / archive_rel)
    if archive_hash != run_info["archive_sha256"] or metadata["licenseName"] != "CC BY-SA 4.0":
        raise ValueError("Source archive/license mismatch")
    previous = []
    for p in ACQ.glob("kaggle_reviews_batch*.json"):
        previous.extend(json.loads(p.read_text(encoding="utf-8")))
    used = {r["source_sha256"] for r in previous}
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 12)
    entries = []
    with zipfile.ZipFile(ACQ / archive_rel) as archive:
        for decision in spec["reviewed"]:
            proposal = proposals[decision["image_id"]]
            raw = archive.read(proposal["archive_entry"])
            source_hash = hashlib.sha256(raw).hexdigest()
            if source_hash != proposal["source_sha256"] or source_hash in used:
                raise ValueError("Source differs from proposal or was already accepted")
            used.add(source_hash)
            source_path = evidence / (decision["image_id"] + "_source.jpg")
            source_path.write_bytes(raw)
            full = Image.open(io.BytesIO(raw)).convert("RGB")
            rect = decision.get("crop", proposal["crop"])
            if not (0 <= rect[0] < rect[2] <= full.width and 0 <= rect[1] < rect[3] <= full.height):
                raise ValueError("Invalid crop bounds")
            crop = full.crop(rect)
            if min(crop.size) < 240:
                raise ValueError("Native crop minimum edge is 240 pixels")
            overlay = crop.copy()
            draw = ImageDraw.Draw(overlay)
            for cls, x1, y1, x2, y2 in decision["boxes"]:
                if cls not in (0, 1, 2) or not (0 <= x1 < x2 <= crop.width and 0 <= y1 < y2 <= crop.height):
                    raise ValueError("Invalid manual XYXY box")
                color = ["#1689ff", "#18b84b", "#fa3030"][cls]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                draw.text((x1, max(0, y1 - 15)), str(cls), font=font, fill=color)
            prefix = evidence / decision["image_id"]
            crop_path = prefix.with_name(prefix.name + "_crop.png")
            overlay_path = prefix.with_name(prefix.name + "_manual_overlay.png")
            head_path = prefix.with_name(prefix.name + "_head_native.png")
            crop.save(crop_path)
            overlay.save(overlay_path)
            crop.crop(decision["head_detail"]).save(head_path)
            head = Image.open(head_path)
            head.resize((head.width * 5, head.height * 5), Image.Resampling.NEAREST).save(prefix.with_name(prefix.name + "_head_nearest.png"))
            rel = lambda p: p.relative_to(ROOT).as_posix()
            entry = dict(
                provider="kaggle", dataset_ref="maitam/vietnamese-traffic-signs", version=8,
                image_id=decision["image_id"], archive_entry=proposal["archive_entry"],
                archive_file=archive_rel, archive_sha256=archive_hash,
                metadata_file=metadata_rel, metadata_sha256=digest(ACQ / metadata_rel),
                source_image=source_path.relative_to(ACQ).as_posix(), source_sha256=source_hash,
                source_canvas=list(full.size), crop=rect, canvas=list(crop.size), crop_sha256=digest(crop_path),
                boxes=decision["boxes"], country="VN",
                location_evidence="VNTS v8 is described by Data Palpating Team as road imagery collected at locations in Vietnam; the exact source frame, native crop, and street context were visually checked. No city is inferred.",
                source_group="kaggle:maitam/vietnamese-traffic-signs",
                page_url="https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs",
                license="CC BY-SA 4.0", license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                creator="Data Palpating Team", maintainer="Mai Tam (maitam)",
                status="corrected" if args.commit else "preview_only",
                visual_reviewed=args.commit, source_verified=True,
                evidence=rel(overlay_path), evidence_sha256=digest(overlay_path),
                head_detail=rel(head_path), head_detail_sha256=digest(head_path),
                review_log=rel(evidence / "manual_review.json"), reason=decision["reason"],
                before={"provider_labels": "Traffic signs; not rider/motorcycle ground truth", "model_proposals_NOT_GT": proposal["proposal_boxes_NOT_GT"]},
                after_manual_boxes_xyxy=decision["boxes"],
                before_proposal_crop=proposal["crop"], before_proposal_canvas=proposal["canvas"],
                after_manual_crop=rect,
                model_assisted_review=True, model_predictions_used_as_ground_truth=False,
                annotator="Codex visual review of native source, crop, head details, and corrected overlay",
                geometry_qa_passed=args.commit,
                geometry_qa=decision.get("geometry_qa", "Separately measured visible extents for each eligible motorcycle and seated rider; no hidden body parts invented."),
                geometry_origin=decision.get("geometry_origin", "manual_pixel_measurement"),
                before_coco_geometry_NOT_GT=decision.get("coco_geometry_NOT_GT"),
                reviewed_at_utc=datetime.now(UTC).isoformat(),
            )
            if args.commit and decision.get("overlay_checked") is not True:
                raise ValueError("Commit requires explicit final overlay review")
            entries.append(entry)
    log = {"accepted_count": len(entries) if args.commit else 0, "preview_count": len(entries), "reviewed": entries, "pending": spec.get("pending", []), "proposal_batch": spec["proposal_batch"], "model_predictions_used_as_ground_truth": False}
    if args.commit:
        write(evidence / "manual_review.json", log)
        write(ledger, entries)
    else:
        write(evidence / "preview_review.json", log)
    print(json.dumps({"committed": args.commit, "entries": len(entries), "evidence": str(evidence), "ledger": str(ledger) if args.commit else None}))


if __name__ == "__main__":
    run()
