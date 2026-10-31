"""Acquire bounded Commons originals and prepare native review crops, never GT."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error
import threading

from PIL import Image, ImageDraw, ImageFont
from sprint_proposals import ROOT, ACQ, review_window, save_json

STOP_DOWNLOADS = threading.Event()


def acquire(source):
    folder = ACQ / "commons_original_resolution_review/sprint_20261030" / str(source["pageid"])
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "source_acquisition.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "downloaded_candidate_only":
            file = folder / receipt["file"]
            if file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == receipt["sha256"]:
                return source["pageid"], receipt, folder
        return source["pageid"], receipt, folder
    receipt = {"pageid": source["pageid"], "requested_url": source["url"], "status": "pending_download"}
    if STOP_DOWNLOADS.is_set():
        receipt.update(status="deferred_rate_limit_not_admitted", attempted_at_utc=datetime.now(UTC).isoformat())
        save_json(receipt_path, receipt)
        return source["pageid"], receipt, folder
    try:
        url = urllib.parse.urlsplit(source["url"])
        if url.scheme != "https" or url.hostname != "upload.wikimedia.org":
            raise ValueError("Original URL is not the published Wikimedia media host")
        req = urllib.request.Request(source["url"], headers={"User-Agent": "VietnamHelmetDatasetCuration/1.0 (research source acquisition)"})
        with urllib.request.urlopen(req, timeout=35) as response:
            if response.status != 200 or urllib.parse.urlsplit(response.geturl()).hostname != "upload.wikimedia.org":
                raise ValueError("Unexpected original media response")
            payload = bytearray()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > 30 * 1024 * 1024:
                    raise ValueError("Original exceeds bounded 30 MiB source limit")
            receipt.update(http_status=response.status, final_url=response.geturl(), content_type=response.headers.get_content_type())
        suffix = Path(url.path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("Unsupported published raster format")
        file = folder / ("source_original" + suffix)
        file.write_bytes(payload)
        receipt.update(file=file.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), sha1=hashlib.sha1(payload).hexdigest())
        if receipt["sha1"] != source["sha1"] or len(payload) != source["size"]:
            raise ValueError("Original does not match published SHA1/byte length")
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            receipt.update(canvas=list(image.size), exif_orientation=image.getexif().get(274, 1))
        if receipt["canvas"] != [source["width"], source["height"]] or receipt["exif_orientation"] != 1:
            raise ValueError("Unexpected original dimensions/orientation; requires separate review")
        receipt.update(status="downloaded_candidate_only", downloaded_at_utc=datetime.now(UTC).isoformat())
    except Exception as error:
        if isinstance(error, urllib.error.HTTPError) and error.code == 429:
            STOP_DOWNLOADS.set()
            receipt["retry_after"] = error.headers.get("Retry-After")
        receipt.update(status="failed_not_admitted", error=str(error), attempted_at_utc=datetime.now(UTC).isoformat())
    save_json(receipt_path, receipt)
    return source["pageid"], receipt, folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="commons_originals_001")
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()
    if not args.batch.startswith("commons_originals_") or not args.batch.replace("_", "").isalnum() or not 1 <= args.limit <= 400:
        raise ValueError("Invalid bounded original review batch")
    out = ACQ / "sprint_5000_5h" / args.batch
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    base = ACQ / "sprint_5000_5h/commons_002"
    screened = json.loads((base / "screening.json").read_text(encoding="utf-8"))
    sources = {r["pageid"]: r for r in json.loads((ACQ / "sources.json").read_text(encoding="utf-8"))}
    queue = []
    for row in screened:
        source = sources[row["pageid"]]
        if source["size"] > 30 * 1024 * 1024:
            continue
        with Image.open(ACQ / source["local_image"]) as thumbnail:
            sx, sy = source["width"] / thumbnail.width, source["height"] / thumbnail.height
        if sx <= 1.25:
            continue
        detections = [{**d, "xyxy": [d["xyxy"][0] * sx, d["xyxy"][1] * sy, d["xyxy"][2] * sx, d["xyxy"][3] * sy]} for d in row["coco_proposals_NOT_GT"]]
        window = review_window(detections, [source["width"], source["height"]])
        if window:
            queue.append((source, window, max((d["xyxy"][3] - d["xyxy"][1] for d in detections if d["class_id"] == 0), default=0)))
    queue.sort(key=lambda r: r[2], reverse=True)
    queue = queue[:args.limit]
    save_json(out / "selection.json", [{"pageid": s["pageid"], "crop": crop, "score_person_height": score} for s, crop, score in queue])
    acquired = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for pageid, receipt, folder in pool.map(acquire, [s for s, _, _ in queue]):
            acquired.append({"pageid": pageid, "receipt": receipt, "folder": str(folder)})
            if len(acquired) % 10 == 0 or len(acquired) == len(queue):
                save_json(out / "acquisitions.json", acquired)
                print(json.dumps({"attempted": len(acquired), "downloaded": sum(r["receipt"].get("status") == "downloaded_candidate_only" for r in acquired), "selected": len(queue)}), flush=True)
    import torch
    from app.detector import Detector

    torch.set_num_threads(3)
    model = Detector()
    model.load()
    if model.model is None:
        raise RuntimeError(model.error)
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 12)
    selections = {s["pageid"]: crop for s, crop, _ in queue}
    proposals = []
    for record in acquired:
        receipt = record["receipt"]
        if receipt.get("status") != "downloaded_candidate_only":
            continue
        folder = Path(record["folder"])
        source = sources[record["pageid"]]
        source_path = folder / receipt["file"]
        crop = Image.open(source_path).convert("RGB").crop(selections[record["pageid"]])
        name = str(record["pageid"])
        crop.save(out / (name + "_crop.png"))
        hints = model.predict(crop, 768, 0.2)
        overlay = crop.copy()
        draw = ImageDraw.Draw(overlay)
        for i, hint in enumerate(hints):
            color = ["#1689ff", "#18b84b", "#fa3030"][hint["class_id"]]
            draw.rectangle(hint["xyxy"], outline=color, width=2)
            draw.text((hint["xyxy"][0], max(0, hint["xyxy"][1] - 15)), f"{i}: class {hint['class_id']}", fill=color, font=font)
        overlay.save(out / (name + "_proposal_overlay.png"))
        proposals.append({"pageid": record["pageid"], "source_page_url": source["descriptionurl"], "source_variant": "original", "source_file": source_path.relative_to(ACQ).as_posix(), "source_sha256": receipt["sha256"], "source_canvas": receipt["canvas"], "original_acquisition_file": (folder / "source_acquisition.json").relative_to(ACQ).as_posix(), "original_acquisition_sha256": hashlib.sha256((folder / "source_acquisition.json").read_bytes()).hexdigest(), "crop": selections[record["pageid"]], "canvas": list(crop.size), "proposal_boxes_NOT_GT": hints, "review_status": "pending", "model_predictions_used_as_ground_truth": False})
        save_json(out / "proposals.json", proposals)
    for start in range(0, len(proposals), 12):
        sheet = Image.new("RGB", (1280, 1080), "#171c23")
        for slot, proposal in enumerate(proposals[start:start + 12]):
            name = str(proposal["pageid"])
            im = Image.open(out / (name + "_proposal_overlay.png")).convert("RGB")
            im.thumbnail((312, 320))
            x, y = slot % 4 * 320, slot // 4 * 360
            sheet.paste(im, (x, y + 24))
            ImageDraw.Draw(sheet).text((x + 3, y + 3), name + " ORIGINAL PENDING", fill="white", font=font)
        sheet.save(out / f"sheet_{start // 12:03d}.png")
    save_json(out / "run.json", {"selected": len(queue), "verified_originals": len(proposals), "accepted": 0, "hint_model": model.identifier, "hint_head_classes": model.model.model.model[-1].nc, "purpose": "Original bytes verified against Commons metadata; all annotations remain unreviewed proposals", "completed_at_utc": datetime.now(UTC).isoformat()})
    print(json.dumps({"verified_original_proposals": len(proposals), "accepted": 0}), flush=True)


if __name__ == "__main__":
    main()
