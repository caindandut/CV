"""Offline Commons proposal scan for photographers already confined to training."""

import argparse
from datetime import datetime, UTC
import hashlib
import json
import random
import time

from PIL import Image, ImageDraw, ImageFont
from curate_v2 import plain
from propose_candidate_crops import train_source_subset
from sprint_proposals import ROOT, ACQ, review_window, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    if not args.batch.startswith("commons_") or not args.batch.replace("_", "").isalnum() or not 1 <= args.limit <= 1500:
        raise ValueError("Invalid bounded Commons batch")
    out = ACQ / "sprint_5000_5h" / args.batch
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    progress = json.loads((ACQ / "progress.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / progress["frozen_dataset"] / "manifest.json").read_text(encoding="utf-8"))
    attrs = json.loads((ROOT / progress["frozen_dataset"] / "attribution.json").read_text(encoding="utf-8"))
    source_file = ACQ / "sources.json"
    sources = [r for r in json.loads(source_file.read_text(encoding="utf-8")) if r.get("download_sha256") and r.get("local_image") and (ACQ / r["local_image"]).is_file()]
    sources = train_source_subset(sources, manifest, attrs)
    used = {r.get("original_pageid", int(r["id"].rsplit("_", 1)[-1].split(".")[0])) for r in manifest if r["new_split"] != "pending" and r["id"].startswith("commons/")}
    for path in (ACQ / "sprint_5000_5h").glob("commons_*/screening.json"):
        used.update(r["pageid"] for r in json.loads(path.read_text(encoding="utf-8")))
    sources = [r for r in sources if r["pageid"] not in used and plain(r["extmetadata"].get("LicenseShortName", {}).get("value", "")).startswith(("CC BY", "CC0", "Public domain"))]
    random.Random(20261030).shuffle(sources)
    sources = sources[:args.limit]
    import torch
    from ultralytics import YOLO
    from app.detector import Detector

    torch.set_num_threads(3)
    general = YOLO(str(ROOT / ".cache/yolo26s.pt"))
    helmet = Detector()
    helmet.load()
    if helmet.model is None:
        raise RuntimeError(helmet.error)
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 12)
    proposals, screened = [], []
    started = time.monotonic()
    for n, source in enumerate(sources, 1):
        path = (ACQ / source["local_image"]).resolve()
        if not path.is_relative_to((ACQ / "images").resolve()):
            raise ValueError("Source path escapes acquisition images")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["download_sha256"]:
            raise ValueError("Commons acquired bytes changed")
        with Image.open(path) as image:
            full = image.convert("RGB")
        pred = general.predict(full, imgsz=640, conf=0.2, classes=[0, 3], device="cpu", verbose=False)[0]
        objects = [{"class_id": int(b.cls.item()), "xyxy": b.xyxy[0].tolist(), "confidence": float(b.conf.item())} for b in pred.boxes]
        window = review_window(objects, full.size)
        screened.append({"pageid": source["pageid"], "coco_proposals_NOT_GT": objects, "crop": window, "status": "unreviewed"})
        if window is not None:
            crop = full.crop(window)
            hints = helmet.predict(crop, 768, 0.2)
            # Keep windows even without a helmet-model hit: bareheaded riders may be missed.
            name = str(source["pageid"])
            (out / (name + "_source.jpg")).write_bytes(raw)
            crop.save(out / (name + "_crop.png"))
            overlay = crop.copy()
            draw = ImageDraw.Draw(overlay)
            for i, hint in enumerate(hints):
                color = ["#1689ff", "#18b84b", "#fa3030"][hint["class_id"]]
                box = hint["xyxy"]
                draw.rectangle(box, outline=color, width=2)
                draw.text((max(0, box[0]), max(0, box[1] - 15)), f"{i}: class {hint['class_id']}", fill=color, font=font)
            overlay.save(out / (name + "_proposal_overlay.png"))
            proposals.append({"pageid": source["pageid"], "title": source["title"], "source_page_url": source["descriptionurl"], "source_image": source["local_image"], "source_sha256": source["download_sha256"], "source_canvas": list(full.size), "crop": window, "canvas": list(crop.size), "proposal_boxes_NOT_GT": hints, "review_status": "pending", "model_predictions_used_as_ground_truth": False})
        if n % 10 == 0 or n == len(sources):
            save_json(out / "screening.json", screened)
            save_json(out / "proposals.json", proposals)
            print(json.dumps({"screened": n, "review_windows": len(proposals), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    for start in range(0, len(proposals), 12):
        sheet = Image.new("RGB", (1280, 1080), "#171c23")
        for slot, proposal in enumerate(proposals[start:start + 12]):
            name = str(proposal["pageid"])
            im = Image.open(out / (name + "_proposal_overlay.png")).convert("RGB")
            im.thumbnail((312, 320))
            x, y = slot % 4 * 320, slot // 4 * 360
            sheet.paste(im, (x, y + 24))
            ImageDraw.Draw(sheet).text((x + 3, y + 3), name + " PENDING", font=font, fill="white")
        sheet.save(out / f"sheet_{start // 12:03d}.png")
    save_json(out / "run.json", {"source": "Wikimedia Commons", "screened": len(screened), "proposals": len(proposals), "accepted": 0, "selection": "photographers already confined to training; no holdout author", "source_metadata_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(), "hint_model": helmet.identifier, "hint_head_classes": helmet.model.model.model[-1].nc, "completed_at_utc": datetime.now(UTC).isoformat(), "purpose": "Unreviewed window and label proposals only; not ground truth"})


if __name__ == "__main__":
    main()
