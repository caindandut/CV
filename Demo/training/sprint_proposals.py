"""Prepare model-assisted review windows; never admit labels as ground truth."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import zipfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ACQ = ROOT / "acquisition_2000"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
sys.path.insert(0, str(ROOT.parent / "backend"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def review_window(detections, size):
    """Use general COCO people/motorcycles solely to suggest an isolated view."""
    width, height = size
    windows = []
    for moto in (r for r in detections if r["class_id"] == 3):
        mb = moto["xyxy"]
        for person in (r for r in detections if r["class_id"] == 0):
            pb = person["xyxy"]
            if pb[3] - pb[1] < 55 or intersection(mb, pb) < area(pb) * 0.08:
                continue
            bounds = [min(mb[0], pb[0]), min(mb[1], pb[1]), max(mb[2], pb[2]), max(mb[3], pb[3])]
            # Discard source-edge truncation from this quick-review queue.
            if bounds[0] < 4 or bounds[1] < 4 or bounds[2] > width - 4 or bounds[3] > height - 4:
                continue
            cw = min(width, max(240, math.ceil(bounds[2] - bounds[0] + 36)))
            ch = min(height, max(240, math.ceil(bounds[3] - bounds[1] + 36)))
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            left = round(max(0, min(width - cw, cx - cw / 2)))
            top = round(max(0, min(height - ch, cy - ch / 2)))
            window = [left, top, left + cw, top + ch]
            others = [r for r in detections if r is not moto and r is not person and intersection(window, r["xyxy"]) > 0.03 * area(r["xyxy"])]
            if any(r["class_id"] == 3 for r in others):
                continue
            # Extra people require visual review; they may be passengers/pedestrians.
            score = area(pb) / (cw * ch) - 0.12 * len(others)
            windows.append((score, window))
    return max(windows, default=(None, None), key=lambda p: -1e9 if p[0] is None else p[0])[1]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if not args.batch.replace("_", "").isalnum() or not 1 <= args.limit <= 4000:
        raise ValueError("Invalid batch name or bounded limit")
    out = ACQ / "sprint_5000_5h" / args.batch
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    meta_path = ACQ / "research_kaggle/maitam_vietnamese_traffic_signs_metadata01.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["ref"] != "maitam/vietnamese-traffic-signs" or metadata["licenseName"] != "CC BY-SA 4.0" or metadata["currentVersionNumber"] != 8:
        raise ValueError("Source identity/license mismatch")
    acquisition = json.loads((ACQ / "research_kaggle/maitam_vnts_v8_acquisition01.json").read_text(encoding="utf-8"))
    archive_path = ACQ / "research_kaggle/maitam_vnts_v8.zip"
    archive_hash = hashlib.file_digest(archive_path.open("rb"), "sha256").hexdigest()
    if archive_hash != acquisition["sha256"]:
        raise ValueError("Archive identity mismatch")
    used = set()
    for ledger in ACQ.glob("kaggle_reviews_batch*.json"):
        for entry in json.loads(ledger.read_text(encoding="utf-8")):
            if entry["dataset_ref"] == metadata["ref"]:
                used.add(entry["image_id"])
    for screening_path in (ACQ / "sprint_5000_5h").glob("maitam_*/screening.json"):
        for entry in json.loads(screening_path.read_text(encoding="utf-8")):
            used.add(entry["image_id"])
    inventory = json.loads((ACQ / "research_kaggle/maitam_v8_inventory01.json").read_text(encoding="utf-8"))
    inventory = [r for r in inventory if Path(r["entry"]).stem not in used]
    random.Random(args.seed).shuffle(inventory)
    selected = []
    for row in inventory:
        if all((int(row["phash"], 16) ^ int(r["phash"], 16)).bit_count() > 8 for r in selected):
            selected.append(row)
            if len(selected) == args.limit:
                break

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
    with zipfile.ZipFile(archive_path) as archive:
        for n, row in enumerate(selected, 1):
            image_id = Path(row["entry"]).stem
            raw = archive.read(row["entry"])
            if sha(raw) != row["sha256"]:
                raise ValueError("Source member hash mismatch")
            with Image.open(io.BytesIO(raw)) as original:
                full = original.convert("RGB")
            pred = general.predict(full, imgsz=640, conf=0.2, classes=[0, 3], device="cpu", verbose=False)[0]
            objects = [{"class_id": int(b.cls.item()), "xyxy": b.xyxy[0].tolist(), "confidence": float(b.conf.item())} for b in pred.boxes]
            window = review_window(objects, full.size)
            screening = {"image_id": image_id, "source_sha256": row["sha256"], "coco_proposals_NOT_GT": objects, "crop": window, "status": "unreviewed"}
            screened.append(screening)
            if window is not None:
                crop = full.crop(window)
                hints = helmet.predict(crop, 768, 0.2)
                if any(r["class_id"] in (1, 2) for r in hints):
                    source_file = out / (image_id + "_source.jpg")
                    source_file.write_bytes(raw)
                    crop.save(out / (image_id + "_crop.png"))
                    overlay = crop.copy()
                    draw = ImageDraw.Draw(overlay)
                    for i, hint in enumerate(hints):
                        color = ["#1689ff", "#18b84b", "#fa3030"][hint["class_id"]]
                        box = hint["xyxy"]
                        draw.rectangle(box, outline=color, width=2)
                        draw.text((max(0, box[0]), max(0, box[1] - 15)), f"{i}: class {hint['class_id']} {hint['confidence']:.2f}", font=font, fill=color)
                    overlay.save(out / (image_id + "_proposal_overlay.png"))
                    proposals.append({"image_id": image_id, "archive_entry": row["entry"], "source_sha256": row["sha256"], "source_canvas": list(full.size), "crop": window, "canvas": list(crop.size), "crop_sha256": sha((out / (image_id + "_crop.png")).read_bytes()), "proposal_boxes_NOT_GT": hints, "review_status": "pending", "model_predictions_used_as_ground_truth": False})
            if n % 10 == 0 or n == len(selected):
                save_json(out / "screening.json", screened)
                save_json(out / "proposals.json", proposals)
                print(json.dumps({"screened": n, "review_windows": len(proposals), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    # Triage sheets are not evidence of a completed annotation review.
    for start in range(0, len(proposals), 12):
        sheet = Image.new("RGB", (1280, 1080), "#171c23")
        for slot, proposal in enumerate(proposals[start:start + 12]):
            im = Image.open(out / (proposal["image_id"] + "_proposal_overlay.png")).convert("RGB")
            im.thumbnail((312, 320))
            x, y = (slot % 4) * 320, (slot // 4) * 360
            sheet.paste(im, (x, y + 24))
            ImageDraw.Draw(sheet).text((x + 4, y + 4), proposal["image_id"] + " PENDING", font=font, fill="white")
        sheet.save(out / f"sheet_{start // 12:03d}.png")
    save_json(out / "run.json", {"source": metadata["ref"], "version": 8, "license": metadata["licenseName"], "archive_sha256": archive_hash, "metadata_sha256": sha(meta_path.read_bytes()), "seed": args.seed, "screened": len(screened), "proposals": len(proposals), "accepted": 0, "helmet_hint_model": helmet.identifier, "helmet_hint_head_classes": helmet.model.model.model[-1].nc, "purpose": "Review candidates only; inference labels are never automatically accepted", "selected_full_frame_phash_distance_minimum": 9})


if __name__ == "__main__":
    run()
