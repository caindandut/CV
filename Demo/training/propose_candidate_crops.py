"""COCO suggests review windows only. Never emits helmet GT or accepts an image."""

import argparse
import json
import math
import os
import re

from curate_v2 import plain
from PIL import Image, ImageDraw, ImageFont
from prepare_dataset import ROOT, digest, write_json


def train_source_subset(sources, manifest, attribution):
    training = {r["id"] for r in manifest if r["new_split"] == "train"}
    holdout = {r["id"] for r in manifest if r["new_split"] in {"val", "test"}}
    authors = {r["artist"] for r in attribution if r["id"] in training}
    authors -= {r["artist"] for r in attribution if r["id"] in holdout}
    return [
        r
        for r in sources
        if plain(r["extmetadata"].get("Artist", {}).get("value", "")) in authors
    ]


def choose_crop(detections, size):
    width, height = size
    candidates = []
    for moto in (r for r in detections if r["name"] == "motorcycle"):
        mx1, my1, mx2, my2 = moto["xyxy"]
        for person in (r for r in detections if r["name"] == "person"):
            px1, py1, px2, py2 = person["xyxy"]
            area = (px2 - px1) * (py2 - py1)
            if area < width * height * 0.02 or not (
                mx1 - 0.3 * (mx2 - mx1) <= (px1 + px2) / 2 <= mx2 + 0.3 * (mx2 - mx1)
            ):
                continue
            overlap = max(0, min(mx2, px2) - max(mx1, px1)) * max(
                0, min(my2, py2) - max(my1, py1)
            )
            if overlap < area * 0.05:
                continue
            candidates.append(
                (area, [min(mx1, px1), min(my1, py1), max(mx2, px2), max(my2, py2)])
            )
    if not candidates:
        return None
    _, (left, top, right, bottom) = max(candidates, key=lambda pair: pair[0])
    cx, cy = (left + right) / 2, (top + bottom) / 2
    crop_w = min(width, max(256, right - left + max(24, (right - left) * 0.1)))
    crop_h = min(height, max(256, bottom - top + max(24, (bottom - top) * 0.1)))
    left = max(0, min(width - crop_w, cx - crop_w / 2))
    top = max(0, min(height - crop_h, cy - crop_h / 2))
    return [
        math.floor(left),
        math.floor(top),
        math.ceil(left + crop_w),
        math.ceil(top + crop_h),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument(
        "--train-source-only",
        action="store_true",
        help="Prioritize authors already confined to training; never move holdout",
    )
    args = parser.parse_args()
    if (
        not re.fullmatch("[a-z0-9_]+", args.batch_name)
        or args.start < 0
        or not 1 <= args.limit <= 200
    ):
        raise ValueError("Invalid bounded review batch")
    folder = ROOT / "acquisition_2000/coco_crop_proposals" / args.batch_name
    if folder.exists():
        raise FileExistsError("Review batch already exists; use another name")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
    import torch

    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(2)
    weights = ROOT / ".cache/yolo26s.pt"
    model = YOLO(str(weights))
    assert model.names[0] == "person" and model.names[3] == "motorcycle"
    progress = json.loads(
        (ROOT / "acquisition_2000/progress.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / progress["frozen_dataset"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    used = {
        int(r["id"].rsplit("_", 1)[-1].split(".")[0])
        for r in manifest
        if r["new_split"] != "pending" and r["id"].startswith("commons/")
    }
    sources = sorted(
        [
            r
            for r in json.loads(
                (ROOT / "acquisition_2000/sources.json").read_text(encoding="utf-8")
            )
            if "download_sha256" in r
        ],
        key=lambda r: r["pageid"],
    )
    if args.train_source_only:
        attribution = json.loads(
            (ROOT / progress["frozen_dataset"] / "attribution.json").read_text(
                encoding="utf-8"
            )
        )
        sources = train_source_subset(sources, manifest, attribution)
    selected = sources[args.start : args.start + args.limit]
    folder.mkdir(parents=True)
    proposals = []
    run = {
        "purpose": "Unreviewed crop proposals only; COCO detections are not training labels",
        "model": "official COCO yolo26s",
        "weights_sha256": digest(weights),
        "ultralytics": ultralytics.__version__,
        "device": "cpu",
        "imgsz": 640,
        "confidence": 0.25,
        "train_source_only": args.train_source_only,
        "parent_dataset": progress["frozen_dataset"],
        "start": args.start,
        "limit": args.limit,
        "processed": 0,
        "proposals": 0,
    }
    for index, source in enumerate(selected):
        if source["pageid"] in used:
            continue
        path = ROOT / "acquisition_2000" / source["local_image"]
        assert path.resolve().is_relative_to(
            (ROOT / "acquisition_2000/images").resolve()
        )
        assert digest(path) == source["download_sha256"]
        with Image.open(path) as image:
            size = image.size
        prediction = model.predict(
            str(path), imgsz=640, conf=0.25, classes=[0, 3], device="cpu", verbose=False
        )[0]
        detections = [
            {
                "name": model.names[int(b.cls.item())],
                "xyxy": b.xyxy[0].tolist(),
                "confidence": float(b.conf.item()),
            }
            for b in prediction.boxes
        ]
        crop = choose_crop(detections, size)
        if crop:
            with Image.open(path) as image:
                image.convert("RGB").crop(crop).save(
                    folder / f"commons_{source['pageid']}.png"
                )
            proposals.append(
                {
                    "index": len(proposals),
                    "pageid": source["pageid"],
                    "source_title": source["title"],
                    "source_sha256": source["download_sha256"],
                    "crop": crop,
                    "canvas": [crop[2] - crop[0], crop[3] - crop[1]],
                    "coco_detections_NOT_GT": detections,
                    "review_status": "unreviewed",
                }
            )
        run.update(processed=index + 1, proposals=len(proposals))
        write_json(folder / "proposals.json", proposals)
        write_json(folder / "run.json", run)
        if (index + 1) % 10 == 0:
            print(
                f"processed={index + 1}/{len(selected)} review_proposals={len(proposals)}",
                flush=True,
            )
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 18)
    for start in range(0, len(proposals), 6):
        sheet = Image.new("RGB", (1280, 1140), "#172029")
        for offset, row in enumerate(proposals[start : start + 6]):
            with Image.open(folder / f"commons_{row['pageid']}.png") as image:
                image.thumbnail((630, 340))
                x = offset % 2 * 640
                y = offset // 2 * 380
                sheet.paste(image, (x, y))
                ImageDraw.Draw(sheet).text(
                    (x + 4, y + 345),
                    f"C{row['index']:03d} | {row['pageid']} | UNREVIEWED",
                    font=font,
                    fill="white",
                )
        sheet.save(folder / f"sheet_{start // 6:03d}.jpg", quality=95)
    print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
