"""Audit the RUNNING demo via HTTP on all validation images; no checkpoint/config changes."""

import argparse
import hashlib
import io
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
import httpx
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from app import config as cfg
from evaluation_metrics import evaluate_image, summarize

DATA = cfg.ROOT.parent / "DataSet2-20260927T023713Z-1-001/DataSet2"
OUT = cfg.ROOT / "evaluation/model_audit_20261013"
BASE = "http://127.0.0.1:8000"


def labels(image):
    with Image.open(image) as im:
        w, h = im.size
    boxes = []
    for row in (DATA / "valid/labels" / (image.stem + ".txt")).read_text().splitlines():
        cls, x, y, bw, bh = map(float, row.split())
        boxes.append(
            {
                "class_id": int(cls),
                "xyxy": [
                    (x - bw / 2) * w,
                    (y - bh / 2) * h,
                    (x + bw / 2) * w,
                    (y + bh / 2) * h,
                ],
            }
        )
    return boxes, (w, h)


def dataset_audit():
    result = {}
    groups = {}
    canonical = {}
    hashes = {}
    for split in ("train", "valid"):
        images = sorted((DATA / split / "images").iterdir())
        counts = Counter()
        invalid = []
        outside = 0
        missing = []
        groups[split] = defaultdict(list)
        canonical[split] = defaultdict(list)
        hashes[split] = defaultdict(list)
        for path in images:
            if not path.is_file():
                continue
            label = DATA / split / "labels" / (path.stem + ".txt")
            if not label.is_file():
                missing.append(path.name)
                continue
            for line_no, row in enumerate(label.read_text().splitlines(), 1):
                try:
                    cls, x, y, w, h = map(float, row.split())
                    if (
                        cls not in (0, 1, 2)
                        or not all(np.isfinite([x, y, w, h]))
                        or min(x, y) < 0
                        or max(x, y) > 1
                        or not (0 < w <= 1 and 0 < h <= 1)
                    ):
                        invalid.append([path.name, line_no])
                    counts[int(cls)] += 1
                    if (
                        x - w / 2 < -0.001
                        or y - h / 2 < -0.001
                        or x + w / 2 > 1.001
                        or y + h / 2 > 1.001
                    ):
                        outside += 1
                except (ValueError, OverflowError):
                    invalid.append([path.name, line_no])
            stem = path.name.split(".rf.")[0]
            canonical[split][stem].append(path.name)
            match = re.match(r"(.+?_(?:MOV|mp4))-\d+(?:_|$)", stem, re.I)
            if match:
                groups[split][match.group(1)].append(path.name)
            hashes[split][hashlib.sha256(path.read_bytes()).hexdigest()].append(
                path.name
            )
        result[split] = {
            "image_count": len(images),
            "class_instance_counts": dict(counts),
            "invalid_rows": invalid,
            "boxes_outside_image": outside,
            "missing_labels": missing,
            "recognized_video_group_count": len(groups[split]),
        }
    shared = sorted(set(groups["train"]) & set(groups["valid"]))
    exact = sorted(set(hashes["train"]) & set(hashes["valid"]))
    result["overlap"] = {
        "recognized_shared_video_groups": shared,
        "validation_images_in_shared_video_groups": sum(
            len(groups["valid"][g]) for g in shared
        ),
        "shared_original_frame_names": sorted(
            set(canonical["train"]) & set(canonical["valid"])
        ),
        "exact_file_duplicates": [
            {
                "sha256": key,
                "train": hashes["train"][key],
                "valid": hashes["valid"][key],
            }
            for key in exact
        ],
        "method": "Video groups inferred only from explicit MOV/mp4 filename roots; same-video frame overlap is a risk indicator, not proof of identical pixels. SHA256 checks exact file duplication. Test split not used.",
    }
    return result


def run_job(client, name, body, media_type="image"):
    start = time.monotonic()
    response = client.post(
        "/api/jobs", data={"media_type": media_type}, files={"file": (name, body)}
    )
    response.raise_for_status()
    job = response.json()
    progress = []
    health_times = []
    deadline = time.monotonic() + 600
    while job["status"] in ("queued", "processing"):
        if time.monotonic() > deadline:
            raise TimeoutError(f"Job {job['id']} exceeded audit wait deadline")
        time.sleep(0.2)
        job = client.get(f"/api/jobs/{job['id']}").raise_for_status().json()
        progress.append(job["progress"])
        if media_type == "video" and len(progress) % 5 == 0:
            t = time.monotonic()
            client.get("/api/health").raise_for_status()
            health_times.append(time.monotonic() - t)
    if job["status"] != "completed":
        raise RuntimeError(job.get("error_message"))
    if (
        job["model_identifier"] != HEALTH["model_identifier"]
        or job["imgsz"] != HEALTH["imgsz"]
        or job["confidence_threshold"] != HEALTH["confidence_threshold"]
    ):
        raise RuntimeError("Backend model/config changed during audit")
    return job, time.monotonic() - start, progress, health_times


def add_counts(target, source):
    for cls in range(3):
        for key in ("tp", "fp", "fn", "gt"):
            target[cls][key] += source[cls][key]


def new_counts():
    return [{"tp": 0, "fp": 0, "fn": 0, "gt": 0} for _ in range(3)]


def main():
    global HEALTH, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise RuntimeError(
            "Audit report already exists; use --output with a new report directory"
        )
    client = httpx.Client(base_url=BASE, timeout=30)
    HEALTH = client.get("/api/health").raise_for_status().json()
    if not HEALTH["database_ready"] or not HEALTH["model_ready"]:
        raise RuntimeError("Backend/MySQL not ready")
    report = {
        "health": HEALTH,
        "dataset_audit": dataset_audit(),
        "method": {
            "confidence": HEALTH["confidence_threshold"],
            "imgsz": HEALTH["imgsz"],
            "iou_thresholds": [0.5, 0.75],
            "matching": "Predictions in descending confidence; match at most one unused GT of same class by highest IoU. Fixed operating-point precision/recall, NOT AP/mAP.",
            "test_split_used": False,
            "checkpoint_modified": False,
            "api_endpoint": BASE,
            "sampling": "All validation images, no selection by score.",
        },
        "validation": [],
        "stress": [],
        "negative": [],
        "videos": [],
    }
    (OUT / "dataset_audit.json").write_text(
        json.dumps(report["dataset_audit"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    images = sorted(
        p
        for p in (DATA / "valid/images").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    totals = {str(t): new_counts() for t in (0.5, 0.75)}
    size_buckets = {s: new_counts() for s in ("small", "medium", "large")}
    density_buckets = {s: new_counts() for s in ("1-3", "4-7", "8+")}
    brightness_buckets = {s: new_counts() for s in ("dark", "normal", "bright")}
    for index, image in enumerate(images, 1):
        gt, (width, height) = labels(image)
        job, duration, _, _ = run_job(
            client, "AUDIT_VAL_" + image.name, image.read_bytes()
        )
        preds = job.get("detections") or []
        evaluation = {str(t): evaluate_image(gt, preds, t) for t in (0.5, 0.75)}
        for t in (0.5, 0.75):
            add_counts(totals[str(t)], evaluation[str(t)]["counts"])
        scale = HEALTH["imgsz"] / max(width, height)
        matched = set(evaluation["0.5"]["matched_gt"])
        for gi, box in enumerate(gt):
            a = (
                (box["xyxy"][2] - box["xyxy"][0])
                * (box["xyxy"][3] - box["xyxy"][1])
                * scale
                * scale
            )
            size = "small" if a < 32**2 else "medium" if a < 96**2 else "large"
            row = size_buckets[size][box["class_id"]]
            row["gt"] += 1
            row["tp" if gi in matched else "fn"] += 1
        density = "1-3" if len(gt) <= 3 else "4-7" if len(gt) <= 7 else "8+"
        with Image.open(image) as im:
            brightness = float(np.asarray(im.convert("L").resize((64, 64))).mean())
        bright = (
            "dark" if brightness < 70 else "bright" if brightness > 180 else "normal"
        )
        add_counts(density_buckets[density], evaluation["0.5"]["counts"])
        add_counts(brightness_buckets[bright], evaluation["0.5"]["counts"])
        row = {
            "image": image.name,
            "job_id": job["id"],
            "width": width,
            "height": height,
            "brightness": brightness,
            "density_bucket": density,
            "ground_truth": gt,
            "predictions": preds,
            "seconds_submission_to_completion": duration,
            "evaluation": evaluation,
        }
        report["validation"].append(row)
        with (OUT / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if index % 10 == 0 or index == len(images):
            print(
                json.dumps({"validation_completed": index, "total": len(images)}),
                flush=True,
            )
    report["metrics"] = {t: summarize(rows) for t, rows in totals.items()}
    report["size_recall"] = size_buckets
    report["density_metrics"] = {k: summarize(v) for k, v in density_buckets.items()}
    report["brightness_metrics"] = {
        k: summarize(v) for k, v in brightness_buckets.items()
    }
    base = max(
        report["validation"],
        key=lambda r: (
            sum(g["class_id"] == 2 for g in r["ground_truth"]) > 0
            and sum(g["class_id"] == 1 for g in r["ground_truth"]) > 0,
            len(r["ground_truth"]),
        ),
    )
    source = DATA / "valid/images" / base["image"]
    with Image.open(source) as im:
        original = im.convert("RGB")
    transforms = {
        "baseline": original,
        "dark_35_percent": ImageEnhance.Brightness(original).enhance(0.35),
        "bright_180_percent": ImageEnhance.Brightness(original).enhance(1.8),
        "blur_radius_3": original.filter(ImageFilter.GaussianBlur(3)),
        "low_resolution_320": original.resize(
            (
                max(1, round(original.width * 320 / max(original.size))),
                max(1, round(original.height * 320 / max(original.size))),
            )
        ),
    }
    for name, image in transforms.items():
        data = io.BytesIO()
        image.save(data, format="JPEG", quality=95)
        sx, sy = image.width / original.width, image.height / original.height
        gt = [
            {
                "class_id": g["class_id"],
                "xyxy": [
                    g["xyxy"][0] * sx,
                    g["xyxy"][1] * sy,
                    g["xyxy"][2] * sx,
                    g["xyxy"][3] * sy,
                ],
            }
            for g in base["ground_truth"]
        ]
        job, duration, _, _ = run_job(
            client, f"AUDIT_STRESS_{name}.jpg", data.getvalue()
        )
        report["stress"].append(
            {
                "case": name,
                "source": base["image"],
                "job_id": job["id"],
                "ground_truth": gt,
                "predictions": job["detections"] or [],
                "evaluation": evaluate_image(gt, job["detections"] or [], 0.5),
                "seconds": duration,
            }
        )
    for name, quality in [("jpeg_quality_15", 15)]:
        data = io.BytesIO()
        original.save(data, format="JPEG", quality=quality)
        job, duration, _, _ = run_job(
            client, f"AUDIT_STRESS_{name}.jpg", data.getvalue()
        )
        report["stress"].append(
            {
                "case": name,
                "source": base["image"],
                "job_id": job["id"],
                "ground_truth": base["ground_truth"],
                "predictions": job["detections"] or [],
                "evaluation": evaluate_image(
                    base["ground_truth"], job["detections"] or [], 0.5
                ),
                "seconds": duration,
            }
        )
    for name, image in [
        ("black", Image.new("RGB", (640, 360))),
        ("gray", Image.new("RGB", (640, 360), (128, 128, 128))),
    ]:
        data = io.BytesIO()
        image.save(data, format="PNG")
        job, duration, _, _ = run_job(
            client, f"AUDIT_NEGATIVE_{name}.png", data.getvalue()
        )
        report["negative"].append(
            {
                "case": name,
                "job_id": job["id"],
                "detections": job["detections"] or [],
                "seconds": duration,
            }
        )
    for name in ["test-xe-may-3s.mp4", "test-ca-doi-mu-va-khong-doi-mu-3s.mp4"]:
        video = cfg.STORAGE / "uploads" / name
        if not video.is_file():
            continue
        with video.open("rb") as stream:
            job, duration, progress, health_times = run_job(
                client, "AUDIT_VIDEO_" + name, stream, "video"
            )
        output = client.get(f"/api/jobs/{job['id']}/output").raise_for_status()
        (OUT / ("result-" + name)).write_bytes(output.content)
        report["videos"].append(
            {
                "input": name,
                "job_id": job["id"],
                "seconds_submission_to_completion": duration,
                "progress_samples": progress,
                "health_response_seconds": health_times,
                "result_file": "result-" + name,
                "note": "No exhaustive frame-by-frame GT; visual case analysis only, not measured video detection accuracy.",
            }
        )
    bad = client.post(
        "/api/jobs",
        data={"media_type": "image"},
        files={"file": ("AUDIT_INVALID.jpg", b"invalid", "image/jpeg")},
    )
    report["invalid_file_status"] = bad.status_code
    (OUT / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "complete": True,
                "report": str(OUT),
                "metrics": report["metrics"],
                "invalid_file_status": bad.status_code,
            }
        ),
        flush=True,
    )
    client.close()


if __name__ == "__main__":
    main()
