"""Small validation-only check of the fixed 0/1/2 filter; no threshold tuning or model changes."""

import json
from collections import Counter
import numpy as np
from app import config as cfg
from app.detector import Detector
from app.media import read_image


def main():
    detector = Detector()
    detector.load()
    if detector.model is None:
        raise RuntimeError(detector.error)
    folder = cfg.ROOT.parent / "DataSet2-20260927T023713Z-1-001/DataSet2/valid/images"
    images = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    indices = sorted(
        set(np.linspace(0, len(images) - 1, min(8, len(images)), dtype=int).tolist())
    )
    if not indices:
        raise RuntimeError("No validation images")
    records = []
    other_total = 0
    for index in indices:
        source = images[index]
        frame = read_image(source)
        raw = detector.model.predict(
            frame,
            imgsz=cfg.IMGSZ,
            conf=cfg.CONFIDENCE,
            iou=0.7,
            rect=False,
            device=detector.device,
            verbose=False,
            save=False,
            classes=None,
        )[0]
        all_boxes = [
            {
                "class_id": int(b.cls.item()),
                "confidence": float(b.conf.item()),
                "xyxy": b.xyxy[0].tolist(),
            }
            for b in raw.boxes
        ]
        expected = [b for b in all_boxes if b["class_id"] in cfg.INFERENCE_CLASS_IDS]
        actual = detector.predict(frame, cfg.IMGSZ, cfg.CONFIDENCE)
        assert len(expected) == len(actual)
        for before, after in zip(expected, actual):
            assert before["class_id"] == after["class_id"]
            assert np.allclose(before["xyxy"], after["xyxy"], atol=1e-4)
            assert abs(before["confidence"] - after["confidence"]) < 1e-6
        extras = [b for b in all_boxes if b["class_id"] not in cfg.INFERENCE_CLASS_IDS]
        other_total += len(extras)
        records.append(
            {
                "image": source.name,
                "detections": len(actual),
                "classes": dict(Counter(b["class_id"] for b in actual)),
                "other_channels_above_threshold": len(extras),
                "other_detections": extras,
                "matches_unfiltered_target_classes": True,
            }
        )
    report = {
        "model_identifier": detector.identifier,
        "device": detector.device,
        "imgsz": cfg.IMGSZ,
        "confidence_threshold": cfg.CONFIDENCE,
        "allowed_class_ids": list(cfg.INFERENCE_CLASS_IDS),
        "head_channels": 80,
        "images_checked": len(records),
        "other_channels_above_threshold": other_total,
        "note": "Small validation-only integration check, not mAP evaluation or proof that extra channels never fire. Threshold unchanged.",
        "records": records,
    }
    (cfg.STORAGE / "outputs/class-filter-validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "images_checked": len(records),
                "target_detections": sum(r["detections"] for r in records),
                "other_channels_above_threshold": other_total,
                "filter_matches": True,
            }
        )
    )


if __name__ == "__main__":
    main()
