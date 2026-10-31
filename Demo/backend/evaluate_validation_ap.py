"""Full validation AP with a low candidate threshold; does not change deployed confidence."""

import argparse
import json
import shutil
from pathlib import Path
import yaml
from app import config as cfg
from app.detector import Detector
from ultralytics.models.yolo.detect.val import DetectionValidator

OUT = cfg.ROOT / "evaluation/model_audit_20261013"
DATA = cfg.ROOT.parent / "DataSet2-20260927T023713Z-1-001/DataSet2"


class ThreeClassValidator(DetectionValidator):
    def postprocess(self, predictions):
        results = super().postprocess(predictions)
        for index, result in enumerate(results):
            mask = (result["cls"] >= 0) & (result["cls"] <= 2)
            results[index] = {key: value[mask] for key, value in result.items()}
        return results


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output.resolve()
    detector = Detector()
    detector.load()
    if detector.model is None:
        raise RuntimeError(detector.error)
    # Keep all Ultralytics label-cache writes under Demo, never in the original dataset.
    copied = OUT / "standard_val_data"
    for kind in ("images", "labels"):
        target = copied / "valid" / kind
        target.mkdir(parents=True, exist_ok=True)
        for source in (DATA / "valid" / kind).iterdir():
            if not source.is_file() or (kind == "labels" and source.suffix != ".txt"):
                continue
            destination = target / source.name
            if destination.exists() and destination.samefile(source):
                destination.unlink()
            if not destination.exists():
                shutil.copy2(source, destination)
    config = {
        "path": str(copied),
        "train": str(DATA / "train/images"),
        "val": "valid/images",
        "names": cfg.NAMES,
        "nc": 3,
    }
    datafile = copied / "data.yaml"
    datafile.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    metrics = detector.model.val(
        validator=ThreeClassValidator,
        data=str(datafile),
        split="val",
        imgsz=cfg.IMGSZ,
        conf=0.001,
        iou=0.7,
        device="cpu",
        batch=1,
        workers=0,
        rect=False,
        plots=True,
        verbose=False,
        project=str(OUT),
        name="standard_ap",
        exist_ok=True,
        save_json=False,
        cache=False,
    )
    box = metrics.box
    report = {
        "model_identifier": detector.identifier,
        "images": 140,
        "split": "validation only",
        "candidate_confidence": 0.001,
        "deployed_confidence_unchanged": cfg.CONFIDENCE,
        "imgsz": cfg.IMGSZ,
        "iou_nms": 0.7,
        "classes": [0, 1, 2],
        "class_filter": "ThreeClassValidator discards IDs outside 0,1,2 after native postprocess, matching demo policy.",
        "metrics": {"mAP50": float(box.map50), "mAP50_95": float(box.map)},
        "per_class": [
            {
                "class_id": int(cls),
                "class_name": cfg.NAMES[int(cls)],
                "AP50": float(box.ap50[i]),
                "AP50_95": float(box.ap[i]),
            }
            for i, cls in enumerate(box.ap_class_index)
        ],
        "speed": metrics.speed,
        "warning": "Current validation has train overlap; these AP values do not establish independent-video generalization.",
    }
    (OUT / "validation_ap.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
