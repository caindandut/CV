"""Read-only checkpoint diagnosis. Does not modify weights or produce labelled predictions."""

import hashlib
import json
import sys
from app import config as cfg
from app import scconv
import torch
from ultralytics import YOLO


def main():
    for name in (
        "GroupBatchnorm2d",
        "SRU",
        "CRU",
        "ScConv",
        "Bottleneck_SCConv",
        "C3k2_SCConv",
    ):
        setattr(sys.modules["__main__"], name, getattr(scconv, name))
    model = YOLO(str(cfg.MODEL_PATH), task="detect")
    net = model.model.cpu().eval()
    head = net.model[-1]
    weights = {
        name: list(tensor.shape)
        for name, tensor in net.state_dict().items()
        if any(
            name.endswith(f"{branch}.{scale}.2.weight")
            for branch in ["cv3", "one2one_cv3"]
            for scale in range(3)
        )
    }
    with torch.inference_mode():
        _, raw = net(torch.zeros(1, 3, cfg.IMGSZ, cfg.IMGSZ))
    shapes = {
        name: list(value["scores"].shape)
        for name, value in raw.items()
        if isinstance(value, dict) and "scores" in value
    }
    report = {
        "checkpoint_path": str(cfg.MODEL_PATH),
        "sha256": hashlib.sha256(cfg.MODEL_PATH.read_bytes()).hexdigest(),
        "checkpoint_version": model.ckpt.get("version"),
        "training_imgsz": model.ckpt.get("train_args", {}).get("imgsz"),
        "model_names": model.names,
        "model_nc_attribute": getattr(net, "nc", None),
        "head_nc": head.nc,
        "classification_weights": weights,
        "raw_score_shapes": shapes,
        "custom_block_count": sum(
            type(m).__name__ == "C3k2_SCConv" for m in net.modules()
        ),
        "matches_required_three_class_head": head.nc == 3
        and all(shape[0] == 3 for shape in weights.values()),
        "note": "Dataset and names describe three classes; head channels are checked separately against serialized weights and a raw forward pass.",
    }
    (cfg.ROOT / "checkpoint_inspection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
