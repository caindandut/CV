import hashlib
import logging
import sys
import importlib.metadata
from . import config as cfg

log = logging.getLogger(__name__)


def validate_class_contract(model, checkpoint_sha256=None):
    names = model.names
    actual = (
        [names[i] for i in range(len(names))]
        if isinstance(names, dict)
        else list(names)
    )
    if actual != cfg.NAMES:
        raise ValueError("Thứ tự hoặc số lớp checkpoint không đúng cấu hình ba lớp")
    head_classes = getattr(model.model.model[-1], "nc", None)
    compatibility = (
        head_classes == 80 and checkpoint_sha256 == cfg.COMPAT_CHECKPOINT_SHA256
    )
    if head_classes != 3 and not compatibility:
        raise ValueError(
            f"Checkpoint có detection head {head_classes} lớp dù model.names khai báo 3 lớp. "
            "Chỉ hỗ trợ head 3 lớp hoặc checkpoint 80 kênh đã được xác nhận cho demo hiện tại."
        )


class Detector:
    def __init__(self):
        self.model = None
        self.error = "Chưa tải được mô hình"
        self.device = None
        self.identifier = None

    def load(self):
        try:
            if not cfg.MODEL_PATH.is_file():
                raise ValueError("Thiếu checkpoint được chỉ định bởi MODEL_PATH")
            if not cfg.FONT_PATH.is_file():
                raise ValueError("Thiếu font Unicode được chỉ định bởi FONT_PATH")
            if importlib.metadata.version("ultralytics") != "8.4.155":
                raise ValueError("Mô hình yêu cầu Ultralytics 8.4.155")
            import torch
            from . import scconv

            # Original notebook serialized these exact classes as __main__.*.
            for name in (
                "GroupBatchnorm2d",
                "SRU",
                "CRU",
                "ScConv",
                "Bottleneck_SCConv",
                "C3k2_SCConv",
            ):
                setattr(sys.modules["__main__"], name, getattr(scconv, name))
            from ultralytics import YOLO

            digest = hashlib.sha256(cfg.MODEL_PATH.read_bytes()).hexdigest()
            model = YOLO(str(cfg.MODEL_PATH), task="detect")
            validate_class_contract(model, checkpoint_sha256=digest)
            count = sum(
                type(m).__name__ == "C3k2_SCConv" for m in model.model.modules()
            )
            if (
                getattr(model.model, "task", model.task) != "detect"
                or type(model.model).__name__ != "DetectionModel"
                or count != 2
            ):
                raise ValueError(
                    "Checkpoint không phải detection YOLO26s với hai khối C3k2_SCConv yêu cầu"
                )
            training = model.ckpt.get("train_args", {})
            if (
                str(training.get("model", "")).replace("\\", "/").split("/")[-1]
                != "yolo26s.pt"
                or model.model.yaml.get("scale") != "s"
            ):
                raise ValueError(
                    "Checkpoint không có bằng chứng kiến trúc YOLO26s yêu cầu"
                )
            self.device = (
                ("0" if torch.cuda.is_available() else "cpu")
                if cfg.DEVICE == "auto"
                else cfg.DEVICE
            )
            model.to("cuda:0" if self.device == "0" else self.device)
            self.identifier = f"YOLO26s-SCConv:{digest}"
            self.model = model
            self.error = None
        except Exception as error:
            log.exception("Model initialization failed")
            self.model = None
            self.error = (
                str(error)
                if isinstance(error, ValueError)
                else "Không thể tải checkpoint SCConv; xem log backend và kiểm tra phiên bản thư viện"
            )

    def predict(self, frame, imgsz, threshold):
        if self.model is None:
            raise ValueError("Chưa tải được mô hình")
        results = self.model.predict(
            frame,
            imgsz=imgsz,
            conf=threshold,
            classes=list(cfg.INFERENCE_CLASS_IDS),
            iou=0.7,
            rect=False,
            device=self.device,
            verbose=False,
            save=False,
        )[0]
        return [
            {
                "class_id": int(box.cls.item()),
                "class_name": cfg.NAMES[int(box.cls.item())],
                "label": cfg.LABELS[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "xyxy": box.xyxy[0].tolist(),
            }
            for box in results.boxes
            if int(box.cls.item()) in cfg.INFERENCE_CLASS_IDS
        ]
