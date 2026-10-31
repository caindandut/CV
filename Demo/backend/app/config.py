import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
(ROOT / ".cache/ultralytics/Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))


def location(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


MODEL_PATH = location(
    os.getenv(
        "MODEL_PATH", "../Report/helmet_yolo26s_scconv_768_dataset2/weights/best.pt"
    )
)
STORAGE = ROOT / "storage"
FONT_PATH = location(os.getenv("FONT_PATH", "assets/fonts/DejaVuSans.ttf"))
IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "768"))
CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "1024")) * 1024 * 1024
MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", "8"))
DEVICE = os.getenv("DEVICE", "auto")
assert IMGSZ > 0 and IMGSZ % 32 == 0, "DEFAULT_IMGSZ phải là bội số dương của 32"
assert 0 <= CONFIDENCE <= 1 and MAX_BYTES > 0 and QUEUE_SIZE > 0
NAMES = ["motorcycle", "rider_with_helmet", "rider_without_helmet"]
LABELS = ["Xe máy", "Người đội mũ bảo hiểm", "Người không đội mũ bảo hiểm"]
COLORS = ["#165BD8", "#00874E", "#CF2434"]
INFERENCE_CLASS_IDS = (0, 1, 2)
# Explicit user-approved compatibility for this unchanged checkpoint only.
COMPAT_CHECKPOINT_SHA256 = (
    "f6b6b0ab4bfe794f52b446107cf2b79d548070641bb805440c3f4a976cbc8f39"
)
