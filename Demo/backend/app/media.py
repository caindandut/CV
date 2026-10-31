import math
import os
import shutil
import warnings
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from . import config as cfg

Image.MAX_IMAGE_PIXELS = cfg.MAX_PIXELS


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Đường dẫn tệp không hợp lệ")
    return path


def ffmpeg_path():
    override = os.getenv("FFMPEG_PATH", "")
    if override:
        path = shutil.which(override) or (
            str(cfg.location(override)) if cfg.location(override).is_file() else None
        )
        return path
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def validate_media(path: Path, kind: str):
    if kind == "image":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    fmt = image.format
                    if fmt not in {"JPEG", "PNG"}:
                        raise ValueError()
                    if image.width * image.height > cfg.MAX_PIXELS:
                        raise ValueError()
                    image.verify()
                with Image.open(path) as image:
                    image.load()
            return "image/jpeg" if fmt == "JPEG" else "image/png"
        except Exception as error:
            raise ValueError(
                "Ảnh không hợp lệ, quá lớn hoặc không phải JPG/PNG có thể giải mã"
            ) from error
    with path.open("rb") as stream:
        header = stream.read(32)
    if (
        len(header) < 12
        or header[4:8] != b"ftyp"
        or not (
            header[8:11] == b"iso"
            or header[8:12] in {b"mp41", b"mp42", b"avc1", b"M4V ", b"dash"}
        )
    ):
        raise ValueError("Video phải là tệp MP4 hợp lệ")
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, frame = cap.read()
        if not ok or frame is None or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Không thể giải mã video hoặc đọc FPS/số frame")
        if frame.shape[0] * frame.shape[1] > cfg.MAX_PIXELS:
            raise ValueError("Kích thước frame vượt giới hạn cấu hình")
        return "video/mp4"
    finally:
        cap.release()


def read_image(path):
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def confidence_text(value):
    return f"{value * 100:.1f}%".replace(".", ",")


@lru_cache(maxsize=64)
def font(size):
    return ImageFont.truetype(str(cfg.FONT_PATH), size)


def annotation(bgr, detections):
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    w, h = image.size
    base = max(12, min(42, round(min(w, h) / 32)))
    thickness = max(2, round(min(w, h) / 280))
    for item in detections:
        cls = item["class_id"]
        color = cfg.COLORS[cls]
        x1, y1, x2, y2 = item["xyxy"]
        x1, x2 = sorted((max(0, min(w - 1, int(x1))), max(0, min(w - 1, int(x2)))))
        y1, y2 = sorted((max(0, min(h - 1, int(y1))), max(0, min(h - 1, int(y2)))))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
        label = f"{cfg.LABELS[cls]} · {confidence_text(item['confidence'])}"
        size = base
        while size > 1 and draw.textbbox((0, 0), label, font=font(size))[2] + 8 > w:
            size -= 1
        f = font(size)
        box = draw.textbbox((0, 0), label, font=f)
        tw, th = box[2] - box[0], box[3] - box[1]
        left = max(0, min(x1, w - tw - 8))
        top = max(0, min(y1 - th - 8, h - th - 8))
        draw.rectangle(
            (left, top, min(w - 1, left + tw + 8), min(h - 1, top + th + 8)), fill=color
        )
        draw.text((left + 4 - box[0], top + 4 - box[1]), label, fill="white", font=f)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
