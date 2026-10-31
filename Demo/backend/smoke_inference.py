"""Real model/media smoke check, independent of MySQL. Writes only under Demo/storage."""

import json
import subprocess
from types import SimpleNamespace
from uuid import uuid4
import cv2
from PIL import Image
from app import config as cfg
from app.media import annotation, read_image, validate_media, ffmpeg_path
from app.worker import Worker


def main():
    root = cfg.ROOT.parent
    source = (
        root
        / "DataSet2-20260927T023713Z-1-001/DataSet2/valid/images/IMG_5031-2_MOV-34_jpg.rf.0262cdba4ccd2b1c9f516ad73c002788.jpg"
    )
    worker = Worker()
    worker.detector.load()
    if worker.detector.model is None:
        raise RuntimeError(worker.detector.error)
    frame = read_image(source)
    detections = worker.detector.predict(frame, cfg.IMGSZ, cfg.CONFIDENCE)
    result = annotation(frame, detections)
    image_out = cfg.STORAGE / "outputs/smoke-image.png"
    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(image_out)
    assert validate_media(image_out, "image") == "image/png"
    # Three frames from one real image: technical pipeline test, not a real road clip.
    clip_id = str(uuid4())
    temp = cfg.STORAGE / "temp" / f"{clip_id}-input.avi"
    video_in = cfg.STORAGE / "uploads/smoke-input.mp4"
    video_out = cfg.STORAGE / "outputs/smoke-video.mp4"
    small = cv2.resize(frame, (640, 360))
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"MJPG"), 2, (640, 360))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(small)
    writer.release()
    executable = ffmpeg_path()
    subprocess.run(
        [
            executable,
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(temp),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video_in),
        ],
        check=True,
        capture_output=True,
    )
    temp.unlink()
    validate_media(video_in, "video")
    progress = []
    worker.change = lambda job_id, **values: progress.append(values)
    worker.video(
        SimpleNamespace(
            id=clip_id, imgsz=cfg.IMGSZ, confidence_threshold=cfg.CONFIDENCE
        ),
        video_in,
        video_out,
    )
    inspection = subprocess.run(
        [executable, "-i", str(video_out), "-hide_banner"],
        capture_output=True,
        text=True,
    ).stderr
    assert "h264" in inspection and "Audio: aac" in inspection
    assert "yuv420p" in inspection
    assert not (cfg.STORAGE / "temp" / f"{clip_id}.avi").exists()
    report = {
        "model_identifier": worker.detector.identifier,
        "device": worker.detector.device,
        "image_source": str(source),
        "image_output": str(image_out),
        "detections": detections,
        "video_output": str(video_out),
        "video_note": "Three repeated frames from a real validation image, with generated audio, solely for pipeline verification.",
        "video_h264": True,
        "video_audio_aac": True,
        "progress": progress,
        "imgsz": cfg.IMGSZ,
        "confidence_threshold": cfg.CONFIDENCE,
    }
    (cfg.STORAGE / "outputs/smoke-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "image_detections": len(detections),
                "video_h264": True,
                "audio_preserved": True,
                "device": worker.detector.device,
            }
        )
    )


if __name__ == "__main__":
    main()
