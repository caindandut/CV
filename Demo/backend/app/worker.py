import logging
import math
import queue
import subprocess
import threading
from uuid import UUID
import cv2
from PIL import Image
from sqlalchemy import update
from filelock import FileLock
from . import config as cfg
from .database import Base, engine, Session, Job, now
from .detector import Detector
from .media import annotation, read_image, safe_path, ffmpeg_path

log = logging.getLogger(__name__)


class Worker:
    def __init__(self):
        self.detector = Detector()
        self.jobs = queue.Queue(maxsize=cfg.QUEUE_SIZE)
        self.stop = threading.Event()
        self.database_ready = False
        self.startup_complete = False
        self.thread = threading.Thread(
            target=self.run, name="inference-worker", daemon=True
        )
        self.lock = FileLock(str(cfg.STORAGE / ".worker.lock"))

    def run(self):
        try:
            self.lock.acquire(timeout=0)
        except Exception:
            self.detector.error = (
                "Đã có worker khác chạy; chỉ khởi động một process backend"
            )
            self.startup_complete = True
            return
        try:
            try:
                Base.metadata.create_all(engine)
                with Session.begin() as db:
                    db.execute(
                        update(Job)
                        .where(Job.status.in_(["queued", "processing"]))
                        .values(
                            status="failed",
                            finished_at=now(),
                            error_message="Xử lý bị gián đoạn do backend khởi động lại. Vui lòng tải tệp lên để chạy lại.",
                        )
                    )
                self.database_ready = True
                for temporary in (cfg.STORAGE / "temp").glob("*.avi"):
                    try:
                        UUID(temporary.stem)
                    except ValueError:
                        continue
                    temporary.unlink(missing_ok=True)
            except Exception:
                log.exception("MySQL startup failed; check DB_* configuration")
            self.detector.load()
            self.startup_complete = True
            while not self.stop.is_set():
                try:
                    job_id = self.jobs.get(timeout=0.3)
                except queue.Empty:
                    continue
                try:
                    self.process(job_id)
                finally:
                    self.jobs.task_done()
        finally:
            self.lock.release()

    def change(self, job_id, **values):
        with Session.begin() as db:
            db.execute(update(Job).where(Job.id == job_id).values(**values))

    def process(self, job_id):
        output = None
        try:
            with Session() as db:
                job = db.get(Job, job_id)
            if job is None:
                return
            self.change(job_id, status="processing", progress=1, started_at=now())
            source = safe_path(cfg.STORAGE, job.input_path)
            suffix = ".png" if job.media_type == "image" else ".mp4"
            relative = f"outputs/{job_id}{suffix}"
            output = safe_path(cfg.STORAGE, relative)
            detections = None
            if job.media_type == "image":
                frame = read_image(source)
                detections = self.detector.predict(
                    frame, job.imgsz, job.confidence_threshold
                )
                result = annotation(frame, detections)
                Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(output)
                with Image.open(output) as check:
                    check.verify()
            else:
                self.video(job, source, output)
            self.change(
                job_id,
                status="completed",
                progress=100,
                finished_at=now(),
                output_path=relative,
                detections=detections,
                error_message=None,
            )
        except Exception as error:
            log.exception("Job %s failed", job_id)
            if output:
                output.unlink(missing_ok=True)
            message = (
                str(error)
                if isinstance(error, ValueError)
                else "Không thể xử lý tệp. Kiểm tra cấu hình hoặc thử tệp khác; chi tiết đã ghi vào log backend."
            )
            try:
                self.change(
                    job_id, status="failed", finished_at=now(), error_message=message
                )
            except Exception:
                log.exception("Could not persist job failure %s", job_id)

    def video(self, job, source, output):
        cap = cv2.VideoCapture(str(source))
        raw = cfg.STORAGE / "temp" / f"{job.id}.avi"
        writer = None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            if not math.isfinite(fps) or fps <= 0 or total <= 0:
                raise ValueError("Video có FPS hoặc số frame không hợp lệ")
            # Pad before OpenCV encoding to avoid silently dropping odd final rows/columns.
            ew, eh = width + width % 2, height + height % 2
            writer = cv2.VideoWriter(
                str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (ew, eh)
            )
            if not writer.isOpened():
                raise ValueError("Không thể tạo tệp video trung gian")
            index, previous = 0, 0
            while True:
                if self.stop.is_set():
                    raise ValueError(
                        "Xử lý bị gián đoạn vì backend đang dừng. Vui lòng tải tệp lên để chạy lại."
                    )
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError("Kích thước frame video không nhất quán")
                detections = self.detector.predict(
                    frame, job.imgsz, job.confidence_threshold
                )
                frame = annotation(frame, detections)
                if ew != width or eh != height:
                    frame = cv2.copyMakeBorder(
                        frame, 0, eh - height, 0, ew - width, cv2.BORDER_REPLICATE
                    )
                writer.write(frame)
                index += 1
                progress = min(94, max(1, int(index / total * 94)))
                if progress != previous:
                    self.change(job.id, progress=progress)
                    previous = progress
            cap.release()
            writer.release()
            writer = None
            if index == 0 or index < total:
                raise ValueError("Video bị hỏng hoặc kết thúc trước số frame khai báo")
            self.change(job.id, progress=95)
            executable = ffmpeg_path()
            if not executable:
                raise ValueError("Thiếu FFmpeg để tạo video H.264")
            command = [
                executable,
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(raw),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-vf",
                "scale=in_range=auto:out_range=tv,format=yuv420p",
                "-color_range",
                "tv",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=1800,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode:
                log.error("FFmpeg: %s", completed.stderr.decode(errors="replace"))
                raise ValueError("FFmpeg không thể tạo video H.264; xem log backend")
            check = cv2.VideoCapture(str(output))
            try:
                ok, _ = check.read()
                if not ok or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != index:
                    raise ValueError("Video kết quả không đọc được đầy đủ")
            finally:
                check.release()
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            raw.unlink(missing_ok=True)
