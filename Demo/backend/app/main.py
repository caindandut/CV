import asyncio
import logging
import queue
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from . import config as cfg
from .database import Session, Job, engine, now
from .media import safe_path, validate_media, ffmpeg_path
from .schemas import JobOut, JobPage, Health
from .worker import Worker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger(__name__)
worker = Worker()


@asynccontextmanager
async def lifespan(app):
    for folder in ("uploads", "outputs", "temp"):
        (cfg.STORAGE / folder).mkdir(parents=True, exist_ok=True)
    worker.thread.start()
    yield
    worker.stop.set()
    # Do not release the process lock while inference still runs.
    await asyncio.to_thread(worker.thread.join, 30)
    engine.dispose()


app = FastAPI(title="Demo nhận diện mũ bảo hiểm", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_error(request, error):
    return JSONResponse(
        {"error": {"code": str(error.status_code), "message": error.detail}},
        status_code=error.status_code,
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request, error):
    return JSONResponse(
        {
            "error": {
                "code": "422",
                "message": "Dữ liệu yêu cầu không hợp lệ. Kiểm tra loại tệp, mã công việc và số trang.",
            }
        },
        status_code=422,
    )


@app.exception_handler(SQLAlchemyError)
async def database_error(request, error):
    log.error(
        "Database operation failed", exc_info=(type(error), error, error.__traceback__)
    )
    return JSONResponse(
        {
            "error": {
                "code": "503",
                "message": "Không kết nối được MySQL. Kiểm tra DB_* trong .env và khởi động lại backend.",
            }
        },
        status_code=503,
    )


@app.exception_handler(Exception)
async def unexpected_error(request, error):
    log.error(
        "Unexpected API error", exc_info=(type(error), error, error.__traceback__)
    )
    return JSONResponse(
        {
            "error": {
                "code": "500",
                "message": "Không thể hoàn tất yêu cầu; xem log backend.",
            }
        },
        status_code=500,
    )


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        # Local demo has no authentication; prevent cross-origin form submissions.
        origin = headers.get(b"origin", b"").decode()
        if (
            scope["method"] == "POST"
            and origin
            and origin
            not in {
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:8000",
                "http://127.0.0.1:8000",
            }
        ):
            return await JSONResponse(
                {
                    "error": {
                        "code": "403",
                        "message": "Nguồn gửi yêu cầu không được phép",
                    }
                },
                status_code=403,
            )(scope, receive, send)
        if scope["method"] != "POST" or scope["path"] != "/api/jobs":
            return await self.app(scope, receive, send)
        length = headers.get(b"content-length")
        limit = cfg.MAX_BYTES + 1024 * 1024
        if length and (not length.isdigit() or int(length) > limit):
            return await JSONResponse(
                {"error": {"code": "413", "message": "Tệp vượt dung lượng cho phép"}},
                status_code=413,
            )(scope, receive, send)
        used = 0

        async def limited_receive():
            nonlocal used
            message = await receive()
            used += len(message.get("body", b""))
            if used > limit:
                raise HTTPException(413, "Tệp vượt dung lượng cho phép")
            return message

        await self.app(scope, limited_receive, send)


app.add_middleware(BodyLimit)


def need_database():
    if not worker.database_ready:
        raise HTTPException(
            503, "MySQL chưa sẵn sàng. Kiểm tra cấu hình DB_* và khởi động lại backend."
        )


def find_job(db, job_id):
    job = db.get(Job, str(job_id))
    if job is None:
        raise HTTPException(404, "Không tìm thấy lượt nhận diện")
    return job


@app.get("/api/health", response_model=Health)
def health():
    db_ready = False
    try:
        with engine.connect() as db:
            db.execute(text("SELECT 1"))
        db_ready = worker.database_ready
    except SQLAlchemyError:
        pass
    return Health(
        database_ready=db_ready,
        model_ready=worker.detector.model is not None,
        model_message=worker.detector.error or "Mô hình đã sẵn sàng",
        model_identifier=worker.detector.identifier,
        device=worker.detector.device,
        font_ready=cfg.FONT_PATH.is_file(),
        ffmpeg_ready=bool(ffmpeg_path()),
        imgsz=cfg.IMGSZ,
        confidence_threshold=cfg.CONFIDENCE,
        max_upload_mb=cfg.MAX_BYTES // 1024 // 1024,
        startup_complete=worker.startup_complete,
    )


@app.post("/api/jobs", response_model=JobOut, status_code=202)
def create_job(
    file: UploadFile = File(...), media_type: Literal["image", "video"] = Form(...)
):
    need_database()
    if worker.detector.model is None:
        raise HTTPException(503, worker.detector.error or "Chưa tải được mô hình")
    if media_type == "video" and not ffmpeg_path():
        raise HTTPException(503, "Thiếu FFmpeg để xử lý video")
    if worker.jobs.full():
        raise HTTPException(429, "Hàng đợi đã đầy. Vui lòng thử lại sau.")
    original = (
        re.sub(
            r"[\x00-\x1f\x7f]",
            "",
            (file.filename or "tep").replace("\\", "/").split("/")[-1],
        )[:255]
        or "tep"
    )
    extension = Path(original).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png"} if media_type == "image" else {".mp4"}
    if extension not in allowed:
        raise HTTPException(415, "Chỉ hỗ trợ JPG, JPEG, PNG cho ảnh và MP4 cho video")
    job_id = str(uuid4())
    relative = f"uploads/{job_id}{extension}"
    path = safe_path(cfg.STORAGE, relative)
    created = False
    try:
        total = 0
        with path.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > cfg.MAX_BYTES:
                    raise HTTPException(413, "Tệp vượt dung lượng cho phép")
                out.write(chunk)
        mime = validate_media(path, media_type)
        job = Job(
            id=job_id,
            original_filename=original,
            media_type=media_type,
            input_path=relative,
            input_mime=mime,
            status="queued",
            progress=0,
            model_identifier=worker.detector.identifier,
            imgsz=cfg.IMGSZ,
            confidence_threshold=cfg.CONFIDENCE,
        )
        with Session.begin() as db:
            db.add(job)
        created = True
        try:
            worker.jobs.put_nowait(job_id)
        except queue.Full:
            worker.change(
                job_id,
                status="failed",
                finished_at=now(),
                error_message="Hàng đợi đã đầy. Vui lòng tải tệp lên để chạy lại.",
            )
            raise HTTPException(429, "Hàng đợi đã đầy. Vui lòng thử lại sau.")
        return JobOut.model_validate(job)
    except ValueError as error:
        raise HTTPException(415, str(error)) from error
    finally:
        file.file.close()
        if not created:
            path.unlink(missing_ok=True)


@app.get("/api/jobs", response_model=JobPage)
def list_jobs(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=50)):
    need_database()
    with Session() as db:
        total = db.scalar(select(func.count()).select_from(Job))
        jobs = db.scalars(
            select(Job)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
        return JobPage(
            items=[JobOut.model_validate(job) for job in jobs],
            page=page,
            page_size=page_size,
            total=total,
        )


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: UUID):
    need_database()
    with Session() as db:
        return JobOut.model_validate(find_job(db, job_id))


def serve(job_id, output, download):
    need_database()
    with Session() as db:
        job = find_job(db, job_id)
    relative = job.output_path if output else job.input_path
    if output and (job.status != "completed" or not relative):
        raise HTTPException(409, "Kết quả chưa sẵn sàng")
    try:
        path = safe_path(cfg.STORAGE, relative)
    except ValueError:
        raise HTTPException(404, "Tệp không hợp lệ")
    if not path.is_file():
        raise HTTPException(404, "Tệp không còn trong storage")
    mime = (
        ("image/png" if job.media_type == "image" else "video/mp4")
        if output
        else job.input_mime
    )
    filename = (f"ket-qua-{job.id}" + path.suffix) if output else job.original_filename
    return FileResponse(
        path,
        media_type=mime,
        filename=filename,
        content_disposition_type="attachment" if download else "inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@app.get("/api/jobs/{job_id}/input")
def input_file(job_id: UUID, download: bool = False):
    return serve(job_id, False, download)


@app.get("/api/jobs/{job_id}/output")
def output_file(job_id: UUID, download: bool = False):
    return serve(job_id, True, download)
