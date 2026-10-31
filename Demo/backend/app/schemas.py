from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, field_serializer


class Detection(BaseModel):
    class_id: int
    class_name: str
    label: str
    confidence: float
    xyxy: list[float]


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    original_filename: str
    media_type: Literal["image", "video"]
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int
    model_identifier: str
    imgsz: int
    confidence_threshold: float
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    detections: list[Detection] | None

    @field_serializer("created_at", "started_at", "finished_at")
    def utc_time(self, value):
        return value.replace(tzinfo=timezone.utc).isoformat() if value else None


class JobPage(BaseModel):
    items: list[JobOut]
    page: int
    page_size: int
    total: int


class Health(BaseModel):
    database_ready: bool
    model_ready: bool
    model_message: str
    model_identifier: str | None
    device: str | None
    font_ready: bool
    ffmpeg_ready: bool
    imgsz: int
    confidence_threshold: float
    max_upload_mb: int
    startup_complete: bool
