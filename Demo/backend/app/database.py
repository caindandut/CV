import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Integer, Float, DateTime, JSON, Text
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

url = URL.create(
    "mysql+pymysql",
    username=os.getenv("DB_USER", "helmet_demo"),
    password=os.getenv("DB_PASSWORD", ""),
    host=os.getenv("DB_HOST", "127.0.0.1"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME", "helmet_demo"),
    query={"charset": "utf8mb4"},
)
engine = create_engine(
    url,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args={"connect_timeout": 3},
    pool_size=4,
    max_overflow=2,
)
Session = sessionmaker(engine, expire_on_commit=False)


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(10))
    input_path: Mapped[str] = mapped_column(String(255))
    output_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_mime: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    model_identifier: Mapped[str] = mapped_column(String(100))
    imgsz: Mapped[int] = mapped_column(Integer)
    confidence_threshold: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    detections: Mapped[list | None] = mapped_column(JSON, nullable=True)
