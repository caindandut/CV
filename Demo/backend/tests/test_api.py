from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from app import main
from app import config as cfg
from app.database import Job
from app.schemas import JobOut


@pytest.fixture
def client():
    # No lifespan: these API failure tests deliberately do not start model/MySQL.
    return TestClient(main.app, raise_server_exceptions=False)


def test_missing_database_and_local_origin(client, monkeypatch):
    monkeypatch.setattr(main.worker, "database_ready", False)
    response = client.get("/api/jobs")
    assert response.status_code == 503
    assert "MySQL" in response.json()["error"]["message"]
    response = client.post("/api/jobs", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403


def test_model_missing_and_upload_size(client, monkeypatch):
    monkeypatch.setattr(main.worker, "database_ready", True)
    monkeypatch.setattr(main.worker.detector, "model", None)
    response = client.post(
        "/api/jobs",
        data={"media_type": "image"},
        files={"file": ("a.jpg", b"123", "image/jpeg")},
    )
    assert response.status_code == 503
    response = client.post(
        "/api/jobs", headers={"Content-Length": str(cfg.MAX_BYTES + 1024 * 1024 + 1)}
    )
    assert response.status_code == 413


def test_invalid_job_id_is_vietnamese(client):
    response = client.get("/api/jobs/not-a-uuid")
    assert response.status_code == 422
    assert "không hợp lệ" in response.json()["error"]["message"]


def test_output_requires_completion_and_supports_range(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.worker, "database_ready", True)
    job = SimpleNamespace(
        id=str(uuid4()),
        output_path="outputs/result.mp4",
        input_path="uploads/a.mp4",
        status="processing",
        media_type="video",
        input_mime="video/mp4",
        original_filename="a.mp4",
    )

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args):
            return job

    monkeypatch.setattr(main, "Session", FakeSession)
    monkeypatch.setattr(cfg, "STORAGE", tmp_path)
    response = client.get(f"/api/jobs/{job.id}/output")
    assert response.status_code == 409
    job.status = "completed"
    (tmp_path / "outputs").mkdir()
    (tmp_path / job.output_path).write_bytes(b"0123456789")
    response = client.get(f"/api/jobs/{job.id}/output", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206 and response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    response = client.get(f"/api/jobs/{job.id}/output?download=true")
    assert "attachment" in response.headers["content-disposition"]
    job.output_path = "../secret"
    assert client.get(f"/api/jobs/{job.id}/output").status_code == 404


def test_utc_serialization():
    job = Job(
        id=str(uuid4()),
        original_filename="a.jpg",
        media_type="image",
        input_path="uploads/a.jpg",
        input_mime="image/jpeg",
        status="queued",
        progress=0,
        model_identifier="model",
        imgsz=768,
        confidence_threshold=0.25,
        created_at=datetime(2026, 10, 13, 10),
    )
    assert (
        JobOut.model_validate(job)
        .model_dump(mode="json")["created_at"]
        .endswith("+00:00")
    )


def test_invalid_image_rejected_without_persistence(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.worker, "database_ready", True)
    monkeypatch.setattr(main.worker.detector, "model", object())
    monkeypatch.setattr(cfg, "STORAGE", tmp_path)
    (tmp_path / "uploads").mkdir()
    response = client.post(
        "/api/jobs",
        data={"media_type": "image"},
        files={"file": ("../fake.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 415
    assert list((tmp_path / "uploads").iterdir()) == []


def test_streamed_upload_limit(client, monkeypatch):
    monkeypatch.setattr(cfg, "MAX_BYTES", 10)
    boundary = "test-boundary"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.jpg"\r\n\r\n'
    ).encode()

    def chunks():
        yield body
        for _ in range(18):
            yield b"a" * 65536

    response = client.post(
        "/api/jobs",
        content=chunks(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413


def test_vietnamese_labels_are_unicode():
    assert cfg.LABELS == [
        "Xe máy",
        "Người đội mũ bảo hiểm",
        "Người không đội mũ bảo hiểm",
    ]
