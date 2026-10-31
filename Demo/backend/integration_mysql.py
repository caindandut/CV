"""End-to-end acceptance test with real inference and MySQL.
Default: isolated MySQL on 3307. --configured-db explicitly uses Demo/.env.
Stop the normal backend first; this test starts one worker and tests restart recovery.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4
import httpx

ROOT = Path(__file__).resolve().parents[1]
CONFIGURED_DB = "--configured-db" in sys.argv
if not CONFIGURED_DB:
    os.environ.update(
        DB_HOST="127.0.0.1",
        DB_PORT="3307",
        DB_NAME="helmet_demo_acceptance",
        DB_USER="root",
        DB_PASSWORD="",
    )
from app import config as cfg
from app.database import Job, Session

BASE = "http://127.0.0.1:8001"


def wait_ready(client, process, require_model=True):
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(
                "Test backend exited; see .cache/integration-backend.log"
            )
        try:
            health = client.get("/api/health").json()
            if health["startup_complete"]:
                assert health["database_ready"], health
                if require_model:
                    assert health["model_ready"], health
                return health
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError("Backend startup timeout")


def main():
    subprocess.run(
        [sys.executable, str(ROOT / "backend/init_db.py")],
        cwd=ROOT,
        env=os.environ,
        check=True,
        capture_output=True,
    )
    logfile = (ROOT / ".cache/integration-backend.log").open("w", encoding="utf-8")

    def start():
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--app-dir",
                str(ROOT / "backend"),
                "--host",
                "127.0.0.1",
                "--port",
                "8001",
                "--workers",
                "1",
            ],
            cwd=ROOT,
            env=os.environ,
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )

    process = start()
    report = {
        "database": "configured MySQL80 from Demo/.env"
        if CONFIGURED_DB
        else "isolated MySQL 8.0.42 on port 3307; not the user's MySQL80 database"
    }
    client = httpx.Client(base_url=BASE, timeout=20)
    try:
        metadata_only = "--metadata-only" in sys.argv
        report["health"] = wait_ready(client, process, require_model=not metadata_only)
        if metadata_only:
            ids = []
            with Session.begin() as db:
                for status in ("queued", "processing"):
                    job_id = str(uuid4())
                    ids.append(job_id)
                    db.add(
                        Job(
                            id=job_id,
                            original_filename="TEST_METADATA_INTERRUPTED.jpg",
                            media_type="image",
                            input_path="uploads/test-only-no-file.jpg",
                            input_mime="image/jpeg",
                            status=status,
                            progress=10,
                            model_identifier="metadata-test-only",
                            imgsz=768,
                            confidence_threshold=0.25,
                        )
                    )
            page = client.get("/api/jobs?page=1&page_size=1").json()
            assert page["page_size"] == 1 and len(page["items"]) == 1
            if not report["health"]["model_ready"]:
                response = client.post(
                    "/api/jobs",
                    data={"media_type": "image"},
                    files={"file": ("a.jpg", b"test", "image/jpeg")},
                )
                assert response.status_code == 503
            process.terminate()
            process.wait(timeout=15)
            process = start()
            wait_ready(client, process, require_model=False)
            for job_id in ids:
                job = client.get(f"/api/jobs/{job_id}").json()
                assert job["status"] == "failed" and "gián đoạn" in job["error_message"]
            report.update(
                metadata_only=True,
                history_persisted=True,
                restart_recovery=True,
                pagination=True,
                inference_verified=False,
                note="No synthetic predictions; only explicitly marked test metadata was inserted.",
            )
            (cfg.STORAGE / "outputs/mysql-metadata-report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "mysql_metadata": True,
                        "history_persisted": True,
                        "restart_recovery": True,
                        "model_ready": report["health"]["model_ready"],
                    }
                )
            )
            return
        image = (
            cfg.ROOT.parent
            / "DataSet2-20260927T023713Z-1-001/DataSet2/valid/images/IMG_5031-2_MOV-34_jpg.rf.0262cdba4ccd2b1c9f516ad73c002788.jpg"
        )
        jobs = []
        for kind, source in [
            ("image", image),
            ("image", image),
            ("video", cfg.STORAGE / "uploads/smoke-input.mp4"),
        ]:
            with source.open("rb") as stream:
                response = client.post(
                    "/api/jobs",
                    data={"media_type": kind},
                    files={"file": (source.name, stream)},
                )
            assert response.status_code == 202, response.text
            jobs.append(response.json())
        progress = {j["id"]: [] for j in jobs}
        pending = {j["id"] for j in jobs}
        completed = {}
        deadline = time.monotonic() + 90
        while pending and time.monotonic() < deadline:
            # Proves health remains available while inference jobs run.
            assert client.get("/api/health").status_code == 200
            for job_id in list(pending):
                result = client.get(f"/api/jobs/{job_id}").json()
                progress[job_id].append(result["progress"])
                if result["status"] in {"completed", "failed"}:
                    assert result["status"] == "completed", result
                    completed[job_id] = result
                    pending.remove(job_id)
            time.sleep(0.1)
        assert not pending
        ordered = [completed[j["id"]] for j in jobs]
        for before, after in zip(ordered, ordered[1:]):
            assert before["finished_at"] <= after["started_at"]
        assert ordered[0]["detections"] and all(
            0 <= d["confidence"] <= 1 for d in ordered[0]["detections"]
        )
        assert ordered[0]["detections"][0]["label"] in cfg.LABELS
        for result in ordered:
            url = f"/api/jobs/{result['id']}/output"
            download = client.get(url + "?download=true")
            assert (
                download.status_code == 200
                and "attachment" in download.headers["content-disposition"]
            )
            suffix = ".png" if result["media_type"] == "image" else ".mp4"
            assert (
                download.content
                == (cfg.STORAGE / "outputs" / (result["id"] + suffix)).read_bytes()
            )
            ranged = client.get(url, headers={"Range": "bytes=0-31"})
            assert ranged.status_code == 206 and len(ranged.content) == 32
        assert client.get("/api/jobs?page=1&page_size=1").json()["page_size"] == 1
        assert client.get(f"/api/jobs/{uuid4()}").status_code == 404
        assert (
            client.post(
                "/api/jobs",
                data={"media_type": "image"},
                files={"file": ("fake.jpg", b"invalid", "image/jpeg")},
            ).status_code
            == 415
        )
        # Insert unfinished test-only rows, terminate, and verify startup recovery.
        recovery = []
        with Session.begin() as db:
            for status in ["queued", "processing"]:
                job_id = str(uuid4())
                recovery.append(job_id)
                db.add(
                    Job(
                        id=job_id,
                        original_filename="KIEM_THU_GIAN_DOAN.jpg",
                        media_type="image",
                        input_path="uploads/missing-test.jpg",
                        input_mime="image/jpeg",
                        status=status,
                        progress=10,
                        model_identifier="acceptance-test",
                        imgsz=768,
                        confidence_threshold=0.25,
                    )
                )
        process.terminate()
        process.wait(timeout=15)
        process = start()
        wait_ready(client, process)
        for job_id in recovery:
            result = client.get(f"/api/jobs/{job_id}").json()
            assert (
                result["status"] == "failed" and "gián đoạn" in result["error_message"]
            )
        for job in jobs:
            assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "completed"
        report.update(
            jobs=ordered,
            progress=progress,
            serial_execution=True,
            history_persisted=True,
            restart_recovery=True,
            invalid_file_rejected=True,
            range_download=True,
            api_responsive=True,
        )
        report["allowed_class_ids"] = list(cfg.INFERENCE_CLASS_IDS)
        report["checkpoint_compatibility"] = (
            "Original 80-channel checkpoint; fixed class filter 0,1,2; weights unchanged."
        )
        filename = (
            "configured-mysql-acceptance-report.json"
            if CONFIGURED_DB
            else "mysql-acceptance-report.json"
        )
        (cfg.STORAGE / "outputs" / filename).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "completed_jobs": len(jobs),
                    "serial_execution": True,
                    "history_persisted": True,
                    "restart_recovery": True,
                    "range_download": True,
                }
            )
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        client.close()
        logfile.close()


if __name__ == "__main__":
    main()
