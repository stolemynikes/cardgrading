"""FastAPI app: serves the frontend and the grading API.

Binds to localhost only (see README/webapp plan for remote-access setup via
Tailscale or a Cloudflare Tunnel — this file doesn't change based on which
front door is used). No accounts, no persistence: uploaded photos and
generated reports live only in a per-job temp directory for the duration of
one grading job.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import detect
from webapp import jobs

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
THRESHOLDS_PATH = REPO_ROOT / "calibration" / "thresholds.json"
TEMP_BASE = Path(tempfile.gettempdir()) / "cardgrading-webapp"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Card Pre-Grader")


@app.on_event("startup")
def _startup() -> None:
    TEMP_BASE.mkdir(parents=True, exist_ok=True)
    jobs.sweep_stale_temp_dirs(TEMP_BASE)


async def _read_and_validate_upload(upload: UploadFile) -> bytes:
    if not (upload.content_type or "").startswith("image/"):
        raise HTTPException(400, f"'{upload.filename}' doesn't look like an image (got {upload.content_type!r})")
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"'{upload.filename}' is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)")
    return data


@app.post("/api/grade")
async def api_grade(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    front_angled: UploadFile | None = File(None),
    back_angled: UploadFile | None = File(None),
):
    if (front_angled is None) != (back_angled is None):
        raise HTTPException(400, "front_angled and back_angled must be provided together, or not at all")

    # Read and validate every upload into memory *before* creating a job or
    # any temp directory. Doing the validation interleaved with disk writes
    # (as an earlier version did) meant a rejected file — bad mime type, too
    # large — left an orphaned job-<id> directory and a permanently "queued"
    # job record behind forever, since neither is cleaned up outside
    # run_job's own try/finally, which never runs if validation fails first.
    has_surface = front_angled is not None and back_angled is not None
    uploads = [("front", front), ("back", back)]
    if has_surface:
        uploads += [("front_angled", front_angled), ("back_angled", back_angled)]
    file_bytes = {name: await _read_and_validate_upload(upload) for name, upload in uploads}

    job_id = jobs.create_job()
    job_root = TEMP_BASE / f"job-{job_id}"
    output_dir = job_root / "output"
    job_root.mkdir(parents=True, exist_ok=True)

    front_path = job_root / "front_upload"
    back_path = job_root / "back_upload"
    front_path.write_bytes(file_bytes["front"])
    back_path.write_bytes(file_bytes["back"])

    surface_paths = None
    if has_surface:
        front_angled_path = job_root / "front_angled_upload"
        back_angled_path = job_root / "back_angled_upload"
        front_angled_path.write_bytes(file_bytes["front_angled"])
        back_angled_path.write_bytes(file_bytes["back_angled"])
        surface_paths = (front_angled_path, back_angled_path)

    thresholds = detect.load_thresholds(THRESHOLDS_PATH)

    jobs.schedule(
        jobs.run_job(job_id, front_path, back_path, thresholds, output_dir, job_root, surface_paths)
    )

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def api_job(job_id: str):
    job = jobs.get_job_and_mark_retrieved(job_id)
    if job is None:
        raise HTTPException(404, "job not found (expired or invalid id)")

    if job.status in ("queued", "running"):
        return {"status": job.status, "stage": job.stage, "message": job.message}
    if job.status == "error":
        return {"status": "error", "message": job.message}
    return {"status": "done", **job.result}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
