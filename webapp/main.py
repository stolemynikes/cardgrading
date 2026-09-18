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

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import centering as centering_stage
from pipeline import detect, regrade
from webapp import jobs, store

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
THRESHOLDS_PATH = REPO_ROOT / "calibration" / "thresholds.json"
TEMP_BASE = Path(tempfile.gettempdir()) / "cardgrading-webapp"

# Finished reports outlive their job and their temp directory. Under the repo
# rather than a temp dir on purpose: these are meant to still be there next
# week, and /tmp is not.
REPORTS_DIR = REPO_ROOT / "reports"

# No upload size limit. This runs on the machine doing the grading, and a
# flatbed scan is as big as it is: 58MB at 1200 dpi, more at 2400, times six
# per side for the photometric flow. Any fixed ceiling here is a number that
# eventually rejects a legitimate scan.

# Photometric stereo needs 3 light directions minimum; more than 6 buys
# nothing but upload time, since the solve is already overdetermined at 4.
MIN_PHOTOMETRIC_SCANS = 3
MAX_PHOTOMETRIC_SCANS = 6

app = FastAPI(title="Card Pre-Grader")


@app.on_event("startup")
def _startup() -> None:
    TEMP_BASE.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs.sweep_stale_temp_dirs(TEMP_BASE)


async def _read_and_validate_upload(upload: UploadFile) -> bytes:
    if not (upload.content_type or "").startswith("image/"):
        raise HTTPException(400, f"'{upload.filename}' doesn't look like an image (got {upload.content_type!r})")
    return await upload.read()


@app.post("/api/grade")
async def api_grade(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    photometric_front: list[UploadFile] = File(default_factory=list),
    photometric_back: list[UploadFile] = File(default_factory=list),
    dpi: float | None = Form(None),
    rotation: str = Form("cw"),
):
    # Which way the card was turned decides which light direction each frame
    # is solved against. An unrecognised value would quietly produce a wrong
    # normal map instead of an error, so it is rejected rather than defaulted.
    if rotation not in ("cw", "ccw"):
        raise HTTPException(400, f"rotation must be 'cw' or 'ccw', got {rotation!r}")

    for label, scans in (("photometric_front", photometric_front), ("photometric_back", photometric_back)):
        if scans and not MIN_PHOTOMETRIC_SCANS <= len(scans) <= MAX_PHOTOMETRIC_SCANS:
            raise HTTPException(
                400,
                f"{label} needs between {MIN_PHOTOMETRIC_SCANS} and {MAX_PHOTOMETRIC_SCANS} scans, got {len(scans)}",
            )
    if dpi is not None and dpi <= 0:
        raise HTTPException(400, "dpi must be positive")

    # Read and validate every upload into memory *before* creating a job or
    # any temp directory. Doing the validation interleaved with disk writes
    # (as an earlier version did) meant a rejected file — bad mime type, too
    # large — left an orphaned job-<id> directory and a permanently "queued"
    # job record behind forever, since neither is cleaned up outside
    # run_job's own try/finally, which never runs if validation fails first.
    uploads = [("front", front), ("back", back)]
    file_bytes = {name: await _read_and_validate_upload(upload) for name, upload in uploads}
    photometric_bytes = {
        side: [await _read_and_validate_upload(scan) for scan in scans]
        for side, scans in (("front", photometric_front), ("back", photometric_back))
    }

    job_id = jobs.create_job()
    job_root = TEMP_BASE / f"job-{job_id}"
    output_dir = job_root / "output"
    job_root.mkdir(parents=True, exist_ok=True)

    front_path = job_root / "front_upload"
    back_path = job_root / "back_upload"
    front_path.write_bytes(file_bytes["front"])
    back_path.write_bytes(file_bytes["back"])

    # Order matters: the scans are the card rotated a further 90 degrees on
    # the glass each time, and the solve maps the k-th scan to the k-th light
    # direction. The index in the filename keeps that order through the
    # upload, which multipart does not otherwise guarantee.
    photometric_paths: dict[str, list[Path] | None] = {"front": None, "back": None}
    for side, scans in photometric_bytes.items():
        if not scans:
            continue
        side_paths = []
        for index, data in enumerate(scans):
            scan_path = job_root / f"{side}_photometric_{index}"
            scan_path.write_bytes(data)
            side_paths.append(scan_path)
        photometric_paths[side] = side_paths

    thresholds = detect.load_thresholds(THRESHOLDS_PATH)

    # The files exactly as they arrived, kept with the report so the capture
    # that prompted a fix is still there to test the fix against.
    uploads = {"front_upload": front_path, "back_upload": back_path}
    for side, side_paths in photometric_paths.items():
        for index, scan_path in enumerate(side_paths or []):
            uploads[f"{side}_photometric_{index}"] = scan_path

    jobs.schedule(
        jobs.run_job(
            job_id,
            front_path,
            back_path,
            thresholds,
            output_dir,
            job_root,
            uploads=uploads,
            dpi=dpi,
            photometric_paths=(photometric_paths["front"], photometric_paths["back"]),
            rotation=rotation,
            reports_dir=REPORTS_DIR,
        )
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


@app.get("/api/reports")
async def api_reports(limit: int = 100):
    """Saved reports, newest first. Summaries only — a full report carries a
    dozen 1500x2100 PNGs and has no business in a listing."""
    return {"reports": store.list_reports(REPORTS_DIR, limit=max(1, min(limit, 500)))}


@app.get("/api/report/{report_id}")
async def api_report(report_id: str):
    """A saved report, in the same shape /api/job returns, so the frontend
    renders a revisited report through exactly the same code path as one it
    just produced."""
    saved = store.load_report(REPORTS_DIR, report_id)
    if saved is None:
        raise HTTPException(404, "report not found")
    return {"status": "done", "report_id": report_id, **saved}


@app.get("/api/centering-tolerances")
async def api_centering_tolerances():
    """PSA-style tolerance tables, so a client placing boundaries by hand can
    show the resulting grade as it drags without a round trip per pixel.

    Served rather than duplicated in the frontend: two copies of a grading
    table is two things to correct when one of them turns out to be wrong,
    which it already has once.
    """
    thresholds = detect.load_thresholds(THRESHOLDS_PATH)
    cfg, capture = thresholds["centering"], thresholds["capture"]
    return {
        "front": cfg["front_tolerances"],
        "back": cfg["back_tolerances"],
        "canonical_width_px": capture["canonical_width_px"],
        "canonical_height_px": capture["canonical_height_px"],
    }


@app.get("/api/report/{report_id}/image/{key}")
async def api_report_image(report_id: str, key: str):
    """One of a report's images as a file rather than a data URI.

    This is how the full-scale detail warps reach the browser: they're tens
    of megabytes, so they're fetched only when someone actually zooms, and
    they get a real cache header because — unlike the report JSON — a saved
    report's pixels never change under the same id.
    """
    path = store.image_path(REPORTS_DIR, report_id, key)
    if path is None:
        raise HTTPException(404, "image not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/report/{report_id}/centering")
async def api_set_centering(report_id: str, overrides: dict = Body(...)):
    """Replace a saved report's centering with hand-placed border widths.

    Widths arrive in canonical-warp pixels measured inward from the card edge
    — the same units the report already shows — so the client can read them
    straight off the overlay it's dragging lines on. Sides left out of the
    payload keep their detected values.

    The overlay images are redrawn from the stored aligned scans, and the
    grade, sub-grades and dings are re-derived, because all three are
    downstream of centering and would otherwise contradict it.
    """
    saved = store.load_report(REPORTS_DIR, report_id)
    if saved is None:
        raise HTTPException(404, "report not found")

    sides = {side: overrides[side] for side in ("front", "back") if overrides.get(side) is not None}
    if not sides:
        raise HTTPException(400, "give border widths for 'front', 'back', or both")

    thresholds = detect.load_thresholds(THRESHOLDS_PATH)
    capture = thresholds["capture"]
    width, height = capture["canonical_width_px"], capture["canonical_height_px"]
    for side, override in sides.items():
        if not isinstance(override, dict):
            raise HTTPException(400, f"{side}: expected an object of border widths")
        borders, edges = centering_stage.split_override(override)
        problem = centering_stage.validate_manual_borders(borders, width, height, edges)
        if problem is not None:
            raise HTTPException(400, f"{side}: {problem}")

    report = saved["report"]
    report["centering"] = centering_stage.regrade_with_manual_borders(
        report.get("centering") or {}, sides, thresholds
    )

    directory = store.report_dir(REPORTS_DIR, report_id)
    # Corner and edge crops are *sized from* the border widths, so moving a
    # boundary invalidates that whole stage. Re-measure it before scoring, or
    # the report ends up reporting corners measured against borders it no
    # longer claims.
    region_crops = regrade.recompute_corners_edges(
        report, regrade.load_aligned(directory / "images", tuple(sides)), thresholds
    )
    regrade.rescore(report, thresholds)

    with tempfile.TemporaryDirectory(prefix="centering-overlay-") as scratch:
        redrawn = _redraw_centering_overlays(directory, Path(scratch), sides)
        scratch_path = Path(scratch)
        for key, crop in region_crops.items():
            crop_path = scratch_path / f"{key}.png"
            if cv2.imwrite(str(crop_path), crop):
                redrawn[key] = crop_path
        if store.update_report(REPORTS_DIR, report_id, report, redrawn) is None:
            raise HTTPException(404, "report not found")

    updated = store.load_report(REPORTS_DIR, report_id)
    return {"status": "done", "report_id": report_id, **updated}


def _redraw_centering_overlays(directory: Path, scratch: Path, sides: dict) -> dict:
    """Redraw the overlay for each corrected side, over its stored warp.

    Returns {image_key: path} for store.update_report, written into `scratch`
    so a failure part-way leaves the stored report untouched. A side whose
    aligned scan is missing is skipped rather than failing the correction —
    the numbers are the measurement, the overlay only illustrates it.
    """
    written = {}
    for side, override in sides.items():
        aligned_path = directory / "images" / f"{side}_aligned.png"
        if not aligned_path.exists():
            continue
        image = cv2.imread(str(aligned_path))
        if image is None:
            continue
        borders, edges = centering_stage.split_override(override)
        overlay = centering_stage.draw_manual_overlay(image, borders, edges)
        out_path = scratch / f"{side}_centering_overlay.png"
        if cv2.imwrite(str(out_path), overlay):
            written[f"{side}_centering_overlay"] = out_path
    return written


@app.delete("/api/report/{report_id}")
async def api_delete_report(report_id: str):
    if not store.delete_report(REPORTS_DIR, report_id):
        raise HTTPException(404, "report not found")
    return {"deleted": report_id}


@app.get("/r/{report_id}")
async def report_permalink(report_id: str):
    """Deep link to a saved report. Serves the same single-page app, which
    reads the id back out of the URL and fetches it — no separate template to
    keep in sync with the one the live report already uses."""
    if not store.is_valid_report_id(report_id):
        raise HTTPException(404, "report not found")
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


class NoStoreCacheStaticFiles(StaticFiles):
    """Static files with `Cache-Control: no-cache` (revalidate every time).

    Without this, mobile Safari happily reuses a cached app.js across visits
    without revalidating — a real user graded a card with a weeks... days-old
    frontend against a newer backend and got "grade null" rendered where the
    new JS would have shown "couldn't measure". `no-cache` still allows ETag
    304s, so repeat loads stay cheap; it just forces the revalidation.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", NoStoreCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})
