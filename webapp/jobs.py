"""In-memory job tracking for the webapp.

Runs grade_card() in a worker thread (it's synchronous OpenCV/numpy code),
tracks stage progress via grade_card's on_stage callback, and owns each job's
temp-directory lifecycle. No persistence: a job's result lives in memory only
until it's fetched once or 15 minutes pass, whichever comes first.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grade import grade_card

MAX_CONCURRENT_JOBS = 2
JOB_TTL_SECONDS = 15 * 60

STAGE_MESSAGES = {
    "detect": "Detecting card…",
    "centering": "Measuring centering…",
    "corners_edges": "Analyzing corners & edges…",
    "surface": "Mapping surface…",
    "scoring": "Assembling grade…",
    "done": "Done",
}

_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_background_tasks: set[asyncio.Task] = set()


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued -> running -> done | error
    stage: str = ""
    message: str = "Queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    retrieved: bool = False


_JOBS: dict[str, Job] = {}


def sweep_stale_temp_dirs(base_dir: Path) -> None:
    """Startup safety net: remove leftover per-job temp dirs from a previous
    run that crashed before its finally block ran."""
    if not base_dir.exists():
        return
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.startswith("job-"):
            shutil.rmtree(child, ignore_errors=True)


def _evict_expired() -> None:
    now = time.time()
    expired = [
        jid
        for jid, job in _JOBS.items()
        if job.status in ("done", "error") and (job.retrieved or now - job.created_at > JOB_TTL_SECONDS)
    ]
    for jid in expired:
        _JOBS.pop(jid, None)


def create_job() -> str:
    _evict_expired()
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = Job(id=job_id)
    return job_id


def get_job(job_id: str) -> Job | None:
    _evict_expired()
    return _JOBS.get(job_id)


def get_job_and_mark_retrieved(job_id: str) -> Job | None:
    """Fetch a job. If it's finished, flag it for eviction on the *next*
    sweep rather than deleting it immediately — a client that polls again
    right after completion (a common race) still gets the result instead of
    a 404."""
    job = get_job(job_id)
    if job and job.status in ("done", "error"):
        job.retrieved = True
    return job


def schedule(coro) -> None:
    """asyncio.create_task() without keeping a reference risks the task being
    garbage-collected mid-flight — hold it in a module-level set until done."""
    task = asyncio.get_running_loop().create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _image_to_data_uri(path: Path) -> str:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _collect_images(output_dir: Path, has_surface: bool) -> dict[str, str]:
    """Base64-encode every overlay/debug image grade_card() wrote, keyed by a
    stable name the frontend knows about. Missing files (e.g. a failed
    capture-quality gate means most of these never got written) are skipped
    silently — the frontend already gets the reason from report.capture_quality.
    """
    candidates: dict[str, Path] = {
        "front_aligned": output_dir / "front_aligned.png",
        "back_aligned": output_dir / "back_aligned.png",
        "front_centering_overlay": output_dir / "front_centering_overlay.png",
        "back_centering_overlay": output_dir / "back_centering_overlay.png",
    }
    for side in ("front", "back"):
        for region in (
            "corner_top_left",
            "corner_top_right",
            "corner_bottom_right",
            "corner_bottom_left",
            "edge_top",
            "edge_right",
            "edge_bottom",
            "edge_left",
        ):
            candidates[f"{side}_{region}"] = output_dir / "corners_edges" / side / f"{region}.png"
    if has_surface:
        for side in ("front", "back"):
            candidates[f"{side}_surface_aligned"] = output_dir / "surface" / f"{side}_aligned.png"
            candidates[f"{side}_surface_defect_map"] = output_dir / "surface" / f"{side}_defect_map.png"
            candidates[f"{side}_surface_annotated"] = output_dir / "surface" / f"{side}_annotated.png"

    return {key: _image_to_data_uri(path) for key, path in candidates.items() if path.exists()}


async def run_job(
    job_id: str,
    front_path: Path,
    back_path: Path,
    thresholds: dict,
    output_dir: Path,
    job_root: Path,
    surface_paths: tuple[Path, Path] | None,
) -> None:
    job = _JOBS.get(job_id)
    if job is None:
        return

    async with _semaphore:
        job.status = "running"
        loop = asyncio.get_running_loop()

        def on_stage(name: str) -> None:
            job.stage = name
            job.message = STAGE_MESSAGES.get(name, name)

        try:
            report = await loop.run_in_executor(
                None,
                lambda: grade_card(
                    front_path,
                    back_path,
                    thresholds,
                    output_dir,
                    surface_paths=surface_paths,
                    verbose=False,
                    on_stage=on_stage,
                ),
            )
            images = _collect_images(output_dir, has_surface=surface_paths is not None)
            job.result = {"report": report, "images": images}
            job.status = "done"
            job.message = "Done"
        except Exception as e:  # noqa: BLE001 - report any failure back to the client, don't crash the server
            job.status = "error"
            job.error = str(e)
            job.message = f"Failed: {e}"
        finally:
            shutil.rmtree(job_root, ignore_errors=True)
