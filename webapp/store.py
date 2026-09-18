"""On-disk store for finished grading reports, keyed by UUID.

The job queue in `jobs.py` is deliberately ephemeral — a result lives in
memory until it's fetched once, then the temp directory is deleted. That's
fine for "grade a card, look at it, close the tab", and useless for
everything else: there's no way back to a report you looked at yesterday,
and no link to send anyone.

So a finished job is also written here, under its job id, and stays until
deleted by hand. Layout:

    reports/<uuid>/report.json     the full report dict
    reports/<uuid>/meta.json       small summary, for listing without
                                   parsing (and loading) every report
    reports/<uuid>/images/<key>.png

Images are stored as files rather than inlined into report.json — a report
carries a dozen-plus 1500x2100 PNGs, and base64 in JSON would make the
listing read tens of megabytes to show a date and a grade.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Report ids come from uuid4().hex. Anything else is a path-traversal attempt
# or a typo, and both should 404 rather than reach the filesystem.
REPORT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

# Image keys become filenames, so they get the same treatment.
IMAGE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,64}$")

# Full-scale warps: stored, but served by URL rather than inlined, because
# they are tens of megabytes each and only wanted when someone zooms.
DETAIL_IMAGE_KEYS = frozenset({"front_detail", "back_detail"}) | frozenset(
    f"{side}_rotation_{index}" for side in ("front", "back") for index in range(6)
)


@dataclass
class ReportSummary:
    report_id: str
    created_at: str
    card_name: str | None
    set_name: str | None
    grade: int | None
    score: int | None
    # Set once a report has been aged out of full-scale storage: everything
    # that was measured is still here, but the zoom-in warps are gone.
    images_pruned: bool = False

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "created_at": self.created_at,
            "card_name": self.card_name,
            "set_name": self.set_name,
            "grade": self.grade,
            "score": self.score,
            "images_pruned": self.images_pruned,
        }


def is_valid_report_id(report_id: str) -> bool:
    return bool(REPORT_ID_PATTERN.match(report_id))


def report_dir(base_dir: Path, report_id: str) -> Path | None:
    """Resolved directory for a report, or None if the id is malformed."""
    if not is_valid_report_id(report_id):
        return None
    return base_dir / report_id


def _summarize(
    report_id: str, report: dict, created_at: str, images_pruned: bool = False
) -> ReportSummary:
    card_id = report.get("card_id") or {}
    grade_estimate = report.get("grade_estimate") or {}
    return ReportSummary(
        report_id=report_id,
        created_at=created_at,
        card_name=card_id.get("card_name"),
        set_name=card_id.get("set_name"),
        grade=grade_estimate.get("overall_grade_rounded"),
        score=grade_estimate.get("score"),
        images_pruned=images_pruned,
    )


def _read_meta(directory: Path) -> dict:
    try:
        return json.loads((directory / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


def save_report(base_dir: Path, report_id: str, report: dict, image_paths: dict[str, Path]) -> ReportSummary:
    """Persist one finished report. Overwrites any existing report at that id.

    Writes into a temp sibling directory and renames it into place, so a
    crash mid-write can't leave a half-written report that later reads as
    corrupt — the id either resolves to a complete report or to nothing.
    """
    if not is_valid_report_id(report_id):
        raise ValueError(f"not a valid report id: {report_id!r}")

    target = base_dir / report_id
    staging = base_dir / f".incoming-{report_id}"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "images").mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = _summarize(report_id, report, created_at)

    (staging / "report.json").write_text(json.dumps(report, indent=2))
    (staging / "meta.json").write_text(json.dumps(summary.to_dict(), indent=2))
    for key, path in image_paths.items():
        if IMAGE_KEY_PATTERN.match(key) and path.exists():
            shutil.copyfile(path, staging / "images" / f"{key}.png")

    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return summary


def _image_to_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.standard_b64encode(path.read_bytes()).decode("ascii")


def load_report(base_dir: Path, report_id: str) -> dict | None:
    """Return {"report": ..., "images": ...} in exactly the shape the job
    endpoint returns, so the frontend has one rendering path for a report it
    just produced and one it's revisiting."""
    directory = report_dir(base_dir, report_id)
    if directory is None or not (directory / "report.json").exists():
        return None

    report = json.loads((directory / "report.json").read_text())
    images_dir = directory / "images"
    images = {}
    if images_dir.is_dir():
        for image_path in sorted(images_dir.glob("*.png")):
            if image_path.stem in DETAIL_IMAGE_KEYS:
                # Tens of megabytes each. Addressed by URL and fetched only
                # when someone zooms; inlining them would put the full-scale
                # warp of both sides into every report response.
                images[image_path.stem] = f"/api/report/{report_id}/image/{image_path.stem}"
            else:
                images[image_path.stem] = _image_to_data_uri(image_path)
    return {
        "report": report,
        "images": images,
        "images_pruned": bool(_read_meta(directory).get("images_pruned")),
    }


def image_path(base_dir: Path, report_id: str, key: str) -> Path | None:
    """On-disk path for one of a report's images, or None if it isn't there.

    Both the id and the key are pattern-checked before touching the
    filesystem — the key becomes a filename, so it gets the same treatment
    as the id it sits under.
    """
    directory = report_dir(base_dir, report_id)
    if directory is None or not IMAGE_KEY_PATTERN.match(key):
        return None
    path = directory / "images" / f"{key}.png"
    return path if path.exists() else None


def update_report(base_dir: Path, report_id: str, report: dict, image_paths: dict[str, Path] | None = None) -> ReportSummary | None:
    """Rewrite a stored report in place, keeping its id, images and created_at.

    Unlike save_report this is an edit, not a fresh write: the report has
    already been looked at and possibly linked to, so its creation time and
    every image it isn't replacing have to survive. Images named in
    `image_paths` are overwritten; the rest are left alone.
    """
    directory = report_dir(base_dir, report_id)
    if directory is None or not (directory / "report.json").exists():
        return None

    meta = _read_meta(directory)
    created_at = meta.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")

    # An edit must not un-prune: the flag describes what's on disk, and
    # rewriting the report doesn't put the deleted warps back.
    summary = _summarize(report_id, report, created_at, bool(meta.get("images_pruned")))
    # Same staged-write reasoning as save_report, at file granularity: a
    # half-written report.json would make the whole report unreadable.
    report_path = directory / "report.json"
    staged = directory / ".report.json.incoming"
    staged.write_text(json.dumps(report, indent=2))
    staged.replace(report_path)
    (directory / "meta.json").write_text(json.dumps(summary.to_dict(), indent=2))

    images_dir = directory / "images"
    for key, path in (image_paths or {}).items():
        if IMAGE_KEY_PATTERN.match(key) and path.exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, images_dir / f"{key}.png")
    return summary


def list_reports(base_dir: Path, limit: int = 100) -> list[dict]:
    """Newest first. Reads only meta.json, never the reports themselves."""
    if not base_dir.is_dir():
        return []

    summaries = []
    for child in base_dir.iterdir():
        if not child.is_dir() or not is_valid_report_id(child.name):
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        try:
            summaries.append(json.loads(meta_path.read_text()))
        except json.JSONDecodeError:
            continue

    summaries.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return summaries[:limit]


# How many reports keep their full-scale images. Everything that was measured
# is kept forever — report.json, meta.json, and every image the report page
# shows inline. What ages out is only what exists to be zoomed into: the
# detail warps and the stored rotation scans.
#
# Sized against the disk this runs on. A photometric report is roughly 140MB
# with those images and 35MB without, so ten full reports is about 1.4GB and
# every report after that costs a quarter of a gigabyte less than it would
# have. Before this existed the reports directory had to be emptied by hand
# five times in one evening.
DEFAULT_KEEP_FULL_IMAGES = 10


def prune_report_images(base_dir: Path, keep_full: int = DEFAULT_KEEP_FULL_IMAGES) -> list[str]:
    """Drop the view-only images from every report but the newest `keep_full`.

    Returns the ids that were pruned this call. Already-pruned reports are
    skipped, so this is cheap to run after every save and safe to run twice.

    Deliberately not a delete: the report, its measurements and its overlays
    survive, so an old report still opens, still shows its grade and still
    re-grades from hand-placed borders. Only the zoom loses resolution, and
    it falls back to the canonical warp on its own.
    """
    if keep_full < 0:
        raise ValueError("keep_full must not be negative")
    if not base_dir.is_dir():
        return []

    dated = []
    for child in base_dir.iterdir():
        if not child.is_dir() or not is_valid_report_id(child.name):
            continue
        meta = _read_meta(child)
        if not meta:
            # No meta.json means a half-written or foreign directory. Leaving
            # it alone is the safe reading — it also can't be ordered, and
            # pruning by guessed age is how the wrong report loses its
            # images.
            continue
        dated.append((meta.get("created_at") or "", child, meta))

    dated.sort(key=lambda item: item[0], reverse=True)

    pruned = []
    for _, directory, meta in dated[keep_full:]:
        if meta.get("images_pruned"):
            continue
        images_dir = directory / "images"
        removed = False
        for key in DETAIL_IMAGE_KEYS:
            path = images_dir / f"{key}.png"
            if path.exists():
                path.unlink()
                removed = True
        # Flag it either way: a report with no detail warps to begin with is
        # already in its pruned state, and marking it stops this walking its
        # images every time a card is graded.
        meta["images_pruned"] = True
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        if removed:
            pruned.append(directory.name)
    return pruned


# How many reports keep the raw files that were uploaded to them.
#
# One, because the point is narrow: to be able to re-run an experiment on the
# card you just graded. Twice in one evening a fix could not be tested against
# the capture that motivated it — a detector bug on the two scans that had
# failed, and an alignment rewrite — because the originals were deleted with
# the job the moment its report appeared, and the operator's scanner software
# had not kept a copy either.
#
# Four, because a calibration session is several cards in a row and keeping
# only the newest would delete the originals of every card but the last one as
# it was uploaded — the exact loss this exists to prevent, arriving faster.
#
# Not many more, because they are large: a six-file set at 1200dpi is about
# 0.8GB as TIFF, so four sets is roughly 3GB. That fits on the machine this
# runs on with room to spare; a history of them would not.
DEFAULT_KEEP_RAW_UPLOADS = 4


def save_uploads(base_dir: Path, report_id: str, uploads: dict[str, Path]) -> int:
    """Keep the files that were uploaded to this report, under their own names.

    Best-effort and never fatal: these are a convenience for re-running work,
    and failing to store them must not cost the report they belong to.
    """
    directory = report_dir(base_dir, report_id)
    if directory is None or not directory.is_dir():
        return 0
    target = directory / "uploads"
    target.mkdir(parents=True, exist_ok=True)
    kept = 0
    for name, path in uploads.items():
        if not IMAGE_KEY_PATTERN.match(name) or not path.exists():
            continue
        shutil.copyfile(path, target / name)
        kept += 1
    return kept


def upload_paths(base_dir: Path, report_id: str) -> list[Path]:
    """The raw files kept for a report, if they are still there."""
    directory = report_dir(base_dir, report_id)
    if directory is None:
        return []
    uploads = directory / "uploads"
    return sorted(uploads.iterdir()) if uploads.is_dir() else []


def prune_report_uploads(base_dir: Path, keep: int = DEFAULT_KEEP_RAW_UPLOADS) -> list[str]:
    """Drop the raw uploads from every report but the newest `keep`."""
    if keep < 0:
        raise ValueError("keep must not be negative")
    if not base_dir.is_dir():
        return []

    dated = []
    for child in base_dir.iterdir():
        if not child.is_dir() or not is_valid_report_id(child.name):
            continue
        meta = _read_meta(child)
        if not meta:
            continue
        dated.append((meta.get("created_at") or "", child))
    dated.sort(key=lambda item: item[0], reverse=True)

    pruned = []
    for _, directory in dated[keep:]:
        uploads = directory / "uploads"
        if uploads.is_dir():
            shutil.rmtree(uploads, ignore_errors=True)
            pruned.append(directory.name)
    return pruned


def delete_report(base_dir: Path, report_id: str) -> bool:
    directory = report_dir(base_dir, report_id)
    if directory is None or not directory.is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return True
