"""Stage 4: surface — turning a defect signal into a sub-grade.

This used to start from a raking-light photo: a high-pass for scratches, a
difference-of-Gaussians for print lines, and a holo mask so iridescent foil
wasn't counted as damage. It needed a vision model on top, because on a
raking-light photo a threshold genuinely cannot separate a scratch from holo
sparkle.

Photometric stereo removed both. A surface-normal map has no albedo in it at
all, so foil and artwork are gone before anything is measured, and what's
left is shape. The grade is computed here, offline and reproducibly, with the
signal's provenance deciding how much weight it carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# The signal's provenance decides how much weight it carries:
#
#   photometric_relief   solved normals. Print cannot leak in. Graded.
#   single_image_relief  the one-capture Card Vision approximation. Print
#                        demonstrably leaks into it (see tests), so it is
#                        measured and shown, never graded.
# ---------------------------------------------------------------------------

GRADED_SOURCES = frozenset({"photometric_relief"})

SOURCE_NOTES = {
    "photometric_relief": "measured from solved surface normals — print and foil are absent from this signal",
    "single_image_relief": "single-capture approximation — print leaks into this signal, so it is reported but not graded",
}


UNGRADED_REASONS = {
    "single_image_relief": (
        "graded only from a photometric capture. This card was measured from a single image, where "
        "printed detail is indistinguishable from real surface relief — the render is shown, but "
        "grading it would be scoring the artwork. Scan the side four times, rotating the card 90 "
        "degrees on the glass each time, to get a gradeable signal."
    ),
}


@dataclass
class SurfaceGrade:
    grade: int | None  # None when the signal isn't trustworthy enough to grade
    defect_area_pct: float
    defect_count: int
    longest_defect_px: int
    source: str
    note: str
    # Why there's no grade, in the report's own words. A bare "n/a" reads the
    # same whether the capture can't support a grade or the card defeated the
    # measurement, and those want different things done about them.
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "reason": self.reason,
            "defect_area_pct": round(self.defect_area_pct, 3),
            "defect_count": self.defect_count,
            "longest_defect_px": self.longest_defect_px,
            "source": self.source,
            "note": self.note,
        }


def relief_defect_stats(relief: np.ndarray, cfg: dict) -> tuple[float, int, int, np.ndarray]:
    """Find defects in a Card Vision relief render.

    The render is centred on mid-gray for a flat surface and has already been
    soft-thresholded at the measured noise floor, so anything still deviating
    is structure. Returns (area %, count, longest extent in px, mask).
    """
    deviation = np.abs(relief.astype(np.int16) - 128).astype(np.uint8)
    _, mask = cv2.threshold(deviation, cfg["relief_defect_threshold"], 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = cfg["min_defect_blob_area_px"]
    kept = np.zeros_like(mask)
    count = 0
    area = 0
    longest = 0
    for i in range(1, num_labels):
        blob_area = stats[i, cv2.CC_STAT_AREA]
        if blob_area < min_area:
            continue
        kept[labels == i] = 255
        count += 1
        area += blob_area
        # A scratch is long and thin; its bounding box diagonal separates it
        # from a dust speck of the same area far better than area alone.
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        longest = max(longest, int((w * w + h * h) ** 0.5))

    total = relief.shape[0] * relief.shape[1]
    return (100.0 * area / total if total else 0.0), count, longest, kept


def _grade_from_bands(value: float, bands: list) -> int:
    """bands is [[max_value, grade], ...] ascending by max_value."""
    for max_value, grade in bands:
        if value <= max_value:
            return int(grade)
    return int(bands[-1][1])


def grade_surface(
    defect_area_pct: float, defect_count: int, longest_defect_px: int, source: str, thresholds: dict
) -> SurfaceGrade:
    """Turn defect statistics into a surface sub-grade."""
    cfg = thresholds["surface"]
    graded = source in GRADED_SOURCES

    grade = None
    if graded:
        grade = _grade_from_bands(defect_area_pct, cfg["grade_bands"])
        # A single long scratch can cover very little area and still be the
        # first thing a grader sees, so length caps the grade independently
        # of how much of the card it covers.
        for min_length, capped in cfg["scratch_length_caps"]:
            if longest_defect_px >= min_length:
                grade = min(grade, int(capped))

    return SurfaceGrade(
        grade=grade,
        reason=None if graded else UNGRADED_REASONS.get(source, "this signal isn't trustworthy enough to grade"),
        defect_area_pct=defect_area_pct,
        defect_count=defect_count,
        longest_defect_px=longest_defect_px,
        source=source,
        note=SOURCE_NOTES.get(source, ""),
    )
