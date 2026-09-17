"""Stage 4: surface — defect detection and grading (scratches, print lines).

Two halves. The first builds a defect visibility map from a raking-light
photo: a high-pass for scratches, difference-of-Gaussians for print lines,
and a holo mask so iridescent foil isn't counted as damage.

The second (below `analyze_surface`) turns a defect signal into a sub-grade.
This used to require a vision model, because on a raking-light photo a
threshold genuinely cannot separate a scratch from holo sparkle. Photometric
stereo removed that premise — a surface-normal map has no albedo in it at
all — so the grade is now computed here, offline and reproducibly, with the
signal's provenance deciding how much weight it carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SurfaceResult:
    defect_area_pct: float
    holo_area_pct: float
    blob_count: int
    defect_map: np.ndarray
    annotated: np.ndarray
    longest_defect_px: int = 0

    def to_dict(self) -> dict:
        return {
            "defect_area_pct": round(self.defect_area_pct, 3),
            "holo_area_pct": round(self.holo_area_pct, 3),
            "blob_count": self.blob_count,
            "longest_defect_px": self.longest_defect_px,
            "note": "raw defect-map signal; see the surface grade for what it was scored as",
        }


def _normalize_robust(values: np.ndarray, high_percentile: float = 99.5) -> np.ndarray:
    """Scale to 0-255 by clipping to a high percentile rather than the true max.

    A plain min/max normalize is wrecked by a single outlier pixel — and this
    pipeline reliably produces one: the perspective-warp seam at the image's
    physical boundary (sub-pixel corner-detection error) has a far higher
    Laplacian/DoG response than any real surface defect, so a literal max
    would crush every genuine scratch toward zero.
    """
    hi = float(np.percentile(values, high_percentile))
    if hi <= 0:
        return np.zeros_like(values, dtype=np.uint8)
    return (np.clip(values, 0, hi) / hi * 255).astype(np.uint8)


def _defect_visibility_map(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """High-pass (Laplacian) catches scratches; difference-of-Gaussians catches
    print lines. Raking light makes both show up as sharp local intensity
    changes against an otherwise flat surface."""
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=cfg["laplacian_ksize"]))
    lap_norm = _normalize_robust(lap)

    blur1 = cv2.GaussianBlur(gray, (0, 0), cfg["dog_sigma1"])
    blur2 = cv2.GaussianBlur(gray, (0, 0), cfg["dog_sigma2"])
    dog = np.abs(blur1.astype(np.float32) - blur2.astype(np.float32))
    dog_norm = _normalize_robust(dog)

    return cv2.max(lap_norm, dog_norm)


def _local_variance(channel: np.ndarray, window: int) -> np.ndarray:
    ch = channel.astype(np.float32)
    mean = cv2.blur(ch, (window, window))
    mean_sq = cv2.blur(ch * ch, (window, window))
    return mean_sq - mean * mean


def _holo_mask(crop_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Flag likely holo-foil regions so iridescent sparkle doesn't get counted
    as a surface defect.

    The signal isn't "high saturation" — plenty of ordinary Pokemon card
    borders/print colors are just as saturated as foil. What's distinctive
    about holo foil is that its hue/saturation shifts at a small spatial
    scale (that's what "iridescent" means), whereas a printed solid color is
    locally uniform. So both checks here measure *local variance* — of the
    saturation channel, and of grayscale intensity — rather than an absolute
    level.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat_var = _local_variance(hsv[:, :, 1], 9)
    sat_mask = (sat_var >= cfg["holo_saturation_variance_threshold"]).astype(np.uint8) * 255

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    local_var = _local_variance(gray, 9)
    var_mask = (local_var >= cfg["holo_local_variance_threshold"]).astype(np.uint8) * 255

    combined = cv2.bitwise_or(sat_mask, var_mask)
    d = cfg["holo_mask_dilate_px"]
    return cv2.dilate(combined, np.ones((d, d), np.uint8))


def analyze_surface(crop_bgr: np.ndarray, thresholds: dict) -> SurfaceResult:
    cfg = thresholds["surface"]
    m = cfg["physical_edge_margin_px"]
    if m > 0:
        crop_bgr = crop_bgr[m:-m, m:-m]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    visibility = _defect_visibility_map(gray, cfg)
    holo_mask = _holo_mask(crop_bgr, cfg)

    visibility_masked = visibility.copy()
    visibility_masked[holo_mask > 0] = 0

    _, defect_mask = cv2.threshold(visibility_masked, cfg["defect_score_threshold"], 255, cv2.THRESH_BINARY)
    defect_mask = defect_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    min_area = cfg["min_defect_blob_area_px"]
    kept_mask = np.zeros_like(defect_mask)
    blob_count = 0
    defect_area = 0
    longest = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            kept_mask[labels == i] = 255
            blob_count += 1
            defect_area += area
            # Bounding-box diagonal: a scratch is long and thin, and length
            # is what caps the grade independently of covered area.
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            longest = max(longest, int((w * w + h * h) ** 0.5))

    total_area = gray.shape[0] * gray.shape[1]
    holo_area = int(np.count_nonzero(holo_mask))
    non_holo_area = max(1, total_area - holo_area)
    defect_area_pct = 100.0 * defect_area / non_holo_area
    holo_area_pct = 100.0 * holo_area / total_area

    annotated = crop_bgr.copy()
    holo_contours, _ = cv2.findContours(holo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, holo_contours, -1, (0, 215, 255), 1)  # amber = masked-out holo region
    defect_contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, defect_contours, -1, (0, 0, 255), 1)  # red = flagged defect

    defect_map = cv2.applyColorMap(kept_mask, cv2.COLORMAP_HOT)

    return SurfaceResult(defect_area_pct, holo_area_pct, blob_count, defect_map, annotated, longest)


# ---------------------------------------------------------------------------
# Deterministic surface grading.
#
# The vision-model step existed for one reason: a fixed pixel threshold on a
# raking-light photo can't tell a scratch from holo sparkle or from ordinary
# print detail, and that genuinely needs judgment about what a card is
# supposed to look like.
#
# Photometric stereo removes the premise. A surface-normal map contains no
# albedo at all — foil, artwork and print lines are gone before anything is
# measured, and what remains is shape. Against that signal a threshold is a
# measurement rather than a guess, so the grade can be computed here,
# offline, reproducibly, with no API key.
#
# The signal's provenance decides how much weight it carries:
#
#   photometric_relief   solved normals. Print cannot leak in. Graded.
#   raking_defect_map    holo-masked defect map from an angled photo. Real
#                        defects show, but so does some print detail, so it
#                        can only ever be an upper bound.
#   single_image_relief  the one-capture Card Vision approximation. Print
#                        demonstrably leaks into it (see tests), so it is
#                        measured and shown, never graded.
# ---------------------------------------------------------------------------

GRADED_SOURCES = frozenset({"photometric_relief", "raking_defect_map"})
UPPER_BOUND_SOURCES = frozenset({"raking_defect_map"})

SOURCE_NOTES = {
    "photometric_relief": "measured from solved surface normals — print and foil are absent from this signal",
    "raking_defect_map": "from the raking-light defect map — print detail can survive the holo mask, so treat it as an upper bound",
    "single_image_relief": "single-capture approximation — print leaks into this signal, so it is reported but not graded",
}


UNGRADED_REASONS = {
    "single_image_relief": (
        "graded only from a photometric or raking-light capture. This card was measured from a single "
        "image, where printed detail is indistinguishable from real surface relief — the render is shown, "
        "but grading it would be scoring the artwork. Scan the side four times, rotating the card 90 "
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
    upper_bound: bool
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
            "upper_bound": self.upper_bound,
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
        upper_bound=source in UPPER_BOUND_SOURCES,
        note=SOURCE_NOTES.get(source, ""),
    )
