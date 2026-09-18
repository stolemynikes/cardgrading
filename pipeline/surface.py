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

from dataclasses import dataclass, field, replace

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
    # Every mark found, named and measured. Kept because "surface 4" is not
    # an answer anyone can check — "a 2.7mm crease at the bottom-left corner
    # caps this at 4" is.
    defects: list = field(default_factory=list)
    # What actually set the grade: the defect kind that capped it, or the
    # area band when nothing capped it harder.
    limited_by: str | None = None
    # Whose standard produced this grade. Recorded because the three published
    # rubrics disagree by several grades on the same defect, so a surface
    # grade without a grader attached is not a complete statement.
    grader: str | None = None
    # The same marks priced by every configured standard. Reference only.
    by_grader: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "reason": self.reason,
            "defect_area_pct": round(self.defect_area_pct, 3),
            "defect_count": self.defect_count,
            "longest_defect_px": self.longest_defect_px,
            "source": self.source,
            "note": self.note,
            "limited_by": self.limited_by,
            "grader": self.grader,
            "by_grader": self.by_grader,
            # Worst first, and capped: a card can carry hundreds of pits and
            # the report has no use for a list that long.
            "defects": [d.to_dict() for d in sorted(self.defects, key=lambda d: d.grade_cap)[:12]],
            "defect_kinds": {
                kind: sum(1 for d in self.defects if d.kind == kind)
                for kind in sorted({d.kind for d in self.defects})
            },
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


# ---------------------------------------------------------------------------
# Defect classification, and the grade ceilings the real graders publish.
#
# Area alone cannot grade a card. A crease and a scuff of the same area are
# several grades apart at every service, because what caps a grade is *what
# the defect is*, not how much of the card it covers. Measured on one card
# that had been creased and scratched deliberately, the crease covered less
# area than the scratches and is the more serious defect by four grades.
#
# The ladder below is taken from TAG's published rubric, which is the closest
# published standard to what this pipeline measures — they grade from
# photometric stereo too — cross-checked against PSA's own grade definitions:
#
#   TAG 10    a light scratch that does not penetrate the gloss; a small pit
#   TAG 9     2-3 small pits, or a longer scratch still not penetrating gloss
#   TAG 8.5   a scratch that penetrates the gloss; very minor scuffing
#   TAG 7.5   a very minor dent
#   TAG 6     a larger dent; significant scuffing in a single area
#   TAG 5     wrinkles on the front
#   TAG 4.5   a minor crease, breaking the stock
#   TAG 4     a wrinkle spanning about half the card
#   TAG 3     a wrinkle spanning about three-quarters of the card
#   TAG 2     full-length wrinkles or heavier creases
#
# PSA agrees on the part that matters most: a light crease caps a card at 5-6,
# and a crease running the full width is "usually an automatic 1".
#
# Grades here are integers on the PSA scale, so TAG's half grades round down —
# a minor crease caps at 4, not 4.5. Rounding down rather than up because this
# tool exists to decide whether a card is worth submitting, and the expensive
# error is telling someone a creased card will come back a 9.
#
# Sources:
#   https://taggrading.com/pages/rubric
#   https://www.psacard.com/gradingstandards

# Physical geometry of each defect kind, in millimetres on the card.
#
# These are shape thresholds, not taste. A crease is a fold in the cardstock:
# the stock has thickness, so a fold has width — measured on a real creased
# card, 1.38mm against 0.14-0.24mm for the scratches on the same card. A
# scratch is the track of something dragged across the surface, so it is as
# narrow as whatever drew it and many times longer than it is wide.
MAX_SCRATCH_WIDTH_MM = 0.5
MIN_CREASE_WIDTH_MM = 0.8
MIN_SCRATCH_ELONGATION = 6.0
MIN_DEFECT_LENGTH_MM = 0.8

# A crease is a fold, so it runs along a line; a dent is a depression, so it
# does not. Both are wide and deep, and that is all that separates them here.
#
# This is the weakest threshold in the file and the most expensive to get
# wrong — a dent caps a card at 7 and a crease at 4. The real creased card
# measured 1.96 against 1.24 for a compact mark of the same depth, which is a
# thin margin resting on one card. A short crease and a dent are genuinely
# hard to tell apart from shape alone; the honest distinction is whether the
# stock is broken, which needs the back of the card to confirm.
MIN_CREASE_ELONGATION = 1.8

# How deep a defect reads in the relief render, where 0 is flat and 127 is the
# top of the scale. A scratch that only burnishes the gloss barely disturbs the
# surface normal; one that cuts into the stock disturbs it a lot. Measured on
# the same card: scratches 29-33, the crease 75.
GLOSS_PENETRATION_DEPTH = 45.0
DEEP_GOUGE_DEPTH = 70.0

# How long a scratch has to be before it stops being a hairline, in
# millimetres — regardless of how deep it is.
#
# Both published ladders separate scratches by extent as well as depth: TAG
# allows "a light scratch" at 10 and "a longer scratch not penetrating the
# gloss" at 9, and PSA separates a hairline (visible only at oblique angles,
# capping at 9) from a visible scratch (readable under normal light, capping
# at 7-8). Depth alone called a 12mm gouge cut with scissors a light scratch,
# because a long shallow-angled cut disturbs the surface normal no more per
# pixel than a hairline does — it just does it for far longer.
#
# 5mm is the point at which a mark stops being something you have to hunt for.
# It is the weakest-supported number in this file: the rubrics describe
# "longer" without giving a length, and the only cards measured against it are
# one pair.
VISIBLE_SCRATCH_LENGTH_MM = 5.0
LONG_SCRATCH_LENGTH_MM = 15.0


@dataclass
class Defect:
    """One measured mark on the card, with the shape that decides what it is.

    Carries no grade of its own. What a mark costs depends on whose standard
    is being applied, and the three published standards disagree by as much as
    four grades on the same defect — a deep scratch is a 5 to PSA and a 4 to
    CGC, a full-length crease is a 1 to PSA and a 2 to both TAG and CGC.
    """

    kind: str  # crease | wrinkle | scratch | dent | pit
    severity: str  # the key a grader's ceiling table is indexed by
    length_mm: float
    width_mm: float
    depth: float
    area_px: int
    centre_mm: tuple[float, float]
    # Where this mark sits on the card, [x, y, w, h] as fractions of the
    # canonical warp — the same units the corner and edge crops use, so the
    # report can draw it back onto the card. Without it a surface ding could
    # say a scratch was 10mm long and never show anyone where to look.
    box: list[float] | None = None
    grade_cap: int = 10  # filled in per grader by grade_surface

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "length_mm": round(self.length_mm, 2),
            "width_mm": round(self.width_mm, 2),
            "depth": round(self.depth, 1),
            "area_px": int(self.area_px),
            "centre_mm": [round(self.centre_mm[0], 1), round(self.centre_mm[1], 1)],
            # float(), not just round(): these come from numpy and the report
            # is written to disk as JSON, which rejects numpy scalars outright.
            "box": None if self.box is None else [round(float(v), 4) for v in self.box],
            "grade_cap": self.grade_cap,
        }


def _classify(length_mm: float, width_mm: float, elongation: float, depth: float) -> tuple[str, str]:
    """What this mark is: its kind, and the severity key the rubrics ladder on.

    Naming only. What each name costs is a question about a grading standard,
    not about the card, and the three standards disagree — see GRADERS.
    """
    if width_mm >= MIN_CREASE_WIDTH_MM and depth >= GLOSS_PENETRATION_DEPTH:
        if elongation >= MIN_CREASE_ELONGATION:
            return "crease", "crease"
        return "dent", "dent"       # deep but not a fold
    if width_mm >= MIN_CREASE_WIDTH_MM:
        # Wide but shallow: the stock is deformed without being broken, which
        # is what the rubrics call a wrinkle rather than a crease.
        return "wrinkle", "wrinkle"
    if elongation >= MIN_SCRATCH_ELONGATION and width_mm <= MAX_SCRATCH_WIDTH_MM:
        if depth >= DEEP_GOUGE_DEPTH or length_mm >= LONG_SCRATCH_LENGTH_MM:
            return "scratch", "scratch_deep"    # cuts into the stock, or runs a long way
        if depth >= GLOSS_PENETRATION_DEPTH or length_mm >= VISIBLE_SCRATCH_LENGTH_MM:
            return "scratch", "scratch_gloss"   # visible without hunting for it
        return "scratch", "scratch_light"       # a hairline
    if depth >= GLOSS_PENETRATION_DEPTH:
        return "dent", "dent"
    return "pit", "pit"


def crease_ceiling(span: float, ladder: list) -> int:
    """The ceiling a crease of this span imposes, from a grader's own ladder.

    `span` is the crease's length as a fraction of the card's long edge, which
    is the unit all three standards describe creases in — "spanning about half
    the card", "travelling across the surface from edge to edge".
    """
    for max_span, ceiling in sorted(ladder, key=lambda entry: -entry[0]):
        if span >= max_span:
            return int(ceiling)
    return int(sorted(ladder, key=lambda entry: entry[0])[0][1])


def ceiling_for(defect: "Defect", grader: dict, card_length_mm: float) -> int:
    """The best grade this grader would give a card carrying this one mark."""
    if defect.kind == "crease":
        span = defect.length_mm / max(card_length_mm, 1e-6)
        return crease_ceiling(span, grader.get("crease_span_ceilings") or [[0.0, 4]])
    return int((grader.get("ceilings") or {}).get(defect.severity, 10))


def classify_defects(relief: np.ndarray, cfg: dict, px_per_mm: float) -> list[Defect]:
    """Every mark in the relief render, measured and named.

    Shape comes from the principal axes of each blob rather than its bounding
    box: a bounding box around a diagonal scratch is nearly square and says
    the scratch is as wide as it is long, which is how a scratch would get
    classified as a crease.
    """
    deviation = np.abs(relief.astype(np.int16) - 128).astype(np.uint8)
    _, mask = cv2.threshold(deviation, cfg["relief_defect_threshold"], 255, cv2.THRESH_BINARY)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    height, width = relief.shape[:2]

    min_area = cfg["min_defect_blob_area_px"]
    defects = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(labels == i)
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        centred = points - points.mean(axis=0)
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        # Extent is the span of the projection onto each principal axis, read
        # between the 1st and 99th percentile. A multiple of sigma was tried
        # first and reads 15% long on anything bar-shaped — sigma of a uniform
        # bar is length/sqrt(12) — which was enough to push a 40mm crease over
        # the half-the-card line and cost the card a grade.
        projected = centred @ axes.T
        spans = np.percentile(projected, 99, axis=0) - np.percentile(projected, 1, axis=0)
        length_mm = float(spans[0]) / px_per_mm
        width_mm = max(float(spans[1]) / px_per_mm, 1e-3)
        if length_mm < MIN_DEFECT_LENGTH_MM:
            continue
        depth = float(deviation[ys, xs].mean())
        kind, severity = _classify(length_mm, width_mm, length_mm / width_mm, depth)
        defects.append(
            Defect(
                kind=kind,
                severity=severity,
                length_mm=length_mm,
                width_mm=width_mm,
                depth=depth,
                area_px=int(stats[i, cv2.CC_STAT_AREA]),
                centre_mm=(float(centroids[i][0]) / px_per_mm, float(centroids[i][1]) / px_per_mm),
                box=[
                    stats[i, cv2.CC_STAT_LEFT] / width,
                    stats[i, cv2.CC_STAT_TOP] / height,
                    stats[i, cv2.CC_STAT_WIDTH] / width,
                    stats[i, cv2.CC_STAT_HEIGHT] / height,
                ],
            )
        )
    return defects


def _grade_from_bands(value: float, bands: list) -> int:
    """bands is [[max_value, grade], ...] ascending by max_value."""
    for max_value, grade in bands:
        if value <= max_value:
            return int(grade)
    return int(bands[-1][1])


def grade_surface(
    defect_area_pct: float,
    defect_count: int,
    longest_defect_px: int,
    source: str,
    thresholds: dict,
    defects: list | None = None,
    grader: str | None = None,
    card_length_mm: float | None = None,
) -> SurfaceGrade:
    """Turn measured defects into a surface sub-grade.

    Two things set the grade and the worse one wins.

    The area band answers "how marked up is this card overall", which is what
    separates a clean card from a scuffed one. On its own it is not a grading
    standard: no service grades by coverage, because a crease and a scuff of
    the same area are several grades apart everywhere.

    So each defect also carries its own ceiling, from the published rubrics
    (see the ladder above). The worst ceiling on the card caps the grade,
    which is how a single 2.7mm crease takes a card that is otherwise clean
    down to a 4 — and is what every real grader does.
    """
    cfg = thresholds["surface"]
    graded = source in GRADED_SOURCES
    defects = defects or []
    grader_key = grader or cfg.get("primary_grader", "psa")
    grader_cfg = (cfg.get("graders") or {}).get(grader_key, {})
    card_length_mm = card_length_mm or _card_length_mm(thresholds)

    grade = None
    limited_by = None
    if graded:
        grade = _grade_from_bands(defect_area_pct, cfg["grade_bands"])
        limited_by = "overall surface wear"

        if defects:
            for defect in defects:
                defect.grade_cap = ceiling_for(defect, grader_cfg, card_length_mm)
            worst = min(defects, key=lambda d: d.grade_cap)
            ceiling = worst.grade_cap

            # Several standards count as well as classify. CGC separates "one
            # light crease" from "one or more light creases" by half a grade,
            # and PSA's low grades are written in terms of "several creases".
            creases = [d for d in defects if d.kind == "crease"]
            if len(creases) > 1:
                ceiling -= int(grader_cfg.get("extra_crease_penalty", 0))

            ceiling = max(1, ceiling)
            if ceiling < grade:
                grade = ceiling
                limited_by = worst.kind

    return SurfaceGrade(
        grade=grade,
        reason=None if graded else UNGRADED_REASONS.get(source, "this signal isn't trustworthy enough to grade"),
        defect_area_pct=defect_area_pct,
        defect_count=defect_count,
        longest_defect_px=longest_defect_px,
        source=source,
        note=SOURCE_NOTES.get(source, ""),
        defects=defects,
        limited_by=limited_by if graded else None,
        grader=grader_key if graded else None,
    )


def _card_length_mm(thresholds: dict) -> float:
    capture = thresholds.get("capture", {})
    width_px = capture.get("canonical_width_px", 1500)
    height_px = capture.get("canonical_height_px", 2100)
    from pipeline.dimensions import NOMINAL_WIDTH_MM

    return height_px * (NOMINAL_WIDTH_MM / width_px)


def compare_graders(
    defect_area_pct: float,
    defect_count: int,
    longest_defect_px: int,
    source: str,
    thresholds: dict,
    defects: list | None = None,
) -> dict:
    """What every configured standard would say about the same measured card.

    The same marks, priced by three different published rubrics. They disagree
    by up to four grades on one defect — a deep scratch is a 5 to PSA and a 4
    to CGC; a full-length crease is a 1 to PSA and a 2 to both TAG and CGC —
    so a single number was always hiding a choice. Reference only: the primary
    grader still drives the card's grade, the same way PSA drives centering.
    """
    cfg = thresholds.get("surface", {})
    graders = cfg.get("graders") or {}
    out = {}
    for key, grader_cfg in graders.items():
        # Each grader prices its own copy: grade_surface writes the ceiling it
        # applied back onto every defect, and they must not tread on each other.
        copies = [replace(d) for d in (defects or [])]
        result = grade_surface(
            defect_area_pct, defect_count, longest_defect_px, source, thresholds, copies, grader=key
        )
        out[key] = {
            "label": grader_cfg.get("label", key.upper()),
            "source": grader_cfg.get("source"),
            "grade": result.grade,
            "limited_by": result.limited_by,
        }
    return out
