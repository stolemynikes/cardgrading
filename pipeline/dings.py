"""DINGS — the defects that actually set the grade.

A report that lists every measurement equally makes the reader do the work
of finding which one mattered. This collapses the whole report down to the
handful of findings that drove the number: the worst corner, the axis that
capped centering, the scratches the vision model called out, a bad cut.
Everything here is already in the report elsewhere; this is a ranking, not
a new measurement.

Named after the idea TAG uses in their reports ("Defects Identified of
Notable Grade Significance") — the concept, not their data or thresholds.
"""

from __future__ import annotations

# A region grading a clean 10 isn't a defect, so nothing at or above this is
# ever listed, even if it happens to be the worst region on the card.
PERFECT_GRADE = 10

CORNER_LABELS = {
    "top_left": "top-left corner",
    "top_right": "top-right corner",
    "bottom_right": "bottom-right corner",
    "bottom_left": "bottom-left corner",
}
EDGE_LABELS = {
    "top": "top edge",
    "right": "right edge",
    "bottom": "bottom edge",
    "left": "left edge",
}


def _ding(attribute: str, side: str, label: str, grade, detail: str, image_key: str | None = None,
          box: list[float] | None = None) -> dict:
    return {
        "attribute": attribute,
        "side": side,
        "label": label,
        "grade": grade,
        # Where it is on the card, when it is somewhere in particular.
        "box": box,
        "detail": detail,
        "image_key": image_key,
    }


def _region_detail(region: dict) -> str:
    """Say which reading actually found the damage.

    A corner is graded from two independent measurements — whitening, and
    physical deformation read from the photometric relief — and the worse one
    sets the grade. Quoting only the whitening figure is how a corner graded 3
    came to report "0.00% whitening, 0 blob(s)": that reading is blind on a
    neutral border, and the relief that saw the damage went unmentioned.
    """
    whitening = region.get("whitening_pct") or 0.0
    wear = region.get("relief_wear_pct")
    parts = []
    if wear is not None and wear >= whitening:
        parts.append(f"{wear:.2f}% of the region is physically deformed")
    if whitening > 0 or wear is None:
        parts.append(f"{whitening:.2f}% whitening, {region.get('blob_count', 0)} blob(s)")
    return " · ".join(parts)


def _corners_edges_dings(side: str, side_data: dict) -> list[dict]:
    """The worst-scoring corner(s)/edge(s) on this side — those are what set
    the side's sub-grade, since the region grades combine by minimum."""
    if not side_data:
        return []

    found = []
    groups = (
        ("corners", side_data.get("corners", {}), CORNER_LABELS, "corner_"),
        ("edges", side_data.get("edges", {}), EDGE_LABELS, "edge_"),
    )
    for attribute, regions, labels, image_prefix in groups:
        # A refused region carries no grade at all, so it can't be the worst
        # one and can't be ranked against the others. It used to serialize
        # the grade it would have had, which made this loop work by accident.
        graded = {key: r for key, r in regions.items() if r.get("grade") is not None}
        if not graded:
            continue
        worst = min(r["grade"] for r in graded.values())
        if worst >= PERFECT_GRADE:
            continue
        for key, region in graded.items():
            if region["grade"] != worst:
                continue
            found.append(
                _ding(
                    attribute,
                    side,
                    labels.get(key, key),
                    region["grade"],
                    _region_detail(region),
                    f"{side}_{image_prefix}{key}",
                    region.get("box"),
                )
            )
    return found


def _centering_dings(side: str, side_data: dict) -> list[dict]:
    """The axis that capped this side's centering grade. An unmeasurable axis
    is not a defect — it's a borderless card or a weak capture, which the
    centering section already explains on its own."""
    if not side_data:
        return []

    axes = [
        ("horizontal", side_data.get("horizontal")),
        ("vertical", side_data.get("vertical")),
    ]
    # Reports written before the measurability check existed have no such key;
    # those measurements were all taken as real, so default to that rather
    # than silently dropping every ding from an older report.
    measurable = [
        (name, axis)
        for name, axis in axes
        if axis and axis.get("measurable", True) and axis.get("grade") is not None
    ]
    if not measurable:
        return []

    worst = min(axis["grade"] for _, axis in measurable)
    if worst >= PERFECT_GRADE:
        return []

    return [
        _ding(
            "centering",
            side,
            f"{name} centering",
            axis["grade"],
            f"{axis.get('ratio', '')} — off-center".lstrip(),
            f"{side}_centering_overlay",
        )
        for name, axis in measurable
        if axis["grade"] == worst
    ]


def _surface_dings(side: str, side_data: dict) -> list[dict]:
    """Surface findings come from the measured defect statistics.

    An ungraded side (the single-capture approximation, where print leaks
    into the signal) produces no ding: reporting a defect count that includes
    printed linework as damage would be worse than reporting nothing.
    """
    if not side_data:
        return []

    grade = side_data.get("grade")
    if grade is None or grade >= PERFECT_GRADE:
        return []

    count = side_data.get("defect_count", 0)
    area = side_data.get("defect_area_pct", 0.0)
    summary = f"{area:.2f}% of the surface, {count} defect(s)"

    # The named mark that capped the grade gets its own entry, because that is
    # the thing to go and look at. A card graded 4 whose ding says only "0.15%
    # of the surface" tells nobody that it has a crease in the bottom-left
    # corner — and the crease is the entire reason for the grade.
    found = []
    limited_by = side_data.get("limited_by")
    defects = side_data.get("defects") or []
    # Lowest ceiling first, then largest — two marks of the same kind cap the
    # card equally, and the one worth walking over to look at is the bigger.
    worst = min(
        defects,
        key=lambda d: (d.get("grade_cap", 10), -d.get("length_mm", 0.0) * d.get("width_mm", 0.0)),
        default=None,
    )
    if worst is not None and limited_by and limited_by != "overall surface wear":
        x, y = worst.get("centre_mm", [0, 0])
        kind = worst.get("kind", "surface")
        cap = worst.get("grade_cap")
        detail = (
            f"{worst.get('length_mm', 0):.1f} x {worst.get('width_mm', 0):.1f}mm at "
            f"{x:.0f}, {y:.0f}mm from the top-left corner — caps this card at {cap}"
        )
        # Some standards count as well as classify, so the card can sit a
        # grade below what any single mark would allow. Saying "caps this card
        # at 5" beside a grade of 4 reads as a contradiction unless the reason
        # is given.
        same_kind = sum(1 for d in defects if d.get("kind") == kind)
        if cap is not None and grade is not None and grade < cap and same_kind > 1:
            detail += f", and a further grade for being {same_kind} of them"
        found.append(_ding("surface", side, kind, grade, detail, f"{side}_card_vision"))
        summary += f", worst is a {kind}"

    found.append(_ding("surface", side, "surface", grade, summary, f"{side}_card_vision"))
    return found


def _dimensions_dings(dimensions: dict | None) -> list[dict]:
    # within_tolerance is three-valued. True is a card that measured fine;
    # False is a miscut, which belongs here. None means the scans disagreed by
    # more than the tolerance they were being judged against, so no verdict
    # was given — and "we could not tell" is not a defect. Listing it put a
    # grade-capping marker on a card nothing had been established about.
    if not dimensions or not dimensions.get("measurable"):
        return []
    if dimensions.get("within_tolerance") is not False:
        return []
    return [
        _ding(
            "dimensions",
            "card",
            "cut / dimensions",
            None,
            dimensions.get("note", "outside cut tolerance"),
            None,
        )
    ]


def collect_dings(report: dict) -> list[dict]:
    """Ranked worst-first. Grade None (dimensions) sorts first: a miscut is a
    hard cap on the grade, not a gradient."""
    dings: list[dict] = []
    dings += _dimensions_dings(report.get("dimensions"))

    centering = report.get("centering") or {}
    corners_edges = report.get("corners_edges") or {}
    surface = report.get("surface") or {}

    for side in ("front", "back"):
        dings += _centering_dings(side, centering.get(side))
        dings += _corners_edges_dings(side, corners_edges.get(side))
        dings += _surface_dings(side, surface.get(side))

    dings.sort(key=lambda d: (d["grade"] is not None, d["grade"] if d["grade"] is not None else 0))
    return dings
