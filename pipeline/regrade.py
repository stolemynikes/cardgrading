"""Re-derive a report's dependent parts after one measurement is corrected.

A report isn't a flat bag of numbers: the overall grade is assembled from the
sub-grades, the per-side breakdown mirrors them, and the dings list is ranked
from all of it. Change centering by hand and three other sections are stale.

Rather than recompute the whole pipeline — which would need the original
uploads, long gone by the time anyone looks at a saved report — this rebuilds
exactly the parts that are downstream of a sub-grade, from the report itself.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline import centering as centering_stage
from pipeline import corners_edges, dings, scoring


def _surface_grade(report: dict) -> int | None:
    """Weakest graded side. Mirrors grade.py: a side with no grade doesn't vote."""
    surface = report.get("surface") or {}
    graded = [
        side for side in surface.values() if isinstance(side, dict) and side.get("grade") is not None
    ]
    return min((side["grade"] for side in graded), default=None)


def _defect_cap(report: dict) -> tuple[int | None, str | None]:
    """The ceiling a named physical defect imposes on the whole card.

    Mirrors grade.py. Re-derived here rather than carried on the report,
    because rescore runs after a hand-placed centering edit and has to rebuild
    the grade from the report's own sections — and dropping this silently
    handed a creased card back its full grade the moment someone dragged a
    centering line.
    """
    surface = report.get("surface") or {}
    capped = [
        (side["grade"], side.get("limited_by"))
        for side in surface.values()
        if isinstance(side, dict)
        and side.get("grade") is not None
        and side.get("limited_by")
        and side["limited_by"] != "overall surface wear"
    ]
    return min(capped, default=(None, None))


DEFAULT_BORDER_FRACTION = 0.04


def border_widths(side_centering: dict, shape: tuple) -> corners_edges.BorderWidths:
    """Crop-sizing widths for one side, mirroring grade.py's own fallback.

    Border widths from an unmeasurable axis are argmax-of-noise, and sizing a
    crop from them produces a postage stamp whose whitening percentage is
    pure noise. Per axis: the measured widths where real, a typical ~4%
    otherwise.
    """
    height, width = shape[:2]
    axis_h = (side_centering or {}).get("horizontal") or {}
    axis_v = (side_centering or {}).get("vertical") or {}

    if axis_h.get("measurable", False):
        left, right = float(axis_h["side_a_px"]), float(axis_h["side_b_px"])
    else:
        left = right = width * DEFAULT_BORDER_FRACTION
    if axis_v.get("measurable", False):
        top, bottom = float(axis_v["side_a_px"]), float(axis_v["side_b_px"])
    else:
        top = bottom = height * DEFAULT_BORDER_FRACTION
    return corners_edges.BorderWidths(left=left, right=right, top=top, bottom=bottom)


def recompute_corners_edges(
    report: dict, aligned: dict, thresholds: dict
) -> dict[str, np.ndarray]:
    """Re-measure corners and edges after the borders moved.

    Every corner and edge crop is *sized from the border widths* — that's how
    a crop is kept inside the border instead of running into the artwork. So
    correcting centering by hand invalidates this whole stage, and leaving it
    alone left a report whose corners were measured against borders it no
    longer claimed. Re-measured on one real card, three of four corners went
    from refused to measurable.

    `aligned` maps a side to its canonical warp. A side with no stored warp
    keeps its existing numbers. Returns the region crops to write back,
    keyed as the report's image store names them.
    """
    block = report.get("corners_edges")
    if not isinstance(block, dict) or not aligned:
        return {}

    cfg = thresholds["corners_edges"]
    centering = report.get("centering") or {}
    images: dict[str, np.ndarray] = {}

    for side, warp in aligned.items():
        if warp is None:
            continue
        widths = border_widths(centering.get(side), warp.shape)
        side_result, overlays = corners_edges.analyze_side(warp, widths, cfg)
        block[side] = side_result.to_dict()
        for key, crop in overlays.items():
            images[f"{side}_{key}"] = crop

    # None means "refused", not "perfect" — a side with nothing measurable
    # drops out of the combination rather than pulling it toward 10.
    grades = [
        block[side].get("grade")
        for side in ("front", "back")
        if isinstance(block.get(side), dict) and block[side].get("grade") is not None
    ]
    block["overall_grade"] = min(grades) if grades else None
    return images


def load_aligned(images_dir, sides=("front", "back")) -> dict:
    """The stored canonical warps, by side. Missing ones come back as None."""
    out = {}
    for side in sides:
        path = images_dir / f"{side}_aligned.png"
        out[side] = cv2.imread(str(path)) if path.exists() else None
    return out


def rescore(report: dict, thresholds: dict) -> dict:
    """Rebuild grade_estimate, subgrades and dings from the report's sections."""
    centering = report.get("centering") or {}
    if centering:
        # Hand-placed boundaries change what every grader's table says, not
        # just PSA's.
        centering["by_grader"] = centering_stage.compare_graders(centering, thresholds)
    corners_edges = report.get("corners_edges") or {}
    defect_cap, defect_cap_reason = _defect_cap(report)
    estimate = scoring.assemble_grade(
        centering.get("overall_grade"),
        corners_edges.get("overall_grade"),
        _surface_grade(report),
        thresholds,
        dimensions_within_tolerance=(report.get("dimensions") or {}).get("within_tolerance"),
        defect_grade_cap=defect_cap,
        defect_cap_reason=f"a {defect_cap_reason}" if defect_cap_reason else None,
    )
    report["grade_estimate"] = estimate.to_dict()

    subgrades = report.setdefault("subgrades", {})
    for side in ("front", "back"):
        side_centering = centering.get(side)
        if isinstance(side_centering, dict):
            subgrades.setdefault(side, {})["centering"] = side_centering.get("grade")

    report["dings"] = dings.collect_dings(report)
    return report
