"""Dimensions: the card's real physical size, in millimetres.

This is the one attribute that needs absolute scale, so it only works on a
capture whose scale is known — a flatbed scan at a stated DPI. A phone photo
has no such scale (distance to the card is unknown), so the measurement is
reported as unavailable rather than guessed.

What it catches is factory miscuts: a card trimmed short on one axis, or cut
out of square ("diamond cut"), both of which cap the grade at real grading
services no matter how clean the surface is. It also catches deliberate
trimming, which is the other reason a grader measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MM_PER_INCH = 25.4

# Nominal trading-card size. Pokemon cards are specified at 63x88mm; the
# 2.5x3.5in figure often quoted (63.5x88.9mm) is the looser US sports-card
# convention, and using it here would flag every normal Pokemon card as
# undersized.
NOMINAL_WIDTH_MM = 63.0
NOMINAL_HEIGHT_MM = 88.0


@dataclass
class DimensionsResult:
    measurable: bool
    width_mm: float | None
    height_mm: float | None
    width_deviation_mm: float | None
    height_deviation_mm: float | None
    squareness_deviation_deg: float | None
    within_tolerance: bool | None
    note: str
    # How much the measurement moved between independent scans of the same
    # card, in millimetres, and how many scans it rests on. A single capture
    # gives no way to know how repeatable its own number is — and measured on
    # this scanner the same card came out 1.8mm different depending only on
    # which way round it was lying, against a tolerance of 0.75mm.
    spread_mm: float | None = field(default=None)
    sample_count: int = field(default=1)

    def to_dict(self) -> dict:
        def r(value: float | None, places: int = 2) -> float | None:
            return None if value is None else round(value, places)

        return {
            "measurable": self.measurable,
            "width_mm": r(self.width_mm),
            "height_mm": r(self.height_mm),
            "nominal_width_mm": NOMINAL_WIDTH_MM,
            "nominal_height_mm": NOMINAL_HEIGHT_MM,
            "width_deviation_mm": r(self.width_deviation_mm),
            "height_deviation_mm": r(self.height_deviation_mm),
            "squareness_deviation_deg": r(self.squareness_deviation_deg),
            "within_tolerance": self.within_tolerance,
            "spread_mm": r(self.spread_mm),
            "sample_count": self.sample_count,
            "note": self.note,
        }


def _side_lengths_px(corners: np.ndarray) -> tuple[float, float]:
    """Mean width and height of the quad, corners ordered tl,tr,br,bl.

    Averaging the two opposite sides rather than taking one of each means a
    single nibbled corner shifts the result by half as much.
    """
    tl, tr, br, bl = corners.astype(np.float64)
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    return (top + bottom) / 2.0, (left + right) / 2.0


def measure_dimensions(
    corners: np.ndarray | None, dpi: float | None, thresholds: dict
) -> DimensionsResult:
    """Physical size of the detected card quad, given the capture's DPI."""
    cfg = thresholds.get("dimensions", {})
    tolerance_mm = cfg.get("tolerance_mm", 0.75)
    squareness_tolerance_deg = cfg.get("squareness_tolerance_deg", 1.0)

    if dpi is None:
        return DimensionsResult(
            False, None, None, None, None, None, None,
            "not measurable without a known capture scale — scan at a fixed DPI and pass --dpi",
        )
    if corners is None:
        return DimensionsResult(
            False, None, None, None, None, None, None, "no card quad was detected to measure"
        )

    mm_per_px = MM_PER_INCH / dpi
    width_px, height_px = _side_lengths_px(corners)
    width_mm = float(width_px * mm_per_px)
    height_mm = float(height_px * mm_per_px)

    # A card scanned in landscape measures 88x63; compare against whichever
    # orientation fits rather than failing a correctly-cut card.
    if width_mm > height_mm:
        width_mm, height_mm = height_mm, width_mm

    width_dev = width_mm - NOMINAL_WIDTH_MM
    height_dev = height_mm - NOMINAL_HEIGHT_MM

    from pipeline.detect import corner_angle_deviations

    # numpy scalars leak out of the angle math and json.dumps rejects them.
    squareness = float(max(corner_angle_deviations(corners.astype(np.float32))))

    size_ok = abs(width_dev) <= tolerance_mm and abs(height_dev) <= tolerance_mm
    square_ok = squareness <= squareness_tolerance_deg
    within = size_ok and square_ok

    if within:
        note = f"within {tolerance_mm}mm of nominal and square to {squareness_tolerance_deg} degrees"
    else:
        problems = []
        if abs(width_dev) > tolerance_mm:
            problems.append(f"width off by {width_dev:+.2f}mm")
        if abs(height_dev) > tolerance_mm:
            problems.append(f"height off by {height_dev:+.2f}mm")
        if not square_ok:
            problems.append(f"corners out of square by {squareness:.2f} degrees (diamond cut)")
        note = "; ".join(problems) + " — miscut or trimmed"

    return DimensionsResult(
        measurable=True,
        width_mm=width_mm,
        height_mm=height_mm,
        width_deviation_mm=width_dev,
        height_deviation_mm=height_dev,
        squareness_deviation_deg=squareness,
        within_tolerance=within,
        note=note,
    )


def measure_from_scans(
    quads: list[np.ndarray], dpi: float | None, thresholds: dict
) -> DimensionsResult:
    """Measure the card from several independent scans of it, not one.

    The single-scan measurement states a confident number on top of something
    it has no way to check. Measured on a real flatbed, the same card came out
    2.9% different — 1.8mm on a 63mm card — depending only on whether it was
    lying portrait or landscape on the glass, against a tolerance of 0.75mm.
    So the same card read "2.13mm miscut" in one run and "within tolerance"
    in the next, decided by which scan happened to be the flat one.

    The median is the figure, because one bad detection shouldn't move it. The
    spread is reported alongside, and when the spread is wider than the
    tolerance being applied, no miscut verdict is given at all: a measurement
    that disagrees with itself by more than the thing it is being judged
    against cannot settle the question, and saying so is the honest answer.
    """
    usable = [q for q in quads if q is not None]
    if len(usable) < 2:
        return measure_dimensions(usable[0] if usable else None, dpi, thresholds)

    cfg = thresholds.get("dimensions", {})
    tolerance_mm = cfg.get("tolerance_mm", 0.75)
    singles = [measure_dimensions(q, dpi, thresholds) for q in usable]
    measured = [s for s in singles if s.measurable]
    if not measured:
        return singles[0]

    widths = sorted(s.width_mm for s in measured)
    heights = sorted(s.height_mm for s in measured)
    width_mm = float(np.median(widths))
    height_mm = float(np.median(heights))
    spread = max(widths[-1] - widths[0], heights[-1] - heights[0])

    width_dev = width_mm - NOMINAL_WIDTH_MM
    height_dev = height_mm - NOMINAL_HEIGHT_MM
    squareness = float(np.median([s.squareness_deviation_deg for s in measured]))
    squareness_tolerance_deg = cfg.get("squareness_tolerance_deg", 1.0)

    if spread > tolerance_mm:
        return DimensionsResult(
            measurable=True,
            width_mm=width_mm,
            height_mm=height_mm,
            width_deviation_mm=width_dev,
            height_deviation_mm=height_dev,
            squareness_deviation_deg=squareness,
            within_tolerance=None,
            note=(
                f"Can't tell — the {len(measured)} scans disagree by {spread:.2f}mm, and the limit "
                f"being checked is {tolerance_mm}mm. Scan the card flat against the glass."
            ),
            spread_mm=spread,
            sample_count=len(measured),
        )

    size_ok = abs(width_dev) <= tolerance_mm and abs(height_dev) <= tolerance_mm
    square_ok = squareness <= squareness_tolerance_deg
    within = size_ok and square_ok
    if within:
        note = (
            f"Correctly cut — within {tolerance_mm}mm of nominal on both axes and square. "
            f"{len(measured)} scans agreed to {spread:.2f}mm."
        )
    else:
        problems = []
        if abs(width_dev) > tolerance_mm:
            problems.append(f"width off by {width_dev:+.2f}mm")
        if abs(height_dev) > tolerance_mm:
            problems.append(f"height off by {height_dev:+.2f}mm")
        if not square_ok:
            problems.append(f"corners out of square by {squareness:.2f} degrees (diamond cut)")
        note = (
            "Miscut or trimmed — "
            + "; ".join(problems)
            + f". {len(measured)} scans agreed to {spread:.2f}mm."
        )

    return DimensionsResult(
        measurable=True,
        width_mm=width_mm,
        height_mm=height_mm,
        width_deviation_mm=width_dev,
        height_deviation_mm=height_dev,
        squareness_deviation_deg=squareness,
        within_tolerance=within,
        note=note,
        spread_mm=spread,
        sample_count=len(measured),
    )
