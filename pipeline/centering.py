"""Stage 2: centering — border-width measurement and PSA-style tolerance grading.

Assumes the input image has already been perspective-corrected to the canonical
size (pipeline.detect.detect_and_normalize), so the physical card edge sits
exactly at the image boundary. Centering is then just: find where the outer
printed border/frame ends and the inner artwork/text panel begins, on each
side, and compare distances from the card edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AxisCentering:
    side_a: str  # "left" or "top"
    side_b: str  # "right" or "bottom"
    side_a_px: float
    side_b_px: float
    side_a_pct: float
    side_b_pct: float
    ratio_str: str
    grade: int
    # False when this axis's two boundaries couldn't be found confidently —
    # the px/pct/grade values above are then argmax-of-noise and must not be
    # shown as measurements.
    measurable: bool = True


@dataclass
class CenteringResult:
    front_horizontal: AxisCentering
    front_vertical: AxisCentering
    front_grade: int | None  # None when the front's borders couldn't be measured
    back_horizontal: AxisCentering
    back_vertical: AxisCentering
    back_grade: int | None
    overall_grade: int | None
    front_measurable: bool = True
    back_measurable: bool = True
    front_confidence: dict | None = None  # per-boundary peak confidences, for debugging
    back_confidence: dict | None = None

    def to_dict(self) -> dict:
        def axis_dict(a: AxisCentering) -> dict:
            return {
                "side_a": a.side_a,
                "side_b": a.side_b,
                "side_a_px": round(a.side_a_px, 1),
                "side_b_px": round(a.side_b_px, 1),
                "side_a_pct": round(a.side_a_pct, 1),
                "side_b_pct": round(a.side_b_pct, 1),
                "ratio": a.ratio_str,
                "grade": a.grade,
                "measurable": a.measurable,
            }

        return {
            "front": {
                "horizontal": axis_dict(self.front_horizontal),
                "vertical": axis_dict(self.front_vertical),
                "grade": self.front_grade,
                "measurable": self.front_measurable,
                "boundary_confidence": self.front_confidence,
            },
            "back": {
                "horizontal": axis_dict(self.back_horizontal),
                "vertical": axis_dict(self.back_vertical),
                "grade": self.back_grade,
                "measurable": self.back_measurable,
                "boundary_confidence": self.back_confidence,
            },
            "overall_grade": self.overall_grade,
        }


def _edge_profile(band: np.ndarray, canny_low: int, canny_high: int, axis: int) -> np.ndarray:
    """Sum of Canny edge response along `axis` (0=per-row, 1=per-column)."""
    edges = cv2.Canny(band, canny_low, canny_high)
    return edges.sum(axis=axis).astype(np.float64)


# The perspective warp leaves a strong straight artifact line within the
# first few pixels of every image edge (interpolation/background bleed).
# It scores near-perfect boundary confidence and once produced a fake
# "60/40 grade 10" from a top=3px/bottom=2px "border". No real card border
# is thinner than ~0.5% of the card dimension, so the boundary search
# starts past the artifact zone.
def _edge_exclusion_px(dim: int) -> int:
    return max(4, int(dim * 0.006))


def _offset_from_start(profile: np.ndarray, search_px: int, start_at: int = 1) -> int:
    """Distance from index 0 to the strongest edge within the first search_px pixels."""
    search_px = min(search_px, len(profile) - 1)
    window = profile[start_at:search_px]
    if window.size == 0:
        return search_px
    return int(np.argmax(window)) + start_at


def _offset_from_end(profile: np.ndarray, search_px: int, start_at: int = 1) -> int:
    """Distance from the last index to the strongest edge within the last search_px pixels."""
    n = len(profile)
    search_px = min(search_px, n - 1)
    window = profile[n - search_px:n - start_at]
    if window.size == 0:
        return search_px
    idx = int(np.argmax(window))
    edge_col = (n - search_px) + idx
    return (n - 1) - edge_col


# A real border→panel boundary is a straight line spanning the whole sampled
# band, so its strongest profile column reaches a large fraction of the
# theoretical maximum (band_height * 255). Scattered art texture on a
# borderless/full-art card peaks far lower, and a boundary lost to blur or
# low contrast peaks near zero. Measured on synthetic scenes at canonical
# resolution: sharp/lightly-blurred bordered cards score 0.52-0.76,
# borderless art ~0.15-0.17, dim or heavily blurred captures ~0.
DEFAULT_MIN_BOUNDARY_CONFIDENCE = 0.35

# Second-chance bar for illumination-normalized re-measurement. Gamma+CLAHE
# recovers boundaries from underexposed captures (a dim bordered card's
# confidence goes 0.0 -> ~0.5) but also inflates art texture (borderless
# scenes rise to ~0.44), so the normalized pass needs a stricter bar than
# the raw pass — 0.5 sits between those two measured outcomes.
DEFAULT_NORMALIZED_MIN_CONFIDENCE = 0.5


def _normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Gamma-correct toward mid-gray, then CLAHE for local contrast — makes
    a border/panel boundary in an underexposed capture visible to Canny."""
    mean = gray.mean()
    if mean > 0:
        gamma = np.log(0.5) / np.log(np.clip(mean / 255.0, 0.05, 0.95))
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        gray = lut[gray]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _peak_confidence(profile: np.ndarray, window: slice, band_px: int) -> float:
    """Strongest edge in the window as a fraction of a perfect straight edge."""
    vals = profile[window]
    if vals.size == 0:
        return 0.0
    return float(vals.max()) / (band_px * 255.0)


def _measure_borders(
    image: np.ndarray, cfg: dict, normalize: bool = False
) -> tuple[tuple[float, float, float, float], dict[str, float]]:
    """Return ((left_px, right_px, top_px, bottom_px), per-boundary confidence).

    The offsets are always computed (argmax of the edge profile picks
    *something* in every image); the confidence dict is what says whether
    each pick is a real border boundary or just the strongest noise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if normalize:
        gray = _normalize_illumination(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    margin = cfg["search_margin_pct"] / 100.0

    # Horizontal (left/right): sample a band around vertical center, look
    # for the strongest vertical edge (column-wise) near each side.
    row_band = (int(h * 0.35), int(h * 0.65))
    band_h = row_band[1] - row_band[0]
    col_profile = _edge_profile(
        gray[row_band[0]:row_band[1], :], cfg["canny_low"], cfg["canny_high"], axis=0
    )
    sp_w = int(w * margin)
    excl_w = _edge_exclusion_px(w)
    left_px = _offset_from_start(col_profile, sp_w, excl_w)
    right_px = _offset_from_end(col_profile, sp_w, excl_w)

    # Vertical (top/bottom): sample a band around horizontal center, look
    # for the strongest horizontal edge (row-wise) near each side.
    col_band = (int(w * 0.35), int(w * 0.65))
    band_w = col_band[1] - col_band[0]
    row_profile = _edge_profile(
        gray[:, col_band[0]:col_band[1]], cfg["canny_low"], cfg["canny_high"], axis=1
    )
    sp_h = int(h * margin)
    excl_h = _edge_exclusion_px(h)
    top_px = _offset_from_start(row_profile, sp_h, excl_h)
    bottom_px = _offset_from_end(row_profile, sp_h, excl_h)

    n = len(col_profile)
    m = len(row_profile)
    confidence = {
        "left": _peak_confidence(col_profile, slice(excl_w, sp_w), band_h),
        "right": _peak_confidence(col_profile, slice(n - sp_w, n - excl_w), band_h),
        "top": _peak_confidence(row_profile, slice(excl_h, sp_h), band_w),
        "bottom": _peak_confidence(row_profile, slice(m - sp_h, m - excl_h), band_w),
    }
    return (float(left_px), float(right_px), float(top_px), float(bottom_px)), confidence


def _grade_from_ratio(worse_pct: float, tolerances: list[dict]) -> int:
    """tolerances: ascending max_ratio per grade, best (10) first."""
    for tier in tolerances:
        if worse_pct <= tier["max_ratio"]:
            return tier["grade"]
    return max(1, tolerances[-1]["grade"] - 2)


def _axis_centering(side_a_name: str, side_b_name: str, px_a: float, px_b: float, tolerances: list[dict]) -> AxisCentering:
    total = px_a + px_b if (px_a + px_b) > 0 else 1.0
    pct_a = 100.0 * px_a / total
    pct_b = 100.0 * px_b / total
    worse = max(pct_a, pct_b)
    grade = _grade_from_ratio(worse, tolerances)
    hi, lo = (pct_a, pct_b) if pct_a >= pct_b else (pct_b, pct_a)
    ratio_str = f"{hi:.0f}/{lo:.0f}"
    return AxisCentering(side_a_name, side_b_name, px_a, px_b, pct_a, pct_b, ratio_str, grade)


def draw_overlay(image: np.ndarray, axis_h: AxisCentering, axis_v: AxisCentering) -> np.ndarray:
    """Draw the detected border boundary lines over a copy of the image.

    Lines are drawn only for measurable axes — an unmeasurable axis's
    positions are argmax-of-noise that would look authoritative but aren't.
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    color = (0, 255, 0)
    thickness = 3
    parts = []
    if axis_h.measurable:
        left, right = int(axis_h.side_a_px), int(axis_h.side_b_px)
        cv2.line(overlay, (left, 0), (left, h), color, thickness)
        cv2.line(overlay, (w - right, 0), (w - right, h), color, thickness)
        parts.append(f"H {axis_h.ratio_str} (g{axis_h.grade})")
    else:
        parts.append("H n/a")
    if axis_v.measurable:
        top, bottom = int(axis_v.side_a_px), int(axis_v.side_b_px)
        cv2.line(overlay, (0, top), (w, top), color, thickness)
        cv2.line(overlay, (0, h - bottom), (w, h - bottom), color, thickness)
        parts.append(f"V {axis_v.ratio_str} (g{axis_v.grade})")
    else:
        parts.append("V n/a")
    if not axis_h.measurable and not axis_v.measurable:
        parts = ["centering unmeasurable"]
    cv2.putText(overlay, "  ".join(parts), (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return overlay


def _measure_side(
    image: np.ndarray, border_cfg: dict
) -> tuple[tuple[float, float, float, float], dict[str, float], bool, bool]:
    """Measure one side's borders, with an illumination-normalized retry.

    Raw measurement first; boundaries below the confidence floor get a
    second chance on a gamma+CLAHE-normalized image against a stricter bar
    (normalization recovers dim captures but also inflates art texture).

    Measurability is decided PER AXIS — horizontal needs left+right,
    vertical needs top+bottom. A real capture measured V confidently while
    H was genuinely invisible (a soft-focus shot of a card back, whose
    blue-frame→blue-swirl left/right transition is the lowest-contrast
    boundary on the card); an all-four rule threw the good axis away.

    Returns (widths, confidences, h_measurable, v_measurable). Widths and
    confidences are per-boundary from whichever pass confirmed that axis
    (raw preferred); unconfirmed axes keep their raw numbers for debugging.
    """
    min_conf = border_cfg.get("min_boundary_confidence", DEFAULT_MIN_BOUNDARY_CONFIDENCE)
    norm_bar = border_cfg.get("normalized_min_boundary_confidence", DEFAULT_NORMALIZED_MIN_CONFIDENCE)

    (l, r, t, b), conf = _measure_borders(image, border_cfg)
    h_ok = conf["left"] >= min_conf and conf["right"] >= min_conf
    v_ok = conf["top"] >= min_conf and conf["bottom"] >= min_conf

    if not (h_ok and v_ok):
        (l_n, r_n, t_n, b_n), conf_n = _measure_borders(image, border_cfg, normalize=True)
        if not h_ok and conf_n["left"] >= norm_bar and conf_n["right"] >= norm_bar:
            l, r = l_n, r_n
            conf = {**conf, "left": conf_n["left"], "right": conf_n["right"]}
            h_ok = True
        if not v_ok and conf_n["top"] >= norm_bar and conf_n["bottom"] >= norm_bar:
            t, b = t_n, b_n
            conf = {**conf, "top": conf_n["top"], "bottom": conf_n["bottom"]}
            v_ok = True

    return (l, r, t, b), conf, h_ok, v_ok


def measure_centering(front: np.ndarray, back: np.ndarray, thresholds: dict) -> CenteringResult:
    cfg = thresholds["centering"]
    border_cfg = cfg["border_detect"]

    # An axis whose boundaries can't be found confidently is unmeasurable —
    # a borderless/full-art card, or a capture too blurry/dim to show that
    # boundary. Reporting its ratio anyway would be fake precision from
    # argmax-of-noise (a real full-art card produced a confident-looking
    # "89/11 grade 3" before this check existed). Measurable axes still
    # grade: a side's grade is the min over its measurable axes, None only
    # when neither axis measures.
    (fl, fr, ft, fb), front_conf, f_h_ok, f_v_ok = _measure_side(front, border_cfg)
    (bl, br, bt, bb), back_conf, b_h_ok, b_v_ok = _measure_side(back, border_cfg)

    def side(h_names, v_names, hpx, vpx, h_ok, v_ok, tolerances):
        axis_h = _axis_centering(*h_names, *hpx, tolerances)
        axis_h.measurable = h_ok
        axis_v = _axis_centering(*v_names, *vpx, tolerances)
        axis_v.measurable = v_ok
        grades = [a.grade for a in (axis_h, axis_v) if a.measurable]
        return axis_h, axis_v, (min(grades) if grades else None)

    front_h, front_v, front_grade = side(
        ("left", "right"), ("top", "bottom"), (fl, fr), (ft, fb), f_h_ok, f_v_ok, cfg["front_tolerances"]
    )
    back_h, back_v, back_grade = side(
        ("left", "right"), ("top", "bottom"), (bl, br), (bt, bb), b_h_ok, b_v_ok, cfg["back_tolerances"]
    )

    measured = [g for g in (front_grade, back_grade) if g is not None]
    overall_grade = min(measured) if measured else None

    return CenteringResult(
        front_horizontal=front_h,
        front_vertical=front_v,
        front_grade=front_grade,
        back_horizontal=back_h,
        back_vertical=back_v,
        back_grade=back_grade,
        overall_grade=overall_grade,
        # Side-level flag means "this side produced a grade at all" — i.e.
        # at least one axis measured. Per-axis truth lives on the axes.
        front_measurable=front_grade is not None,
        back_measurable=back_grade is not None,
        front_confidence={k: round(v, 3) for k, v in front_conf.items()},
        back_confidence={k: round(v, 3) for k, v in back_conf.items()},
    )
