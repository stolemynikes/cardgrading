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


@dataclass
class CenteringResult:
    front_horizontal: AxisCentering
    front_vertical: AxisCentering
    front_grade: int
    back_horizontal: AxisCentering
    back_vertical: AxisCentering
    back_grade: int
    overall_grade: int

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
            }

        return {
            "front": {
                "horizontal": axis_dict(self.front_horizontal),
                "vertical": axis_dict(self.front_vertical),
                "grade": self.front_grade,
            },
            "back": {
                "horizontal": axis_dict(self.back_horizontal),
                "vertical": axis_dict(self.back_vertical),
                "grade": self.back_grade,
            },
            "overall_grade": self.overall_grade,
        }


def _edge_profile(band: np.ndarray, canny_low: int, canny_high: int, axis: int) -> np.ndarray:
    """Sum of Canny edge response along `axis` (0=per-row, 1=per-column)."""
    edges = cv2.Canny(band, canny_low, canny_high)
    return edges.sum(axis=axis).astype(np.float64)


def _offset_from_start(profile: np.ndarray, search_px: int) -> int:
    """Distance from index 0 to the strongest edge within the first search_px pixels."""
    search_px = min(search_px, len(profile) - 1)
    window = profile[1:search_px]
    if window.size == 0:
        return search_px
    return int(np.argmax(window)) + 1


def _offset_from_end(profile: np.ndarray, search_px: int) -> int:
    """Distance from the last index to the strongest edge within the last search_px pixels."""
    n = len(profile)
    search_px = min(search_px, n - 1)
    window = profile[n - search_px:n - 1]
    if window.size == 0:
        return search_px
    idx = int(np.argmax(window))
    edge_col = (n - search_px) + idx
    return (n - 1) - edge_col


def _measure_borders(image: np.ndarray, cfg: dict) -> tuple[float, float, float, float]:
    """Return (left_px, right_px, top_px, bottom_px) border widths."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    margin = cfg["search_margin_pct"] / 100.0

    # Horizontal (left/right): sample a band around vertical center, look
    # for the strongest vertical edge (column-wise) near each side.
    row_band = (int(h * 0.35), int(h * 0.65))
    col_profile = _edge_profile(
        gray[row_band[0]:row_band[1], :], cfg["canny_low"], cfg["canny_high"], axis=0
    )
    left_px = _offset_from_start(col_profile, int(w * margin))
    right_px = _offset_from_end(col_profile, int(w * margin))

    # Vertical (top/bottom): sample a band around horizontal center, look
    # for the strongest horizontal edge (row-wise) near each side.
    col_band = (int(w * 0.35), int(w * 0.65))
    row_profile = _edge_profile(
        gray[:, col_band[0]:col_band[1]], cfg["canny_low"], cfg["canny_high"], axis=1
    )
    top_px = _offset_from_start(row_profile, int(h * margin))
    bottom_px = _offset_from_end(row_profile, int(h * margin))

    return float(left_px), float(right_px), float(top_px), float(bottom_px)


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
    """Draw the detected border boundary lines over a copy of the image for debugging."""
    overlay = image.copy()
    h, w = overlay.shape[:2]
    left, right = int(axis_h.side_a_px), int(axis_h.side_b_px)
    top, bottom = int(axis_v.side_a_px), int(axis_v.side_b_px)
    color = (0, 255, 0)
    thickness = 3
    cv2.line(overlay, (left, 0), (left, h), color, thickness)
    cv2.line(overlay, (w - right, 0), (w - right, h), color, thickness)
    cv2.line(overlay, (0, top), (w, top), color, thickness)
    cv2.line(overlay, (0, h - bottom), (w, h - bottom), color, thickness)
    text = f"H {axis_h.ratio_str} (g{axis_h.grade})  V {axis_v.ratio_str} (g{axis_v.grade})"
    cv2.putText(overlay, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return overlay


def measure_centering(front: np.ndarray, back: np.ndarray, thresholds: dict) -> CenteringResult:
    cfg = thresholds["centering"]
    border_cfg = cfg["border_detect"]

    fl, fr, ft, fb = _measure_borders(front, border_cfg)
    bl, br, bt, bb = _measure_borders(back, border_cfg)

    front_h = _axis_centering("left", "right", fl, fr, cfg["front_tolerances"])
    front_v = _axis_centering("top", "bottom", ft, fb, cfg["front_tolerances"])
    front_grade = min(front_h.grade, front_v.grade)

    back_h = _axis_centering("left", "right", bl, br, cfg["back_tolerances"])
    back_v = _axis_centering("top", "bottom", bt, bb, cfg["back_tolerances"])
    back_grade = min(back_h.grade, back_v.grade)

    overall_grade = min(front_grade, back_grade)

    return CenteringResult(
        front_horizontal=front_h,
        front_vertical=front_v,
        front_grade=front_grade,
        back_horizontal=back_h,
        back_vertical=back_v,
        back_grade=back_grade,
        overall_grade=overall_grade,
    )
