"""Whitening detection — specifically, not firing on cardstock that is fine.

The regression these guard was found by running a real Base Set scan through
the stage: it reported 25-36% whitening on all sixteen regions of a card a
grading service had passed with zero corner or edge defects. Synthetic cards
never caught it, because their borders are noiseless gradients and the failure
needs ordinary scan noise on a flat colour to appear.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline import corners_edges

CFG = json.loads((Path(__file__).resolve().parent.parent / "calibration" / "thresholds.json").read_text())[
    "corners_edges"
]

SIZE = 120


def _flat_border(colour: tuple[int, int, int], noise: float = 4.0, seed: int = 0) -> np.ndarray:
    """A patch of undamaged cardstock with realistic sensor/JPEG noise."""
    rng = np.random.default_rng(seed)
    base = np.full((SIZE, SIZE, 3), colour, dtype=np.float32)
    return np.clip(base + rng.normal(0, noise, base.shape), 0, 255).astype(np.uint8)


def _with_chip(crop: np.ndarray, radius: int = 20, seed: int = 5) -> np.ndarray:
    """Add exposed white cardstock in one corner.

    Speckled rather than a solid disc, because that is what chipping actually
    looks like — the printed layer tears away unevenly — and because a solid
    disc is a poor test: its interior sits at the local mean, so the adaptive
    stage only ever sees its rim.
    """
    worn = crop.copy()
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    inside = (yy ** 2 + xx ** 2) <= radius ** 2
    speckle = rng.random((SIZE, SIZE)) < 0.55
    worn[inside & speckle] = (240, 242, 244)
    return worn


YELLOW_BORDER = (40, 200, 230)   # BGR — a Base Set front
BLUE_BORDER = (120, 50, 30)      # BGR — a card back


def test_clean_yellow_border_is_not_whitening():
    result, _ = corners_edges.analyze_region("top_left", _flat_border(YELLOW_BORDER), CFG)
    assert result.whitening_pct < 0.5
    assert result.grade == 10


def test_clean_blue_back_is_not_whitening():
    result, _ = corners_edges.analyze_region("top_left", _flat_border(BLUE_BORDER, seed=1), CFG)
    assert result.whitening_pct < 0.5
    assert result.grade == 10


def test_noisier_capture_still_reads_clean():
    """The failure scaled with noise, so the guard has to as well."""
    result, _ = corners_edges.analyze_region("top_left", _flat_border(YELLOW_BORDER, noise=9.0, seed=2), CFG)
    assert result.whitening_pct < 1.0


def test_a_real_chip_is_still_detected():
    """The gate must not have bought quiet by going blind."""
    clean, _ = corners_edges.analyze_region("top_left", _flat_border(YELLOW_BORDER), CFG)
    chipped, _ = corners_edges.analyze_region("top_left", _with_chip(_flat_border(YELLOW_BORDER)), CFG)
    assert chipped.whitening_pct > clean.whitening_pct + 1.0
    assert chipped.grade < 10


def test_chip_detected_on_a_dark_back_too():
    chipped, _ = corners_edges.analyze_region("top_left", _with_chip(_flat_border(BLUE_BORDER, seed=1)), CFG)
    assert chipped.whitening_pct > 1.0
    assert chipped.grade < 10


def test_bigger_chip_grades_worse():
    small, _ = corners_edges.analyze_region("top_left", _with_chip(_flat_border(YELLOW_BORDER), radius=12), CFG)
    large, _ = corners_edges.analyze_region("top_left", _with_chip(_flat_border(YELLOW_BORDER), radius=40), CFG)
    assert large.whitening_pct > small.whitening_pct
    assert large.grade <= small.grade
