"""Working out how each rotation scan was turned, from the scan itself.

This was the one solve input that was asserted rather than measured. It
depended on file order and on the operator remembering which way they turned
the card, it failed silently when either was wrong, and the failure produced
a confident render of a badly damaged card: the solve differences the card
against rotated copies of itself, so the artwork ghosts and a clean card
comes back with 18% of its area flagged as defect.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import grade as grade_module
from webapp import main

THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())
CARD_ASPECT = 63.0 / 88.0


def _card_image(seed: int = 0) -> np.ndarray:
    """A card with asymmetric content, so its 180-degree turn is tellable."""
    height = 740
    width = round(height * CARD_ASPECT)
    card = np.full((height, width, 3), 210, np.uint8)
    rng = np.random.default_rng(seed)
    # Distinctive marks near the top only — the asymmetry the match relies on.
    card[40:220, 40 : width - 40] = rng.integers(0, 120, (180, width - 80, 3), dtype=np.uint8)
    card[300:700, 60 : width - 60] = 150
    return card


def _scan(tmp_path, name: str, ccw_on_glass: int, seed: int = 0):
    """A scan of that card turned `ccw_on_glass` degrees on the glass."""
    card = _card_image(seed)
    frame = np.full((1000, 1000, 3), 12, np.uint8)
    h, w = card.shape[:2]
    top, left = (1000 - h) // 2, (1000 - w) // 2
    frame[top : top + h, left : left + w] = card
    codes = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    if ccw_on_glass % 360:
        frame = cv2.rotate(frame, codes[ccw_on_glass % 360])
    path = tmp_path / name
    cv2.imwrite(str(path), frame)
    return path


@pytest.fixture
def reference():
    """The main flat capture, which every rotation scan is matched against."""
    card = _card_image()
    canonical = (THRESHOLDS["capture"]["canonical_width_px"], THRESHOLDS["capture"]["canonical_height_px"])
    return cv2.resize(card, canonical, interpolation=cv2.INTER_AREA)


class TestMeasuredRotation:
    @pytest.mark.parametrize("turned", [0, 90, 180, 270])
    def test_each_quarter_turn_is_recovered(self, tmp_path, reference, turned):
        """A card turned `turned` degrees clockwise on the glass has to be
        turned that far counter-clockwise to stand upright again."""
        # A clockwise turn on the glass is a negative counter-clockwise one.
        path = _scan(tmp_path, f"s{turned}.png", ccw_on_glass=-turned)
        warp, ccw, confident, _ = grade_module.normalise_scan(path, reference, THRESHOLDS)
        assert warp is not None
        assert confident, "orientation should be unambiguous on an asymmetric card"
        assert ccw % 360 == turned % 360, f"measured {ccw}, card was turned {turned}"

    def test_a_turned_scan_comes_back_upright(self, tmp_path, reference):
        """Not just the angle — the warp itself has to be usable."""
        path = _scan(tmp_path, "s.png", ccw_on_glass=180)
        warp, _, _, _ = grade_module.normalise_scan(path, reference, THRESHOLDS)
        assert grade_module._orientation_match(warp, reference) > 0.9

    def test_upside_down_is_told_apart_from_upright(self, tmp_path, reference):
        """The case geometry cannot decide: both are portrait, both pass the
        aspect gate, and only the content separates them."""
        upright = _scan(tmp_path, "up.png", ccw_on_glass=0)
        flipped = _scan(tmp_path, "down.png", ccw_on_glass=180)
        assert grade_module.normalise_scan(upright, reference, THRESHOLDS)[1] % 360 == 0
        assert grade_module.normalise_scan(flipped, reference, THRESHOLDS)[1] % 360 == 180

    def test_a_scan_with_no_card_is_refused(self, tmp_path, reference):
        blank = tmp_path / "blank.png"
        cv2.imwrite(str(blank), np.full((1000, 1000, 3), 12, np.uint8))
        warp, ccw, confident, _ = grade_module.normalise_scan(blank, reference, THRESHOLDS)
        assert warp is None and ccw is None and confident is False

    def test_an_unmatchable_card_is_reported_as_unconfident(self, tmp_path, reference):
        """Falls back to the declared order rather than picking at random."""
        other = _scan(tmp_path, "other.png", ccw_on_glass=0, seed=99)
        flat = np.full_like(reference, 128)
        _, _, confident, _ = grade_module.normalise_scan(other, flat, THRESHOLDS)
        assert confident is False


class TestAzimuthsFollowTheMeasurement:
    def test_file_order_no_longer_decides_the_light(self, tmp_path, reference):
        """The set scanned 0, 90, 180, 270 but handed over in a shuffled
        order still gets each frame the light direction it was taken under."""
        turns = [0, 90, 180, 270]
        measured = {}
        for turned in turns:
            path = _scan(tmp_path, f"s{turned}.png", ccw_on_glass=-turned)
            measured[turned] = grade_module.normalise_scan(path, reference, THRESHOLDS)[1] % 360
        assert measured == {t: t for t in turns}

    def test_the_turn_direction_no_longer_decides_it_either(self, tmp_path, reference):
        """A counter-clockwise turn on the glass needs a clockwise rotation
        to undo — and comes back as that, measured, without anyone declaring
        which way the card went round."""
        for glass_ccw in (90, 270):
            path = _scan(tmp_path, f"ccw{glass_ccw}.png", ccw_on_glass=glass_ccw)
            applied = grade_module.normalise_scan(path, reference, THRESHOLDS)[1] % 360
            assert applied == (-glass_ccw) % 360


class TestOrientationMatch:
    def test_an_image_matches_itself(self, reference):
        assert grade_module._orientation_match(reference, reference) == pytest.approx(1.0, abs=1e-5)

    def test_a_half_turn_scores_lower_than_the_match(self, reference):
        turned = cv2.rotate(reference, cv2.ROTATE_180)
        assert grade_module._orientation_match(turned, reference) < grade_module.MIN_ORIENTATION_CONFIDENCE

    def test_a_flat_image_does_not_divide_by_zero(self, reference):
        flat = np.full_like(reference, 200)
        assert grade_module._orientation_match(flat, reference) == 0.0
