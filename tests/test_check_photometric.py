"""The scanner-capability check, fed from a saved report's warps.

The check normally takes raw scans and warps them itself. Getting two raw
scans off a Windows machine and onto this one is friction, and the web app
already stores a perspective-corrected warp of anything it grades — so the
same measurement can run off those instead, with the detection step skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import check_photometric  # noqa: E402
from webapp import main  # noqa: E402

THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())


def _shaded_card(direction: int, relief: float, size=(600, 420)) -> np.ndarray:
    """A card with one ridge across it, lit from one side or the other.

    `direction` flips the lighting. A scanner that lights off-axis produces
    exactly this: the ridge's bright and dark flanks swap when the card turns.
    """
    h, w = size
    card = np.full((h, w, 3), 170, np.uint8)
    for offset in range(-6, 7):
        shade = int(relief * offset * direction)
        card[h // 2 + offset, :] = np.clip(170 + shade, 0, 255)
    return card


@pytest.fixture
def pair(tmp_path):
    def build(relief: float):
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        cv2.imwrite(str(a), _shaded_card(1, relief))
        # The second scan is the card turned 180 degrees, so what the app
        # stored is upside down relative to the first.
        cv2.imwrite(str(b), cv2.rotate(_shaded_card(-1, relief), cv2.ROTATE_180))
        return a, b

    return build


class TestPreWarpedInput:
    def test_a_directional_scanner_is_recognised(self, pair):
        a, b = pair(relief=9.0)
        pct, _ = check_photometric.measure(a, b, THRESHOLDS, pre_warped=True)
        assert pct >= check_photometric.GOOD_MODULATION_PCT

    def test_a_coaxial_scanner_reads_flat(self, pair):
        """No relief signal at all: rotating the card changes nothing, which
        is the whole failure mode a CIS unit is suspected of."""
        a, b = pair(relief=0.0)
        pct, _ = check_photometric.measure(a, b, THRESHOLDS, pre_warped=True)
        assert pct < check_photometric.MARGINAL_MODULATION_PCT

    def test_the_second_warp_is_turned_back(self, tmp_path):
        """Without it the two are compared upside down against each other and
        every card reads as wildly directional — a false pass."""
        image = np.zeros((40, 30, 3), np.uint8)
        image[0, 0] = (255, 255, 255)
        path = tmp_path / "w.png"
        cv2.imwrite(str(path), image)
        turned = check_photometric._warp(path, True, THRESHOLDS, pre_warped=True)
        assert tuple(turned[-1, -1]) == (255, 255, 255)

    def test_the_first_warp_is_left_alone(self, tmp_path):
        image = np.zeros((40, 30, 3), np.uint8)
        image[0, 0] = (255, 255, 255)
        path = tmp_path / "w.png"
        cv2.imwrite(str(path), image)
        kept = check_photometric._warp(path, False, THRESHOLDS, pre_warped=True)
        assert tuple(kept[0, 0]) == (255, 255, 255)

    def test_mismatched_warps_are_refused_rather_than_compared(self, tmp_path):
        """Warps from different pipeline stages aren't comparable, and a
        silent resize would fabricate a difference that isn't lighting."""
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        cv2.imwrite(str(a), np.zeros((400, 300, 3), np.uint8))
        cv2.imwrite(str(b), np.zeros((800, 600, 3), np.uint8))
        with pytest.raises(SystemExit) as excinfo:
            check_photometric.measure(a, b, THRESHOLDS, pre_warped=True)
        assert "different sizes" in str(excinfo.value)

    def test_detection_is_skipped(self, tmp_path):
        """A pre-warped card fills its frame edge to edge, so there is no
        background for the contour finder and it would refuse the image."""
        flat = np.full((600, 420, 3), 170, np.uint8)
        path = tmp_path / "flat.png"
        cv2.imwrite(str(path), flat)
        assert check_photometric._warp(path, False, THRESHOLDS, pre_warped=True).shape == flat.shape
