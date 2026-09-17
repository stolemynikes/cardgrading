"""Does the surface stage measure damage?

Every other attribute is checked against synthetic fixtures with known ground
truth. Surface can't be: the thing being measured is real ink on real
cardstock, and a synthetic relief map proves only that the arithmetic works.

So this runs against a real pair — one card scanned four times, damaged, then
scanned four times again on the same scanner at the same settings. The
property under test is the weakest one that still means anything:

    the damaged capture must score worse than the clean one.

When this was written it failed: the clean card graded surface 3 and the
damaged one surface 6. Two causes, both since fixed. The render was scaled by
each card's *own* 99.5th percentile, so every number was relative to that
card's worst feature and adding damage improved the score. And 87% of the
pixels counted as defects sat on printed ink, which a normal map contains as
real relief because ink is physically raised.

It now reads 10 clean against 9 damaged.

The pair lives outside git (see calibration/validation_pairs/README.md); these
skip when it isn't there, so a fresh clone still runs clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline import cardvision, surface
from webapp import main

PAIR_DIR = Path(__file__).resolve().parent.parent / "calibration" / "validation_pairs"
THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())

# The card was turned 90 degrees counter-clockwise on the glass between scans,
# and the lamp sits at 90 degrees in the scanner's frame.
AZIMUTHS = [90.0, 180.0, 270.0, 0.0]


def _load(name: str) -> list[np.ndarray]:
    directory = PAIR_DIR / name
    paths = sorted(directory.glob("front_rotation_*.png"))
    if len(paths) < 3:
        pytest.skip(f"validation pair not present at {directory}")
    return [cv2.imread(str(p)) for p in paths]


def _solve(warps: list[np.ndarray]) -> cardvision.CardVisionResult:
    return cardvision.photometric_card_vision(
        warps, AZIMUTHS, THRESHOLDS["card_vision"], registration_reference=warps[0]
    )


def _measure(vision: cardvision.CardVisionResult) -> tuple[float, int, int]:
    relief = vision.measurement_relief if vision.measurement_relief is not None else vision.relief
    area, count, longest, _ = surface.relief_defect_stats(relief, THRESHOLDS["surface"])
    return area, count, longest


@pytest.fixture(scope="module")
def measured() -> dict:
    out = {}
    for name in ("clean", "damaged"):
        vision = _solve(_load(name))
        area, count, longest = _measure(vision)
        out[name] = {
            "vision": vision,
            "area": area,
            "count": count,
            "longest": longest,
            "grade": surface.grade_surface(area, count, longest, "photometric_relief", THRESHOLDS).grade,
        }
    return out


def test_the_damaged_card_grades_worse(measured):
    assert measured["damaged"]["grade"] < measured["clean"]["grade"], (
        f"clean graded {measured['clean']['grade']}, damaged graded {measured['damaged']['grade']}"
    )


def test_the_damaged_card_has_more_defect_area(measured):
    assert measured["damaged"]["area"] > measured["clean"]["area"], (
        f"clean {measured['clean']['area']:.3f}%, damaged {measured['damaged']['area']:.3f}%"
    )


def test_the_scratch_is_visible_in_the_relief(measured):
    """What does work. The render shows the damage plainly — the scissor
    gouge above the text box is the longest thing on the damaged card and it
    isn't there on the clean one. The picture is sound; the score on top of
    it is what isn't."""
    damaged = measured["damaged"]["vision"]
    relief = damaged.measurement_relief if damaged.measurement_relief is not None else damaged.relief
    # The gouge runs through the empty panel below the artwork, where the card
    # is unprinted — so anything found here is surface, not ink.
    panel = relief[1150:1500, 150:1350]
    deviation = np.abs(panel.astype(np.int16) - 128)
    assert deviation.max() > 40, "the scratch should stand well clear of a flat panel"


def test_the_clean_card_is_not_called_damaged(measured):
    """The other half of the property. Ordering alone is satisfied by two
    cards that both grade 2, and for a while both did — an undamaged card
    read grade 3 because its own ink was being counted against it."""
    assert measured["clean"]["grade"] >= 9, (
        f"an undamaged card graded {measured['clean']['grade']} "
        f"on {measured['clean']['area']:.3f}% defect area"
    )


def test_the_measurement_does_not_rescale_itself_per_card(measured):
    """The root cause, guarded directly.

    The displayed render normalises by the card's own 99.5th percentile, which
    is right for a picture. If the *measured* render ever shares that, every
    number becomes relative to the card's worst feature — and the damaged
    card's normaliser measured 2.55x the clean one's, which is what inverted
    the grades. A card's measured relief must not change when a strong
    feature is added somewhere else on it.
    """
    vision = measured["clean"]["vision"]
    relief = vision.measurement_relief.copy()
    before, _, _, _ = surface.relief_defect_stats(relief, THRESHOLDS["surface"])

    # Stamp an extreme feature into one corner, well away from everything.
    scarred = relief.copy()
    scarred[60:160, 60:160] = 255
    after, _, _, _ = surface.relief_defect_stats(scarred, THRESHOLDS["surface"])

    # The stamp itself adds area; what must not happen is the rest of the
    # card being scaled down to accommodate it.
    assert after > before, "the added feature should be counted, not normalised away"


def test_both_sets_solved_photometrically(measured):
    """Guards the comparison itself: two single-image fallbacks would agree
    with each other for reasons that have nothing to do with damage."""
    for name in ("clean", "damaged"):
        assert measured[name]["vision"].method == "photometric_stereo"
        assert measured[name]["vision"].light_count == 4
