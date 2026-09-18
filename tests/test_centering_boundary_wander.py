"""Refusing an axis whose "border" isn't a border.

One physical card, scanned twice on the same scanner at the same settings,
measured 19/81 on one pass and 67/33 on the other — and reported centering
grade 5 and grade 8 for a card whose centering had not changed between them.

Both readings arrived with a variation figure saying the sample bands
disagreed: 159px and 99px of wander along sides whose borders were 52px and
77px wide. The figure was recorded and nothing acted on it.

A printed border edge is a straight line parallel to the card edge, so on a
card whose border the detector really found, that wander is near zero.
Measured: synthetic cards with straight printed borders run 0.00-0.03 of the
border width, including one cut deliberately off-centre; the real card ran
0.74 to 3.06. The gap is two orders of magnitude, which is what makes the
threshold safe to set.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import centering

W, H = 1500, 2100

THRESHOLDS = {
    "centering": {
        "front_tolerances": [
            {"grade": 10, "max_ratio": 55},
            {"grade": 8, "max_ratio": 65},
            {"grade": 6, "max_ratio": 75},
            {"grade": 3, "max_ratio": 90},
        ],
        "back_tolerances": [{"grade": 10, "max_ratio": 75}, {"grade": 7, "max_ratio": 90}],
        "border_detect": {
            "canny_low": 40,
            "canny_high": 120,
            "search_margin_pct": 12,
            "min_boundary_confidence": 0.35,
        },
    }
}


def _straight(bw_left=60, bw_right=60, bw_top=60, bw_bottom=60) -> np.ndarray:
    """A card with straight printed borders — what the detector is for."""
    rng = np.random.default_rng(3)
    img = np.full((H, W, 3), 210, np.float32)
    img[bw_top:H - bw_bottom, bw_left:W - bw_right] = 120
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _stepped(inset: int = 180, border: int = 60) -> np.ndarray:
    """Straight, confident edges — at two different insets down one side.

    This, not a wavy line, is the real failure. A merely wavy boundary is
    already caught: it weakens every band's edge, and the confidence gate
    refuses the axis. What got through was a card where each band found a
    strong straight edge and the bands disagreed about *which* edge — a title
    bar and an art panel sitting at different insets, so the samples were
    individually confident and collectively meaningless.
    """
    rng = np.random.default_rng(3)
    img = np.full((H, W, 3), 210, np.float32)
    img[border:H - border, border:W - border] = 120
    img[700:1400, border:inset] = 210
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestAStraightBorderStillMeasures:
    @pytest.mark.parametrize(
        "kwargs",
        [{}, dict(bw_left=40, bw_right=80),
         dict(bw_left=120, bw_right=120, bw_top=120, bw_bottom=120)],
        ids=["even", "off-centre", "wide borders"],
    )
    def test_it_is_not_refused(self, kwargs):
        card = _straight(**kwargs)
        result = centering.measure_centering(card, card, THRESHOLDS)
        assert result.front_horizontal.measurable
        assert result.front_vertical.measurable
        assert result.front_grade is not None

    def test_its_wander_is_a_tiny_fraction_of_the_border(self):
        """The headroom the threshold sits in. Anything near the limit would
        make this test fragile, and it isn't close."""
        card = _straight()
        result = centering.measure_centering(card, card, THRESHOLDS)
        for axis in (result.front_horizontal, result.front_vertical):
            narrower = min(axis.side_a_px, axis.side_b_px)
            assert axis.variation_px / narrower < 0.1


class TestAWanderingBoundaryIsRefused:
    def test_it_stops_being_a_grade(self):
        card = _stepped()
        result = centering.measure_centering(card, card, THRESHOLDS)
        assert result.front_horizontal.measurable is False

    def test_the_reason_says_what_is_wrong_and_what_to_do(self):
        card = _stepped()
        result = centering.measure_centering(card, card, THRESHOLDS)
        reason = result.front_horizontal.reason
        assert reason and "wanders" in reason
        assert "by hand" in reason, "the manual adjustment is the answer, so it should be offered"

    def test_the_numbers_are_still_reported(self):
        """Refused is not deleted. The widths and the wander are what let
        someone see *why* it was refused, and the manual overlay seeds its
        lines from them."""
        card = _stepped()
        axis = centering.measure_centering(card, card, THRESHOLDS).front_horizontal
        assert axis.side_a_px > 0
        assert axis.variation_px > 0
        assert centering.axis_dict(axis)["reason"]

    def test_a_refused_axis_does_not_vote(self):
        """A side's grade is the min over its *measurable* axes. A refused
        axis must not drag the grade down with a number nobody trusts."""
        card = _stepped()
        result = centering.measure_centering(card, card, THRESHOLDS)
        if result.front_vertical.measurable:
            assert result.front_grade == result.front_vertical.grade
        else:
            assert result.front_grade is None


class TestTheThresholdIsConfigurable:
    def test_a_looser_limit_lets_a_wandering_axis_through(self):
        card = _stepped()
        loose = {"centering": {**THRESHOLDS["centering"], "max_boundary_wander": 99.0}}
        assert centering.measure_centering(card, card, loose).front_horizontal.measurable

    def test_a_tighter_limit_refuses_a_straight_one(self):
        card = _straight()
        tight = {"centering": {**THRESHOLDS["centering"], "max_boundary_wander": 0.0001}}
        assert centering.measure_centering(card, card, tight).front_horizontal.measurable is False
