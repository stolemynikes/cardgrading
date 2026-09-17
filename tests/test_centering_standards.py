"""The rules behind the centering grade, as opposed to the measurement of it.

Three of these come from reading what other centering tools do and what PSA
actually publishes: the 5% front leeway is a published allowance we were
ignoring, "the most off-center part of the card" is a point rather than an
average, and every comparable tool reports more than one grading service.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import centering
from webapp import main

THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())
FRONT = THRESHOLDS["centering"]["front_tolerances"]
BACK = THRESHOLDS["centering"]["back_tolerances"]


class TestFrontLeeway:
    """PSA: "A 5% leeway is given to the front centering minimum standards for
    cards which grade PSA 7 or better." Front only, grade 7 and up."""

    def test_a_card_just_past_the_line_keeps_the_better_grade(self):
        # The real case this came from: 58px against 108px is 65.06%, a hair
        # past the 65 limit for an 8.
        assert centering._grade_from_ratio(65.06, FRONT) == 7
        assert centering._grade_from_ratio(65.06, FRONT, 5.0, 7.0) == 8

    def test_the_leeway_is_five_points_not_five_percent_relative(self):
        """55 x 1.05 would be 57.75; the published figure sits in a table of
        percentage points, so five points is the reading."""
        assert centering._grade_from_ratio(59.9, FRONT, 5.0, 7.0) == 10
        assert centering._grade_from_ratio(60.1, FRONT, 5.0, 7.0) == 9

    def test_it_does_not_reach_below_grade_seven(self):
        strict = centering._grade_from_ratio(84.0, FRONT)
        assert centering._grade_from_ratio(84.0, FRONT, 5.0, 7.0) == strict

    def test_backs_get_no_leeway(self):
        """Published for the front only — and the 10,000-card study found
        zero backs outside the published back tolerance, so there is no
        evidence of an unpublished one either."""
        axis = centering._axis_centering("left", "right", 10, 90, BACK, 0.0, 7.0)
        assert axis.leeway_applied is False

    def test_the_strict_grade_is_reported_alongside(self):
        axis = centering._axis_centering("left", "right", 58, 108, FRONT, 5.0, 7.0)
        assert axis.grade == 8
        assert axis.strict_grade == 7
        assert axis.leeway_applied is True

    def test_a_card_inside_the_table_is_not_marked_as_using_leeway(self):
        axis = centering._axis_centering("left", "right", 100, 100, FRONT, 5.0, 7.0)
        assert axis.leeway_applied is False
        assert axis.grade == axis.strict_grade == 10

    def test_the_serialized_axis_carries_both(self):
        d = centering.axis_dict(centering._axis_centering("left", "right", 58, 108, FRONT, 5.0, 7.0))
        assert d["grade"] == 8 and d["strict_grade"] == 7 and d["leeway_applied"] is True


class TestMostOffCentrePoint:
    """PSA grades "the percent of difference at the most off-center part of
    the card". Sampling only the middle reports a skew-cut card at its best."""

    @staticmethod
    def _card(skew_px: int) -> np.ndarray:
        """A bordered card whose left border widens by `skew_px` down the side."""
        h, w = 2100, 1500
        image = np.full((h, w, 3), 240, np.uint8)
        for y in range(h):
            left = 100 + int(skew_px * y / h)
            image[y, left : w - 100] = (40, 60, 90)
        return image

    def test_a_square_cut_reads_the_same_wherever_it_is_sampled(self):
        cfg = THRESHOLDS["centering"]["border_detect"]
        (left, right, _, _), _, variation = centering._measure_borders(self._card(0), cfg)
        assert variation["left"] < 3
        assert abs(left - right) < 6

    def test_a_skewed_cut_is_caught_and_reported(self):
        cfg = THRESHOLDS["centering"]["border_detect"]
        card = self._card(20)
        (left, _, _, _), _, variation = centering._measure_borders(card, cfg)
        one_band = centering._measure_borders(card, {**cfg, "sample_bands": 1})[0][0]
        assert variation["left"] > 5, "the wander down the side is measured"
        assert left > one_band, "reads the card at its worst point, not its middle"

    def test_narrow_bands_keep_a_skewed_border_measurable(self):
        """The real payoff. A slanted edge smears the Canny peak across the
        band it's summed over; a shorter band contains less of the slant, so
        the peak stays sharp enough to clear the confidence floor."""
        cfg = THRESHOLDS["centering"]["border_detect"]
        card = self._card(20)
        floor = cfg["min_boundary_confidence"]
        wide = centering._measure_borders(card, {**cfg, "sample_bands": 1})[1]["left"]
        narrow = centering._measure_borders(card, cfg)[1]["left"]
        assert wide < floor, "one wide band refuses this card"
        assert narrow >= floor, "five narrow bands measure it"

    def test_sampling_one_band_restores_the_old_behaviour(self):
        cfg = {**THRESHOLDS["centering"]["border_detect"], "sample_bands": 1}
        (left, _, _, _), _, variation = centering._measure_borders(self._card(20), cfg)
        assert variation["left"] == 0, "a single band has nothing to vary against"

    def test_bands_stay_clear_of_the_corners(self):
        bands = centering._sample_bands(2100, 3)
        assert min(start for start, _ in bands) >= 2100 * centering.SAMPLE_SPAN[0] - 1
        assert max(end for _, end in bands) <= 2100 * centering.SAMPLE_SPAN[1] + 1

    def test_bands_do_not_overlap(self):
        bands = centering._sample_bands(2100, 3)
        assert all(bands[i][1] <= bands[i + 1][0] for i in range(len(bands) - 1))

    def test_low_confidence_samples_cannot_win_the_worst_vote(self):
        """Taking the max over noisy picks would reliably select the noise."""
        samples = [
            {"left": 50.0, "right": 50.0, "confidence": {"left": 0.9, "right": 0.9}},
            {"left": 400.0, "right": 10.0, "confidence": {"left": 0.01, "right": 0.01}},
        ]
        chosen, _, _ = centering._worst_sample(samples, ("left", "right"), 0.35)
        assert chosen["left"] == 50.0

    def test_with_nothing_confident_a_sample_is_still_returned(self):
        samples = [
            {"left": 10.0, "right": 20.0, "confidence": {"left": 0.01, "right": 0.01}},
            {"left": 11.0, "right": 21.0, "confidence": {"left": 0.02, "right": 0.02}},
        ]
        chosen, _, _ = centering._worst_sample(samples, ("left", "right"), 0.35)
        assert chosen is not None


class TestHalfGrades:
    def test_a_half_grade_table_returns_halves(self):
        tag = THRESHOLDS["centering"]["graders"]["tag"]["front"]
        assert centering._grade_from_ratio(62.0, tag) == 8.5
        assert centering._grade_from_ratio(63.0, tag) == 8, "62.5 is the 8.5 limit, so 63 is an 8"

    def test_whole_grades_still_serialize_as_integers(self):
        """Emitting 7.0 where every previous report said 7 would churn every
        stored report and every test for nothing."""
        d = centering.axis_dict(centering._axis_centering("left", "right", 100, 100, FRONT))
        assert d["grade"] == 10
        assert json.dumps(d["grade"]) == "10"

    def test_a_half_grade_survives_serialization(self):
        tag = THRESHOLDS["centering"]["graders"]["tag"]["front"]
        d = centering.axis_dict(centering._axis_centering("left", "right", 38, 62, tag))
        assert json.dumps(d["grade"]) == "8.5"


class TestConventionalRatio:
    """PSA and every third-party tool print the larger share first. We lead
    with left/right because which side the card shifted toward is half the
    information — so report both rather than pick."""

    def test_both_orderings_are_emitted(self):
        d = centering.axis_dict(centering._axis_centering("left", "right", 35, 65, FRONT))
        assert d["ratio"] == "35/65"
        assert d["ratio_conventional"] == "65/35"

    def test_a_centred_card_reads_the_same_either_way(self):
        d = centering.axis_dict(centering._axis_centering("left", "right", 100, 100, FRONT))
        assert d["ratio"] == d["ratio_conventional"] == "50/50"


class TestGraderComparison:
    def _centering(self, front_pct: float, back_pct: float | None = 50.0) -> dict:
        def side(pct):
            if pct is None:
                return {"horizontal": {"measurable": False}, "vertical": {"measurable": False}}
            return {
                "horizontal": {"side_a_pct": 100 - pct, "side_b_pct": pct, "measurable": True},
                "vertical": {"side_a_pct": 50.0, "side_b_pct": 50.0, "measurable": True},
            }

        return {"front": side(front_pct), "back": side(back_pct)}

    def test_every_configured_grader_reports(self):
        out = centering.compare_graders(self._centering(55.0), THRESHOLDS)
        assert set(out) == set(THRESHOLDS["centering"]["graders"])

    def test_graders_disagree_where_their_tables_do(self):
        """BGS grades both axes against its front table and is stricter than
        PSA at the top; the same card lands on different grades."""
        out = centering.compare_graders(self._centering(58.0, 50.0), THRESHOLDS)
        assert out["psa"]["front"] != out["bgs"]["front"]

    def test_an_unmeasurable_side_does_not_vote(self):
        """An unmeasurable back must not silently become a perfect one."""
        out = centering.compare_graders(self._centering(58.0, None), THRESHOLDS)
        assert out["psa"]["back"] is None
        assert out["psa"]["grade"] == out["psa"]["front"]

    def test_a_grader_with_no_back_table_reports_none(self):
        out = centering.compare_graders(self._centering(55.0, 80.0), THRESHOLDS)
        assert out["sgc"]["back"] is None
        assert out["sgc"]["front"] is not None

    def test_the_worse_axis_is_the_one_that_counts(self):
        block = self._centering(52.0)
        block["front"]["vertical"] = {"side_a_pct": 25.0, "side_b_pct": 75.0, "measurable": True}
        out = centering.compare_graders(block, THRESHOLDS)
        assert out["psa"]["front"] <= 7

    def test_tag_grades_on_halves(self):
        out = centering.compare_graders(self._centering(62.0), THRESHOLDS)
        assert out["tag"]["front"] == 8.5

    def test_psa_leeway_applies_in_the_comparison_too(self):
        out = centering.compare_graders(self._centering(65.06), THRESHOLDS)
        assert out["psa"]["front"] == 8

    def test_only_psa_gets_a_leeway(self):
        """The 5% allowance is PSA's published rule, not a general one."""
        graders = THRESHOLDS["centering"]["graders"]
        assert all(g.get("front_leeway_points", 0) == 0 for key, g in graders.items() if key != "psa")

    def test_every_table_states_where_it_came_from(self):
        """These are third-party transcriptions, and a table taken on trust is
        how the back tolerances were wrong here before."""
        for grader in THRESHOLDS["centering"]["graders"].values():
            assert grader.get("source")

    def test_tables_are_ordered_best_grade_first(self):
        """_grade_from_ratio walks the list and returns the first tier that
        fits; an out-of-order table would silently hand out wrong grades."""
        for key, grader in THRESHOLDS["centering"]["graders"].items():
            for side in ("front", "back"):
                table = grader.get(side) or []
                grades = [tier["grade"] for tier in table]
                ratios = [tier["max_ratio"] for tier in table]
                assert grades == sorted(grades, reverse=True), f"{key}.{side} grades"
                assert ratios == sorted(ratios), f"{key}.{side} ratios"

    def test_nothing_measurable_produces_no_verdict(self):
        out = centering.compare_graders(self._centering(None, None), THRESHOLDS)
        assert out["psa"]["grade"] is None
