"""Grade assembly: heuristic combination, fitted-weights path, and fitting."""

import pytest

from pipeline import scoring


class TestHeuristic:
    def test_perfect_card(self):
        ge = scoring.assemble_grade(10, 10, 10, {})
        assert ge.overall_grade == 10.0
        assert ge.overall_grade_rounded == 10

    def test_worst_subgrade_dominates(self):
        # A single weak category must cap the overall near it — a 10/10/4
        # card is not a 8; PSA rarely grades more than ~1 above the worst.
        ge = scoring.assemble_grade(10, 10, 4, {})
        assert ge.overall_grade <= 5.0

    def test_missing_surface_grades_from_two_subgrades(self):
        ge = scoring.assemble_grade(9, 7, None, {})
        assert ge.surface_grade is None
        assert 7.0 <= ge.overall_grade <= 8.0
        assert "excludes" in ge.note and "surface" in ge.note

    def test_missing_centering_grades_from_remaining_subgrades(self):
        # Borderless/full-art card: centering unmeasurable — the overall
        # estimate must exclude it rather than crash or fabricate.
        ge = scoring.assemble_grade(None, 7, 8, {})
        assert ge.centering_grade is None
        assert 7.0 <= ge.overall_grade <= 8.0
        assert "centering" in ge.note

    def test_missing_centering_and_surface(self):
        ge = scoring.assemble_grade(None, 6, None, {})
        assert ge.overall_grade == pytest.approx(6.0)
        assert "centering" in ge.note and "surface" in ge.note

    def test_clamped_to_valid_range(self):
        ge = scoring.assemble_grade(1, 1, 1, {})
        assert 1.0 <= ge.overall_grade <= 10.0

    def test_rounded_matches_precise(self):
        ge = scoring.assemble_grade(8, 6, 7, {})
        assert ge.overall_grade_rounded == round(ge.overall_grade)


class TestFittedWeights:
    THRESHOLDS = {
        "scoring": {
            "fitted_weights": {
                "weights": {"centering": 0.2, "corners_edges": 0.2, "surface": 0.2, "min_sub_grade": 0.4},
                "intercept": 0.0,
            },
            "fit_metadata": {"n_samples": 12, "train_mae": 0.5, "loo_mae": 0.7},
        }
    }

    def test_fitted_path_used_when_available(self):
        ge = scoring.assemble_grade(10, 10, 5, self.THRESHOLDS)
        # 0.2*10 + 0.2*10 + 0.2*5 + 0.4*5 = 7.0
        assert ge.overall_grade == pytest.approx(7.0)
        assert "fit on 12 calibrated cards" in ge.note

    def test_falls_back_to_heuristic_without_surface(self):
        # Fitted weights need all three sub-grades; without surface the
        # heuristic must be used instead of crashing or guessing.
        ge = scoring.assemble_grade(10, 10, None, self.THRESHOLDS)
        assert "fit on" not in ge.note

    def test_falls_back_to_heuristic_without_centering(self):
        ge = scoring.assemble_grade(None, 10, 8, self.THRESHOLDS)
        assert "fit on" not in ge.note


class TestFitLinearWeights:
    def test_too_few_samples_returns_none(self):
        rows = [
            {"centering": 9, "corners_edges": 8, "surface": 9, "overall_actual": 8}
        ] * (scoring.MIN_CALIBRATION_SAMPLES - 1)
        assert scoring.fit_linear_weights(rows) is None

    def test_fit_recovers_simple_relationship(self):
        # overall == min(subgrades) exactly; the fit should predict well on
        # its own training data.
        import itertools

        rows = [
            {"centering": c, "corners_edges": ce, "surface": s, "overall_actual": min(c, ce, s)}
            for c, ce, s in itertools.product([4, 6, 8, 10], repeat=3)
        ]
        fitted = scoring.fit_linear_weights(rows)
        assert fitted is not None
        pred = scoring._fitted_overall(fitted, 10, 10, 4)
        assert pred == pytest.approx(4.0, abs=0.75)


class TestScore:
    THRESHOLDS = {"dimensions": {"miscut_grade_cap": 8.0}}

    def test_score_tracks_the_unrounded_estimate(self):
        """The point of the score is to separate cards the 1-10 grade lumps
        together, so two cards that round to the same grade must not get the
        same score."""
        strong = scoring.assemble_grade(10, 9, 9, self.THRESHOLDS)
        weak = scoring.assemble_grade(9, 9, 9, self.THRESHOLDS)
        assert strong.overall_grade_rounded == weak.overall_grade_rounded
        assert strong.score > weak.score

    def test_score_stays_in_range(self):
        assert scoring.grade_to_score(10.0) == scoring.MAX_SCORE
        assert scoring.grade_to_score(1.0) == scoring.MIN_SCORE
        assert scoring.grade_to_score(-5.0) == scoring.MIN_SCORE
        assert scoring.grade_to_score(99.0) == scoring.MAX_SCORE


class TestDimensionsCap:
    THRESHOLDS = {"dimensions": {"miscut_grade_cap": 8.0}}

    def test_miscut_caps_an_otherwise_clean_card(self):
        clean = scoring.assemble_grade(10, 10, 10, self.THRESHOLDS)
        miscut = scoring.assemble_grade(10, 10, 10, self.THRESHOLDS, dimensions_within_tolerance=False)
        assert clean.overall_grade == 10.0
        assert miscut.overall_grade == 8.0
        assert "cut tolerance" in miscut.note

    def test_cap_never_raises_a_worse_grade(self):
        capped = scoring.assemble_grade(4, 4, 4, self.THRESHOLDS, dimensions_within_tolerance=False)
        assert capped.overall_grade == 4.0

    def test_unmeasured_dimensions_change_nothing(self):
        """A phone photo can't measure the card, and 'unmeasured' must not be
        treated as either pass or fail."""
        baseline = scoring.assemble_grade(10, 10, 10, self.THRESHOLDS)
        unmeasured = scoring.assemble_grade(10, 10, 10, self.THRESHOLDS, dimensions_within_tolerance=None)
        within = scoring.assemble_grade(10, 10, 10, self.THRESHOLDS, dimensions_within_tolerance=True)
        assert baseline.overall_grade == unmeasured.overall_grade == within.overall_grade


class TestADefectCapsTheWholeCard:
    """A crease is not a surface problem, it is a card problem.

    Every published standard says so — PSA: "even a light crease usually caps
    you at PSA 6 or below, no matter how perfect everything else looks." The
    heuristic on its own cannot express that: it lets the overall grade sit a
    point above the worst sub-grade, so a card creased end to end came out a
    2 where PSA calls a full-width crease "usually an automatic 1".
    """

    def test_a_defect_ceiling_binds_the_overall_grade(self):
        ge = scoring.assemble_grade(10, 10, 5, {}, defect_grade_cap=5, defect_cap_reason="a crease")
        assert ge.overall_grade == pytest.approx(5.0)

    def test_a_full_length_crease_reaches_one(self):
        ge = scoring.assemble_grade(10, 10, 1, {}, defect_grade_cap=1, defect_cap_reason="a crease")
        assert ge.overall_grade == pytest.approx(1.0)

    def test_the_reason_is_stated(self):
        ge = scoring.assemble_grade(10, 10, 4, {}, defect_grade_cap=4, defect_cap_reason="a crease")
        assert "crease" in ge.note
        assert "caps the card" in ge.note

    def test_no_cap_leaves_the_heuristic_alone(self):
        """Ordinary surface wear is not a named defect: the sub-grade already
        says it, and capping on it twice would double-count."""
        with_cap = scoring.assemble_grade(8, 8, 4, {}, defect_grade_cap=None)
        assert with_cap.overall_grade > 4.0

    def test_a_cap_above_the_estimate_changes_nothing(self):
        plain = scoring.assemble_grade(9, 9, 9, {})
        capped = scoring.assemble_grade(9, 9, 9, {}, defect_grade_cap=10, defect_cap_reason="a pit")
        assert capped.overall_grade == pytest.approx(plain.overall_grade)
        assert "caps the card" not in capped.note

    def test_it_stacks_with_the_miscut_cap(self):
        """Two independent caps, both of which bind."""
        ge = scoring.assemble_grade(
            10, 10, 5, {"dimensions": {"miscut_grade_cap": 8.0}},
            dimensions_within_tolerance=False, defect_grade_cap=4, defect_cap_reason="a crease",
        )
        assert ge.overall_grade == pytest.approx(4.0)
        assert "crease" in ge.note
        assert "cut tolerance" in ge.note
