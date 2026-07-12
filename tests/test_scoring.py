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
