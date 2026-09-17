"""Dimension measurement — only meaningful when the capture's scale is known."""

from __future__ import annotations

import numpy as np

from pipeline import dimensions

THRESHOLDS = {"dimensions": {"tolerance_mm": 0.75, "squareness_tolerance_deg": 1.0, "miscut_grade_cap": 8.0}}
DPI = 1200.0
PX_PER_MM = DPI / dimensions.MM_PER_INCH


def _quad(width_mm: float, height_mm: float, skew_px: float = 0.0) -> np.ndarray:
    w = width_mm * PX_PER_MM
    h = height_mm * PX_PER_MM
    return np.array([[0, 0], [w, skew_px], [w, h + skew_px], [0, h]], dtype=np.float32)


def test_nominal_card_passes():
    result = dimensions.measure_dimensions(_quad(63.0, 88.0), DPI, THRESHOLDS)
    assert result.measurable
    assert result.within_tolerance
    assert round(result.width_mm) == 63
    assert round(result.height_mm) == 88


def test_trimmed_card_flagged():
    result = dimensions.measure_dimensions(_quad(61.0, 88.0), DPI, THRESHOLDS)
    assert result.within_tolerance is False
    assert "width off by" in result.note


def test_diamond_cut_flagged_at_correct_size():
    """Right dimensions, wrong angles — a card cut out of square still caps."""
    result = dimensions.measure_dimensions(_quad(63.0, 88.0, skew_px=80.0), DPI, THRESHOLDS)
    assert result.within_tolerance is False
    assert "out of square" in result.note


def test_landscape_scan_is_not_reported_as_miscut():
    """A card scanned on its side measures 88x63; orientation is a property of
    how it was placed on the glass, not of the card."""
    result = dimensions.measure_dimensions(_quad(88.0, 63.0), DPI, THRESHOLDS)
    assert result.within_tolerance


def test_unknown_scale_is_unmeasurable_not_guessed():
    result = dimensions.measure_dimensions(_quad(63.0, 88.0), None, THRESHOLDS)
    assert not result.measurable
    assert result.within_tolerance is None
    assert result.width_mm is None


def test_no_quad_is_unmeasurable():
    assert not dimensions.measure_dimensions(None, DPI, THRESHOLDS).measurable


def test_result_is_json_serializable():
    """numpy scalars leak out of the angle math; json.dumps rejects them."""
    import json

    json.dumps(dimensions.measure_dimensions(_quad(63.0, 88.0), DPI, THRESHOLDS).to_dict())


class TestMeasuringFromSeveralScans:
    """One scan states a confident number it has no way to check.

    Measured on a real flatbed, the same card came out 2.9% different — 1.8mm
    on a 63mm card — depending only on whether it was lying portrait or
    landscape on the glass, against a 0.75mm tolerance. The same card read
    "2.13mm miscut" in one run and "within tolerance" in the next, decided by
    which scan happened to be the flat capture.
    """

    THRESHOLDS = {"dimensions": {"tolerance_mm": 0.75, "squareness_tolerance_deg": 1.0}}
    DPI = 1200.0

    @staticmethod
    def _quad(width_mm: float, height_mm: float, dpi: float = 1200.0) -> np.ndarray:
        w = width_mm / 25.4 * dpi
        h = height_mm / 25.4 * dpi
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    def test_the_median_is_the_figure(self):
        """One bad detection shouldn't move the answer."""
        quads = [self._quad(63.0, 88.0), self._quad(63.1, 88.1), self._quad(58.0, 81.0)]
        result = dimensions.measure_from_scans(quads, self.DPI, self.THRESHOLDS)
        assert abs(result.width_mm - 63.1) < 0.2

    def test_scans_that_agree_still_give_a_verdict(self):
        quads = [self._quad(63.0, 88.0), self._quad(63.1, 88.05), self._quad(62.95, 87.95)]
        result = dimensions.measure_from_scans(quads, self.DPI, self.THRESHOLDS)
        assert result.within_tolerance is True
        assert result.spread_mm < 0.75
        assert result.sample_count == 3

    def test_scans_that_disagree_give_no_verdict_at_all(self):
        """The real case: 1.8mm of scatter judged against a 0.75mm tolerance.
        A measurement that disagrees with itself by more than the thing it is
        being judged against cannot settle the question."""
        quads = [self._quad(63.0, 88.0), self._quad(61.2, 85.5), self._quad(63.0, 88.0), self._quad(61.2, 85.5)]
        result = dimensions.measure_from_scans(quads, self.DPI, self.THRESHOLDS)
        assert result.measurable is True, "the size is still reported"
        assert result.within_tolerance is None, "but no miscut verdict is given"
        assert result.spread_mm > 0.75
        assert "disagree" in result.note

    def test_a_genuinely_trimmed_card_is_still_caught(self):
        """Refusing a verdict when scans disagree must not become refusing
        every verdict — consistent scans of a small card still fail."""
        quads = [self._quad(60.0, 85.0), self._quad(60.05, 85.05), self._quad(59.95, 84.95)]
        result = dimensions.measure_from_scans(quads, self.DPI, self.THRESHOLDS)
        assert result.within_tolerance is False
        assert "Miscut or trimmed" in result.note

    def test_the_spread_is_reported(self):
        quads = [self._quad(63.0, 88.0), self._quad(63.4, 88.0)]
        result = dimensions.measure_from_scans(quads, self.DPI, self.THRESHOLDS)
        assert abs(result.spread_mm - 0.4) < 0.05
        assert result.to_dict()["spread_mm"] is not None

    def test_a_single_scan_behaves_exactly_as_before(self):
        quads = [self._quad(63.0, 88.0)]
        multi = dimensions.measure_from_scans(quads, self.DPI, self.THRESHOLDS)
        single = dimensions.measure_dimensions(quads[0], self.DPI, self.THRESHOLDS)
        assert multi.to_dict() == single.to_dict()
        assert multi.sample_count == 1
        assert multi.spread_mm is None, "one scan has nothing to compare itself against"

    def test_no_scans_at_all_is_unmeasurable(self):
        assert dimensions.measure_from_scans([], self.DPI, self.THRESHOLDS).measurable is False

    def test_an_unknown_dpi_is_still_unmeasurable(self):
        quads = [self._quad(63.0, 88.0), self._quad(63.0, 88.0)]
        assert dimensions.measure_from_scans(quads, None, self.THRESHOLDS).measurable is False
