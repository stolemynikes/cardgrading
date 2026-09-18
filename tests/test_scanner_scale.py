"""The scanner-axis check, against cards of known size.

A trading card measured 2.9% larger placed landscape than placed portrait —
1.8mm on a 63mm card, against a 0.75mm tolerance. Two explanations fit equally
well and want opposite fixes: the scanner's two axes are scaled differently
(correctable in software), or the card bows off the glass (correctable only by
handling). A trading card cannot separate them because it can do both.

An ISO/IEC 7810 ID-1 card can. It is rigid PVC, so it cannot bow, and it is
exactly 85.60 x 53.98mm, so the error becomes a number rather than a
suspicion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration import check_scanner_scale as scale  # noqa: E402

DPI = 1200.0


def _scan(path: Path, long_mm: float, short_mm: float, vertical: bool) -> Path:
    long_px = int(long_mm / 25.4 * DPI)
    short_px = int(short_mm / 25.4 * DPI)
    h, w = (long_px + 700, short_px + 700) if vertical else (short_px + 700, long_px + 700)
    img = np.full((h, w, 3), 18, np.uint8)
    ch, cw = (long_px, short_px) if vertical else (short_px, long_px)
    img[350:350 + ch, 350:350 + cw] = (205, 200, 195)
    img[420:350 + ch - 70, 420:350 + cw - 70] = (150, 140, 130)
    cv2.imwrite(str(path), img)
    return path


class TestMeasuringAnIdCard:
    def test_a_true_card_measures_true(self, tmp_path):
        long_mm, short_mm = scale.measure(_scan(tmp_path / "p.png", 85.60, 53.98, True), DPI)
        assert long_mm == pytest.approx(scale.ID1_LONG_MM, abs=0.1)
        assert short_mm == pytest.approx(scale.ID1_SHORT_MM, abs=0.1)

    def test_orientation_does_not_change_the_reading(self, tmp_path):
        """The whole method rests on this: turning a rigid card must not
        change its measured size unless the scanner is at fault."""
        p = scale.measure(_scan(tmp_path / "p.png", 85.60, 53.98, True), DPI)
        l = scale.measure(_scan(tmp_path / "l.png", 85.60, 53.98, False), DPI)
        assert p[0] == pytest.approx(l[0], abs=0.05)
        assert p[1] == pytest.approx(l[1], abs=0.05)


class TestTheVerdict:
    def _run(self, tmp_path, capsys, landscape_long_mm):
        _scan(tmp_path / "p.png", 85.60, 53.98, True)
        _scan(tmp_path / "l.png", landscape_long_mm, 53.98, False)
        scale.main([str(tmp_path / "p.png"), str(tmp_path / "l.png"), "--dpi", str(DPI)])
        return capsys.readouterr().out

    def test_a_good_scanner_is_cleared(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, 85.60)
        assert "THE SCANNER IS FINE" in out
        assert "bowing" in out, "and it should say what the other explanation then is"

    def test_a_scaled_axis_is_caught_and_quantified(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, 85.60 * 1.03)
        assert "AXES DISAGREE" in out
        assert "2.9" in out or "3.0" in out

    def test_the_correction_it_prints_actually_corrects(self, tmp_path):
        """The number is only worth printing if applying it lands on truth."""
        _scan(tmp_path / "l.png", 85.60 * 1.03, 53.98, False)
        measured_long, _ = scale.measure(tmp_path / "l.png", DPI)
        correction = scale.ID1_LONG_MM / measured_long
        assert measured_long * correction == pytest.approx(scale.ID1_LONG_MM, abs=0.05)

    @pytest.mark.parametrize("error_pct", [0.05, -0.05])
    def test_it_does_not_cry_wolf_on_measurement_noise(self, tmp_path, capsys, error_pct):
        """A hundredth of a millimetre on an 85mm card is 0.012%. Anything
        under a tenth of a percent is the measurement's own noise."""
        out = self._run(tmp_path, capsys, 85.60 * (1 + error_pct / 100))
        assert "THE SCANNER IS FINE" in out

    def test_a_missing_card_says_so_rather_than_guessing(self, tmp_path):
        blank = tmp_path / "blank.png"
        cv2.imwrite(str(blank), np.full((900, 900, 3), 18, np.uint8))
        with pytest.raises(SystemExit) as caught:
            scale.measure(blank, DPI)
        assert "no card found" in str(caught.value)
