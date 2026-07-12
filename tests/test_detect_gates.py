"""Capture-quality gate semantics: hard (geometry) vs soft (image quality).

Hard-gate failures mean the detected quad isn't a usable image of the card,
so grading must block and demand a retake. Soft failures grade anyway with a
warning. These tests pin that classification and the gate math itself —
especially the aspect-ratio gate, which silently measured the always-canonical
warped image (and therefore could never fail) until it was pointed at the
pre-warp corner quad.
"""

import numpy as np
import pytest

from pipeline import detect

CAP_CFG = {
    "card_aspect_ratio": [63, 88],
    "aspect_ratio_tolerance_pct": 3.0,
    "aspect_ratio_tolerance_upright_pct": 8.0,
    "max_corner_angle_deviation_deg": 2.0,
    "min_input_shortest_side_px": 1200,
}


def quad(w: float, h: float, skew: float = 0.0) -> np.ndarray:
    """A card quad of the given size; skew shifts the top edge sideways."""
    return np.array(
        [[skew, 0], [w + skew, 0], [w, h], [0, h]], dtype=np.float32
    )


class TestHardSoftClassification:
    def test_geometry_gates_are_hard(self):
        for name in ["card_detection", "tilt", "aspect_ratio"]:
            assert detect.QualityGate(name, False, "x").hard is True

    def test_quality_gates_are_soft(self):
        for name in ["resolution", "glare", "uneven_lighting"]:
            assert detect.QualityGate(name, False, "x").hard is False

    def test_hard_failures_property_filters(self):
        gates = [
            detect.QualityGate("resolution", False, "soft fail"),
            detect.QualityGate("tilt", False, "hard fail"),
            detect.QualityGate("glare", True, "pass"),
        ]
        result = detect.DetectResult(ok=False, warped=None, gates=gates)
        assert [g.name for g in result.hard_failures] == ["tilt"]
        assert [g.name for g in result.failures] == ["resolution", "tilt"]

    def test_to_dict_carries_hard_flag(self):
        gates = [detect.QualityGate("tilt", False, "x"), detect.QualityGate("glare", True, "y")]
        d = detect.DetectResult(ok=False, warped=None, gates=gates).to_dict()
        assert d["gates"][0]["hard"] is True
        assert d["gates"][1]["hard"] is False


class TestAspectRatioGate:
    def test_card_shaped_quad_passes(self):
        # 63:88 exactly, at an arbitrary scale
        gate = detect.check_aspect_ratio(quad(630, 880), CAP_CFG)
        assert gate.passed

    def test_wrong_rectangle_fails(self):
        # A rectangle that boxed card+background together (real measured
        # failure: 28% off card ratio, with perfect 90-degree corners that
        # sail through the tilt gate).
        gate = detect.check_aspect_ratio(quad(515, 1000), CAP_CFG)
        assert not gate.passed
        assert gate.value > 20

    def test_measures_the_quad_not_the_warp(self):
        # The historical bug: measuring the warped image made the gate a
        # constant. Two very differently shaped quads must produce different
        # gate values.
        a = detect.check_aspect_ratio(quad(630, 880), CAP_CFG)
        b = detect.check_aspect_ratio(quad(880, 630), CAP_CFG)
        assert a.value != b.value

    def test_perspective_foreshortening_tolerated_when_upright(self):
        # ~5% aspect deviation with clean right-angle corners: plausibly a
        # slightly off-overhead camera, which perspective_correct fixes.
        # Strict without the tilt_ok flag, relaxed with it.
        squished = quad(630 * 0.95, 880)
        assert not detect.check_aspect_ratio(squished, CAP_CFG).passed
        assert detect.check_aspect_ratio(squished, CAP_CFG, tilt_ok=True).passed

    def test_gross_deviation_blocked_even_when_upright(self):
        # The real minAreaRect card+background box measured 28% off — must
        # stay blocked no matter how clean its corners are.
        assert not detect.check_aspect_ratio(quad(515, 1000), CAP_CFG, tilt_ok=True).passed


class TestTiltGate:
    def test_straight_quad_passes(self):
        gate = detect.check_tilt(quad(630, 880), CAP_CFG)
        assert gate.passed

    def test_skewed_quad_fails(self):
        gate = detect.check_tilt(quad(630, 880, skew=120), CAP_CFG)
        assert not gate.passed


class TestResolutionGate:
    def test_below_minimum_fails_soft(self):
        img = np.zeros((360, 270, 3), dtype=np.uint8)
        gate = detect.check_resolution(img, CAP_CFG)
        assert not gate.passed
        assert gate.hard is False

    def test_at_minimum_passes(self):
        img = np.zeros((1600, 1200, 3), dtype=np.uint8)
        assert detect.check_resolution(img, CAP_CFG).passed


class TestJsonSafety:
    def test_numpy_types_are_coerced(self):
        # json.dumps rejects np.bool_/np.float64 — QualityGate must coerce.
        import json

        gate = detect.QualityGate("tilt", np.bool_(True), "x", np.float64(1.5))
        json.dumps({"passed": gate.passed, "value": gate.value})
