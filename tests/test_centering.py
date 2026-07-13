"""Centering measurability: real borders measure, absent borders say so.

Root-caused from the first real end-to-end card (a borderless full-art
promo): _measure_borders picks the argmax of a Canny edge profile, which
returns the strongest *noise* when no border boundary exists — that produced
a confident-looking "89/11 grade 3" for a card whose centering is simply not
measurable this way, and the garbage border widths then shrank the
corner/edge crops to 12x12 pixels.
"""

import numpy as np
import pytest

from pipeline import centering

THRESHOLDS = {
    "centering": {
        "front_tolerances": [
            {"grade": 10, "max_ratio": 55},
            {"grade": 8, "max_ratio": 65},
            {"grade": 6, "max_ratio": 75},
            {"grade": 3, "max_ratio": 90},
        ],
        "back_tolerances": [
            {"grade": 10, "max_ratio": 75},
            {"grade": 7, "max_ratio": 90},
        ],
        "border_detect": {
            "canny_low": 40,
            "canny_high": 120,
            "search_margin_pct": 12,
            "min_boundary_confidence": 0.35,
        },
    }
}

W, H = 1500, 2100


def bordered_card(bw_left=60, bw_right=60, bw_top=60, bw_bottom=60, border_val=210, panel_val=120):
    rng = np.random.default_rng(3)
    img = np.full((H, W, 3), border_val, np.float32)
    img[bw_top:H - bw_bottom, bw_left:W - bw_right] = panel_val
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def borderless_card():
    """Full-art style: textured dark artwork edge to edge, no frame line."""
    rng = np.random.default_rng(3)
    img = rng.normal(60, 25, (H, W, 3))
    for _ in range(400):
        y, x = rng.integers(0, H), rng.integers(0, W)
        img[max(0, y - 2):y + 2, max(0, x - 2):x + 2] = 200
    return np.clip(img, 0, 255).astype(np.uint8)


class TestMeasurable:
    def test_bordered_both_sides_measurable(self):
        card = bordered_card()
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert r.front_measurable and r.back_measurable
        assert r.front_grade is not None
        assert r.overall_grade is not None

    def test_centered_card_grades_10(self):
        card = bordered_card()
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert r.front_grade == 10

    def test_offcenter_card_measures_offset(self):
        # left border 2x the right: 67/33 -> front grade 6 band
        card = bordered_card(bw_left=90, bw_right=45)
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert r.front_measurable
        assert 63 <= r.front_horizontal.side_a_pct <= 71

    def test_borderless_front_is_unmeasurable(self):
        r = centering.measure_centering(borderless_card(), bordered_card(), THRESHOLDS)
        assert r.front_measurable is False
        assert r.front_grade is None
        # back is a normal bordered design and must still measure
        assert r.back_measurable is True
        assert r.back_grade is not None
        # overall uses what's measurable rather than going n/a entirely
        assert r.overall_grade == r.back_grade

    def test_both_unmeasurable_overall_none(self):
        card = borderless_card()
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert r.overall_grade is None

    def test_dim_capture_recovered_by_normalization(self):
        # Underexposed but real border (70 vs 45 gray levels): the raw Canny
        # pass sees nothing, but the gamma+CLAHE second-chance pass recovers
        # it — this was the real "back centering 99/1" failure mode, where a
        # dim capture hid a perfectly normal border.
        card = bordered_card(border_val=70, panel_val=45)
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert r.front_measurable
        assert r.front_grade == 10

    def test_normalization_does_not_rescue_borderless(self):
        # The second-chance pass must not turn borderless art into a fake
        # border — CLAHE inflates art texture too, hence its stricter bar.
        card = borderless_card()
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert r.front_measurable is False


def one_axis_card():
    """Vertical boundaries strong, horizontal ones invisible: border and
    panel share the same value along left/right (like a card back whose
    blue frame melts into blue swirl art in a soft capture), while the
    top/bottom border-panel transition has real contrast."""
    rng = np.random.default_rng(3)
    img = np.full((H, W, 3), 200, np.float32)
    # inner panel darker ONLY vertically: rows 60..H-60 keep border value at
    # the left/right columns so no vertical boundary edge exists
    img[60:H - 60, :] = 120
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestPerAxisMeasurability:
    """Real case: a soft capture of a card back measured top/bottom at 0.54
    and 0.82 confidence while left/right were ~0 (the blue-frame-to-swirl
    transition is the lowest-contrast boundary on the card). The old
    all-four-boundaries rule threw the good vertical measurement away."""

    def test_vertical_measures_while_horizontal_does_not(self):
        card = one_axis_card()
        r = centering.measure_centering(card, bordered_card(), THRESHOLDS)
        assert r.front_vertical.measurable is True
        assert r.front_horizontal.measurable is False
        # side still grades, from the measurable axis alone
        assert r.front_grade == r.front_vertical.grade
        # side-level flag means "produced a grade at all"
        assert r.front_measurable is True

    def test_axis_flags_in_dict(self):
        r = centering.measure_centering(one_axis_card(), bordered_card(), THRESHOLDS)
        d = r.to_dict()
        assert d["front"]["horizontal"]["measurable"] is False
        assert d["front"]["vertical"]["measurable"] is True
        assert d["front"]["grade"] is not None

    def test_overlay_draws_only_measurable_axis_lines(self):
        card = one_axis_card()
        r = centering.measure_centering(card, bordered_card(), THRESHOLDS)
        overlay = centering.draw_overlay(card, r.front_horizontal, r.front_vertical)
        green = (
            (overlay[:, :, 1].astype(int) > 200)
            & (overlay[:, :, 0].astype(int) < 100)
            & (overlay[:, :, 2].astype(int) < 100)
        )
        # horizontal boundary lines exist (drawn for the V axis) but no
        # full-height vertical lines (H axis unmeasurable): check columns —
        # a vertical line would make some column almost entirely green
        col_green = green.sum(axis=0)
        row_green = green.sum(axis=1)
        assert row_green.max() > W * 0.9, "V-axis boundary lines are drawn"
        assert col_green.max() < H * 0.5, "no H-axis boundary lines for the unmeasurable axis"

    def test_to_dict_carries_measurability(self):
        r = centering.measure_centering(borderless_card(), bordered_card(), THRESHOLDS)
        d = r.to_dict()
        assert d["front"]["measurable"] is False
        assert d["front"]["grade"] is None
        assert d["back"]["measurable"] is True
        assert "boundary_confidence" in d["front"]


class TestEdgeArtifactExclusion:
    """The perspective warp leaves a strong straight line within a few
    pixels of every image edge. On a real capture it scored 0.54-0.82
    boundary confidence and produced a fake '60/40 grade 10' from a
    top=3px/bottom=2px 'border'. The boundary search must skip that zone."""

    def artifact_card(self, with_real_border: bool):
        rng = np.random.default_rng(3)
        img = rng.normal(60, 25, (H, W, 3))  # unmeasurable art texture
        if with_real_border:
            img = np.full((H, W, 3), 200, np.float32)
            img[60:H - 60, 60:W - 60] = 120
            img += rng.normal(0, 6, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # bright warp-artifact line hugging every edge (2px in)
        img[:2, :] = 255
        img[-2:, :] = 255
        img[:, :2] = 255
        img[:, -2:] = 255
        return img

    def test_artifact_alone_does_not_measure(self):
        card = self.artifact_card(with_real_border=False)
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert r.front_grade is None
        assert not r.front_horizontal.measurable and not r.front_vertical.measurable

    def test_real_border_still_measures_despite_artifact(self):
        card = self.artifact_card(with_real_border=True)
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert r.front_grade == 10
        # and the measured widths are the real ~60px border, not the artifact
        assert r.front_horizontal.side_a_px > 30


class TestOverlay:
    def test_unmeasurable_overlay_has_no_boundary_lines(self):
        card = borderless_card()
        r = centering.measure_centering(card, card, THRESHOLDS)
        assert not r.front_horizontal.measurable and not r.front_vertical.measurable
        overlay = centering.draw_overlay(card, r.front_horizontal, r.front_vertical)
        # no green boundary lines drawn — only the red note text differs
        green = (
            (overlay[:, :, 1].astype(int) > 200)
            & (overlay[:, :, 0].astype(int) < 100)
            & (overlay[:, :, 2].astype(int) < 100)
        )
        assert green.sum() < 50
