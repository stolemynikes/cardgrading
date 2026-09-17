"""Corners and edges after the borders move, and why a refusal says so.

Every corner and edge crop is *sized from* the measured border widths —
that's what keeps a crop inside the border instead of running into artwork.
Correcting centering by hand therefore invalidates the whole stage, and for a
while it was left alone: the report showed corners measured against borders
it no longer claimed.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline import corners_edges, regrade
from webapp import main, store

REPORT_ID = "0123456789abcdef0123456789abcdef"
THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())


def _card(border_px: int = 120) -> np.ndarray:
    """A bordered card: flat mid-grey border, textured panel inside it."""
    h, w = 2100, 1500
    image = np.full((h, w, 3), 150, np.uint8)
    rng = np.random.default_rng(0)
    panel = rng.integers(0, 255, (h - 2 * border_px, w - 2 * border_px, 3), dtype=np.uint8)
    image[border_px:-border_px, border_px:-border_px] = panel
    return image


class TestTiledUniformity:
    """The gate used to measure one median over the whole crop, which
    penalised a long strip for its length: a border that changes colour along
    the side is locally as flat as a corner square, but half of it sits far
    from the strip's single median."""

    @staticmethod
    def _graded_strip() -> np.ndarray:
        """A locally flat border whose colour changes along its length — a
        patterned border, which is what a Japanese card's frame actually is.
        Every point on it is flat; the strip as a whole is not."""
        strip = np.zeros((77, 1300, 3), np.uint8)
        strip[:, :430] = (40, 60, 90)
        strip[:, 430:870] = (130, 40, 160)
        strip[:, 870:] = (30, 150, 70)
        return strip

    def test_a_locally_flat_strip_is_not_refused_for_being_long(self):
        strip = self._graded_strip()
        whole = corners_edges._local_uniformity(strip)
        tiled = corners_edges._border_uniformity(strip)
        assert whole < 0.55, "the old whole-crop measure refuses this"
        assert tiled >= 0.55, "measured tile by tile it passes, as its corners always did"

    def test_artwork_is_still_refused(self):
        """The gate exists to keep full-art cards out; that must survive."""
        rng = np.random.default_rng(1)
        art = rng.integers(0, 255, (77, 1300, 3), dtype=np.uint8)
        assert corners_edges._border_uniformity(art) < 0.55

    def test_a_square_crop_is_one_tile(self):
        square = np.full((98, 98, 3), 130, np.uint8)
        assert len(corners_edges._square_tiles(square)) == 1
        assert corners_edges._border_uniformity(square) == corners_edges._local_uniformity(square)

    def test_a_long_strip_is_cut_into_squares(self):
        tiles = corners_edges._square_tiles(np.zeros((77, 1300, 3), np.uint8))
        assert len(tiles) > 10
        assert all(abs(t.shape[0] - 77) < 2 for t in tiles), "tiles keep the strip's thickness"

    def test_tiling_works_on_either_orientation(self):
        assert len(corners_edges._square_tiles(np.zeros((1300, 77, 3), np.uint8))) > 10

    def test_one_bad_tile_does_not_condemn_the_side(self):
        """Median, not mean: a logo or foil stamp shouldn't refuse an edge."""
        strip = np.full((77, 1300, 3), 130, np.uint8)
        rng = np.random.default_rng(2)
        strip[:, 600:700] = rng.integers(0, 255, (77, 100, 3), dtype=np.uint8)
        assert corners_edges._border_uniformity(strip) >= 0.55

    def test_an_empty_crop_is_refused_rather_than_crashing(self):
        assert corners_edges._border_uniformity(np.zeros((0, 10, 3), np.uint8)) == 0.0


class TestRefusalReasons:
    def test_a_refused_region_says_why(self):
        rng = np.random.default_rng(3)
        art = rng.integers(0, 255, (98, 98, 3), dtype=np.uint8)
        result, _ = corners_edges.analyze_region("top_left", art, THRESHOLDS["corners_edges"])
        assert result.measurable is False
        assert "uniform border" in result.reason

    def test_a_refused_region_has_no_grade_at_all(self):
        """It used to serialize the grade it would have had, gated only by
        `measurable` — which reads as a perfect 10 to anything consuming the
        report without knowing to check the flag alongside it."""
        rng = np.random.default_rng(4)
        art = rng.integers(0, 255, (98, 98, 3), dtype=np.uint8)
        result, _ = corners_edges.analyze_region("top_left", art, THRESHOLDS["corners_edges"])
        assert result.to_dict()["grade"] is None

    def test_a_measured_region_carries_no_reason(self):
        flat = np.full((98, 98, 3), 140, np.uint8)
        result, _ = corners_edges.analyze_region("top_left", flat, THRESHOLDS["corners_edges"])
        assert result.measurable is True
        assert result.to_dict()["reason"] is None
        assert result.to_dict()["grade"] is not None

    def test_a_side_with_every_region_refused_explains_the_group(self):
        rng = np.random.default_rng(5)
        art = rng.integers(0, 255, (2100, 1500, 3), dtype=np.uint8)
        side, _ = corners_edges.analyze_side(
            art, corners_edges.BorderWidths(60, 60, 60, 60), THRESHOLDS["corners_edges"]
        )
        d = side.to_dict()
        assert d["corners_grade"] is None and d["corners_reason"]
        assert d["edges_grade"] is None and d["edges_reason"]

    def test_a_measured_side_explains_nothing(self):
        side, _ = corners_edges.analyze_side(
            _card(), corners_edges.BorderWidths(120, 120, 120, 120), THRESHOLDS["corners_edges"]
        )
        d = side.to_dict()
        assert d["corners_reason"] is None and d["edges_reason"] is None

    def test_an_ungradeable_surface_says_what_would_fix_it(self):
        from pipeline import surface

        ungraded = surface.grade_surface(0.5, 3, 40, "single_image_relief", THRESHOLDS)
        assert ungraded.grade is None
        assert "rotating the card 90" in ungraded.to_dict()["reason"]

    def test_a_graded_surface_carries_no_reason(self):
        from pipeline import surface

        graded = surface.grade_surface(0.5, 3, 40, "photometric_relief", THRESHOLDS)
        assert graded.grade is not None
        assert graded.to_dict()["reason"] is None


class TestBorderWidthFallback:
    def test_measured_axes_are_used(self):
        side = {
            "horizontal": {"side_a_px": 58, "side_b_px": 108, "measurable": True},
            "vertical": {"side_a_px": 54, "side_b_px": 64, "measurable": True},
        }
        widths = regrade.border_widths(side, (2100, 1500))
        assert (widths.left, widths.right, widths.top, widths.bottom) == (58, 108, 54, 64)

    def test_an_unmeasurable_axis_falls_back_per_axis(self):
        """Widths from a refused axis are argmax-of-noise; sizing a crop from
        them produced postage stamps whose percentages were pure noise."""
        side = {
            "horizontal": {"side_a_px": 900, "side_b_px": 900, "measurable": False},
            "vertical": {"side_a_px": 54, "side_b_px": 64, "measurable": True},
        }
        widths = regrade.border_widths(side, (2100, 1500))
        assert widths.left == widths.right == 1500 * 0.04
        assert (widths.top, widths.bottom) == (54, 64), "the good axis is kept"

    def test_a_missing_side_still_produces_widths(self):
        widths = regrade.border_widths(None, (2100, 1500))
        assert widths.left > 0 and widths.top > 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path / "reports")
    aligned = tmp_path / "front_aligned.png"
    cv2.imwrite(str(aligned), _card())
    report = {
        "centering": {
            "front": {
                "horizontal": {"side_a_px": 40.0, "side_b_px": 40.0, "measurable": True,
                               "side_a_pct": 50.0, "side_b_pct": 50.0, "ratio": "50/50", "grade": 10},
                "vertical": {"side_a_px": 40.0, "side_b_px": 40.0, "measurable": True,
                             "side_a_pct": 50.0, "side_b_pct": 50.0, "ratio": "50/50", "grade": 10},
                "grade": 10,
                "measurable": True,
            },
            "back": None,
            "overall_grade": 10,
        },
        "corners_edges": {"front": {"grade": None}, "back": {"grade": None}, "overall_grade": None},
        "surface": {"front": {"grade": None, "upper_bound": False}},
        "dimensions": {"measurable": False, "within_tolerance": None},
        "grade_estimate": {"overall_grade_rounded": 10, "score": 1000},
        "subgrades": {"front": {}, "back": {}},
        "dings": [],
    }
    store.save_report(tmp_path / "reports", REPORT_ID, report, {"front_aligned": aligned})
    with TestClient(main.app) as test_client:
        yield test_client


class TestRecomputeOnSave:
    BORDERS = {"left": 120, "right": 120, "top": 120, "bottom": 120}

    def test_corners_are_remeasured_against_the_new_borders(self, client):
        """The crops were sized from 40px borders and landed on artwork; at
        the real 120px they sit on the border and measure."""
        body = client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS}).json()
        front = body["report"]["corners_edges"]["front"]
        assert front["corners_grade"] is not None
        assert front["edges_grade"] is not None

    def test_the_overall_corners_edges_grade_follows(self, client):
        body = client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS}).json()
        assert body["report"]["corners_edges"]["overall_grade"] is not None

    def test_the_card_grade_follows_it(self, client):
        body = client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS}).json()
        assert body["report"]["grade_estimate"]["corners_edges_grade"] is not None

    def test_the_region_crops_are_rewritten(self, client, tmp_path):
        """Stale crops are the same bug wearing a different hat: the report
        would show evidence images cut from borders it no longer claims."""
        client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS})
        images = tmp_path / "reports" / REPORT_ID / "images"
        assert (images / "front_corner_top_left.png").exists()
        assert (images / "front_edge_left.png").exists()
        crop = cv2.imread(str(images / "front_corner_top_left.png"))
        assert min(crop.shape[:2]) > 40, "sized from the corrected border, not the old one"

    def test_a_side_with_no_stored_warp_is_left_alone(self, client):
        """Only the front warp was saved; the back keeps whatever it had."""
        before = client.get(f"/api/report/{REPORT_ID}").json()["report"]["corners_edges"]["back"]
        body = client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS}).json()
        assert body["report"]["corners_edges"]["back"] == before

    def test_a_report_with_no_corners_edges_block_survives(self, tmp_path):
        report = {"centering": {}, "corners_edges": None}
        assert regrade.recompute_corners_edges(report, {"front": _card()}, THRESHOLDS) == {}

    def test_missing_warps_are_reported_as_none(self, tmp_path):
        assert regrade.load_aligned(tmp_path) == {"front": None, "back": None}
