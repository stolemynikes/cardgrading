"""Hand-placed centering boundaries, end to end.

The detector has a failure mode no confidence gate catches: finding *an*
edge, confidently enough to pass, that isn't the border. Modern cards stack
an artwork boundary, an inner frame band and a thin rule within a few
millimetres. When that happens the boundaries get placed by hand instead,
and everything downstream of centering has to move with them.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline import centering
from webapp import main, store

REPORT_ID = "0123456789abcdef0123456789abcdef"

THRESHOLDS = json.loads((main.THRESHOLDS_PATH).read_text())


def _centering_block(front_right_px: float = 122.0) -> dict:
    """What the detector produced for the real card that prompted this: the
    left boundary exactly right, the right one 14px too far in."""
    tolerances = THRESHOLDS["centering"]["front_tolerances"]
    axis_h = centering._axis_centering("left", "right", 58.0, front_right_px, tolerances)
    axis_v = centering._axis_centering("top", "bottom", 54.0, 64.0, tolerances)
    side = {
        "horizontal": centering.axis_dict(axis_h),
        "vertical": centering.axis_dict(axis_v),
        "grade": min(axis_h.grade, axis_v.grade),
        "measurable": True,
        "boundary_confidence": {"left": 0.744, "right": 0.37, "top": 0.782, "bottom": 0.893},
    }
    return {
        "front": side,
        "back": json.loads(json.dumps(side)),
        "overall_grade": side["grade"],
    }


def _report() -> dict:
    return {
        "card_id": {"card_name": "Metagross", "set_name": "Crimson Rift"},
        "centering": _centering_block(),
        "corners_edges": {"front": {}, "back": {}, "overall_grade": None},
        "surface": {
            "front": {"grade": None, "source": "single_image_relief", "upper_bound": False},
            "back": {"grade": None, "source": "single_image_relief", "upper_bound": False},
        },
        "dimensions": {"measurable": False, "within_tolerance": None},
        "grade_estimate": {"overall_grade_rounded": 7, "score": 700},
        "subgrades": {
            "front": {"centering": 7, "corners": None, "edges": None, "surface": None},
            "back": {"centering": 7, "corners": None, "edges": None, "surface": None},
        },
        "dings": [],
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path / "reports")
    reports_dir = tmp_path / "reports"
    aligned = tmp_path / "front_aligned.png"
    cv2.imwrite(str(aligned), np.full((2100, 1500, 3), 200, np.uint8))
    store.save_report(reports_dir, REPORT_ID, _report(), {"front_aligned": aligned})
    with TestClient(main.app) as test_client:
        yield test_client


# --- the arithmetic ---


class TestRegrade:
    def test_corrected_boundary_changes_the_ratio(self):
        updated = centering.regrade_with_manual_borders(
            _centering_block(), {"front": {"left": 58, "right": 108, "top": 54, "bottom": 64}}, THRESHOLDS
        )
        assert updated["front"]["horizontal"]["ratio"] == "35/65"

    def test_a_side_left_out_of_the_payload_is_untouched(self):
        before = _centering_block()
        updated = centering.regrade_with_manual_borders(
            before, {"front": {"left": 58, "right": 108, "top": 54, "bottom": 64}}, THRESHOLDS
        )
        assert updated["back"] == before["back"]
        assert "manual" not in updated["back"]

    def test_hand_placed_boundaries_are_always_measurable(self):
        """The confidence score says how sure the *detector* was. Once a
        person has said where the edge is, that question is settled."""
        refused = _centering_block()
        refused["front"]["horizontal"]["measurable"] = False
        refused["front"]["measurable"] = False
        updated = centering.regrade_with_manual_borders(
            refused, {"front": {"left": 58, "right": 108, "top": 54, "bottom": 64}}, THRESHOLDS
        )
        assert updated["front"]["horizontal"]["measurable"] is True
        assert updated["front"]["measurable"] is True

    def test_stale_confidences_are_dropped_not_carried_over(self):
        updated = centering.regrade_with_manual_borders(
            _centering_block(), {"front": {"left": 58, "right": 108, "top": 54, "bottom": 64}}, THRESHOLDS
        )
        assert updated["front"]["boundary_confidence"] is None

    def test_overall_is_the_weaker_side(self):
        updated = centering.regrade_with_manual_borders(
            _centering_block(),
            {"front": {"left": 10, "right": 300, "top": 54, "bottom": 64}},
            THRESHOLDS,
        )
        assert updated["overall_grade"] == min(updated["front"]["grade"], updated["back"]["grade"])

    def test_the_manual_path_emits_the_same_keys_as_the_detector(self):
        """Guards the schema the report template reads. An earlier bug in a
        sibling module read a key the serializer doesn't emit."""
        before = _centering_block()
        updated = centering.regrade_with_manual_borders(
            before, {"front": {"left": 58, "right": 108, "top": 54, "bottom": 64}}, THRESHOLDS
        )
        detected_keys = set(before["front"]["horizontal"])
        assert detected_keys <= set(updated["front"]["horizontal"])


class TestValidation:
    def test_opposite_borders_may_not_meet(self):
        assert centering.validate_manual_borders(
            {"left": 800, "right": 800, "top": 10, "bottom": 10}, 1500, 2100
        )

    def test_negative_width_is_rejected(self):
        assert centering.validate_manual_borders({"left": -1, "right": 10, "top": 10, "bottom": 10}, 1500, 2100)

    def test_missing_edge_is_rejected(self):
        assert centering.validate_manual_borders({"left": 10, "right": 10, "top": 10}, 1500, 2100)

    def test_non_numeric_is_rejected(self):
        assert centering.validate_manual_borders(
            {"left": "wide", "right": 10, "top": 10, "bottom": 10}, 1500, 2100
        )

    def test_nan_is_rejected(self):
        """float('nan') passes every comparison it's given, so it has to be
        excluded by a positive test rather than a bounds check."""
        assert centering.validate_manual_borders(
            {"left": float("nan"), "right": 10, "top": 10, "bottom": 10}, 1500, 2100
        )

    def test_a_plausible_card_passes(self):
        assert centering.validate_manual_borders({"left": 58, "right": 108, "top": 54, "bottom": 64}, 1500, 2100) is None


# --- the endpoint ---


class TestEndpoint:
    BORDERS = {"left": 58, "right": 108, "top": 54, "bottom": 64}

    def test_correction_is_persisted(self, client):
        client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS})
        fetched = client.get(f"/api/report/{REPORT_ID}").json()
        assert fetched["report"]["centering"]["front"]["horizontal"]["ratio"] == "35/65"
        assert fetched["report"]["centering"]["front"]["manual"] is True

    def test_response_carries_the_rerendered_report(self, client):
        body = client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS}).json()
        assert body["report"]["centering"]["front"]["horizontal"]["ratio"] == "35/65"
        assert "front_aligned" in body["images"]

    def test_subgrades_follow_centering(self, client):
        body = client.post(
            f"/api/report/{REPORT_ID}/centering",
            json={"front": {"left": 10, "right": 400, "top": 54, "bottom": 64}},
        ).json()
        front = body["report"]["centering"]["front"]
        assert body["report"]["subgrades"]["front"]["centering"] == front["grade"]

    def test_overall_grade_is_recomputed(self, client):
        before = client.get(f"/api/report/{REPORT_ID}").json()["report"]["grade_estimate"]
        body = client.post(
            f"/api/report/{REPORT_ID}/centering",
            json={"front": {"left": 10, "right": 400, "top": 54, "bottom": 64}},
        ).json()
        assert body["report"]["grade_estimate"]["overall_grade_rounded"] < before["overall_grade_rounded"]

    def test_listing_summary_follows_the_new_grade(self, client):
        client.post(
            f"/api/report/{REPORT_ID}/centering",
            json={"front": {"left": 10, "right": 400, "top": 54, "bottom": 64}},
        )
        row = client.get("/api/reports").json()["reports"][0]
        assert row["grade"] == client.get(f"/api/report/{REPORT_ID}").json()["report"]["grade_estimate"]["overall_grade_rounded"]

    def test_created_at_survives_the_edit(self, client, tmp_path):
        before = json.loads((tmp_path / "reports" / REPORT_ID / "meta.json").read_text())["created_at"]
        client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS})
        after = json.loads((tmp_path / "reports" / REPORT_ID / "meta.json").read_text())["created_at"]
        assert after == before

    def test_overlay_is_redrawn_over_the_stored_warp(self, client, tmp_path):
        client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS})
        overlay = cv2.imread(str(tmp_path / "reports" / REPORT_ID / "images" / "front_centering_overlay.png"))
        assert overlay is not None
        # Green lines land where the widths say, and nowhere else: the pixel
        # at the hand-placed left boundary is drawn, one well inside is not.
        assert overlay[1000, 58].tolist() == [0, 255, 0]
        assert overlay[1000, 400].tolist() != [0, 255, 0]

    def test_no_scratch_files_left_in_the_report_directory(self, client, tmp_path):
        client.post(f"/api/report/{REPORT_ID}/centering", json={"front": self.BORDERS})
        leftovers = [p.name for p in (tmp_path / "reports" / REPORT_ID).iterdir() if p.name.startswith(".")]
        assert leftovers == []

    def test_overlapping_borders_are_refused(self, client):
        res = client.post(
            f"/api/report/{REPORT_ID}/centering",
            json={"front": {"left": 800, "right": 800, "top": 10, "bottom": 10}},
        )
        assert res.status_code == 400

    def test_a_refused_correction_leaves_the_report_alone(self, client):
        client.post(
            f"/api/report/{REPORT_ID}/centering",
            json={"front": {"left": 800, "right": 800, "top": 10, "bottom": 10}},
        )
        assert client.get(f"/api/report/{REPORT_ID}").json()["report"]["centering"]["front"]["horizontal"]["ratio"] == "32/68"

    def test_empty_payload_is_refused(self, client):
        assert client.post(f"/api/report/{REPORT_ID}/centering", json={}).status_code == 400

    def test_unknown_report_is_404(self, client):
        res = client.post("/api/report/{}/centering".format("f" * 32), json={"front": self.BORDERS})
        assert res.status_code == 404

    def test_a_malformed_id_never_reaches_the_filesystem(self, client):
        res = client.post("/api/report/..%2F..%2Fetc/centering", json={"front": self.BORDERS})
        assert res.status_code == 404

    def test_tolerance_tables_are_served_for_the_client_to_grade_with(self, client):
        body = client.get("/api/centering-tolerances").json()
        assert body["front"] == THRESHOLDS["centering"]["front_tolerances"]
        assert body["back"] == THRESHOLDS["centering"]["back_tolerances"]
        assert body["canonical_width_px"] == THRESHOLDS["capture"]["canonical_width_px"]


class TestCardEdgeInsets:
    """The eighth line. The warp is defined by the detected corners, so the
    card edge is the image edge by construction — but corner detection can be
    a pixel or two out, and at 1500px across a 63mm card two pixels is 0.08mm,
    enough to move a ratio across a grade line."""

    def test_an_inset_narrows_the_border_it_sits_behind(self):
        square = {"borders": {"left": 100, "right": 100, "top": 100, "bottom": 100}}
        offset = {
            "borders": {"left": 100, "right": 100, "top": 100, "bottom": 100},
            "edges": {"left": 20, "right": 0, "top": 0, "bottom": 0},
        }
        # The widths themselves are what's graded, so an inset with unchanged
        # widths must not change the ratio — it moves both lines together.
        a = centering.regrade_with_manual_borders(_centering_block(), {"front": square}, THRESHOLDS)
        b = centering.regrade_with_manual_borders(_centering_block(), {"front": offset}, THRESHOLDS)
        assert a["front"]["horizontal"]["ratio"] == b["front"]["horizontal"]["ratio"]

    def test_the_inset_is_recorded_on_the_report(self):
        updated = centering.regrade_with_manual_borders(
            _centering_block(),
            {
                "front": {
                    "borders": {"left": 100, "right": 100, "top": 100, "bottom": 100},
                    "edges": {"left": 20, "right": 0, "top": 0, "bottom": 0},
                }
            },
            THRESHOLDS,
        )
        assert updated["front"]["card_edge_px"]["left"] == 20.0

    def test_the_bare_four_widths_still_work(self):
        """The first version of this API had no card-edge concept; a payload
        in that shape still has to grade rather than 400."""
        updated = centering.regrade_with_manual_borders(
            _centering_block(), {"front": {"left": 58, "right": 108, "top": 54, "bottom": 64}}, THRESHOLDS
        )
        assert updated["front"]["horizontal"]["ratio"] == "35/65"
        assert updated["front"]["card_edge_px"] == {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}

    def test_inset_plus_border_may_not_span_the_card(self):
        assert centering.validate_manual_borders(
            {"left": 700, "right": 700, "top": 10, "bottom": 10},
            1500,
            2100,
            {"left": 60, "right": 60, "top": 0, "bottom": 0},
        )

    def test_a_negative_inset_is_rejected(self):
        assert centering.validate_manual_borders(
            {"left": 58, "right": 108, "top": 54, "bottom": 64},
            1500,
            2100,
            {"left": -5, "right": 0, "top": 0, "bottom": 0},
        )

    def test_endpoint_accepts_the_eight_line_payload(self, client):
        res = client.post(
            f"/api/report/{REPORT_ID}/centering",
            json={
                "front": {
                    "borders": {"left": 58, "right": 108, "top": 54, "bottom": 64},
                    "edges": {"left": 3, "right": 2, "top": 0, "bottom": 0},
                }
            },
        )
        assert res.status_code == 200
        assert res.json()["report"]["centering"]["front"]["card_edge_px"]["left"] == 3.0

    def test_overlay_draws_the_boundary_at_inset_plus_width(self, client, tmp_path):
        client.post(
            f"/api/report/{REPORT_ID}/centering",
            json={
                "front": {
                    "borders": {"left": 58, "right": 108, "top": 54, "bottom": 64},
                    "edges": {"left": 20, "right": 0, "top": 0, "bottom": 0},
                }
            },
        )
        overlay = cv2.imread(str(tmp_path / "reports" / REPORT_ID / "images" / "front_centering_overlay.png"))
        assert overlay[1000, 78].tolist() == [0, 255, 0], "border line sits at inset + width"
        assert overlay[1000, 20].tolist() == [0, 170, 255], "card edge drawn in its own colour"

    def test_a_flush_card_edge_draws_no_line(self, client, tmp_path):
        """Zero inset is the normal case; drawing it would paint a line down
        every image border of every report."""
        client.post(f"/api/report/{REPORT_ID}/centering", json={"front": TestEndpoint.BORDERS})
        overlay = cv2.imread(str(tmp_path / "reports" / REPORT_ID / "images" / "front_centering_overlay.png"))
        assert overlay[1000, 0].tolist() != [0, 170, 255]


class TestDetailWarp:
    """A second warp at the capture's own scale, for looking at rather than
    measuring from."""

    THRESHOLDS = THRESHOLDS

    @staticmethod
    def _corners(width: float, height: float) -> np.ndarray:
        return np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)

    def test_a_high_resolution_capture_keeps_its_detail(self):
        from pipeline import detect

        capture = np.full((5200, 3700, 3), 180, np.uint8)
        out = detect.detail_warp(capture, self._corners(3656, 5133), THRESHOLDS)
        assert out is not None
        assert out.shape[1] > THRESHOLDS["capture"]["canonical_width_px"]

    def test_the_canonical_aspect_is_held(self):
        """Corner error must not stretch the detail view, or the overlay
        drawn on top of it would no longer line up."""
        from pipeline import detect

        capture = np.full((5200, 3700, 3), 180, np.uint8)
        out = detect.detail_warp(capture, self._corners(3656, 5000), THRESHOLDS)
        canonical = THRESHOLDS["capture"]["canonical_width_px"] / THRESHOLDS["capture"]["canonical_height_px"]
        assert abs(out.shape[1] / out.shape[0] - canonical) < 0.002

    def test_a_capture_at_canonical_resolution_produces_nothing(self):
        """Upscaling adds bytes and no detail."""
        from pipeline import detect

        capture = np.full((2200, 1600, 3), 180, np.uint8)
        assert detect.detail_warp(capture, self._corners(1500, 2100), THRESHOLDS) is None

    def test_an_undetected_card_produces_nothing(self):
        from pipeline import detect

        assert detect.detail_warp(np.zeros((10, 10, 3), np.uint8), None, THRESHOLDS) is None

    def test_the_long_edge_is_capped(self):
        from pipeline import detect

        capture = np.full((30000, 21000, 3), 180, np.uint8)
        out = detect.detail_warp(capture, self._corners(20000, 28000), THRESHOLDS)
        assert max(out.shape[:2]) <= detect.MAX_DETAIL_LONG_EDGE_PX


class TestDetailDelivery:
    """Tens of megabytes each: stored, addressed by URL, never inlined."""

    def test_detail_images_are_urls_not_data_uris(self, client, tmp_path):
        detail = tmp_path / "front_detail.png"
        cv2.imwrite(str(detail), np.full((5133, 3656, 3), 200, np.uint8))
        store.save_report(tmp_path / "reports", REPORT_ID, _report(), {"front_detail": detail})
        images = client.get(f"/api/report/{REPORT_ID}").json()["images"]
        assert images["front_detail"] == f"/api/report/{REPORT_ID}/image/front_detail"

    def test_the_url_serves_the_image(self, client, tmp_path):
        detail = tmp_path / "front_detail.png"
        cv2.imwrite(str(detail), np.full((5133, 3656, 3), 200, np.uint8))
        store.save_report(tmp_path / "reports", REPORT_ID, _report(), {"front_detail": detail})
        res = client.get(f"/api/report/{REPORT_ID}/image/front_detail")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"

    def test_a_missing_image_is_404(self, client):
        assert client.get(f"/api/report/{REPORT_ID}/image/back_detail").status_code == 404

    def test_an_image_key_cannot_escape_the_report_directory(self, client):
        assert store.image_path(main.REPORTS_DIR, REPORT_ID, "../../etc/passwd") is None


class TestManualMatchesAutomatic:
    """A measurement is a measurement however it was arrived at. The two paths
    diverged once — the manual one skipped PSA's front leeway — and that's the
    kind of bug a user sees directly, as an axis and a comparison table
    disagreeing about the same card."""

    def test_the_manual_path_applies_the_front_leeway(self):
        updated = centering.regrade_with_manual_borders(
            _centering_block(), {"front": {"left": 58, "right": 108, "top": 54, "bottom": 64}}, THRESHOLDS
        )
        axis = updated["front"]["horizontal"]
        assert axis["grade"] == 8 and axis["strict_grade"] == 7 and axis["leeway_applied"] is True

    def test_it_agrees_with_the_grader_comparison(self, client):
        body = client.post(
            f"/api/report/{REPORT_ID}/centering",
            json={"front": {"left": 58, "right": 108, "top": 54, "bottom": 64}},
        ).json()
        block = body["report"]["centering"]
        assert block["by_grader"]["psa"]["front"] == block["front"]["grade"]

    def test_the_back_still_gets_no_leeway(self):
        updated = centering.regrade_with_manual_borders(
            _centering_block(), {"back": {"left": 58, "right": 108, "top": 54, "bottom": 64}}, THRESHOLDS
        )
        assert updated["back"]["horizontal"]["leeway_applied"] is False
