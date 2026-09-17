"""grade_card() from images to report, with nothing stubbed out.

Every other test here exercises one stage. That left the seams untested, and
the seams are where the last three user-visible failures came from: a stage
reading a variable the stage that sets it hadn't run yet, a serializer
emitting None into code that called min() on it, and a return shape changing
under its callers. All three passed the whole suite and broke on the first
real card.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from grade import grade_card
from webapp import main

THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())
CARD_ASPECT = 63.0 / 88.0
DPI = 1200.0


def _card(seed: int = 0) -> np.ndarray:
    """A card-shaped image with a printed border and an asymmetric panel."""
    height = int(88.0 / 25.4 * DPI / 4)
    width = int(round(height * CARD_ASPECT))
    card = np.full((height, width, 3), 225, np.uint8)
    border = int(width * 0.08)
    inner = (height - 2 * border, width - 2 * border)
    rng = np.random.default_rng(seed)
    fine = rng.integers(40, 200, (*inner, 3), dtype=np.uint8)
    # Artwork at two scales. The fine noise is what the corner, edge and
    # surface measurements read. The coarse blocks are what survives being
    # downsampled to a thumbnail — without them two different seeds average
    # to the same flat gray, and every pair of "different" cards correlates
    # at 0.99, which is the reading the same-side check is looking for.
    coarse = cv2.resize(
        rng.integers(0, 255, (11, 8, 3), dtype=np.uint8), (inner[1], inner[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    card[border:-border, border:-border] = (fine * 0.45 + coarse * 0.55).astype(np.uint8)
    # Asymmetric mark, so a half-turn is distinguishable from upright.
    card[border : border + 40, border : border + 200] = 20
    return card


def _scan(tmp_path, name: str, ccw_on_glass: int = 0, seed: int = 0):
    card = _card(seed)
    frame = np.full((1400, 1400, 3), 10, np.uint8)
    h, w = card.shape[:2]
    top, left = (1400 - h) // 2, (1400 - w) // 2
    frame[top : top + h, left : left + w] = card
    codes = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    if ccw_on_glass % 360:
        frame = cv2.rotate(frame, codes[ccw_on_glass % 360])
    path = tmp_path / name
    cv2.imwrite(str(path), frame)
    return path


@pytest.fixture
def capture(tmp_path):
    return {
        "front": _scan(tmp_path, "front.png"),
        "back": _scan(tmp_path, "back.png", seed=1),
        "out": tmp_path / "out",
    }


class TestFlatCaptureOnly:
    def test_it_produces_a_report(self, capture, tmp_path):
        report = grade_card(capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False)
        assert report["grade_estimate"]["overall_grade_rounded"] is not None

    def test_every_section_is_present(self, capture):
        report = grade_card(capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False)
        for section in ("capture_quality", "capture_pair", "card_vision", "centering",
                        "corners_edges", "surface", "dimensions", "grade_estimate",
                        "subgrades", "dings"):
            assert section in report, f"missing {section}"

    def test_the_report_is_json_serializable(self, capture):
        """numpy scalars and arrays leak out of the measurement code, and the
        report is written to disk as JSON."""
        report = grade_card(capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False)
        json.dumps(report)

    def test_the_stages_fire_in_the_order_the_ui_lists_them(self, capture):
        """The progress display walks a fixed list; a stage running out of
        order means the UI reports the wrong thing, and — as happened — a
        stage reading a variable a later stage sets."""
        from webapp.jobs import STAGE_MESSAGES

        seen = []
        grade_card(capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False,
                   on_stage=seen.append)
        known = [s for s in seen if s in STAGE_MESSAGES]
        expected_order = [s for s in STAGE_MESSAGES if s in known]
        assert known == expected_order, f"stages fired {known}, UI lists {expected_order}"

    def test_dimensions_are_unmeasurable_without_a_dpi(self, capture):
        report = grade_card(capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False)
        assert report["dimensions"]["measurable"] is False

    def test_two_real_sides_raise_no_same_side_warning(self, capture):
        report = grade_card(capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False)
        assert report["capture_pair"]["same_side_suspected"] is False

    def test_the_same_file_twice_still_grades_but_says_so(self, capture):
        """The check warns; it must not turn a gradeable pair into an error.
        Grading one side against both tolerance tables is something the user
        has deliberately asked for."""
        report = grade_card(capture["front"], capture["front"], THRESHOLDS, capture["out"], verbose=False)
        assert report["capture_pair"]["same_side_suspected"] is True
        assert report["grade_estimate"]["overall_grade_rounded"] is not None


class TestWithRotationScans:
    """The path that has broken most often — and the one no test ran."""

    @pytest.fixture
    def photometric(self, tmp_path):
        return [_scan(tmp_path, f"rot{i}.png", ccw_on_glass=-90 * i) for i in range(4)]

    def test_a_full_photometric_run_completes(self, capture, photometric):
        report = grade_card(
            capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False,
            dpi=DPI, photometric_paths=(photometric, None),
        )
        assert report["card_vision"]["front"]["method"] == "photometric_stereo"
        assert report["card_vision"]["front"]["light_count"] == 4

    def test_dimensions_use_every_scan(self, capture, photometric):
        """The flat capture plus four rotations is five measurements of the
        same card; resting the verdict on one made the same card read
        '2.13mm miscut' in one run and 'within tolerance' in the next."""
        report = grade_card(
            capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False,
            dpi=DPI, photometric_paths=(photometric, None),
        )
        assert report["dimensions"]["sample_count"] > 1
        assert report["dimensions"]["spread_mm"] is not None

    def test_it_still_serializes(self, capture, photometric):
        report = grade_card(
            capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False,
            dpi=DPI, photometric_paths=(photometric, None),
        )
        json.dumps(report)

    def test_the_measured_rotations_are_recorded(self, capture, photometric):
        report = grade_card(
            capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False,
            dpi=DPI, photometric_paths=(photometric, None),
        )
        assert sorted(report["card_vision"]["front"]["rotations_deg"]) == [0, 90, 180, 270]

    def test_a_short_set_falls_back_and_says_why(self, capture, photometric):
        report = grade_card(
            capture["front"], capture["back"], THRESHOLDS, capture["out"], verbose=False,
            dpi=DPI, photometric_paths=(photometric[:2], None),
        )
        assert report["card_vision"]["front"]["method"] == "single_image"
        assert "at least 3" in report["card_vision"]["front"]["fallback_reason"]


class TestRotationScansAreKept:
    """The four normalised warps are the only inputs the photometric solve
    has. Discarding them meant every change to the solve cost a rescan — four
    times over one evening, each one twenty minutes of the card going back on
    the glass."""

    @pytest.fixture
    def photometric(self, tmp_path):
        return [_scan(tmp_path, f"rot{i}.png", ccw_on_glass=-90 * i) for i in range(4)]

    def test_each_scan_is_written_alongside_the_report(self, capture, photometric):
        out = capture["out"]
        grade_card(
            capture["front"], capture["back"], THRESHOLDS, out, verbose=False,
            dpi=DPI, photometric_paths=(photometric, None),
        )
        for index in range(4):
            assert (out / f"front_rotation_{index}.png").exists(), f"rotation {index} not kept"

    def test_they_are_the_normalised_warps_not_the_raw_scans(self, capture, photometric):
        """Normalised, so a re-solve starts where the last one did rather than
        redoing detection and orientation from scratch."""
        out = capture["out"]
        grade_card(
            capture["front"], capture["back"], THRESHOLDS, out, verbose=False,
            dpi=DPI, photometric_paths=(photometric, None),
        )
        kept = cv2.imread(str(out / "front_rotation_0.png"))
        canonical = (THRESHOLDS["capture"]["canonical_height_px"], THRESHOLDS["capture"]["canonical_width_px"])
        assert kept.shape[:2] == canonical

    def test_a_side_with_no_rotation_scans_writes_none(self, capture, photometric):
        out = capture["out"]
        grade_card(
            capture["front"], capture["back"], THRESHOLDS, out, verbose=False,
            dpi=DPI, photometric_paths=(photometric, None),
        )
        assert not (out / "back_rotation_0.png").exists()

    def test_they_are_served_by_url_not_inlined(self):
        """Six warps a side at 6MB each has no business in a report response."""
        from webapp import jobs, store

        for index in range(4):
            assert f"front_rotation_{index}" in jobs.DETAIL_IMAGE_KEYS
            assert f"front_rotation_{index}" in store.DETAIL_IMAGE_KEYS
