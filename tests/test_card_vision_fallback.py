"""Why the photometric solve didn't run.

"single_image" alone can't distinguish "no rotation scans were attached" from
"four were attached and one of them was rejected". Those are entirely
different problems — one is a capture that never happened, the other is a
capture that did and was silently thrown away — and the report looked
identical either way.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import grade as grade_module
from webapp import main

THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())


# 63x88mm, the ratio the aspect gate checks against.
CARD_ASPECT = 63.0 / 88.0


def _scan_on_background(tmp_path, name: str, blank: bool = False, index: int = 0):
    """A card-shaped rectangle on a dark background, turned on the glass the
    way the `index`-th scan of a rotation set would be.

    Square frame so a quarter-turn doesn't crop it, and the card is actually
    rotated per scan — `load_derotated` undoes that turn before detection, so
    a set of identical un-rotated frames would arrive at Stage 1 sideways and
    fail the aspect gate, which is a property of the fixture rather than of
    the code under test.
    """
    height = 740
    width = round(height * CARD_ASPECT)
    frame = np.full((900, 900, 3), 12, np.uint8)
    if not blank:
        top = (900 - height) // 2
        left = (900 - width) // 2
        frame[top : top + height, left : left + width] = 200
        frame[top + 120 : top + 620, left + 60 : left + width - 60] = 90
    # Counter-clockwise on the glass, matching the direction these tests pass.
    for _ in range(index % 4):
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    path = tmp_path / name
    cv2.imwrite(str(path), frame)
    return path


@pytest.fixture
def warped():
    return np.full((2100, 1500, 3), 180, np.uint8)


class TestFallbackReason:
    def test_no_scans_says_so(self, tmp_path, warped):
        vision = grade_module.build_card_vision(
            "front", warped, None, tmp_path, THRESHOLDS, 90.0, "ccw", verbose=False
        )
        assert vision.method == "single_image"
        assert "no rotation scans" in vision.to_dict()["fallback_reason"]

    def test_too_few_scans_says_how_many(self, tmp_path, warped):
        scans = [_scan_on_background(tmp_path, f"s{i}.png", index=i) for i in range(2)]
        vision = grade_module.build_card_vision(
            "front", warped, scans, tmp_path, THRESHOLDS, 90.0, "ccw", verbose=False
        )
        assert "at least 3" in vision.to_dict()["fallback_reason"]
        assert "2 scan(s)" in vision.to_dict()["fallback_reason"]

    def test_a_rejected_scan_is_named(self, tmp_path, warped):
        """The failure that actually happened: a full set attached, one scan
        unusable, and a report that looked like nothing had been uploaded."""
        scans = [_scan_on_background(tmp_path, f"s{i}.png", index=i) for i in range(3)]
        scans.append(_scan_on_background(tmp_path, "bad.png", blank=True, index=3))
        vision = grade_module.build_card_vision(
            "front", warped, scans, tmp_path, THRESHOLDS, 90.0, "ccw", verbose=False
        )
        assert any("bad.png" in entry for entry in vision.to_dict()["dropped_scans"])

    def test_the_bad_scan_is_dropped_not_the_whole_set(self, tmp_path, warped):
        """This used to drop all four, on the reasoning that solving three
        would change which light direction each surviving frame was
        attributed to. That reasoning stopped being true once the azimuths
        came from each scan's *measured* rotation instead of its position in
        the file list — a dropped scan now takes its own light direction with
        it and moves nobody else's. Three non-collinear directions solve, and
        a four-scan set carries exactly one spare."""
        # Built at their final positions: each scan's turn on the glass has to
        # match where it sits in the list, or the fixture — not the code —
        # decides what the rotations come out as.
        scans = [
            _scan_on_background(tmp_path, "s0.png", index=0),
            _scan_on_background(tmp_path, "bad.png", blank=True, index=1),
            _scan_on_background(tmp_path, "s2.png", index=2),
            _scan_on_background(tmp_path, "s3.png", index=3),
        ]
        vision = grade_module.build_card_vision(
            "front", warped, scans, tmp_path, THRESHOLDS, 90.0, "ccw", verbose=False
        )
        assert vision.method == "photometric_stereo"
        assert vision.light_count == 3
        assert vision.to_dict()["fallback_reason"] is None

        # The property that makes dropping safe: each surviving frame keeps
        # the rotation it would have had in a complete set, rather than
        # sliding up to fill the gap. Compared against the full run rather
        # than against hand-written numbers, because the point is that they
        # agree — not what they happen to be.
        full = grade_module.build_card_vision(
            "front",
            warped,
            [_scan_on_background(tmp_path, f"f{i}.png", index=i) for i in range(4)],
            tmp_path,
            THRESHOLDS,
            90.0,
            "ccw",
            verbose=False,
        )
        assert vision.rotations_deg == [full.rotations_deg[i] for i in (0, 2, 3)]

    def test_a_successful_solve_carries_no_reason(self, tmp_path, warped):
        scans = [_scan_on_background(tmp_path, f"s{i}.png", index=i) for i in range(4)]
        vision = grade_module.build_card_vision(
            "front", warped, scans, tmp_path, THRESHOLDS, 90.0, "ccw", verbose=False
        )
        assert vision.method == "photometric_stereo", vision.to_dict().get("fallback_reason")
        assert vision.light_count == 4
        assert vision.to_dict()["fallback_reason"] is None

    def test_a_normal_map_is_written_only_on_the_photometric_path(self, tmp_path, warped):
        """Its presence on disk is the other way to tell the two apart."""
        scans = [_scan_on_background(tmp_path, f"s{i}.png", index=i) for i in range(4)]
        grade_module.build_card_vision(
            "front", warped, scans, tmp_path, THRESHOLDS, 90.0, "ccw", verbose=False
        )
        assert (tmp_path / "front_card_vision_normals.png").exists()
