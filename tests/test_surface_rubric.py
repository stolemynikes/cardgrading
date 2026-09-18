"""Surface grading against the published rubrics.

Area alone is not a grading standard. No service grades by how much of the
card is marked: a crease and a scuff covering the same area are several
grades apart everywhere, because what caps a grade is what the defect *is*.

The ceilings here come from TAG's published rubric — the closest published
standard to what this pipeline measures, since they grade from photometric
stereo too — cross-checked against PSA's grade definitions.

    https://taggrading.com/pages/rubric
    https://www.psacard.com/gradingstandards

These are synthetic marks with known geometry, so they test the ladder rather
than any one card. The real card pair is in test_surface_measures_damage.py.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pipeline import surface
from webapp import main

THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())
CFG = THRESHOLDS["surface"]
PX_PER_MM = THRESHOLDS["capture"]["canonical_width_px"] / 63.0
W = THRESHOLDS["capture"]["canonical_width_px"]
H = THRESHOLDS["capture"]["canonical_height_px"]


def _mm(v: float) -> int:
    return int(round(v * PX_PER_MM))


def _card(marks) -> np.ndarray:
    """A flat relief render carrying the given marks.

    marks: (length_mm, width_mm, depth, centre_mm_xy, angle_deg)
    """
    relief = np.full((H, W), 128, np.uint8)
    for length_mm, width_mm, depth, (cx, cy), angle in marks:
        box = ((_mm(cx), _mm(cy)), (max(_mm(length_mm), 2), max(_mm(width_mm), 2)), angle)
        cv2.drawContours(relief, [np.int32(cv2.boxPoints(box))], 0, int(128 + depth), -1)
    return relief


def _classify(marks) -> list[surface.Defect]:
    return surface.classify_defects(_card(marks), CFG, PX_PER_MM)


def _grade(marks, grader: str | None = None) -> surface.SurfaceGrade:
    relief = _card(marks)
    area, count, longest, _ = surface.relief_defect_stats(relief, CFG)
    defects = surface.classify_defects(relief, CFG, PX_PER_MM)
    return surface.grade_surface(
        area, count, longest, "photometric_relief", THRESHOLDS, defects=defects, grader=grader
    )


def _compare(marks) -> dict:
    relief = _card(marks)
    area, count, longest, _ = surface.relief_defect_stats(relief, CFG)
    defects = surface.classify_defects(relief, CFG, PX_PER_MM)
    return surface.compare_graders(area, count, longest, "photometric_relief", THRESHOLDS, defects)


# One mark per row, and what each published standard says it costs.
#
#   PSA   hairline scratches cap at 9, visible ones at 7-8, deep ones that
#         gouge the print layer at 5-6, and a light crease at 5-6. A crease
#         running the full width is "usually an automatic 1".
#   TAG   a light scratch not penetrating the gloss still allows 10, one that
#         penetrates caps at 8.5, a minor dent 7.5, a wrinkle 5, a minor
#         crease 4.5, half the card 4, three-quarters 3, full length 2.
#   CGC   a light surface scratch is allowed at 9.5; "noticeable surface
#         flaws ... scuffing, scratches or one light crease" is already 4.5;
#         one moderate crease is 3; heavier creasing travelling edge to edge
#         is 2; severe creasing that breaks the surface is 1.
#
# Half grades round down throughout: the expensive error for a tool that
# decides whether to pay for a submission is optimism.
# A hairline: shallow *and* short. Length matters on its own — TAG allows "a
# light scratch" at 10 and "a longer scratch not penetrating the gloss" at 9,
# and PSA separates a hairline from a scratch you can read under normal light.
LIGHT_SCRATCH = (3.0, 0.2, 32, (30, 44), 0)
GLOSS_SCRATCH = (3.0, 0.2, 55, (30, 44), 0)
LONG_BUT_SHALLOW = (8.0, 0.2, 32, (30, 44), 0)
DEEP_GOUGE = (3.0, 0.2, 80, (30, 44), 0)
DENT = (1.5, 1.2, 80, (30, 44), 0)
WRINKLE = (6.0, 1.4, 30, (30, 44), 0)
MINOR_CREASE = (6.0, 1.4, 80, (30, 44), 0)


class TestWhatTheMarkIs:
    """Shape decides the name. A crease is a fold in stock, so it has width;
    a scratch is the track of something dragged, so it is narrow and long."""

    def test_a_narrow_shallow_line_is_a_scratch(self):
        (defect,) = _classify([(8.0, 0.2, 35, (30, 44), 0)])
        assert defect.kind == "scratch"

    def test_a_wide_deep_fold_is_a_crease(self):
        (defect,) = _classify([(6.0, 1.4, 80, (30, 44), 0)])
        assert defect.kind == "crease"

    def test_a_wide_shallow_fold_is_a_wrinkle_not_a_crease(self):
        """TAG separates them: a wrinkle deforms the stock, a crease breaks
        it, and that is two grades."""
        (defect,) = _classify([(6.0, 1.4, 30, (30, 44), 0)])
        assert defect.kind == "wrinkle"

    def test_a_compact_deep_mark_is_a_dent(self):
        (defect,) = _classify([(1.5, 1.2, 80, (30, 44), 0)])
        assert defect.kind == "dent"

    def test_a_diagonal_scratch_is_not_mistaken_for_a_crease(self):
        """A bounding box around a 45-degree scratch is nearly square and
        would call it as wide as it is long. Shape comes from the principal
        axes for exactly this reason."""
        (defect,) = _classify([(10.0, 0.2, 35, (30, 44), 45)])
        assert defect.kind == "scratch"
        assert defect.width_mm < surface.MAX_SCRATCH_WIDTH_MM


class TestTheCeilings:
    """Each standard's own published numbers, not one blended ladder."""

    @pytest.mark.parametrize("grader,expected", [("psa", 9), ("tag", 10), ("cgc", 9)])
    def test_a_light_scratch(self, grader, expected):
        assert _grade([LIGHT_SCRATCH], grader).grade == expected

    @pytest.mark.parametrize("grader,expected", [("psa", 7), ("tag", 8), ("cgc", 8)])
    def test_a_scratch_through_the_gloss(self, grader, expected):
        assert _grade([GLOSS_SCRATCH], grader).grade == expected

    @pytest.mark.parametrize("grader,expected", [("psa", 7), ("tag", 8), ("cgc", 8)])
    def test_length_alone_makes_a_scratch_visible(self, grader, expected):
        """A long shallow-angled cut disturbs the surface normal no more per
        pixel than a hairline — it just does it for far longer. Judging on
        depth alone called a 12mm gouge cut with scissors a light scratch."""
        assert _grade([LONG_BUT_SHALLOW], grader).grade == expected

    @pytest.mark.parametrize("grader,expected", [("psa", 5), ("tag", 5), ("cgc", 4)])
    def test_a_deep_gouge(self, grader, expected):
        """The widest disagreement between the three on a single mark."""
        assert _grade([DEEP_GOUGE], grader).grade == expected

    @pytest.mark.parametrize("grader,expected", [("psa", 7), ("tag", 7), ("cgc", 6)])
    def test_a_dent(self, grader, expected):
        assert _grade([DENT], grader).grade == expected

    @pytest.mark.parametrize("grader,expected", [("psa", 6), ("tag", 5), ("cgc", 5)])
    def test_a_wrinkle(self, grader, expected):
        assert _grade([WRINKLE], grader).grade == expected

    @pytest.mark.parametrize("grader,expected", [("psa", 5), ("tag", 4), ("cgc", 4)])
    def test_a_minor_crease(self, grader, expected):
        assert _grade([MINOR_CREASE], grader).grade == expected

    @pytest.mark.parametrize(
        "span_mm,psa,tag,cgc",
        [
            (30.0, 5, 4, 3),   # about a third of the card
            (50.0, 3, 4, 2),   # over half
            (70.0, 2, 3, 2),   # about three-quarters
            (86.0, 1, 2, 2),   # edge to edge
        ],
    )
    def test_a_crease_costs_more_the_further_it_runs(self, span_mm, psa, tag, cgc):
        """All three ladder creases by span, and all three disagree on it.
        PSA is alone in putting a full-length crease at 1."""
        marks = [(span_mm, 1.4, 80, (31, 44), 90)]
        assert _grade(marks, "psa").grade == psa
        assert _grade(marks, "tag").grade == tag
        assert _grade(marks, "cgc").grade == cgc

    def test_a_second_crease_costs_a_grade_where_the_standard_counts_them(self):
        """CGC separates "one light crease" (4.5) from "one or more light
        creases" (4), and PSA's low grades are written as "several creases".
        TAG's ladder is written by span alone, so it doesn't move."""
        two = [(6.0, 1.4, 80, (18, 30), 0), (6.0, 1.4, 80, (44, 62), 0)]
        assert _grade(two, "psa").grade == 4
        assert _grade(two, "cgc").grade == 3
        assert _grade(two, "tag").grade == _grade([MINOR_CREASE], "tag").grade

    def test_the_worst_defect_sets_the_grade(self):
        """A card carrying both a crease and a light scratch grades as the
        crease. Real graders do not average defects."""
        graded = _grade([MINOR_CREASE, (5.0, 0.2, 32, (40, 60), 0)])
        assert graded.grade == 5
        assert graded.limited_by == "crease"

    def test_a_clean_card_is_not_capped(self):
        graded = _grade([])
        assert graded.grade == 10
        assert graded.limited_by == "overall surface wear"

    def test_heavy_scuffing_still_grades_on_area(self):
        """The area band is not replaced by the ladder — it is the floor. A
        card covered in marks too small to name individually still grades
        badly, which is what "surface wear" means."""
        rng = np.random.default_rng(0)
        marks = [(1.2, 0.6, 40, (float(rng.uniform(4, 59)), float(rng.uniform(4, 84))), 0) for _ in range(900)]
        graded = _grade(marks)
        assert graded.grade <= 5
        assert graded.limited_by == "overall surface wear"

    def test_a_grade_never_falls_below_one(self):
        many = [(86.0, 1.4, 80, (10 + 12 * i, 44), 90) for i in range(4)]
        assert _grade(many, "psa").grade == 1
        assert _grade(many, "cgc").grade >= 1


class TestComparingTheStandards:
    def test_every_configured_standard_is_reported(self):
        compared = _compare([MINOR_CREASE])
        assert set(compared) == {"psa", "tag", "cgc"}
        for key, entry in compared.items():
            assert entry["label"]
            assert entry["source"], f"{key} must cite where its ladder came from"

    def test_they_disagree_and_the_report_shows_it(self):
        """The point of showing all three: one number was hiding a choice."""
        compared = _compare([DEEP_GOUGE])
        assert compared["psa"]["grade"] == 5
        assert compared["cgc"]["grade"] == 4

    def test_comparing_does_not_disturb_the_primary_grade(self):
        """Each grader prices its own copy of the defects. Sharing them let
        whichever grader ran last leave its ceilings behind on the list the
        report then displayed."""
        relief = _card([MINOR_CREASE])
        area, count, longest, _ = surface.relief_defect_stats(relief, CFG)
        defects = surface.classify_defects(relief, CFG, PX_PER_MM)
        primary = surface.grade_surface(area, count, longest, "photometric_relief", THRESHOLDS, defects)
        caps_before = [d.grade_cap for d in defects]
        surface.compare_graders(area, count, longest, "photometric_relief", THRESHOLDS, defects)
        assert [d.grade_cap for d in defects] == caps_before
        assert primary.grade == 5

    def test_the_primary_grader_is_named_on_the_grade(self):
        """A surface grade without a standard attached is not a complete
        statement when the standards differ by four grades."""
        graded = _grade([DEEP_GOUGE])
        assert graded.grader == "psa"
        assert graded.to_dict()["grader"] == "psa"


class TestTheReport:
    def test_the_defects_are_named_in_the_report(self):
        """'surface 4' is not an answer anyone can check. 'a 6mm crease at
        (30,44)mm caps this at 4' is."""
        graded = _grade([MINOR_CREASE])
        payload = graded.to_dict()
        assert payload["limited_by"] == "crease"
        assert payload["defect_kinds"]["crease"] == 1
        worst = payload["defects"][0]
        assert worst["kind"] == "crease"
        # The ceiling shown is the primary grader's — PSA puts a light crease
        # at 5-6 where TAG and CGC both put it at 4.5.
        assert worst["grade_cap"] == 5
        assert worst["centre_mm"] == pytest.approx([30, 44], abs=1.5)

    def test_the_list_is_capped_so_a_scuffed_card_does_not_flood_it(self):
        rng = np.random.default_rng(1)
        marks = [(1.2, 0.6, 40, (float(rng.uniform(4, 59)), float(rng.uniform(4, 84))), 0) for _ in range(400)]
        assert len(_grade(marks).to_dict()["defects"]) <= 12

    def test_an_ungraded_source_names_no_limit(self):
        relief = _card([MINOR_CREASE])
        area, count, longest, _ = surface.relief_defect_stats(relief, CFG)
        graded = surface.grade_surface(area, count, longest, "single_image_relief", THRESHOLDS,
                                       defects=surface.classify_defects(relief, CFG, PX_PER_MM))
        assert graded.grade is None
        assert graded.limited_by is None
        assert graded.to_dict()["defect_kinds"]["crease"] == 1


def test_a_defect_survives_being_written_to_disk():
    """The report is stored as JSON, and json.dumps rejects numpy scalars
    outright. Blob statistics come straight from numpy, so a defect that
    hasn't been cast takes the whole report down at save time — after the
    grading work is already done."""
    import json as _json

    relief = _card([MINOR_CREASE])
    defects = surface.classify_defects(relief, CFG, PX_PER_MM)
    assert defects, "fixture must produce at least one defect"
    payload = [d.to_dict() for d in defects]
    _json.dumps(payload)                      # must not raise
    for value in payload[0]["box"]:
        assert type(value) is float, f"box carries a {type(value).__name__}, not a float"
    assert type(payload[0]["area_px"]) is int
