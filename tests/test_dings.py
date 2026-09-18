"""DINGS — the ranking that decides what a reader sees first."""

from __future__ import annotations

from pipeline import centering, dings


def _region(grade: int, whitening: float = 1.0, blobs: int = 2) -> dict:
    return {"grade": grade, "whitening_pct": whitening, "blob_count": blobs}


def _side(corner_grades: dict, edge_grades: dict) -> dict:
    return {
        "corners": {name: _region(g) for name, g in corner_grades.items()},
        "edges": {name: _region(g) for name, g in edge_grades.items()},
    }


CLEAN_CORNERS = {"top_left": 10, "top_right": 10, "bottom_right": 10, "bottom_left": 10}
CLEAN_EDGES = {"top": 10, "right": 10, "bottom": 10, "left": 10}


def _report(**overrides) -> dict:
    report = {
        "centering": {
            "front": {
                "horizontal": {"measurable": True, "grade": 10, "ratio": "50/50"},
                "vertical": {"measurable": True, "grade": 10, "ratio": "50/50"},
            },
            "back": {
                "horizontal": {"measurable": True, "grade": 10, "ratio": "50/50"},
                "vertical": {"measurable": True, "grade": 10, "ratio": "50/50"},
            },
        },
        "corners_edges": {"front": _side(CLEAN_CORNERS, CLEAN_EDGES), "back": _side(CLEAN_CORNERS, CLEAN_EDGES)},
        "surface": None,
        "vision_flat": None,
        "dimensions": None,
    }
    report.update(overrides)
    return report


def test_perfect_card_has_no_dings():
    assert dings.collect_dings(_report()) == []


def test_only_the_worst_region_is_listed():
    corners = {**CLEAN_CORNERS, "top_left": 6, "bottom_right": 8}
    report = _report(
        corners_edges={"front": _side(corners, CLEAN_EDGES), "back": _side(CLEAN_CORNERS, CLEAN_EDGES)}
    )
    found = dings.collect_dings(report)
    assert [d["label"] for d in found] == ["top-left corner"]


def test_tied_worst_regions_are_both_listed():
    corners = {**CLEAN_CORNERS, "top_left": 7, "bottom_right": 7}
    report = _report(
        corners_edges={"front": _side(corners, CLEAN_EDGES), "back": _side(CLEAN_CORNERS, CLEAN_EDGES)}
    )
    assert len(dings.collect_dings(report)) == 2


def test_corners_and_edges_rank_separately():
    """They're separate attributes, so a clean-cornered card with a bad edge
    must still report the edge — a single card-wide minimum would hide it."""
    report = _report(
        corners_edges={
            "front": _side({**CLEAN_CORNERS, "top_left": 9}, {**CLEAN_EDGES, "left": 5}),
            "back": _side(CLEAN_CORNERS, CLEAN_EDGES),
        }
    )
    labels = [d["label"] for d in dings.collect_dings(report)]
    assert labels == ["left edge", "top-left corner"]


def test_unmeasurable_centering_is_not_a_defect():
    """A borderless card has no border to be off-center; that's explained in
    the centering section, not listed as a flaw."""
    report = _report(
        centering={
            "front": {
                "horizontal": {"measurable": False, "grade": 1, "ratio": "89/11"},
                "vertical": {"measurable": False, "grade": 1, "ratio": "88/12"},
            },
            "back": {
                "horizontal": {"measurable": True, "grade": 10, "ratio": "50/50"},
                "vertical": {"measurable": True, "grade": 10, "ratio": "50/50"},
            },
        }
    )
    assert dings.collect_dings(report) == []


def test_measured_surface_defects_are_listed():
    report = _report(
        surface={
            "front": {"grade": 6, "defect_area_pct": 1.25, "defect_count": 4,
                      "longest_defect_px": 820},
        }
    )
    surface_dings = [d for d in dings.collect_dings(report) if d["attribute"] == "surface"]
    assert len(surface_dings) == 1
    assert "1.25%" in surface_dings[0]["detail"]


def test_the_defect_that_capped_the_grade_gets_its_own_ding():
    """A card graded 4 whose only ding says "0.15% of the surface" tells
    nobody it has a crease in the bottom-left corner — and the crease is the
    entire reason for the grade. The point of a ding is to say where to look."""
    report = _report(
        surface={
            "front": {
                "grade": 4, "defect_area_pct": 0.15, "defect_count": 53, "longest_defect_px": 214,
                "limited_by": "crease",
                "defects": [
                    {"kind": "crease", "length_mm": 2.7, "width_mm": 1.35,
                     "centre_mm": [1.2, 87.4], "grade_cap": 4},
                    {"kind": "scratch", "length_mm": 9.9, "width_mm": 0.24,
                     "centre_mm": [29.1, 57.6], "grade_cap": 9},
                ],
            },
        }
    )
    surface_dings = [d for d in dings.collect_dings(report) if d["attribute"] == "surface"]
    assert len(surface_dings) == 2
    named = surface_dings[0]
    assert "crease" in named["label"]
    assert "87" in named["detail"], "the position on the card is what sends someone to look"
    assert "caps this card at 4" in named["detail"]


def test_plain_surface_wear_gets_no_named_ding():
    """Nothing to point at: the area band is the whole finding."""
    report = _report(
        surface={
            "front": {"grade": 5, "defect_area_pct": 3.4, "defect_count": 900,
                      "longest_defect_px": 120, "limited_by": "overall surface wear",
                      "defects": [{"kind": "pit", "length_mm": 1.0, "width_mm": 0.5,
                                   "centre_mm": [30, 40], "grade_cap": 9}]},
        }
    )
    surface_dings = [d for d in dings.collect_dings(report) if d["attribute"] == "surface"]
    assert len(surface_dings) == 1


def test_a_region_ding_cites_the_reading_that_found_the_damage():
    """A corner graded 3 that reports "0.00% whitening" names the one
    measurement that saw nothing. Whitening is blind on a neutral border;
    relief is what found it."""
    report = _report(
        corners_edges={
            "front": _side({**CLEAN_CORNERS, "top_left": 3}, CLEAN_EDGES),
            "back": _side(CLEAN_CORNERS, CLEAN_EDGES),
        }
    )
    report["corners_edges"]["front"]["corners"]["top_left"].update(
        {"whitening_pct": 0.0, "relief_wear_pct": 29.69, "blob_count": 0}
    )
    corner_dings = [d for d in dings.collect_dings(report) if d["attribute"] == "corners"]
    assert corner_dings
    assert "29.69% of the region is physically deformed" in corner_dings[0]["detail"]
    assert "0.00% whitening" not in corner_dings[0]["detail"]


def test_a_region_ding_still_cites_whitening_when_that_found_it():
    report = _report(
        corners_edges={
            "front": _side({**CLEAN_CORNERS, "top_left": 5}, CLEAN_EDGES),
            "back": _side(CLEAN_CORNERS, CLEAN_EDGES),
        }
    )
    report["corners_edges"]["front"]["corners"]["top_left"].update(
        {"whitening_pct": 4.2, "relief_wear_pct": 0.0, "blob_count": 3}
    )
    corner_dings = [d for d in dings.collect_dings(report) if d["attribute"] == "corners"]
    assert "4.20% whitening" in corner_dings[0]["detail"]


def test_ungraded_surface_produces_no_ding():
    """The single-capture approximation leaks print into the signal, so its
    defect count would include printed linework — worse than saying nothing."""
    report = _report(
        surface={"front": {"grade": None, "defect_area_pct": 9.9, "defect_count": 300,
                           "longest_defect_px": 1400}}
    )
    assert dings.collect_dings(report) == []


def test_worst_first_with_miscut_at_the_top():
    report = _report(
        corners_edges={"front": _side({**CLEAN_CORNERS, "top_left": 4}, CLEAN_EDGES), "back": _side(CLEAN_CORNERS, CLEAN_EDGES)},
        dimensions={"measurable": True, "within_tolerance": False, "note": "width off by -2.00mm — miscut or trimmed"},
    )
    found = dings.collect_dings(report)
    assert found[0]["attribute"] == "dimensions"
    assert found[1]["grade"] == 4


def test_axis_keys_match_what_centering_actually_emits():
    """Regression: this module read `ratio_str` while the report carries
    `ratio`, so any card with a measurable off-center axis raised KeyError.
    The unit fixtures above can't catch that on their own — they'd just
    encode whichever names this module happens to use — so assert against the
    real serializer instead."""
    axis = centering.AxisCentering(
        side_a="left", side_b="right", side_a_px=141.0, side_b_px=56.0,
        side_a_pct=71.6, side_b_pct=28.4, ratio_str="72/28", grade=6,
    )
    result = centering.CenteringResult(
        front_horizontal=axis, front_vertical=axis, front_grade=6,
        back_horizontal=axis, back_vertical=axis, back_grade=6, overall_grade=6,
    )
    found = dings.collect_dings({"centering": result.to_dict()})
    assert len(found) == 4
    assert all("72/28" in d["detail"] for d in found)


def test_older_report_without_measurability_still_ranks():
    report = _report()
    for side in ("front", "back"):
        for axis in ("horizontal", "vertical"):
            del report["centering"][side][axis]["measurable"]
    report["centering"]["front"]["horizontal"]["grade"] = 6
    assert [d["label"] for d in dings.collect_dings(report)] == ["horizontal centering"]


class TestUngradedRegionsAreNotRanked:
    """A refused region carries no grade at all. It used to serialize the
    grade it would have had — gated only by a `measurable` flag — which made
    the ranking here work by accident. Reporting None instead crashed every
    grading run with "'<' not supported between instances of 'NoneType' and
    'int'" the moment any region was refused."""

    @staticmethod
    def _regions(graded: dict, refused: list) -> dict:
        out = {
            name: {"grade": grade, "whitening_pct": 1.0, "blob_count": 1, "measurable": True}
            for name, grade in graded.items()
        }
        for name in refused:
            out[name] = {"grade": None, "whitening_pct": 0.0, "blob_count": 0, "measurable": False}
        return out

    def test_a_refused_region_does_not_crash_the_ranking(self):
        side = {
            "corners": self._regions({"top_left": 7}, ["top_right", "bottom_left", "bottom_right"]),
            "edges": self._regions({}, ["top", "right", "bottom", "left"]),
        }
        found = dings._corners_edges_dings("front", side)
        assert [d["grade"] for d in found] == [7]

    def test_a_group_with_nothing_graded_contributes_nothing(self):
        side = {"corners": self._regions({}, ["top_left"]), "edges": {}}
        assert dings._corners_edges_dings("front", side) == []

    def test_a_refused_region_cannot_be_named_as_the_worst(self):
        side = {
            "corners": self._regions({"top_left": 9, "top_right": 6}, ["bottom_left"]),
            "edges": {},
        }
        labels = [d["label"] for d in dings._corners_edges_dings("front", side)]
        assert all("bottom" not in label.lower() for label in labels)

    def test_an_axis_without_a_grade_is_skipped(self):
        side = {
            "horizontal": {"grade": None, "measurable": True, "ratio": "50/50"},
            "vertical": {"grade": 6, "measurable": True, "ratio": "70/30"},
        }
        assert [d["grade"] for d in dings._centering_dings("front", side)] == [6]

    def test_a_side_with_no_graded_axis_contributes_nothing(self):
        side = {
            "horizontal": {"grade": None, "measurable": True},
            "vertical": {"grade": None, "measurable": True},
        }
        assert dings._centering_dings("front", side) == []


class TestNoVerdictIsNotADefect:
    """`within_tolerance` is three-valued. None means the scans disagreed by
    more than the tolerance they were judged against, so no verdict was
    given — and "we could not tell" is not a defect. It was being listed as a
    grade-capping ding on a card nothing had been established about."""

    def test_a_card_with_no_verdict_is_not_dinged(self):
        assert dings._dimensions_dings(
            {"measurable": True, "within_tolerance": None, "spread_mm": 2.35, "note": "can't tell"}
        ) == []

    def test_a_genuine_miscut_is_still_dinged(self):
        found = dings._dimensions_dings(
            {"measurable": True, "within_tolerance": False, "note": "Miscut or trimmed — width off by -2.13mm"}
        )
        assert len(found) == 1 and "Miscut" in found[0]["detail"]

    def test_a_correctly_cut_card_is_not_dinged(self):
        assert dings._dimensions_dings({"measurable": True, "within_tolerance": True, "note": "fine"}) == []

    def test_an_unmeasurable_card_is_not_dinged(self):
        assert dings._dimensions_dings({"measurable": False, "within_tolerance": None, "note": "no dpi"}) == []


class TestRegionDingsCarryTheirPlace:
    """So the report can draw the defect back onto the card, rather than only
    showing a crop with no sense of where on the card it came from."""

    def _side(self):
        region = {"grade": 7, "whitening_pct": 1.2, "blob_count": 2, "measurable": True,
                  "box": [0.01, 0.02, 0.07, 0.05]}
        return {"corners": {"top_left": region}, "edges": {}}

    def test_the_box_reaches_the_ding(self):
        found = dings._corners_edges_dings("front", self._side())
        assert found[0]["box"] == [0.01, 0.02, 0.07, 0.05]

    def test_a_region_without_a_box_still_dings(self):
        side = self._side()
        del side["corners"]["top_left"]["box"]
        found = dings._corners_edges_dings("front", side)
        assert len(found) == 1 and found[0]["box"] is None


def test_the_named_defect_is_the_biggest_of_the_worst():
    """Two marks with the same ceiling cap the card equally; the one worth
    walking over to look at is the bigger one."""
    report = _report(
        surface={
            "front": {
                "grade": 4, "defect_area_pct": 0.15, "defect_count": 2, "longest_defect_px": 200,
                "limited_by": "crease",
                "defects": [
                    {"kind": "crease", "length_mm": 2.2, "width_mm": 1.1,
                     "centre_mm": [1.0, 1.0], "grade_cap": 5},
                    {"kind": "crease", "length_mm": 2.7, "width_mm": 1.35,
                     "centre_mm": [1.2, 87.4], "grade_cap": 5},
                ],
            },
        }
    )
    named = [d for d in dings.collect_dings(report) if d["attribute"] == "surface"][0]
    assert "87" in named["detail"], "the larger crease is the one to point at"


def test_a_count_penalty_is_explained_not_left_as_a_contradiction():
    """"caps this card at 5" beside a grade of 4 reads as a contradiction
    unless the reason — a second crease, which PSA and CGC both charge for —
    is given."""
    report = _report(
        surface={
            "front": {
                "grade": 4, "defect_area_pct": 0.15, "defect_count": 2, "longest_defect_px": 200,
                "limited_by": "crease",
                "defects": [
                    {"kind": "crease", "length_mm": 2.7, "width_mm": 1.35,
                     "centre_mm": [1.2, 87.4], "grade_cap": 5},
                    {"kind": "crease", "length_mm": 2.2, "width_mm": 1.1,
                     "centre_mm": [1.0, 1.0], "grade_cap": 5},
                ],
            },
        }
    )
    named = [d for d in dings.collect_dings(report) if d["attribute"] == "surface"][0]
    assert "caps this card at 5" in named["detail"]
    assert "2 of them" in named["detail"]
