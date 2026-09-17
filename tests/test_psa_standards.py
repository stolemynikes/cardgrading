"""The centering tolerance tables, pinned to PSA's published standards.

Source: https://www.psacard.com/gradingstandards — the grade definitions state
a front and reverse tolerance for every grade from 10 down to 2. These are the
only numbers in the whole pipeline that come from a grading service rather than
from our own calibration, so they are worth asserting literally: a typo here
mis-grades every card, silently and plausibly.

The back table is the one that bites. Only Gem Mint 10 is tight (75/25); from
Mint 9 downward PSA allows 90/10. An earlier table tightened the back
progressively by grade, which read as reasonable and was wrong — it graded a
card measuring 88/12 on the reverse a 7 where PSA allows a 9.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import centering

THRESHOLDS = json.loads((Path(__file__).resolve().parent.parent / "calibration" / "thresholds.json").read_text())
FRONT = THRESHOLDS["centering"]["front_tolerances"]
BACK = THRESHOLDS["centering"]["back_tolerances"]

# grade -> (front worst-allowed, reverse worst-allowed), verbatim from PSA
PSA = {
    10: (55, 75),
    9: (60, 90),
    8: (65, 90),
    7: (70, 90),
    6: (80, 90),
    5: (85, 90),
    4: (85, 90),
    3: (90, 90),
    2: (90, 90),
}


@pytest.mark.parametrize("grade,expected", sorted(PSA.items()))
def test_front_tolerance_matches_psa(grade, expected):
    actual = {t["grade"]: t["max_ratio"] for t in FRONT}
    assert actual[grade] == expected[0]


@pytest.mark.parametrize("grade,expected", sorted(PSA.items()))
def test_back_tolerance_matches_psa(grade, expected):
    actual = {t["grade"]: t["max_ratio"] for t in BACK}
    assert actual[grade] == expected[1]


def test_tolerances_are_monotonic():
    """A worse grade may never demand better centering than a better one."""
    for table in (FRONT, BACK):
        by_grade = sorted(table, key=lambda t: -t["grade"])
        ratios = [t["max_ratio"] for t in by_grade]
        assert ratios == sorted(ratios), "tolerance must loosen as the grade drops"


def test_a_slightly_off_reverse_still_grades_nine():
    """The regression the corrected back table fixes: 88/12 on the reverse is
    within PSA's 90/10 allowance for a Mint 9, not a 7."""
    assert centering._grade_from_ratio(88.0, BACK) == 9


def test_gem_mint_reverse_is_the_only_tight_one():
    assert centering._grade_from_ratio(76.0, BACK) == 9     # just past 75/25
    assert centering._grade_from_ratio(74.0, BACK) == 10


def test_front_boundaries_land_on_the_right_side():
    assert centering._grade_from_ratio(55.0, FRONT) == 10
    assert centering._grade_from_ratio(55.1, FRONT) == 9
    assert centering._grade_from_ratio(80.0, FRONT) == 6
    assert centering._grade_from_ratio(80.1, FRONT) == 5
