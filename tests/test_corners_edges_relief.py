"""Corner and edge wear read from the photometric relief.

Whitening is found as a local brightness spike against a darker border. That
works on a classic dark-bordered card and finds nothing at all on a modern
silver-bordered one: measured on a real card whitened deliberately along two
edges, the whitening map read **0.0% on every region of both the clean and
the damaged capture**. The same regions read 0.001% against 1.209% in relief.

Worse, the gate protecting the whitening measurement was being tripped by the
damage. A crop that isn't uniform border is refused, because whitening
measured off artwork isn't a measurement of the card — and a whitened edge is
less uniform than a clean one. The deliberately whitened edge pushed its own
uniformity from above the floor to 0.54 and was refused, so the card's worst
edge reported "can't measure" instead of a bad grade.

Relief needs no border at all: the albedo is divided out before anything is
measured. So it grades where whitening refuses, and the two are taken at their
worst — they fail in opposite directions, whitening being blind to a light
border and relief being blind to a stain that hasn't deformed anything.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pipeline import corners_edges as ce
from webapp import main

THRESHOLDS = json.loads(main.THRESHOLDS_PATH.read_text())
CFG = THRESHOLDS["corners_edges"]
W = THRESHOLDS["capture"]["canonical_width_px"]
H = THRESHOLDS["capture"]["canonical_height_px"]
BORDERS = ce.BorderWidths(left=62, right=95, top=60, bottom=80)


def _flat_relief() -> np.ndarray:
    return np.full((H, W), 128, np.uint8)


def _uniform_border_card() -> np.ndarray:
    """A card the whitening path can read: a flat, uniform, darkish border."""
    card = np.full((H, W, 3), 40, np.uint8)
    card[140:-160, 100:-120] = (190, 170, 150)
    return card


def _busy_border_card() -> np.ndarray:
    """A border the uniformity gate refuses — patterned, like a modern card.

    Blocks small enough to vary *within* a corner crop: uniformity is measured
    per crop, so a coarse pattern reads as perfectly uniform inside any one of
    them.
    """
    rng = np.random.default_rng(4)
    blocks = rng.integers(0, 255, (H // 40, W // 40, 3), dtype=np.uint8)
    return cv2.resize(blocks, (W, H), interpolation=cv2.INTER_NEAREST)


def _yellow_border_card() -> np.ndarray:
    """A border the whitening path can actually read.

    Its load-bearing test is desaturation — exposed cardstock is far less
    saturated than the ink over it — so it needs a *saturated* border. Against
    a yellow Base Set border a white chip gates through; against the grey or
    silver border of a modern card, whose saturation is already near zero,
    nothing can be less saturated than the border and the whole method is
    blind. That is not a tuning problem, and it is why relief matters here.
    """
    card = np.full((H, W, 3), (0, 200, 255), np.uint8)
    card[140:-160, 100:-120] = (190, 170, 150)
    return card


class TestReliefGradesWhereWhiteningCannot:
    def test_a_refused_crop_is_still_graded_from_relief(self):
        card = _busy_border_card()
        refused, _ = ce.analyze_side(card, BORDERS, CFG)
        graded, _ = ce.analyze_side(card, BORDERS, CFG, relief=_flat_relief())

        assert any(not r.measurable for r in refused.corners.values()), "fixture must refuse without relief"
        assert all(r.measurable for r in graded.corners.values())
        assert all(r.measurable for r in graded.edges.values())

    def test_and_says_it_was_graded_from_relief_alone(self):
        graded, _ = ce.analyze_side(_busy_border_card(), BORDERS, CFG, relief=_flat_relief())
        reasons = [r.reason for r in graded.corners.values() if r.reason]
        assert reasons, "a crop the whitening path can't read should say so"
        assert any("relief only" in reason for reason in reasons)

    def test_a_region_the_whitening_path_refused_reads_clean_from_relief(self):
        """The point of the refusal was that a patterned border produces a
        meaningless whitening number. Relief has no such problem — a flat
        relief is a flat card whatever is printed on it.

        Only the refused regions are checked. Where the crop *did* pass the
        uniformity gate the whitening path still runs, and on random pattern
        it still produces the false positives the gate exists to limit; that
        is the pre-existing behaviour, not something relief claims to fix.
        """
        card = _busy_border_card()
        without, _ = ce.analyze_side(card, BORDERS, CFG)
        refused = {name for name, r in without.corners.items() if not r.measurable}
        refused |= {name for name, r in without.edges.items() if not r.measurable}
        assert refused, "fixture must refuse at least one region without relief"

        graded, _ = ce.analyze_side(card, BORDERS, CFG, relief=_flat_relief())
        for name in refused:
            region = graded.corners.get(name) or graded.edges.get(name)
            assert region.measurable, f"{name} should be measurable from relief"
            assert region.grade == 10, f"{name} has a flat relief and should read clean"


class TestDamageIsFound:
    @staticmethod
    def _relief_with(region_slice) -> np.ndarray:
        relief = _flat_relief()
        relief[region_slice] = 220
        return relief

    def test_a_deformed_corner_drops_the_corner_grade(self):
        relief = self._relief_with((slice(0, 220), slice(0, 220)))
        result, _ = ce.analyze_side(_uniform_border_card(), BORDERS, CFG, relief=relief)
        assert result.corners["top_left"].grade < 7
        assert result.corners["top_right"].grade == 10, "only the damaged corner should move"

    def test_a_deformed_edge_drops_the_edge_grade(self):
        relief = self._relief_with((slice(0, 60), slice(400, 1100)))
        result, _ = ce.analyze_side(_uniform_border_card(), BORDERS, CFG, relief=relief)
        assert result.edges["top"].grade < 10
        assert result.edges["bottom"].grade == 10

    def test_the_measured_wear_is_reported(self):
        relief = self._relief_with((slice(0, 220), slice(0, 220)))
        result, _ = ce.analyze_side(_uniform_border_card(), BORDERS, CFG, relief=relief)
        payload = result.corners["top_left"].to_dict()
        assert payload["relief_wear_pct"] > 10
        assert result.corners["top_right"].to_dict()["relief_wear_pct"] == 0.0

    def test_corners_and_edges_use_their_own_bands(self):
        """A corner crop is small and touches two warp seams, so it carries
        far more false relief than a long edge crop. One shared table sized
        for corners made every edge unreadable: a deliberately whitened edge
        measured 1.21%, 811x its own clean reading, and still sat inside a
        floor set at 2.5%."""
        corners = ce._relief_wear_bands(CFG, is_corner=True)
        edges = ce._relief_wear_bands(CFG, is_corner=False)
        assert corners and edges and corners != edges
        top_corner = corners[0].get("max_pct", corners[0].get("max_whitening_pct"))
        top_edge = edges[0].get("max_pct", edges[0].get("max_whitening_pct"))
        assert top_edge < top_corner


class TestItStaysOptional:
    def test_no_relief_behaves_exactly_as_before(self):
        card = _uniform_border_card()
        without, _ = ce.analyze_side(card, BORDERS, CFG)
        assert without.corners_grade is not None
        for region in [*without.corners.values(), *without.edges.values()]:
            assert region.relief_wear_pct is None

    def test_a_mismatched_relief_is_ignored_rather_than_crashing(self):
        """A side whose relief came out at a different size than its warp —
        a fallback path, a stale render — must not take the stage down."""
        card = _uniform_border_card()
        result, _ = ce.analyze_side(card, BORDERS, CFG, relief=np.full((100, 100), 128, np.uint8))
        assert result.corners_grade is not None
        assert all(r.relief_wear_pct is None for r in result.corners.values())

    def test_the_worse_of_the_two_readings_wins(self):
        """They fail in opposite directions, so neither may rescue the other.
        A chip that has lifted colour without deforming the stock shows in
        whitening and not in relief."""
        card = _yellow_border_card()
        # Small relative to the crop: a corner crop is about 48px square here,
        # and a chip filling it entirely becomes the crop's own median, which
        # is precisely what the gate measures everything against.
        card[4:18, 4:18] = (250, 250, 250)
        result, _ = ce.analyze_side(card, BORDERS, CFG, relief=_flat_relief())
        assert result.corners["top_left"].grade < 10, "a flat relief must not wash out real whitening"


class TestWhiteningIsBlindOnANeutralBorder:
    """Why relief was added rather than the whitening thresholds retuned.

    The absolute gate asks for pixels markedly *less saturated* than the crop's
    own median. On a yellow or blue border that isolates exposed cardstock
    exactly as intended. On a grey, silver or white border the median
    saturation is already near zero, so no pixel can clear the test and the
    method reports 0.0% however badly the card is chipped.
    """

    @staticmethod
    def _chipped(border) -> np.ndarray:
        crop = np.full((200, 200, 3), border, np.uint8)
        crop[20:120, 20:120] = (250, 250, 250)
        return crop

    @pytest.mark.parametrize("border", [(0, 200, 255), (150, 60, 20)], ids=["yellow", "blue"])
    def test_a_saturated_border_shows_its_chip(self, border):
        mask = ce._whitening_mask(
            ce._whitening_visibility_map(self._chipped(border), CFG), self._chipped(border), CFG
        )
        assert mask.sum() > 0

    @pytest.mark.parametrize("border", [(40, 40, 40), (245, 245, 245)], ids=["grey", "white"])
    def test_a_neutral_border_hides_the_same_chip(self, border):
        crop = self._chipped(border)
        mask = ce._whitening_mask(ce._whitening_visibility_map(crop, CFG), crop, CFG)
        assert mask.sum() == 0, "documents the blind spot rather than claiming it is fixed"

    def test_relief_sees_the_chip_the_neutral_border_hid(self):
        deformed = np.full((200, 200), 128, np.uint8)
        deformed[20:120, 20:120] = 220
        assert ce.relief_wear_pct(deformed, CFG) > 20
        assert ce.relief_wear_pct(np.full((200, 200), 128, np.uint8), CFG) == 0.0
