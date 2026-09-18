"""Finding a dark card on a dark background.

A Pokemon card *back* is deep blue, and it was scanned on black card stock.
Both read about the same brightness, so the only thing separating them is
colour — and the detector's threshold step works on distance from the sampled
background colour, where that separation is small compared with the artwork's.

Measured on the real scan:

    background       9
    blue border     35     <- the card's actual edge
    artwork        235
    Otsu cut       104     <- above the border, so the border read as background

The detector confidently returned the *artwork* as the card: 56.55 x 81.33mm
for a 63 x 88mm card, a warp cropped inside the card's own border with no
edges in it at all. Every centering, corner and edge number on that side was
then measured against a boundary that was not the card's.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline import detect

W, H = 1200, 1680


def _dark_card_on_dark_background(border_bgr, background_bgr=(50, 50, 50)) -> np.ndarray:
    """A card whose border barely separates from the background, with bright
    artwork inside — the shape that makes Otsu cut in the wrong place."""
    rng = np.random.default_rng(7)
    img = np.full((H, W, 3), background_bgr, np.uint8)
    y0, x0, ch, cw = 180, 150, 1300, 930
    img[y0:y0 + ch, x0:x0 + cw] = border_bgr                    # the card's border
    art = rng.integers(90, 250, (ch - 200, cw - 160, 3), dtype=np.uint8)
    img[y0 + 100:y0 + ch - 100, x0 + 80:x0 + cw - 80] = art     # bright artwork
    return img, (x0, y0, cw, ch)


class TestTheBorderIsFoundNotTheArtwork:
    def test_a_blue_border_on_black_stock(self):
        """The real case: B=79 G=41 R=33 border on a neutral background."""
        img, (x0, y0, cw, ch) = _dark_card_on_dark_background((79, 41, 33))
        corners = detect.find_card_contour(img)
        assert corners is not None
        quad = corners.astype(float)
        width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
        height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        assert width == pytest.approx(cw, abs=25), f"found {width:.0f}px wide, card is {cw}px"
        assert height == pytest.approx(ch, abs=25), f"found {height:.0f}px tall, card is {ch}px"

    def test_it_does_not_settle_for_the_artwork(self):
        """The failure it replaces: a quad inside the card's own border."""
        img, (_, _, cw, ch) = _dark_card_on_dark_background((79, 41, 33))
        corners = detect.find_card_contour(img)
        area = cv2.contourArea(corners.astype(np.float32))
        assert area > 0.8 * cw * ch, "the found quad is well inside the card"

    @pytest.mark.parametrize(
        "border",
        [(79, 41, 33), (70, 45, 40), (95, 55, 45)],
        ids=["measured", "darker", "lighter"],
    )
    def test_across_a_range_of_dark_borders(self, border):
        img, (_, _, cw, ch) = _dark_card_on_dark_background(border)
        corners = detect.find_card_contour(img)
        assert corners is not None
        assert cv2.contourArea(corners.astype(np.float32)) > 0.8 * cw * ch


class TestTheSweepReachesFarEnough:
    def test_it_goes_below_otsu_as_well_as_above(self):
        assert min(detect.THRESHOLD_MULTIPLIERS) < 1.0
        assert max(detect.THRESHOLD_MULTIPLIERS) > 1.0

    def test_it_reaches_a_third_of_otsu(self):
        """0.4 was tried first and was not enough: on the real scan it put the
        threshold at 41 against a border sitting at 35. The border was at
        one third of Otsu, so the sweep has to reach past that."""
        assert min(detect.THRESHOLD_MULTIPLIERS) <= 0.25

    def test_a_bright_card_on_a_dark_background_still_works(self):
        """The ordinary case, which the extra thresholds must not disturb."""
        img = np.full((H, W, 3), 25, np.uint8)
        img[180:1480, 150:1080] = (205, 200, 195)
        corners = detect.find_card_contour(img)
        assert corners is not None
        quad = corners.astype(float)
        width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
        assert width == pytest.approx(930, abs=20)
