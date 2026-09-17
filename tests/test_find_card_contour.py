"""Card contour detection on synthetic scenes with known ground truth."""

import cv2
import numpy as np
import pytest

from pipeline import detect


def make_scene(bg_value=25, card_bright=200, size=(1600, 1200), card_frac=0.55, noise=None):
    """A card-proportioned bright rectangle centered on a dark matte background."""
    h, w = size
    img = np.full((h, w, 3), bg_value, dtype=np.uint8)
    card_h = int(h * card_frac)
    card_w = int(card_h * 63 / 88)
    y0 = (h - card_h) // 2
    x0 = (w - card_w) // 2
    img[y0:y0 + card_h, x0:x0 + card_w] = (card_bright, card_bright - 30, card_bright - 60)
    if noise is not None:
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, (x0, y0, card_w, card_h)


def quad_matches(corners, x0, y0, w, h, tol=15):
    expected = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], dtype=np.float32)
    return np.abs(corners - expected).max() <= tol


class TestCleanBackground:
    def test_detects_centered_card(self):
        img, (x0, y0, w, h) = make_scene()
        corners = detect.find_card_contour(img)
        assert corners is not None
        assert quad_matches(corners, x0, y0, w, h)

    def test_detected_quad_passes_geometry_gates(self):
        img, _ = make_scene()
        corners = detect.find_card_contour(img)
        cfg = {"card_aspect_ratio": [63, 88], "aspect_ratio_tolerance_pct": 3.0,
               "max_corner_angle_deviation_deg": 2.0}
        assert detect.check_tilt(corners, cfg).passed
        assert detect.check_aspect_ratio(corners, cfg).passed

    def test_empty_background_returns_none(self):
        img = np.full((1600, 1200, 3), 25, dtype=np.uint8)
        assert detect.find_card_contour(img) is None


class TestBusyBackground:
    """The real-world failure this detector was rebuilt for: a textured
    background bright enough that Otsu's threshold merges card and
    background into one blob. The multi-threshold candidate sweep must
    still isolate the card."""

    def make_textured_scene(self):
        rng = np.random.default_rng(42)
        h, w = 1600, 1200
        # medium-bright noisy texture (like fabric/concrete), distinctly
        # dimmer than the card but far from black
        img = rng.integers(60, 140, size=(h, w, 3), dtype=np.uint8).astype(np.uint8)
        card_h = int(h * 0.55)
        card_w = int(card_h * 63 / 88)
        y0 = (h - card_h) // 2
        x0 = (w - card_w) // 2
        img[y0:y0 + card_h, x0:x0 + card_w] = (230, 200, 170)
        return img, (x0, y0, card_w, card_h)

    def test_card_isolated_from_texture(self):
        img, (x0, y0, w, h) = self.make_textured_scene()
        corners = detect.find_card_contour(img)
        assert corners is not None
        assert quad_matches(corners, x0, y0, w, h, tol=25)

    def test_busy_scene_quad_is_card_shaped(self):
        img, _ = self.make_textured_scene()
        corners = detect.find_card_contour(img)
        cfg = {"card_aspect_ratio": [63, 88], "aspect_ratio_tolerance_pct": 3.0}
        assert detect.check_aspect_ratio(corners, cfg).passed


class TestBackgroundThatIsntOneColour:
    """A backing sheet smaller than the scanner bed.

    The real failure: a black card-stock backing sheet that didn't cover the
    whole glass left a wedge of bare white lid showing at one edge. The
    background colour was estimated as the *mean* of the four image corners,
    so black stock (~14) and a white lid (~245) averaged to a mid-grey that
    matched neither. Every pixel then measured as far from background, the
    whole frame came out as one blob, and two scans of a four-scan set were
    reported as having no card in them at all — which dropped the photometric
    solve and cost the card its surface grade.
    """

    BG = 14
    LID = 245

    def scene(self, lid=None, card_at=None, card_value=205, size=(1400, 1400)):
        h, w = size
        img = np.full((h, w, 3), self.BG, np.uint8)
        if lid == "left_edge":
            img[:, : w // 5] = self.LID
        elif lid == "left_half":  # two of the four corners land on the lid
            img[:, : w // 2] = self.LID
        elif lid == "one_corner":
            img[int(h * 0.8) :, : w // 5] = self.LID

        card_h, card_w = 740, 530
        y0, x0 = card_at if card_at else ((h - card_h) // 2, (w - card_w) // 2)
        card = np.full((card_h, card_w, 3), card_value, np.uint8)
        card[90:600, 45:485] = (120, 90, 70)
        img[y0 : y0 + card_h, x0 : x0 + card_w] = card
        return img, (x0, y0, card_w, card_h)

    @pytest.mark.parametrize("lid", [None, "left_edge", "left_half", "one_corner"])
    def test_the_card_is_found_however_much_lid_shows(self, lid):
        img, (x0, y0, w, h) = self.scene(lid=lid)
        corners = detect.find_card_contour(img)
        assert corners is not None, "no contour found at all"
        assert quad_matches(corners, x0, y0, w, h, tol=20)

    def test_the_whole_frame_is_never_the_answer(self):
        """What it used to return: a quad the size of the image, which then
        failed the aspect gate and read as 'no card in this scan'."""
        img, _ = self.scene(lid="left_edge")
        corners = detect.find_card_contour(img)
        area = cv2.contourArea(corners.astype(np.float32))
        assert area < 0.5 * img.shape[0] * img.shape[1]

    def test_a_card_pushed_into_a_corner_still_works(self):
        """The regression this guards: sampling four corner colours instead
        of one mean means a card occupying a corner puts its own colour into
        the background list. Both models are searched and scored, so the
        card-likeness penalty decides rather than a rule about corners."""
        img, (x0, y0, w, h) = self.scene(card_at=(10, 10))
        corners = detect.find_card_contour(img)
        assert corners is not None
        assert quad_matches(corners, x0, y0, w, h, tol=20)

    def test_matching_corners_collapse_to_one_background(self):
        """An evenly-backed scan must not pay for the second search: at a
        full 4200x5600 it costs about a second per call, and detection runs
        32 times in a photometric job."""
        img, _ = self.scene()
        assert len(detect._estimate_background_colors(img)) == 1

    def test_a_two_tone_background_is_kept_as_two(self):
        img, _ = self.scene(lid="left_edge")
        colors = detect._estimate_background_colors(img)
        assert len(colors) == 2
        assert min(c.mean() for c in colors) < 60      # the black sheet
        assert max(c.mean() for c in colors) > 200     # the bare lid

    def test_lighting_falloff_across_one_sheet_stays_one_background(self):
        """A CIS scanner's light falls off across the bed, so one sheet reads
        differently corner to corner. That must not be mistaken for two
        backgrounds — the tolerance has to swallow it."""
        img, (x0, y0, w, h) = self.scene()
        gradient = np.linspace(0, 35, img.shape[1], dtype=np.float32)[None, :, None]
        img = np.clip(img.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
        assert len(detect._estimate_background_colors(img)) == 1
        assert quad_matches(detect.find_card_contour(img), x0, y0, w, h, tol=20)
