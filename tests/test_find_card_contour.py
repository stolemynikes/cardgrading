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
