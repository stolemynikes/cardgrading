"""Card Vision tests.

The load-bearing claim of the photometric path is that it shows physical
relief and *not* print, so that's what these check: a synthetic card with a
zero-height high-contrast printed band and a real scratch, rendered under
several light directions.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline import cardvision

CFG = {
    "light_elevation_deg": 70.0,
    "normal_smoothing_sigma": 1.0,
    "relief_gain": 1.0,
    "albedo_sigma": 12.0,
    "chroma_suppression": 24.0,
    "single_image_gain": 1.0,
    "flat_band": 18,
}

H, W = 300, 220
BAND_ROWS = slice(60, 80)
SCRATCH_ROW = 200
SCRATCH_COLS = slice(40, 180)


def _synthetic_scan_set(elevation_deg: float = 70.0) -> tuple[list[np.ndarray], list[float]]:
    """Four differently-lit captures of one card, already de-rotated.

    Albedo carries a bright printed band with no height; the height field
    carries a horizontal scratch with no color.
    """
    albedo = np.full((H, W, 3), 120.0, dtype=np.float32)
    albedo[BAND_ROWS, :] = 245.0

    height = np.zeros((H, W), dtype=np.float32)
    height[SCRATCH_ROW, SCRATCH_COLS] = -1.0

    gy, gx = np.gradient(height)
    normals = np.dstack([-gx, -gy, np.ones_like(height)])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    azimuths = [0.0, 90.0, 180.0, 270.0]
    lights = cardvision.light_vectors(azimuths, elevation_deg)
    images = []
    for light in lights:
        shading = np.clip(normals @ light.astype(np.float32), 0, None)[:, :, None]
        images.append(np.clip(albedo * shading, 0, 255).astype(np.uint8))
    return images, azimuths


def _contrast_at(relief: np.ndarray, rows) -> float:
    """How far this band of the render sits from flat mid-gray."""
    return float(np.abs(relief[rows].astype(np.float32) - 128).max())


def test_photometric_shows_relief_and_hides_print():
    images, azimuths = _synthetic_scan_set()
    result = cardvision.photometric_card_vision(images, azimuths, CFG)

    scratch_contrast = _contrast_at(result.relief, slice(SCRATCH_ROW - 2, SCRATCH_ROW + 3))
    print_contrast = _contrast_at(result.relief, BAND_ROWS)

    assert result.method == "photometric_stereo"
    assert result.light_count == 4
    # The printed band is 2x the brightness of its surroundings in every input
    # image and must still leave essentially no trace in the render.
    assert scratch_contrast > 10 * max(print_contrast, 1.0)


def test_single_image_shows_relief_but_leaks_print():
    """The documented weakness of the fallback, pinned so it can't be quietly
    claimed to be as good as the photometric path."""
    images, _ = _synthetic_scan_set()
    result = cardvision.single_image_card_vision(images[0], CFG)

    assert result.method == "single_image"
    assert result.light_count == 1
    assert _contrast_at(result.relief, slice(SCRATCH_ROW - 2, SCRATCH_ROW + 3)) > 20
    assert _contrast_at(result.relief, BAND_ROWS) > 20


def test_card_vision_falls_back_below_three_lights():
    images, azimuths = _synthetic_scan_set()
    assert cardvision.card_vision(images[:2], azimuths[:2], CFG).method == "single_image"
    assert cardvision.card_vision(images, None, CFG).method == "single_image"
    assert cardvision.card_vision(images, azimuths, CFG).method == "photometric_stereo"


def test_photometric_rejects_mismatched_azimuths():
    images, azimuths = _synthetic_scan_set()
    with pytest.raises(ValueError):
        cardvision.photometric_card_vision(images, azimuths[:3], CFG)


def test_scanner_azimuths_track_the_cards_rotation():
    """Rotating the card clockwise moves the (fixed) lamp counter-clockwise in
    the card's own frame, so the azimuths must advance, not retreat."""
    assert cardvision.scanner_azimuths(4, 90.0, clockwise=True) == [90.0, 180.0, 270.0, 0.0]
    assert cardvision.scanner_azimuths(4, 90.0, clockwise=False) == [90.0, 0.0, 270.0, 180.0]


def test_flat_card_does_not_get_amplified_into_noise():
    """A genuinely featureless surface must render flat. Without the floor on
    the autoscale divisor, normalizing by a near-zero percentile would stretch
    sensor noise across the full range and invent defects."""
    rng = np.random.default_rng(0)
    flat = np.full((H, W, 3), 120.0, dtype=np.float32)
    images = [np.clip(flat + rng.normal(0, 0.3, flat.shape), 0, 255).astype(np.uint8) for _ in range(4)]
    result = cardvision.photometric_card_vision(images, [0.0, 90.0, 180.0, 270.0], CFG)
    assert result.roughness_pct < 1.0


def test_register_to_reference_recovers_a_shift():
    images, _ = _synthetic_scan_set()
    reference = images[0]
    shifted = np.roll(reference, 3, axis=1)
    registered = cardvision.register_to_reference(reference, shifted)
    # Compare away from the wrapped column the roll introduced.
    before = np.abs(shifted[:, 20:-20].astype(int) - reference[:, 20:-20].astype(int)).mean()
    after = np.abs(registered[:, 20:-20].astype(int) - reference[:, 20:-20].astype(int)).mean()
    assert after < before / 2


class TestNoiseFloorFromFlatRegions:
    """The floor decides what survives to be rendered, and it was being read
    off the print rather than off the card. Measured on a real scan the
    whole-card median put it at 0.222 while the quietest tiles sat at 0.109 —
    and a scratch six grey levels deep is 0.024 of deviation."""

    @staticmethod
    def _field(noise=0.01, print_amplitude=0.25, seed=0):
        """A field that is quiet over most of its area and loud where the
        'print' is — which is what a card's high-pass residual looks like."""
        rng = np.random.default_rng(seed)
        field = rng.normal(0, noise, (640, 640)).astype(np.float32)
        field[:, 320:] += rng.normal(0, print_amplitude, (640, 320))
        return field

    def test_the_floor_is_read_from_the_quiet_side(self):
        field = self._field()
        sigma = cardvision._noise_sigma(field)
        assert sigma < 0.05, f"the floor followed the print instead of the card ({sigma:.4f})"

    def test_print_does_not_drag_the_floor_up(self):
        quiet = cardvision._noise_sigma(self._field(print_amplitude=0.05))
        loud = cardvision._noise_sigma(self._field(print_amplitude=0.6))
        assert abs(quiet - loud) < 0.01, "louder print moved the floor, which is the bug"

    def test_a_shallow_defect_survives_the_floor(self):
        """Shallow relative to the print beside it — 20 grey levels against a
        0.25 print amplitude. Under the whole-card floor this was erased."""
        field = self._field()
        field[300:303, 40:400] = -20 / 255.0
        rendered = cardvision._autoscale(field, 1.0)
        groove = np.abs(rendered[300:303, 40:400].astype(float) - 128).mean()
        around = np.abs(rendered[260:280, 40:400].astype(float) - 128).mean()
        assert groove > around + 2, f"the groove rendered no stronger than the flat card ({groove:.1f} vs {around:.1f})"

    def test_the_old_whole_card_floor_would_have_erased_it(self):
        """Pins the regression this fixes: the median over everything sits
        above the defect, so a 3-sigma threshold on it removes the defect."""
        field = self._field()
        field[300:303, 40:400] = -20 / 255.0
        whole_card = cardvision.MAD_TO_SIGMA * float(np.median(np.abs(field - np.median(field))))
        assert 3 * whole_card > 20 / 255.0, "the old estimate no longer buries this defect"
        assert 3 * cardvision._noise_sigma(field) < 20 / 255.0, "the new one leaves it standing"

    def test_a_field_smaller_than_one_tile_still_measures(self):
        assert cardvision._noise_sigma(np.full((8, 8), 0.01, np.float32)) >= 0.0

    def test_an_empty_field_is_zero_rather_than_an_error(self):
        assert cardvision._noise_sigma(np.zeros((0, 0), np.float32)) == 0.0


class TestSoftKnee:
    """A hard clip drove every strong edge to pure black or white, so nothing
    distinguished a deep gouge from a printed rule."""

    def test_strong_features_no_longer_clip(self):
        rng = np.random.default_rng(1)
        field = rng.normal(0, 0.01, (256, 256)).astype(np.float32)
        field[100:110, 50:200] = 3.0  # far past anything the scale expects
        rendered = cardvision._autoscale(field, 1.0)
        assert rendered.max() < 255, "a strong feature still saturates to pure white"
        assert rendered.min() > 0

    def test_ordinary_relief_is_still_mapped_linearly(self):
        """tanh(x) is x for small x, so the change costs nothing in the range
        the render actually lives in."""
        rng = np.random.default_rng(2)
        field = rng.normal(0, 0.01, (256, 256)).astype(np.float32)
        field[100:110, 50:200] = 0.08
        field[150:160, 50:200] = 0.16
        rendered = cardvision._autoscale(field, 1.0).astype(float)
        weak = rendered[100:110, 50:200].mean() - 128
        strong = rendered[150:160, 50:200].mean() - 128
        assert strong > weak * 1.5, "twice the relief still renders noticeably stronger"

    def test_a_flat_field_stays_mid_grey(self):
        flat = np.zeros((128, 128), np.float32)
        assert set(np.unique(cardvision._autoscale(flat, 1.0))) == {128}


class TestWarpSeamIsBlanked:
    """`perspective_correct` maps the detected quad onto the canonical
    rectangle, and corner detection is not sub-pixel perfect — so the
    outermost pixels are an interpolated blend of card edge and background
    rather than card. In a relief render that seam became a bright line right
    around the card: measured on a real render, deviation 76 at the second
    column falling to 8 by the eighth."""

    CFG = {"physical_edge_margin_px": 8}

    def test_the_seam_band_is_flattened(self):
        noisy = np.full((200, 150), 30, np.uint8)
        blanked = cardvision.blank_warp_seam(noisy, self.CFG)
        assert set(np.unique(blanked[:8, :])) == {128}
        assert set(np.unique(blanked[-8:, :])) == {128}
        assert set(np.unique(blanked[:, :8])) == {128}
        assert set(np.unique(blanked[:, -8:])) == {128}

    def test_the_card_itself_is_untouched(self):
        noisy = np.full((200, 150), 30, np.uint8)
        blanked = cardvision.blank_warp_seam(noisy, self.CFG)
        assert set(np.unique(blanked[8:-8, 8:-8])) == {30}

    def test_mid_grey_rather_than_cropped(self):
        """Cropping would break alignment with everything else in the
        canonical frame; mid-grey is this render's own word for 'flat', which
        is the honest thing to say about a band with no measurement in it."""
        blanked = cardvision.blank_warp_seam(np.zeros((200, 150), np.uint8), self.CFG)
        assert blanked.shape == (200, 150)

    def test_no_margin_configured_changes_nothing(self):
        noisy = np.full((200, 150), 30, np.uint8)
        assert np.array_equal(cardvision.blank_warp_seam(noisy, {}), noisy)

    def test_the_seam_no_longer_counts_as_relief(self):
        """It was inflating roughness_pct and could register as a defect."""
        relief = np.full((400, 300), 128, np.uint8)
        relief[:6, :] = 255
        relief[-6:, :] = 0
        before = cardvision._roughness_pct(relief, {"flat_band": 18})
        after = cardvision._roughness_pct(cardvision.blank_warp_seam(relief, self.CFG), {"flat_band": 18})
        assert before > 0 and after == 0


class TestTheDisplayedReliefIsFlattened:
    """A smooth ripple grows sevenfold from the card's centre to its corners —
    3.09 grey levels against 21.99 on a real capture — and it is not the card:
    two captures of the same card correlate -0.071 on that component.

    It is removed from the picture and kept in the measurement. Removing it
    from the measurement costs 27% of the defect area on a genuinely damaged
    card; leaving it in the picture shows prominent structure the grade does
    not reflect.
    """

    CFG = {"display_flatten_sigma_px": 100.0}

    @staticmethod
    def _relief_with_ripple() -> np.ndarray:
        h, w = 2100, 1500
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # a smooth swell, of the kind the solve produces
        ripple = 28 * np.sin(xx / 420.0) * np.cos(yy / 500.0)
        relief = np.clip(128 + ripple, 0, 255).astype(np.uint8)
        relief[900:915, 400:1100] = 235          # a scratch: long, thin, sharp
        relief[1400:1440, 300:360] = 30          # a dent: compact
        return relief

    def test_the_ripple_is_removed(self):
        before = self._relief_with_ripple()
        after = cardvision.flatten_display_relief(before, self.CFG)

        def swell(img):
            # measured clear of the stamped defects: a 40px blur would smear
            # them into the reading and hide what the filter actually did
            field = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 40) - 128
            return float(field[100:800, 100:1400].std())

        assert swell(after) < swell(before) / 4, f"{swell(before):.2f} -> {swell(after):.2f}"

    def test_a_scratch_survives_it(self):
        """A crease is about a millimetre wide and the filter is four, so
        nothing gradeable is large enough to be touched."""
        after = cardvision.flatten_display_relief(self._relief_with_ripple(), self.CFG)
        assert after[900:915, 400:1100].max() > 200

    def test_a_dent_survives_it(self):
        after = cardvision.flatten_display_relief(self._relief_with_ripple(), self.CFG)
        assert after[1400:1440, 300:360].min() < 70

    def test_a_flat_card_stays_flat(self):
        flat = np.full((2100, 1500), 128, np.uint8)
        assert set(np.unique(cardvision.flatten_display_relief(flat, self.CFG))) == {128}

    def test_it_can_be_turned_off(self):
        before = self._relief_with_ripple()
        assert np.array_equal(cardvision.flatten_display_relief(before, {"display_flatten_sigma_px": 0}), before)

    def test_the_measurement_is_not_flattened(self):
        """The line that must not move. The measured render carries the ripple
        because removing it there costs real defect sensitivity, and because
        the ripple is below the defect threshold anyway."""
        import inspect

        source = inspect.getsource(cardvision.photometric_card_vision)
        assert "flatten_display_relief(render(" in source
        assert "flatten_display_relief(swept" not in source
        assert "flatten_display_relief(_blank_border" not in source
