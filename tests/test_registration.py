"""Aligning the frames of a photometric set to each other.

Photometric stereo is extremely sensitive to this: a two-pixel shift turns
every high-contrast print edge into a fake ridge in the normal map. On a real
set it left the card's own edge standing in five separate places within
fourteen pixels of where it belonged, and inflated a clean card's defect area
to 22% — because the frames differ in *size* as well as position, and a
rotation-and-translation fit cannot express that.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline import cardvision


# The canonical warp's own size. The plausibility bound is a *fraction* of
# the frame, so a pixel offset only means something against a real one: 25px
# is 1.2% of a 2100px warp and well within detection error, but 5% of a
# thumbnail and rightly refused.
CANONICAL = (2100, 1500)


def _card(size=CANONICAL) -> np.ndarray:
    """Something with plenty of print edges to register on."""
    height, width = size
    card = np.full((height, width, 3), 200, np.uint8)
    rng = np.random.default_rng(0)
    for y in range(120, height - 120, 140):
        card[y : y + 52, 100 : width - 100] = rng.integers(30, 120, 3, dtype=np.uint8).tolist()
    card[height // 2 : height // 2 + 280, 160 : width - 160] = 60
    return card


def _transform(image, dx=0.0, dy=0.0, scale=1.0, degrees=0.0):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def _misalignment(a, b) -> float:
    """Mean absolute difference over the interior, in grey levels."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)[40:-40, 40:-40]
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)[40:-40, 40:-40]
    return float(np.abs(ga - gb).mean())


class TestRecovery:
    def test_a_translation_is_corrected(self):
        reference = _card()
        moved = _transform(reference, dx=3.0, dy=-2.0)
        fixed = cardvision.register_to_reference(reference, moved)
        assert _misalignment(fixed, reference) < _misalignment(moved, reference) * 0.5

    def test_a_scale_difference_is_corrected(self):
        """The one the old rigid fit could not express. Each frame is warped
        from its own detection of the card, and those quads differ in size."""
        reference = _card()
        grown = _transform(reference, scale=1.008)
        fixed = cardvision.register_to_reference(reference, grown)
        assert _misalignment(fixed, reference) < _misalignment(grown, reference) * 0.6

    def test_a_small_rotation_is_corrected(self):
        reference = _card()
        turned = _transform(reference, degrees=0.3)
        fixed = cardvision.register_to_reference(reference, turned)
        assert _misalignment(fixed, reference) < _misalignment(turned, reference) * 0.6

    def test_an_already_aligned_frame_is_left_alone(self):
        """Six degrees of freedom will happily spend themselves on noise; the
        correction has to be near-identity when there is nothing to correct."""
        reference = _card()
        fixed = cardvision.register_to_reference(reference, reference.copy())
        assert _misalignment(fixed, reference) < 1.0


class TestPlausibilityGuard:
    """Identity is the prior. A fit outside these bounds is the fit drifting,
    and applying it makes the alignment worse than leaving it alone."""

    SHAPE = CANONICAL

    def test_identity_is_plausible(self):
        assert cardvision._plausible_registration(np.eye(2, 3, dtype=np.float32), self.SHAPE)

    def test_a_large_scale_change_is_refused(self):
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0, 0] = warp[1, 1] = 1.2
        assert not cardvision._plausible_registration(warp, self.SHAPE)

    def test_a_large_shift_is_refused(self):
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0, 2] = 200.0
        assert not cardvision._plausible_registration(warp, self.SHAPE)

    def test_shear_is_refused(self):
        """Perpendicular columns mean the card kept its shape. Shear means the
        fit started deforming it to chase brightness."""
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0, 1] = 0.2
        assert not cardvision._plausible_registration(warp, self.SHAPE)

    def test_a_realistic_correction_is_accepted(self):
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0, 0] = warp[1, 1] = 1.005
        warp[0, 2], warp[1, 2] = 3.0, -2.0
        assert cardvision._plausible_registration(warp, self.SHAPE)


class TestFeatures:
    def test_shading_is_removed_before_matching(self):
        """The frames differ in shading by design — that difference is the
        whole signal — so matching raw brightness asks the fit to align the
        one thing that is not supposed to align."""
        card = cv2.cvtColor(_card(), cv2.COLOR_BGR2GRAY)
        lit = np.clip(card.astype(np.float32) + np.linspace(-40, 40, card.shape[1]), 0, 255).astype(np.uint8)
        a = cardvision._registration_features(card)
        b = cardvision._registration_features(lit)
        assert np.abs(a - b).mean() < np.abs(card.astype(np.float32) - lit.astype(np.float32)).mean() * 0.2


class TestStackedEdges:
    def test_frames_of_different_sizes_stop_stacking(self):
        """The symptom that started this: the card edge appearing several
        times over, a few pixels apart, because the frames were different
        sizes and only translation was being corrected."""
        reference = _card()
        frames = [reference] + [_transform(reference, scale=s, dx=d) for s, d in ((1.004, 2.0), (0.996, -3.0))]
        aligned = [reference] + [cardvision.register_to_reference(reference, f) for f in frames[1:]]

        def edge_ridges(stack):
            mean = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in stack], axis=0)
            profile = np.abs(np.diff(mean.mean(axis=0)))[60:120]
            floor = profile.max() * 0.4
            return sum(
                1
                for x in range(1, len(profile) - 1)
                if profile[x] > profile[x - 1] and profile[x] >= profile[x + 1] and profile[x] > floor
            )

        assert edge_ridges(aligned) <= edge_ridges(frames)


class TestCaptureRange:
    """How far out of place a frame can start and still be brought back.

    ECC is a local optimiser, and the high-pass it matches on is deliberately
    spiky, which left it a capture range of about ten pixels — it corrected a
    10px offset and silently did nothing to a 15px one. The offsets a real
    rotation set actually produces were around 25px, so every frame in every
    set was going through unregistered, and the card's own text came out of
    the solve embossed twice.
    """

    def _shifted(self, dy: float):
        card = _card()
        return card, _transform(card, dy=dy)

    @pytest.mark.parametrize("dy", [2, 10, 20, 30, 40])
    def test_offsets_up_to_forty_pixels_are_corrected(self, dy):
        reference, moved = self._shifted(dy)
        fixed = cardvision.register_to_reference(reference, moved)
        assert _misalignment(fixed, reference) < _misalignment(moved, reference) * 0.3

    def test_an_implausible_offset_is_still_refused(self):
        """Past a point it isn't detection error any more, and forcing a fit
        would be worse than leaving the frame where the canonical warp put it."""
        reference, moved = self._shifted(600)
        fixed = cardvision.register_to_reference(reference, moved)
        assert np.array_equal(fixed, moved), "an implausible fit must be declined, not applied"

    def test_the_seed_estimate_has_no_capture_limit(self):
        """Phase correlation reads the whole image at once, which is what
        gives ECC a starting point close enough to refine from."""
        reference, moved = self._shifted(35)
        ref_f = cardvision._registration_features(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
        img_f = cardvision._registration_features(cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY))
        seed = cardvision._estimate_seed(ref_f, img_f)
        assert abs(abs(float(seed[1, 2])) - 35) < 5, f"estimated {seed[1, 2]}, actual 35"

    def test_the_seed_finds_scale_too(self):
        """Translation alone is not a close enough start: a 4% scale error is
        eighty pixels of displacement at the card's edges, and ECC never
        walked to it from identity."""
        reference = _card()
        stretched = _transform(reference, scale=1.04)
        ref_f = cardvision._registration_features(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
        img_f = cardvision._registration_features(cv2.cvtColor(stretched, cv2.COLOR_BGR2GRAY))
        seed = cardvision._estimate_seed(ref_f, img_f)
        found = float(np.hypot(seed[0, 0], seed[1, 0]))
        assert abs(found - 1.04) < 0.015, f"seeded {found:.4f}, actual 1.04"

    def test_lighting_does_not_move_the_estimate(self):
        """The frames differ in shading by design; the fit must not chase it."""
        reference = _card()
        moved = _transform(reference, dy=20)
        height, width = moved.shape[:2]
        ramp = np.linspace(-40, 40, width, dtype=np.float32)[None, :, None]
        shaded = np.clip(moved.astype(np.float32) + ramp, 0, 255).astype(np.uint8)
        fixed = cardvision.register_to_reference(reference, shaded)
        before = _misalignment(_transform(reference, dy=20), reference)
        # Compare geometry only — the ramp itself is not registration's to fix.
        def geometry(a, b):
            fa = cardvision._registration_features(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))[40:-40, 40:-40]
            fb = cardvision._registration_features(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))[40:-40, 40:-40]
            return float(np.abs(fa - fb).mean())
        assert geometry(fixed, reference) < geometry(shaded, reference) * 0.3
        assert before > 0


class TestSetAlignment:
    def test_a_whole_rotation_set_comes_into_agreement(self):
        """The end-to-end property: four frames, each from its own detection
        of the card and each lit differently, agreeing on where the card is."""
        reference = _card()
        offsets = [(0.0, 0.0, 1.0), (8.0, -22.0, 1.008), (-14.0, 25.0, 0.994), (5.0, 18.0, 1.012)]
        frames = [_transform(reference, dx=dx, dy=dy, scale=s) for dx, dy, s in offsets]

        def disagreement(stack):
            features = [
                cardvision._registration_features(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in stack
            ]
            return float(np.stack(features).std(axis=0)[60:-60, 60:-60].mean())

        aligned = [cardvision.register_to_reference(reference, f) for f in frames]
        assert disagreement(aligned) < disagreement(frames) * 0.25


class TestBoundsMatchRealSets:
    """The bounds were guesswork once, and the guess was wrong.

    Measured across one real four-scan set, the detector found the same card
    at widths of 2875, 2896, 2980 and 2993 pixels — a 4.1% spread. The fits
    ECC proposed to correct that were right, and a 3% bound refused two of
    them, which left half the set unregistered and the card's own print
    embossed twice in the render. These pin the bounds to what a real set
    actually needs.
    """

    SHAPE = CANONICAL

    def _fit(self, scale_x, scale_y, dx, dy, shear=0.0):
        return np.array([[scale_x, shear, dx], [0.0, scale_y, dy]], dtype=np.float32)

    def test_the_two_fits_a_real_set_needed_are_accepted(self):
        assert cardvision._plausible_registration(self._fit(1.0401, 1.0394, -55.1, -38.2), self.SHAPE)
        assert cardvision._plausible_registration(self._fit(1.0330, 1.0222, -39.0, -17.8), self.SHAPE)

    def test_a_four_percent_scale_correction_is_allowed(self):
        """The spread a real set produced. Refusing it is refusing the set."""
        assert cardvision._plausible_registration(self._fit(1.041, 1.041, 0, 0), self.SHAPE)

    def test_a_runaway_fit_is_still_refused(self):
        assert not cardvision._plausible_registration(self._fit(1.3, 1.3, 400, 400), self.SHAPE)

    def test_a_fit_past_the_widened_bound_is_still_refused(self):
        assert not cardvision._plausible_registration(self._fit(1.12, 1.12, 0, 0), self.SHAPE)

    def test_shear_stays_tight(self):
        """These are rectified images of a flat card. A fit that wants to skew
        one is chasing brightness, not geometry — widening the scale bound is
        no reason to widen this one."""
        assert not cardvision._plausible_registration(self._fit(1.0, 1.0, 0, 0, shear=0.1), self.SHAPE)

    def test_a_frame_that_needs_four_percent_is_actually_corrected(self):
        """End to end, not just the bound: the warp the detector's own spread
        produces has to come back into line."""
        reference = _card()
        stretched = _transform(reference, scale=1.041, dx=-20, dy=-15)
        fixed = cardvision.register_to_reference(reference, stretched)
        assert _misalignment(fixed, reference) < _misalignment(stretched, reference) * 0.3
