"""The area-averaging pre-pass in perspective_correct.

`warpPerspective` has no area-averaging mode — INTER_LINEAR reads a 2x2
neighbourhood wherever it lands, so it can only properly average a 2x
reduction. Past that it point-samples, and an oversampled scan arrives at the
detectors no cleaner than a modest one: measured through this function, fine
texture plateaued at ~14 standard deviation whether the input was downscaled
2x, 4x or 8x. That surviving texture is what the whitening and surface stages
read as defects.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline import detect

SIZE = 2400
TARGET = 300


def _textured_card(seed: int = 0) -> np.ndarray:
    """Flat grey carrying paper-fibre noise and a halftone-like screen —
    the content a real scan oversamples."""
    rng = np.random.default_rng(seed)
    base = np.full((SIZE, SIZE), 128.0) + rng.normal(0, 18, (SIZE, SIZE))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    base += 14 * np.sin(2 * np.pi * xx / 16) * np.sin(2 * np.pi * yy / 16)
    return np.clip(np.dstack([base] * 3), 0, 255).astype(np.uint8)


def _full_frame_corners(dim: int = SIZE) -> np.ndarray:
    return np.array([[0, 0], [dim - 1, 0], [dim - 1, dim - 1], [0, dim - 1]], dtype=np.float32)


def _interior_std(image: np.ndarray, margin: int = 4) -> float:
    """Excludes the warp's border pixels, which BORDER_CONSTANT leaves black
    and which would otherwise dominate the statistic on a small output."""
    return float(image[margin:-margin, margin:-margin, 0].std())


def _warp_without_prefilter(image, corners, size):
    width, height = size
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(corners, dst), (width, height))


def test_heavy_downscale_is_prefiltered():
    image, corners = _textured_card(), _full_frame_corners()
    reduced, scaled = detect._prefilter_for_downscale(image, corners, (TARGET, TARGET))
    assert reduced.shape[0] < image.shape[0]
    # The card should now span roughly the output size, leaving the warp a
    # ~1:1 job rather than an 8x reduction.
    assert abs(scaled[2][0] - (TARGET - 1)) < 2


def test_upscale_is_left_alone():
    """A low-resolution capture must not be resized at all — there is nothing
    to average, and touching it would only add interpolation."""
    image, corners = _textured_card(), _full_frame_corners()
    reduced, scaled = detect._prefilter_for_downscale(image, corners, (SIZE * 2, SIZE * 2))
    assert reduced is image
    assert np.array_equal(scaled, corners)


def test_prefilter_removes_texture_the_warp_alone_leaves():
    image, corners = _textured_card(), _full_frame_corners()
    before = _interior_std(_warp_without_prefilter(image, corners, (TARGET, TARGET)))
    after = _interior_std(detect.perspective_correct(image, corners, (TARGET, TARGET)))
    # Averaging an 8x reduction should approach source/8; the un-prefiltered
    # path stalls around 14 regardless.
    assert before > 10.0
    assert after < before / 3


def test_texture_now_falls_with_downscale():
    """The property that was missing: more input resolution should mean less
    surviving noise, not the same amount."""
    image, corners = _textured_card(), _full_frame_corners()
    results = [
        _interior_std(detect.perspective_correct(image, corners, (SIZE // f, SIZE // f)))
        for f in (2, 4, 8)
    ]
    assert results[0] > results[1] > results[2]


def test_geometry_is_unchanged_by_the_prefilter():
    """Rescaling the corners must not move the card.

    A marker placed at a known fraction of the source must land at the same
    fraction of the output — a fraction of a pixel of corner error here is a
    fraction of a millimetre of centering error downstream.
    """
    image = np.full((SIZE, SIZE, 3), 40, np.uint8)
    # A bright block spanning 25%-75% of the frame in both axes.
    image[SIZE // 4:3 * SIZE // 4, SIZE // 4:3 * SIZE // 4] = 230

    warped = detect.perspective_correct(image, _full_frame_corners(), (TARGET, TARGET))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 135)

    for lo, hi, label in ((xs.min(), xs.max(), "x"), (ys.min(), ys.max(), "y")):
        assert abs(lo / TARGET - 0.25) < 0.01, f"{label} start drifted"
        assert abs(hi / TARGET - 0.75) < 0.01, f"{label} end drifted"
