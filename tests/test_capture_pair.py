"""The offline same-side check.

The failure this guards against is silent and complete: upload the front
twice and the pipeline produces a confident report in which every "back"
number was measured on the front and scored against PSA's looser back
tolerance table. Nothing else notices — both images detect, warp and grade
perfectly well, because they are both perfectly good images of a card.

The vision identify stage already cross-checks this, but it needs an API key
and is skipped without one, which is the normal case here.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from grade import check_capture_pair

W, H = 1500, 2100


def _card(seed: int) -> np.ndarray:
    """A card-like warp: a border, a bright art panel, and blocky artwork.

    Blocky rather than per-pixel noise on purpose — the check downsamples to
    a thumbnail, and pixel noise would average away to a flat gray that
    correlates with every other flat gray.
    """
    rng = np.random.default_rng(seed)
    card = np.full((H, W, 3), 40, np.uint8)
    card[80:H - 80, 80:W - 80] = 230
    blocks = rng.integers(0, 255, (14, 10, 3), dtype=np.uint8)
    card[200:1600, 150:W - 150] = cv2.resize(
        blocks, (W - 300, 1400), interpolation=cv2.INTER_NEAREST
    )
    return card


def _write(tmp_path, name: str, image: np.ndarray):
    path = tmp_path / name
    cv2.imwrite(str(path), image)
    return path


def test_two_sides_of_one_card_pass_quietly(tmp_path):
    front, back = _card(1), _card(2)
    result = check_capture_pair(
        _write(tmp_path, "f.png", front), _write(tmp_path, "b.png", back), front, back
    )
    assert result["same_side_suspected"] is False
    assert result["identical_files"] is False
    assert result["note"] is None


def test_the_same_file_twice_is_caught(tmp_path):
    front = _card(1)
    path = _write(tmp_path, "f.png", front)
    result = check_capture_pair(path, path, front, front)
    assert result["identical_files"] is True
    assert result["same_side_suspected"] is True
    assert "same file" in result["note"]


def test_two_different_scans_of_the_same_side_are_caught(tmp_path):
    """The harder case, and the one the hash can't see: the card was lifted
    off the glass and scanned again, so the bytes differ — but it's still the
    same side, and the back sub-grades would be measured on the front."""
    front = _card(1)
    rescan = np.clip(front.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    rescan = np.roll(rescan, 3, axis=0)

    result = check_capture_pair(
        _write(tmp_path, "f.png", front), _write(tmp_path, "b.png", rescan), front, rescan
    )
    assert result["identical_files"] is False
    assert result["same_side_suspected"] is True
    assert "same side" in result["note"]


def test_the_two_cases_are_far_apart(tmp_path):
    """How much room the threshold actually has.

    These two cards share a border, an art panel and a layout and differ only
    in artwork, which is the hardest case this check can be given without
    real front/back pairs — at thumbnail size the shared layout is most of
    what survives. A real front and back share less. So the number asserted
    here is a floor on the separation, not the typical one."""
    front, back = _card(1), _card(2)
    same_side = check_capture_pair(
        _write(tmp_path, "a.png", front), _write(tmp_path, "b.png", front), front, front
    )["similarity"]
    two_sides = check_capture_pair(
        _write(tmp_path, "c.png", front), _write(tmp_path, "d.png", back), front, back
    )["similarity"]
    assert same_side > 0.99
    assert two_sides < 0.8
    # ...and the threshold is above the worst case, not inside it.
    from grade import SAME_SIDE_CORRELATION

    assert two_sides < SAME_SIDE_CORRELATION < same_side


def test_it_warns_rather_than_refusing(tmp_path):
    """Re-grading one side against both tolerance tables is a legitimate
    thing to do on purpose — the user asked for it explicitly during
    testing. So this reports, and never raises."""
    front = _card(1)
    path = _write(tmp_path, "f.png", front)
    result = check_capture_pair(path, path, front, front)
    assert isinstance(result, dict)
    assert result["note"]
