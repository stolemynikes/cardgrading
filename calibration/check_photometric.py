#!/usr/bin/env python3
"""Does this scanner actually light the card from an angle?

Photometric stereo only works if the lamp reaches the card off-axis. CCD
flatbeds do this well — the lamp sits below and to one side of the sensor
line. CIS flatbeds (Canon LiDe, most cheap USB-powered units) put an LED
strip almost flush against the glass, and the flatter that light is, the
less a scratch or dent changes the pixel it falls on. It is not worth
guessing which side of that line a given scanner falls on, so this measures
it, in about two minutes and two scans.

    scan the card once, rotate it 180 degrees on the glass, scan again

    python calibration/check_photometric.py scan_0.png scan_180.png

The 180-degree pair is the sharpest possible test. Rotating by half a turn
reverses the light direction relative to the card while leaving everything
else — optics, focus, the card itself — identical. Any pixel whose
brightness *flips* between the two scans is being shaded by geometry.
Coaxial light produces no flip, only sensor noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import cardvision, detect  # noqa: E402

# Modulation is the share of a pixel's brightness that flips with the light
# direction. Measured at a high percentile, since a card is mostly flat and
# the interesting pixels are the few that aren't. Thresholds are calibrated
# against what the solve needs, not against any particular scanner.
GOOD_MODULATION_PCT = 2.0
MARGINAL_MODULATION_PCT = 0.8


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a scanner's directional-lighting response.")
    parser.add_argument("scan_a", type=Path, help="Scan of the card at its starting orientation")
    parser.add_argument("scan_b", type=Path, help="Scan of the same side, card rotated 180 degrees on the glass")
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(__file__).resolve().parent / "thresholds.json",
    )
    parser.add_argument("--output", type=Path, help="Write a visualization of the flipped component here")
    parser.add_argument(
        "--pre-warped",
        action="store_true",
        help="The inputs are already perspective-corrected card images — the warps a saved report "
        "stores — rather than raw scans. Skips detection; the second is still turned back 180 degrees.",
    )
    return parser.parse_args(argv)


def _warp(path: Path, rotate_180: bool, thresholds: dict, pre_warped: bool = False) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"could not read {path}")

    if pre_warped:
        # Already perspective-corrected — the warps a finished report stores.
        # Detection ordered their corners geometrically, so a card that was
        # turned 180 degrees on the glass came out of the warp upside down;
        # turning it back here puts the two in the same frame, which is the
        # only thing the raw path's rotation was for.
        return cv2.rotate(image, cv2.ROTATE_180) if rotate_180 else image

    if rotate_180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    result = detect.detect_and_normalize(image, thresholds)
    if result.warped is None or result.hard_failures:
        reasons = ", ".join(g.name for g in result.hard_failures) or "no card found"
        raise SystemExit(f"couldn't detect the card in {path.name} ({reasons})")
    return result.warped


def measure(
    scan_a: Path, scan_b: Path, thresholds: dict, pre_warped: bool = False
) -> tuple[float, np.ndarray]:
    """Returns (modulation percentage, signed difference image)."""
    warp_a = _warp(scan_a, False, thresholds, pre_warped)
    warp_b = _warp(scan_b, True, thresholds, pre_warped)
    if warp_a.shape != warp_b.shape:
        raise SystemExit(
            f"these two warps are different sizes ({warp_a.shape[:2]} vs {warp_b.shape[:2]}) — "
            "they have to come from the same stage of the same pipeline to be compared"
        )
    warp_b = cardvision.register_to_reference(warp_a, warp_b)

    gray_a = cv2.cvtColor(warp_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(warp_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Trim the physical boundary: the warp seam there differs between the two
    # scans for reasons that have nothing to do with lighting, and it would
    # dominate a percentile taken over the whole frame.
    margin = 20
    gray_a = gray_a[margin:-margin, margin:-margin]
    gray_b = gray_b[margin:-margin, margin:-margin]

    difference = gray_a - gray_b
    # Real relief is spatially coherent — a scratch is a line, a dent is a
    # patch — while sensor noise is not. A light blur removes the noise and
    # leaves the structure, which matters because the percentile below is
    # deliberately reading the extreme tail.
    smoothed = cv2.GaussianBlur(difference, (0, 0), 1.0)

    total = gray_a + gray_b
    # Only where there is light to modulate — a black region has no headroom
    # to show shading either way.
    lit = total > 40
    modulation = np.abs(smoothed[lit]) / total[lit]

    # 99.9th, not 99th: the defects being tested for cover well under 1% of a
    # card's area, so a 99th percentile reads the flat cardstock between them
    # and calls a perfectly good scanner dead.
    return float(np.percentile(modulation, 99.9) * 100.0), difference


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    thresholds = detect.load_thresholds(args.thresholds)
    modulation_pct, difference = measure(args.scan_a, args.scan_b, thresholds, args.pre_warped)

    print(f"directional modulation: {modulation_pct:.2f}% of brightness (99.9th percentile)")
    if modulation_pct >= GOOD_MODULATION_PCT:
        verdict = "usable — this scanner lights the card off-axis, photometric stereo will resolve relief"
    elif modulation_pct >= MARGINAL_MODULATION_PCT:
        verdict = (
            "marginal — there is some directional signal, so photometric stereo will find deep dents "
            "and creases but probably not fine scratches"
        )
    else:
        verdict = (
            "too flat — the light is effectively coaxial, so rotating the card changes nothing. "
            "This scanner can't support a surface grade; keep it for centering, corners/edges "
            "and dimensions, where it is still the better capture"
        )
    print(f"verdict: {verdict}")

    if args.output:
        # Mid-gray means "did not change when the light reversed". Anything
        # that isn't mid-gray is being shaded by shape.
        visualization = np.clip(128 + difference * 4, 0, 255).astype(np.uint8)
        cv2.imwrite(str(args.output), visualization)
        print(f"flipped-component visualization written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
