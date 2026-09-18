#!/usr/bin/env python3
"""Does this scanner measure the same object differently along its two axes?

A card measured on this scanner came out 2.9% larger placed landscape than
placed portrait — 1.8mm on a 63mm card, against a grading tolerance of 0.75mm.
Two explanations fit that equally well and want opposite fixes:

    the scanner   its two axes are scaled differently, correctable in software
    the card      trading cards are flexible and bow off the glass, correctable
                  only by holding them flatter

A trading card cannot tell them apart, because it can do both. An ISO/IEC 7810
ID-1 card can: bank card, driving licence, national ID, transit card, loyalty
card — they are all rigid PVC and all exactly 85.60 x 53.98mm, to hundredths
of a millimetre. Rigid removes the bowing explanation; a known size turns the
error into a number.

    .venv/bin/python calibration/check_scanner_scale.py portrait.tif landscape.tif --dpi 1200

Scan the same card twice at the same settings, once with its long edge along
the scan direction and once across it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import detect  # noqa: E402
from pipeline.dimensions import MM_PER_INCH  # noqa: E402

# ISO/IEC 7810 ID-1. Every bank card, driving licence and transit card in the
# world is made to this, which is what makes one a measuring stick.
ID1_LONG_MM = 85.60
ID1_SHORT_MM = 53.98

# Below this the two orientations agree and the scanner is not the problem.
# A hundredth of a millimetre on an 85mm card is 0.012%, so anything under a
# tenth of a percent is the measurement's own noise rather than a real scale
# error.
AGREEMENT_PCT = 0.1


def measure(path: Path, dpi: float) -> tuple[float, float]:
    """The long and short side of the card in this scan, in millimetres."""
    image = detect.load_image(path) if hasattr(detect, "load_image") else cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"could not read {path}")
    corners = detect.find_card_contour(image)
    if corners is None:
        raise SystemExit(f"no card found in {path} — is it flat on the glass with background all round?")

    quad = corners.astype(np.float64)
    top = float(np.linalg.norm(quad[1] - quad[0]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    a = (top + bottom) / 2.0 * MM_PER_INCH / dpi
    b = (left + right) / 2.0 * MM_PER_INCH / dpi
    return max(a, b), min(a, b)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("portrait", type=Path, help="the card with its long edge along the scan direction")
    parser.add_argument("landscape", type=Path, help="the same card turned 90 degrees")
    parser.add_argument("--dpi", type=float, required=True)
    args = parser.parse_args(argv)

    p_long, p_short = measure(args.portrait, args.dpi)
    l_long, l_short = measure(args.landscape, args.dpi)

    print(f"true size (ISO/IEC 7810 ID-1): {ID1_LONG_MM} x {ID1_SHORT_MM} mm\n")
    print(f"{'placement':12s} {'long side':>12s} {'short side':>12s} {'long err':>10s} {'short err':>10s}")
    for label, (lo, sh) in (("portrait", (p_long, p_short)), ("landscape", (l_long, l_short))):
        print(f"{label:12s} {lo:9.2f} mm {sh:9.2f} mm "
              f"{100*(lo-ID1_LONG_MM)/ID1_LONG_MM:9.2f}% {100*(sh-ID1_SHORT_MM)/ID1_SHORT_MM:9.2f}%")

    # The card's long edge lies along a different scanner axis in each scan, so
    # measuring it both ways isolates the axes from the card.
    swing = 100.0 * (l_long - p_long) / ID1_LONG_MM
    print(f"\nthe same physical edge measured {swing:+.2f}% different between the two placements")

    if abs(swing) < AGREEMENT_PCT:
        print("\n-> THE SCANNER IS FINE. It measures the same object the same way on both axes.")
        print("   The size difference seen on trading cards is the cards bowing off the glass.")
        print("   Fix is physical: weigh them flat, or scan through the lid closed.")
    else:
        scale = ID1_LONG_MM / l_long
        print(f"\n-> THE SCANNER'S AXES DISAGREE by {abs(swing):.2f}%.")
        print(f"   Correct the axis that reads long by x{scale:.5f} "
              f"(i.e. an effective {args.dpi/scale:.0f} dpi on that axis, not {args.dpi:.0f}).")
        print("   This is correctable in software, and recovers the card's outer ring for measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
