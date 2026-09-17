"""Grade a fictional card and save the result as the demo report.

Runs the real pipeline end to end — detection, centering, corners/edges,
dimensions, photometric-stereo Card Vision, DINGS — against a card drawn by
`demo_card.py`. Nothing here is mocked: the numbers in the demo report are
what the pipeline actually measured, so the demo doubles as a smoke test of
the whole chain.

    python webapp/seed_demo.py

The report id is fixed, so re-running replaces the demo rather than piling up
copies of it.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from grade import grade_card  # noqa: E402
from pipeline import detect  # noqa: E402
from webapp import demo_card, store  # noqa: E402
from webapp.jobs import _image_paths  # noqa: E402

DEMO_REPORT_ID = "deadbeefdeadbeefdeadbeefdeadbeef"
REPORTS_DIR = REPO_ROOT / "reports"

# The scanner's lamp is fixed; the card turns. Four quarter-turns give the
# four light directions the solve needs. The DPI has to match the resolution
# the card is drawn at, or the dimensions stage reports a miscut that isn't
# there.
SCAN_DPI = demo_card.SCAN_DPI
LAMP_AZIMUTH = 90.0
QUARTER_TURNS = 4
LIGHT_ELEVATION = 70.0


def write_captures(staging: Path) -> dict:
    """Render every capture the pipeline can consume, and return their paths."""
    sides = {}
    for side, is_front in (("front", True), ("back", False)):
        albedo = demo_card.draw_front() if is_front else demo_card.draw_back()
        albedo = demo_card.apply_wear(albedo, is_front)
        height = demo_card.height_field(is_front)

        # The flat capture: lit from straight on, which is what an ordinary
        # scan looks like.
        flat_path = staging / f"{side}.png"
        cv2.imwrite(str(flat_path), demo_card.light(albedo, height, LAMP_AZIMUTH, 88.0))

        # The photometric set: same card, turned a further 90 degrees on the
        # glass each time, so the (fixed) lamp lands from a new direction.
        scan_paths = []
        for turn in range(QUARTER_TURNS):
            azimuth = (LAMP_AZIMUTH + 90.0 * turn) % 360.0
            lit = demo_card.light(albedo, height, azimuth, LIGHT_ELEVATION)
            scan_path = staging / f"{side}_scan{turn}.png"
            cv2.imwrite(str(scan_path), demo_card.rotate_on_glass(lit, turn))
            scan_paths.append(scan_path)

        sides[side] = {"flat": flat_path, "scans": scan_paths}
    return sides


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        captures = write_captures(staging)
        output_dir = staging / "output"

        thresholds = detect.load_thresholds(REPO_ROOT / "calibration" / "thresholds.json")
        report = grade_card(
            captures["front"]["flat"],
            captures["back"]["flat"],
            thresholds,
            output_dir,
            verbose=False,
            dpi=SCAN_DPI,
            photometric_paths=(captures["front"]["scans"], captures["back"]["scans"]),
            lamp_azimuth=LAMP_AZIMUTH,
            rotation="cw",
        )

        if report.get("grade_estimate") is None:
            print("the demo card failed a hard capture gate — nothing saved", file=sys.stderr)
            return 1

        summary = store.save_report(
            REPORTS_DIR, DEMO_REPORT_ID, report, _image_paths(output_dir, has_surface=False)
        )

    estimate = report["grade_estimate"]
    print(f"demo report saved: {REPORTS_DIR / DEMO_REPORT_ID}")
    print(f"  score:      {summary.score} (grade {summary.grade})")
    print(f"  centering:  {estimate['centering_grade']}   corners/edges: {estimate['corners_edges_grade']}")
    print(f"  dimensions: {report['dimensions']['note']}")
    for side in ("front", "back"):
        vision = report["card_vision"][side]
        print(f"  {side} Card Vision: {vision['method']} ({vision['light_count']} lights)")
    print(f"  dings:      {len(report['dings'])}")
    for ding in report["dings"][:6]:
        print(f"    - {ding['side']} {ding['label']}: {ding['detail']}")
    print(f"\nopen: /r/{DEMO_REPORT_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
