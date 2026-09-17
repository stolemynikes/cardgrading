#!/usr/bin/env python3
"""Card pre-grader CLI.

Every sub-grade — centering, corners/edges, surface, dimensions — is measured
deterministically, so a card grades end to end with no API key and no network.
The one optional model call identifies the card, which drives the market
lookup and the full-art note; skipping it costs a name, never a grade.

Usage:
    python grade.py front.png back.png
                     [--output-dir output] [--thresholds calibration/thresholds.json]

Card identification uses whichever credentials are configured:
ANTHROPIC_API_KEY (or an `ant auth login` profile) for Claude, else
GEMINI_API_KEY for Gemini's free tier. With neither, it's skipped with a note
in the report and everything else runs unchanged.

The per-card orchestration lives in grade_card() so calibration/calibrate.py
can reuse it to batch-run the pipeline against cards with known PSA grades.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import market
from llm import vision
from pipeline import cardvision, centering, corners_edges, detect, dimensions, dings, scoring, surface

REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-grade a Pokemon card from flatbed/overhead scans.")
    parser.add_argument("front", type=Path, help="Path to the front overhead photo")
    parser.add_argument("back", type=Path, help="Path to the back overhead photo")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "output", help="Directory to write reports/debug images"
    )
    parser.add_argument(
        "--thresholds", type=Path, default=REPO_ROOT / "calibration" / "thresholds.json", help="Path to thresholds.json"
    )
    parser.add_argument(
        "--dpi",
        type=float,
        help="Scan resolution of the flat captures. Only a fixed-DPI scan has a known scale, "
        "so this is what makes the card's physical dimensions measurable (miscut/trim detection).",
    )
    parser.add_argument(
        "--photometric-front",
        nargs="+",
        type=Path,
        metavar="SCAN",
        help="3+ scans of the front, the card rotated a further 90 degrees on the glass each time, "
        "in rotation order. Solves a true surface-normal map for Card Vision.",
    )
    parser.add_argument(
        "--photometric-back", nargs="+", type=Path, metavar="SCAN", help="Same, for the back."
    )
    parser.add_argument(
        "--lamp-azimuth",
        type=float,
        default=90.0,
        help="Direction the scanner's lamp lights from, in the first scan's card frame (degrees, "
        "counter-clockwise from the card's right edge). Default 90 = along the card's long axis.",
    )
    parser.add_argument(
        "--rotation",
        choices=("cw", "ccw"),
        default="cw",
        help="Which way the card was turned between photometric scans (default: clockwise).",
    )
    return parser.parse_args(argv)


def load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def _log(verbose: bool, message: str = "") -> None:
    if verbose:
        print(message)


def print_gate_report(label: str, result: detect.DetectResult, verbose: bool = True) -> None:
    status = "PASS" if result.ok else "FAIL — retake recommended"
    _log(verbose, f"\n[{label}] capture quality: {status}")
    for gate in result.gates:
        mark = "ok" if gate.passed else "FAIL"
        _log(verbose, f"  [{mark:4}] {gate.name}: {gate.detail}")


def print_axis(label: str, axis: centering.AxisCentering, verbose: bool = True) -> None:
    _log(
        verbose,
        f"  {label}: {axis.ratio_str} ({axis.side_a}={axis.side_a_pct:.1f}% / {axis.side_b}={axis.side_b_pct:.1f}%) -> grade {axis.grade}",
    )


def print_side_regions(label: str, side: corners_edges.SideResult, verbose: bool = True) -> None:
    _log(verbose, f" {label}:")
    for name, region in side.corners.items():
        _log(verbose, f"  corner {name:14} whitening={region.whitening_pct:.2f}%  blobs={region.blob_count}  grade={region.grade}")
    for name, region in side.edges.items():
        _log(verbose, f"  edge   {name:14} whitening={region.whitening_pct:.2f}%  blobs={region.blob_count}  grade={region.grade}")
    _log(verbose, f"  {label} corners/edges grade: {side.grade}")


def save_region_overlays(card_dir: Path, side_label: str, overlays: dict) -> None:
    side_dir = card_dir / "corners_edges" / side_label
    side_dir.mkdir(parents=True, exist_ok=True)
    for region_name, overlay_img in overlays.items():
        cv2.imwrite(str(side_dir / f"{region_name}.png"), overlay_img)


def surface_grade_for_side(vision, thresholds: dict) -> surface.SurfaceGrade:
    """Grade one side's surface from the best signal available.

    Solved surface normals contain no albedo, so print and foil cannot be
    mistaken for damage. The single-capture Card Vision approximation is
    measured and shown but never graded: print demonstrably leaks into it.
    """
    # measurement_relief, not relief: the displayed render's gain and its
    # autoscale are viewing preferences, and a viewing preference must not
    # move a measurement.
    if vision is None:
        return surface.grade_surface(0.0, 0, 0, "single_image_relief", thresholds)

    measured = vision.measurement_relief if vision.measurement_relief is not None else vision.relief
    area, count, longest, _ = surface.relief_defect_stats(measured, thresholds["surface"])
    source = "photometric_relief" if vision.method == "photometric_stereo" else "single_image_relief"
    # Classified on both paths: the single-image render isn't trustworthy
    # enough to grade, but naming what it found is still worth showing.
    px_per_mm = thresholds["capture"]["canonical_width_px"] / dimensions.NOMINAL_WIDTH_MM
    defects = surface.classify_defects(measured, thresholds["surface"], px_per_mm)
    return surface.grade_surface(area, count, longest, source, thresholds, defects=defects)


# The most rotation scans a set can hold, and so the most warps kept with a
# report. Matches the upload cap in webapp/main.py.
MAX_PHOTOMETRIC_SCANS_KEPT = 6


def load_derotated(path: Path, index: int, direction: str):
    """Load the index-th photometric scan and undo the physical rotation.

    Every scan in the set has the card turned a further 90 degrees on the
    glass, so the card sits at a different angle in each raw file. Undoing
    that here means Stage 1 sees a normally-oriented card in all of them —
    otherwise the perspective correction would stretch a landscape card into
    the portrait canonical frame, and a 180-degree scan would come out
    upside down.

    This is the *declared* rotation: it trusts the file order and the stated
    turn direction. `normalise_scan` works it out from the pixels instead,
    and this is kept as the fallback for a scan that can't be resolved.
    """
    image = load_image(path)
    code = cv2.ROTATE_90_COUNTERCLOCKWISE if direction == "cw" else cv2.ROTATE_90_CLOCKWISE
    for _ in range(index % 4):
        image = cv2.rotate(image, code)
    return image


def _rotate_ccw(image, degrees: int):
    codes = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    return image if degrees % 360 == 0 else cv2.rotate(image, codes[degrees % 360])


def _orientation_match(warp, reference) -> float:
    """Normalised cross-correlation of two warps, thumbnail-sized.

    Distinguishing a card from the same card turned 180 degrees is the one
    thing geometry can't do — both are portrait and both pass the aspect
    gate — so it takes content. Downscaled hard, because what's wanted is
    "is this the same picture the same way up", not a pixel comparison.
    """
    def prepare(image):
        small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (120, 168), interpolation=cv2.INTER_AREA)
        centred = small.astype(np.float32) - float(small.mean())
        norm = float(np.linalg.norm(centred))
        return centred / norm if norm > 0 else centred

    return float((prepare(warp) * prepare(reference)).sum())


# Below this correlation the two orientations are indistinguishable and the
# declared order is the better guess. A card against itself scores ~1.0; a
# card against its own 180-degree turn scores well under 0.5 on any card with
# asymmetric artwork, which is all of them.
MIN_ORIENTATION_CONFIDENCE = 0.55


# Above this correlation the two uploads are the same picture, not two sides
# of one card.
#
# The same side scanned twice — card lifted off the glass and put back down —
# still correlates above 0.99. The other end is not as far away as it looks:
# two *different* pictures that share a card's layout (same border, same art
# panel, different artwork) measure 0.74, because at thumbnail size the
# layout is most of what's left. A real front and back share less than that,
# but 0.74 is the closest thing to a worst case measurable without a stack of
# real pairs, so the threshold sits with margin above it rather than at the
# "near zero" a front and back would actually give.
SAME_SIDE_CORRELATION = 0.90


def check_capture_pair(front_path: Path, back_path: Path, front_warp, back_warp) -> dict:
    """Are these two uploads really two different sides of one card?

    Worth checking because the failure is silent and expensive: grading the
    front twice produces a complete, confident report in which every "back"
    number was measured on the front and scored against PSA's looser back
    tolerances. Nothing else in the pipeline notices — both images detect,
    warp and grade perfectly well.

    The vision identify stage already cross-checks this, but it needs an API
    key and is skipped without one, which is the normal case here. This is
    the offline version: a file hash catches the same file submitted twice,
    and the same thumbnail correlation that settles rotation catches a second
    scan of the same side.

    A warning, never a refusal. Re-grading one side against both tolerance
    tables is a legitimate thing to do deliberately, and the operator is
    standing right there.
    """
    front_bytes, back_bytes = front_path.read_bytes(), back_path.read_bytes()
    identical = hashlib.sha256(front_bytes).digest() == hashlib.sha256(back_bytes).digest()
    similarity = float(_orientation_match(front_warp, back_warp))
    suspected = identical or similarity >= SAME_SIDE_CORRELATION

    if identical:
        note = "The front and back uploads are the same file, so the back was graded on the front."
    elif suspected:
        note = (
            f"The front and back scans look like the same side of the card ({similarity:.0%} match). "
            "If that's not deliberate, the back sub-grades were measured on the front."
        )
    else:
        note = None

    return {
        "identical_files": identical,
        "similarity": round(similarity, 3),
        "same_side_suspected": suspected,
        "note": note,
    }


def normalise_scan(path: Path, reference_warp, thresholds: dict):
    """Warp one rotation scan upright, working out its rotation from the image.

    Returns (warp, ccw_degrees_applied, confident, diagnostics). The rotation is what the
    scan had to be turned counter-clockwise to stand the card upright, which
    is exactly the angle the card was turned clockwise on the glass — and
    that is what sets the light's azimuth in the card's frame.

    Derived rather than declared because the declared version is a trap: it
    depends on file order and on the operator remembering which way they
    turned the card, it fails silently when either is wrong, and the failure
    looks like a plausible render of a badly damaged card.
    """
    image = load_image(path)
    candidates = []
    for ccw in (0, 90, 180, 270):
        result = detect.detect_and_normalize(_rotate_ccw(image, ccw), thresholds)
        if result.warped is None or result.hard_failures:
            continue
        candidates.append((_orientation_match(result.warped, reference_warp), ccw, result))

    if not candidates:
        return None, None, False, {}
    candidates.sort(reverse=True, key=lambda c: c[0])
    score, ccw, result = candidates[0]
    runner_up = candidates[1][0] if len(candidates) > 1 else -1.0
    # Confident only if the winner is clearly the winner. A symmetrical card
    # back, or a scan too dark to correlate, should fall back rather than
    # pick one at random.
    confident = score >= MIN_ORIENTATION_CONFIDENCE and score - runner_up >= 0.1

    # What the detector actually found in this scan, before the warp
    # normalised it away. A set whose frames don't line up can fail here just
    # as easily as in the alignment: an auto-crop that clips a card edge
    # gives a truncated quad, and the warp then stretches that scan's content
    # to fill a frame it doesn't belong in.
    quad = result.contour.astype(np.float64)
    diagnostics = {
        "contour": result.contour,
        "orientation_score": round(float(score), 3),
        "runner_up_score": round(float(runner_up), 3),
        "quad_width_px": round(float((np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2), 1),
        "quad_height_px": round(float((np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2), 1),
        "scan_size_px": [int(image.shape[1]), int(image.shape[0])],
    }
    return result.warped, ccw, confident, diagnostics


def build_card_vision(
    side_label: str,
    flat_warped,
    photometric_paths: list[Path] | None,
    output_dir: Path,
    thresholds: dict,
    lamp_azimuth: float,
    rotation: str,
    verbose: bool = True,
):
    """Card Vision for one side: photometric stereo if the scan set is there,
    the single-image approximation otherwise.

    Falls back rather than failing — a bad scan in the set costs the render
    its fidelity, not its existence, and the method is recorded in the
    report either way so the UI can label what the viewer is looking at.
    """
    cfg = thresholds["card_vision"]
    warps = []
    rotations: list[int] = []
    detected: list[dict] = []
    quads: list = []
    dropped: list[str] = []
    fallback_reason = None
    if not photometric_paths:
        fallback_reason = "no rotation scans were provided for this side"
    elif len(photometric_paths) < 3:
        fallback_reason = (
            f"photometric stereo needs at least 3 light directions; {len(photometric_paths)} scan(s) "
            "were provided"
        )
        _log(verbose, f"  {side_label}: {fallback_reason} — using single-image Card Vision")
    else:
        for i, scan_path in enumerate(photometric_paths):
            warp, ccw, confident, scan_diagnostics = normalise_scan(scan_path, flat_warped, thresholds)
            if warp is None:
                # Drop the scan, not the set. Photometric stereo needs three
                # non-collinear light directions, and a four-scan set has one
                # to spare — throwing the other three away for the sake of the
                # one that didn't detect cost a whole capture session's
                # surface grade over a single card that sat slightly off the
                # glass.
                dropped.append(f"scan {i + 1} of {len(photometric_paths)} ({scan_path.name})")
                _log(verbose, f"  {side_label}: no card contour in {dropped[-1]} — dropping that scan")
                continue
            if not confident:
                # The pixels didn't settle it — a near-symmetrical side, or a
                # scan too dark to correlate. Fall back on what the operator
                # declared, which is what this used to do for every scan.
                declared = detect.detect_and_normalize(load_derotated(scan_path, i, rotation), thresholds)
                if declared.warped is not None and not declared.hard_failures:
                    step = 90 if rotation == "cw" else -90
                    warp, ccw = declared.warped, (step * i) % 360
                    _log(verbose, f"  {side_label}: scan {i + 1} orientation unresolved — using the declared order")
            warps.append(warp)
            rotations.append(ccw)
            scan_diagnostics["name"] = scan_path.name
            scan_diagnostics["rotation_deg"] = ccw
            # The quad itself is for the dimensions stage, not the report.
            quads.append(scan_diagnostics.pop("contour", None))
            detected.append(scan_diagnostics)

    if dropped and len(warps) < 3:
        fallback_reason = (
            f"no card contour was found in {', '.join(dropped)}, leaving {len(warps)} of "
            f"{len(photometric_paths)} scans usable — photometric stereo needs at least 3."
        )
        _log(verbose, f"  {side_label}: {fallback_reason}")

    if len(warps) >= 3:
        # A card turned clockwise by theta moves the lamp clockwise by theta
        # in the card's frame, and the counter-clockwise rotation needed to
        # stand that scan upright is exactly theta — so the measured rotation
        # gives the azimuth directly, with no assumption about file order or
        # which way the operator turned the card.
        azimuths = [(lamp_azimuth + ccw) % 360.0 for ccw in rotations]
        # The flat capture leads, with no light of its own to contribute: it
        # is there as the registration reference, so the solved relief comes
        # out in the same frame as the image the report cross-fades it
        # against. Measured before this, the two were three pixels and half a
        # percent of scale apart, which is visible on a slider.
        vision = cardvision.photometric_card_vision(
            warps, azimuths, cfg, registration_reference=flat_warped
        )
        vision.rotations_deg = rotations
        vision.dropped_scans = dropped or None
        # Fold what the detector found in each scan in with how well that
        # scan then aligned: together they say whether a misaligned set is a
        # capture problem or an alignment one.
        for frame, found in zip(vision.registration or [], detected):
            frame.update(found)
        _log(verbose, f"  {side_label}: measured card rotations {rotations} degrees")
        for frame in vision.registration or []:
            _log(verbose, f"    {frame}")
    else:
        vision = cardvision.single_image_card_vision(flat_warped, cfg)
        vision.fallback_reason = fallback_reason
        vision.dropped_scans = dropped or None

    cv2.imwrite(str(output_dir / f"{side_label}_card_vision.png"), vision.relief)
    # The normalised rotation scans, kept so the solve can be re-run later
    # without the card going back on the glass. They are the only inputs the
    # photometric path has, and discarding them meant every change to the
    # solve cost a rescan — four times, over one evening.
    for index, warp in enumerate(warps):
        cv2.imwrite(str(output_dir / f"{side_label}_rotation_{index}.png"), warp)
    if vision.normal_map is not None:
        cv2.imwrite(str(output_dir / f"{side_label}_card_vision_normals.png"), vision.normal_map)
    if vision.albedo is not None:
        cv2.imwrite(str(output_dir / f"{side_label}_card_vision_albedo.png"), vision.albedo)

    _log(
        verbose,
        f"  {side_label}: {vision.method} from {vision.light_count} light direction(s), "
        f"relief off-flat {vision.roughness_pct:.2f}% of area",
    )
    vision.scan_quads = quads
    return vision


def grade_card(
    front_path: Path,
    back_path: Path,
    thresholds: dict,
    output_dir: Path,
    verbose: bool = True,
    on_stage: Callable[[str], None] | None = None,
    dpi: float | None = None,
    photometric_paths: tuple[list[Path] | None, list[Path] | None] = (None, None),
    lamp_azimuth: float = 90.0,
    rotation: str = "cw",
) -> dict:
    """Run the full pipeline on one card and return the report dict.

    Writes debug images and report.json under output_dir. Does not raise on
    a failed capture-quality gate or a skipped vision review — those are
    recorded in the report instead, so this can run unattended over a batch
    of cards (see calibration/calibrate.py).

    on_stage, if given, is called with a short stage-name string ("detect",
    "identify", "dimensions", "card_vision", "centering", "corners_edges",
    "surface", "scoring", "done") right before
    each stage starts — for callers that want progress reporting (the
    webapp's job polling) without parsing verbose print output. Stage names
    are deliberately UI-agnostic; the caller maps them to human-readable text.
    """

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    stage("detect")
    front_img = load_image(front_path)
    back_img = load_image(back_path)

    front_result = detect.detect_and_normalize(front_img, thresholds)
    back_result = detect.detect_and_normalize(back_img, thresholds)

    print_gate_report("front", front_result, verbose)
    print_gate_report("back", back_result, verbose)

    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "front_image": str(front_path),
        "back_image": str(back_path),
        # The scale every millimetre in this report rests on. Recorded because
        # a wrong one is indistinguishable from a miscut card: the dimensions
        # stage reported a card 5.34mm narrow and 8.52mm short — a 9% error on
        # both axes at once, which is a scale mistake, not a trim — and the
        # report gave no way to check the number it had been handed.
        "capture_dpi": dpi,
        "capture_quality": {
            "front": front_result.to_dict(),
            "back": back_result.to_dict(),
        },
    }

    if front_result.warped is not None:
        cv2.imwrite(str(output_dir / "front_aligned.png"), front_result.warped)
    if back_result.warped is not None:
        cv2.imwrite(str(output_dir / "back_aligned.png"), back_result.warped)

    # A second warp at the capture's own scale, for looking at rather than
    # measuring from. The canonical 1500x2100 is the measurement resolution —
    # every threshold in thresholds.json is calibrated against it — but it
    # throws away most of a 1200dpi scan, and zooming into it just
    # interpolates. This keeps the detail that was actually captured.
    for label, image, result in (("front", front_img, front_result), ("back", back_img, back_result)):
        detail = detect.detail_warp(image, result.contour, thresholds)
        if detail is not None:
            cv2.imwrite(str(output_dir / f"{label}_detail.png"), detail)

    front_blocked = front_result.warped is None or front_result.hard_failures
    back_blocked = back_result.warped is None or back_result.hard_failures
    if front_blocked or back_blocked:
        # Hard gates (card detection, tilt, aspect ratio) mean the detected
        # geometry itself is wrong — the warp isn't a usable image of the
        # card, so grading it would produce meaningless numbers. Soft gates
        # (resolution, glare, uneven lighting) don't block: grading proceeds
        # and the caller surfaces them as a "grade may be less reliable"
        # warning instead of forcing a retake.
        _log(verbose, "\nCard geometry check failed on one or both photos — retake required.")
        report["centering"] = None
        report["corners_edges"] = None
        report["surface"] = None
        report["grade_estimate"] = None
        (output_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report

    # Both sides warped, so the pair can be checked against itself before
    # anything is measured from it.
    report["capture_pair"] = check_capture_pair(
        front_path, back_path, front_result.warped, back_result.warped
    )
    if report["capture_pair"]["note"]:
        _log(verbose, f"\n[capture] {report['capture_pair']['note']}")

    # Identify-first, like the commercial AI graders: know what the card IS
    # before measuring it. One vision call covers identification plus a
    # sanity check of the capture pair itself (same side shot twice, front/
    # back swapped, mismatched cards). Optional like every vision stage.
    stage("identify")
    report["card_id"] = None
    report["market"] = None
    try:
        ident, ident_model = vision.identify_card(
            output_dir / "front_aligned.png", output_dir / "back_aligned.png"
        )
    except vision.VisionUnavailable as e:
        _log(verbose, f"\n[identify] skipped: {e}")
    else:
        report["card_id"] = {**ident.model_dump(), "model": ident_model}
        _log(
            verbose,
            f"\n[identify] {ident.card_name} — {ident.set_name or 'set unknown'}"
            f"{' #' + ident.collector_number if ident.collector_number else ''}"
            f" (full-art={ident.is_full_art}, holo={ident.is_holo}, confidence={ident.confidence})",
        )
        if ident.front_image_side == "back" or ident.back_image_side == "front":
            _log(verbose, "  WARNING: the photos may be swapped or show the same side twice")
        if not ident.looks_like_same_card:
            _log(verbose, "  WARNING: front and back photos may be different cards")
        report["market"] = market.lookup_prices(ident.card_name, ident.collector_number, ident.set_name)
        if report["market"]:
            m = report["market"]
            _log(verbose, f"  market match: {m['matched_name']} ({m['matched_set']} #{m['matched_number']}) prices={m['prices']}")

    # Card Vision runs before centering so the relief render is available to
    # every later stage and to the report regardless of which optional
    # stages ran. It never blocks: worst case it is the single-image
    # approximation of the flat capture we already have.
    stage("card_vision")
    _log(verbose, "\n[card vision]")
    front_vision = build_card_vision(
        "front", front_result.warped, photometric_paths[0], output_dir, thresholds, lamp_azimuth, rotation, verbose
    )
    back_vision = build_card_vision(
        "back", back_result.warped, photometric_paths[1], output_dir, thresholds, lamp_azimuth, rotation, verbose
    )
    report["card_vision"] = {"front": front_vision.to_dict(), "back": back_vision.to_dict()}

    stage("dimensions")
    # After Card Vision, not before: every rotation scan it normalised is an
    # independent measurement of the same card, and they disagree by more
    # than the tolerance on this hardware. Resting the verdict on whichever
    # scan happened to be the flat capture made the same card read '2.13mm
    # miscut' one run and 'within tolerance' the next.
    front_quads = [front_result.contour] + list(getattr(front_vision, "scan_quads", None) or [])
    front_dimensions = dimensions.measure_from_scans(front_quads, dpi, thresholds)
    report["dimensions"] = front_dimensions.to_dict()
    _log(verbose, "\n[dimensions]")
    if front_dimensions.measurable:
        _log(
            verbose,
            f"  {front_dimensions.width_mm:.2f} x {front_dimensions.height_mm:.2f} mm "
            f"(nominal {dimensions.NOMINAL_WIDTH_MM} x {dimensions.NOMINAL_HEIGHT_MM}) — {front_dimensions.note}",
        )
    else:
        _log(verbose, f"  {front_dimensions.note}")


    stage("centering")
    result = centering.measure_centering(front_result.warped, back_result.warped, thresholds)
    report["centering"] = result.to_dict()
    # What every other grading service's published table would say about the
    # same measurement. Reference only — PSA still drives the grade.
    report["centering"]["by_grader"] = centering.compare_graders(report["centering"], thresholds)

    _log(verbose, "\n[centering]")
    for side_label, axis_h, axis_v, side_grade in [
        ("front", result.front_horizontal, result.front_vertical, result.front_grade),
        ("back", result.back_horizontal, result.back_vertical, result.back_grade),
    ]:
        _log(verbose, f" {side_label}:")
        for axis_label, axis in (("horizontal", axis_h), ("vertical  ", axis_v)):
            if axis.measurable:
                print_axis(f"  {axis_label}", axis, verbose)
            else:
                _log(verbose, f"    {axis_label}: unmeasurable — boundary not visible")
        if side_grade is not None:
            _log(verbose, f"  {side_label} centering grade: {side_grade}")
        else:
            _log(verbose, f"  {side_label} centering unmeasurable — borderless/full-art card, or border not visible in this capture")
    _log(verbose, f"\n overall centering grade: {result.overall_grade if result.overall_grade is not None else 'n/a'}")

    front_overlay = centering.draw_overlay(front_result.warped, result.front_horizontal, result.front_vertical)
    back_overlay = centering.draw_overlay(back_result.warped, result.back_horizontal, result.back_vertical)
    cv2.imwrite(str(output_dir / "front_centering_overlay.png"), front_overlay)
    cv2.imwrite(str(output_dir / "back_centering_overlay.png"), back_overlay)

    def side_borders(axis_h, axis_v, image) -> corners_edges.BorderWidths:
        # Border widths from an unmeasurable axis are argmax-of-noise —
        # sizing the corner/edge crops from them produced postage-stamp
        # crops whose whitening percentages were pure noise. Per axis: use
        # the measured widths where real, a typical ~4% default otherwise.
        h, w = image.shape[:2]
        if axis_h.measurable:
            left, right = axis_h.side_a_px, axis_h.side_b_px
        else:
            left = right = w * 0.04
        if axis_v.measurable:
            top, bottom = axis_v.side_a_px, axis_v.side_b_px
        else:
            top = bottom = h * 0.04
        return corners_edges.BorderWidths(left=left, right=right, top=top, bottom=bottom)

    front_borders = side_borders(result.front_horizontal, result.front_vertical, front_result.warped)
    back_borders = side_borders(result.back_horizontal, result.back_vertical, back_result.warped)
    stage("corners_edges")
    ce_result, ce_overlays = corners_edges.analyze_corners_edges(
        front_result.warped, back_result.warped, front_borders, back_borders, thresholds
    )
    report["corners_edges"] = ce_result.to_dict()

    _log(verbose, "\n[corners & edges]")
    print_side_regions("front", ce_result.front, verbose)
    print_side_regions("back", ce_result.back, verbose)
    _log(verbose, f"\n overall corners/edges grade: {ce_result.overall_grade}")

    save_region_overlays(output_dir, "front", ce_overlays["front"])
    save_region_overlays(output_dir, "back", ce_overlays["back"])

    stage("surface")
    _log(verbose, "\n[surface] grading from the Card Vision relief")
    surface_grades = {
        "front": surface_grade_for_side(front_vision, thresholds),
        "back": surface_grade_for_side(back_vision, thresholds),
    }
    report["surface"] = {side: surface_grades[side].to_dict() for side in ("front", "back")}

    for side in ("front", "back"):
        sg = surface_grades[side]
        grade_text = "not graded" if sg.grade is None else f"grade {sg.grade}"
        _log(
            verbose,
            f"  {side}: defect area={sg.defect_area_pct:.3f}%, blobs={sg.defect_count}, "
            f"longest={sg.longest_defect_px}px -> {grade_text} ({sg.source})",
        )

    # The weakest side sets the sub-grade.
    graded = [sg for sg in surface_grades.values() if sg.grade is not None]
    surface_grade = min((sg.grade for sg in graded), default=None)

    stage("scoring")
    grade_estimate = scoring.assemble_grade(
        result.overall_grade, ce_result.overall_grade, surface_grade, thresholds,
        dimensions_within_tolerance=front_dimensions.within_tolerance,
    )
    report["grade_estimate"] = grade_estimate.to_dict()

    # Per-side, per-attribute breakdown. The overall grade already combines
    # these; this is the same data split the way a grader reads a card —
    # front and back are separate surfaces with separate wear.
    report["subgrades"] = {
        "front": {
            "centering": result.front_grade,
            "corners": ce_result.front.corners_grade,
            "edges": ce_result.front.edges_grade,
            "surface": surface_grades["front"].grade,
        },
        "back": {
            "centering": result.back_grade,
            "corners": ce_result.back.corners_grade,
            "edges": ce_result.back.edges_grade,
            "surface": surface_grades["back"].grade,
        },
    }
    report["dings"] = dings.collect_dings(report)
    _log(verbose, f"\n[dings] {len(report['dings'])} defect(s) of notable grade significance")
    for ding in report["dings"][:8]:
        grade_str = "n/a" if ding["grade"] is None else f"grade {ding['grade']}"
        _log(verbose, f"  {ding['side']} {ding['label']} ({grade_str}): {ding['detail']}")

    _log(verbose, "\n[grade estimate]")
    _log(verbose, f"  centering:      {grade_estimate.centering_grade}")
    _log(verbose, f"  corners/edges:  {grade_estimate.corners_edges_grade}")
    _log(verbose, f"  surface:        {grade_estimate.surface_grade if grade_estimate.surface_grade is not None else 'n/a'}")
    _log(verbose, f"  overall (est.): {grade_estimate.overall_grade_rounded} ({grade_estimate.overall_grade:.2f})")
    _log(verbose, f"  score:          {grade_estimate.score} / {scoring.MAX_SCORE}")
    _log(verbose, f"  note: {grade_estimate.note}")

    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    stage("done")
    return report


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    thresholds = detect.load_thresholds(args.thresholds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    card_dir = args.output_dir / f"{args.front.stem}_{timestamp}"

    report = grade_card(
        args.front,
        args.back,
        thresholds,
        card_dir,
        dpi=args.dpi,
        photometric_paths=(args.photometric_front, args.photometric_back),
        lamp_azimuth=args.lamp_azimuth,
        rotation=args.rotation,
    )

    print(f"\nReport and debug images saved to {card_dir}")

    if report["centering"] is None:
        print("Partial report only — capture quality gate(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
