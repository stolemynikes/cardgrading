#!/usr/bin/env python3
"""Card pre-grader CLI (Phase 1: detect/normalize + centering; Phase 2: corners/edges;
Phase 3: surface defect map; Phase 4: vision-model surface judgment + grade assembly).

Usage:
    python grade.py front.png back.png [--surface front_angled.png back_angled.png]
                     [--output-dir output] [--thresholds calibration/thresholds.json]

Surface vision review runs automatically when --surface is passed, using
whichever vision-model credentials are configured: ANTHROPIC_API_KEY (or an
`ant auth login` profile) for Claude, else GEMINI_API_KEY for Gemini's free
tier. If no credentials are available, it's skipped with a note in the
report — the rest of the pipeline still runs.

The per-card orchestration lives in grade_card() so calibration/calibrate.py
can reuse it to batch-run the pipeline against cards with known PSA grades.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

from llm import vision
from pipeline import centering, corners_edges, detect, scoring, surface

REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-grade a Pokemon card from flatbed/overhead scans.")
    parser.add_argument("front", type=Path, help="Path to the front overhead photo")
    parser.add_argument("back", type=Path, help="Path to the back overhead photo")
    parser.add_argument(
        "--surface",
        nargs=2,
        type=Path,
        metavar=("FRONT_ANGLED", "BACK_ANGLED"),
        help="Angled raking-light photos of front/back for surface defect visualization (indicative only)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "output", help="Directory to write reports/debug images"
    )
    parser.add_argument(
        "--thresholds", type=Path, default=REPO_ROOT / "calibration" / "thresholds.json", help="Path to thresholds.json"
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


def run_surface_side(label: str, angled_path: Path, card_dir: Path, thresholds: dict, verbose: bool = True) -> dict | None:
    angled_img = load_image(angled_path)
    align_result = detect.align_for_surface(angled_img, thresholds)
    print_gate_report(f"{label} (angled)", align_result, verbose)
    # Same hard/soft split as the flat shots: only a missing warp or a
    # geometry failure skips surface analysis. A soft failure (resolution)
    # proceeds — the defect map is indicative-only anyway.
    if align_result.warped is None or align_result.hard_failures:
        _log(verbose, f"  skipping surface analysis for {label} — could not align the angled photo")
        return {
            "aligned": False,
            "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in align_result.gates],
            "vision_judgment": None,
        }

    result = surface.analyze_surface(align_result.warped, thresholds)
    surf_dir = card_dir / "surface"
    surf_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(surf_dir / f"{label}_aligned.png"), align_result.warped)
    cv2.imwrite(str(surf_dir / f"{label}_defect_map.png"), result.defect_map)
    cv2.imwrite(str(surf_dir / f"{label}_annotated.png"), result.annotated)

    _log(
        verbose,
        f"  {label}: defect area={result.defect_area_pct:.2f}% (of non-holo area), "
        f"holo masked={result.holo_area_pct:.2f}%, blobs={result.blob_count} — indicative only, not a grade",
    )

    result_dict = result.to_dict()
    result_dict["aligned"] = True
    try:
        judgment, model_used = vision.judge_surface(
            surf_dir / f"{label}_aligned.png", surf_dir / f"{label}_defect_map.png"
        )
    except vision.VisionUnavailable as e:
        _log(verbose, f"  {label} vision review skipped: {e}")
        result_dict["vision_judgment"] = None
    else:
        result_dict["vision_judgment"] = {**judgment.model_dump(), "model": model_used}
        _log(verbose, f"  {label} vision judgment ({model_used}): surface grade={judgment.surface_grade} (confidence={judgment.confidence})")
        for defect in judgment.defects_found:
            _log(verbose, f"    - {defect}")

    return result_dict


def grade_card(
    front_path: Path,
    back_path: Path,
    thresholds: dict,
    output_dir: Path,
    surface_paths: tuple[Path, Path] | None = None,
    verbose: bool = True,
    on_stage: Callable[[str], None] | None = None,
) -> dict:
    """Run the full pipeline on one card and return the report dict.

    surface_paths, if given, is (front_angled_path, back_angled_path).
    Writes debug images and report.json under output_dir. Does not raise on
    a failed capture-quality gate or a skipped vision review — those are
    recorded in the report instead, so this can run unattended over a batch
    of cards (see calibration/calibrate.py).

    on_stage, if given, is called with a short stage-name string ("detect",
    "centering", "corners_edges", "surface", "vision_flat", "scoring",
    "done") right before
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
        "capture_quality": {
            "front": front_result.to_dict(),
            "back": back_result.to_dict(),
        },
    }

    if front_result.warped is not None:
        cv2.imwrite(str(output_dir / "front_aligned.png"), front_result.warped)
    if back_result.warped is not None:
        cv2.imwrite(str(output_dir / "back_aligned.png"), back_result.warped)

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

    stage("centering")
    result = centering.measure_centering(front_result.warped, back_result.warped, thresholds)
    report["centering"] = result.to_dict()

    _log(verbose, "\n[centering]")
    for side_label, measurable, axis_h, axis_v, side_grade in [
        ("front", result.front_measurable, result.front_horizontal, result.front_vertical, result.front_grade),
        ("back", result.back_measurable, result.back_horizontal, result.back_vertical, result.back_grade),
    ]:
        _log(verbose, f" {side_label}:")
        if measurable:
            print_axis("  horizontal", axis_h, verbose)
            print_axis("  vertical  ", axis_v, verbose)
            _log(verbose, f"  {side_label} centering grade: {side_grade}")
        else:
            _log(verbose, f"  {side_label} centering unmeasurable — borderless/full-art card, or border not visible in this capture")
    _log(verbose, f"\n overall centering grade: {result.overall_grade if result.overall_grade is not None else 'n/a'}")

    front_overlay = centering.draw_overlay(
        front_result.warped, result.front_horizontal, result.front_vertical, result.front_measurable
    )
    back_overlay = centering.draw_overlay(
        back_result.warped, result.back_horizontal, result.back_vertical, result.back_measurable
    )
    cv2.imwrite(str(output_dir / "front_centering_overlay.png"), front_overlay)
    cv2.imwrite(str(output_dir / "back_centering_overlay.png"), back_overlay)

    def side_borders(measurable: bool, axis_h, axis_v, image) -> corners_edges.BorderWidths:
        if measurable:
            return corners_edges.BorderWidths(
                left=axis_h.side_a_px, right=axis_h.side_b_px,
                top=axis_v.side_a_px, bottom=axis_v.side_b_px,
            )
        # Border widths from an unmeasurable side are argmax-of-noise — sizing
        # the corner/edge crops from them produced postage-stamp crops whose
        # whitening percentages were pure noise. Fall back to a typical
        # border width (~4% of the card dimension) instead.
        h, w = image.shape[:2]
        return corners_edges.BorderWidths(left=w * 0.04, right=w * 0.04, top=h * 0.04, bottom=h * 0.04)

    front_borders = side_borders(result.front_measurable, result.front_horizontal, result.front_vertical, front_result.warped)
    back_borders = side_borders(result.back_measurable, result.back_horizontal, result.back_vertical, back_result.warped)
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

    surface_grade = None
    if surface_paths:
        stage("surface")
        front_angled_path, back_angled_path = surface_paths
        _log(verbose, "\n[surface] (indicative only — needs vision-model review)")
        front_surface = run_surface_side("front", front_angled_path, output_dir, thresholds, verbose)
        back_surface = run_surface_side("back", back_angled_path, output_dir, thresholds, verbose)
        report["surface"] = {"front": front_surface, "back": back_surface}

        front_judgment = (front_surface or {}).get("vision_judgment")
        back_judgment = (back_surface or {}).get("vision_judgment")
        if front_judgment and back_judgment:
            surface_grade = min(front_judgment["surface_grade"], back_judgment["surface_grade"])
    else:
        report["surface"] = None
        _log(verbose, "\n[surface] skipped — pass --surface front_angled.png back_angled.png to run it")

    # Vision opinion on the flat captures themselves — an independent take on
    # corners/edges wear (not tied to the whitening thresholds) plus an
    # upper-bound surface estimate. Runs whenever vision credentials exist;
    # skipped silently otherwise, same contract as the raking-light judgment.
    stage("vision_flat")
    report["vision_flat"] = None
    try:
        front_flat, flat_model = vision.judge_flat(output_dir / "front_aligned.png", "front")
        back_flat, _ = vision.judge_flat(output_dir / "back_aligned.png", "back")
    except vision.VisionUnavailable as e:
        _log(verbose, f"\n[vision] flat-shot review skipped: {e}")
    else:
        report["vision_flat"] = {
            "front": {**front_flat.model_dump(), "model": flat_model},
            "back": {**back_flat.model_dump(), "model": flat_model},
        }
        _log(verbose, f"\n[vision] flat-shot opinion ({flat_model}):")
        for side_label, judgment in (("front", front_flat), ("back", back_flat)):
            _log(
                verbose,
                f"  {side_label}: corners={judgment.corners_grade} edges={judgment.edges_grade} "
                f"surface<={judgment.surface_grade} (confidence={judgment.confidence})",
            )

    # No raking-light surface judgment (2-shot flow, or the angled shots
    # failed)? The flat-shot surface estimate fills in — clearly labeled as
    # an upper bound, since flat lighting hides shallow scratches.
    surface_from_flat = False
    if surface_grade is None and report["vision_flat"] is not None:
        surface_grade = min(
            report["vision_flat"]["front"]["surface_grade"],
            report["vision_flat"]["back"]["surface_grade"],
        )
        surface_from_flat = True

    stage("scoring")
    grade_estimate = scoring.assemble_grade(
        result.overall_grade, ce_result.overall_grade, surface_grade, thresholds,
        surface_from_flat=surface_from_flat,
    )
    report["grade_estimate"] = grade_estimate.to_dict()

    _log(verbose, "\n[grade estimate]")
    _log(verbose, f"  centering:      {grade_estimate.centering_grade}")
    _log(verbose, f"  corners/edges:  {grade_estimate.corners_edges_grade}")
    _log(verbose, f"  surface:        {grade_estimate.surface_grade if grade_estimate.surface_grade is not None else 'n/a'}")
    _log(verbose, f"  overall (est.): {grade_estimate.overall_grade_rounded} ({grade_estimate.overall_grade:.2f})")
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
        surface_paths=tuple(args.surface) if args.surface else None,
    )

    print(f"\nReport and debug images saved to {card_dir}")

    if report["centering"] is None:
        print("Partial report only — capture quality gate(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
