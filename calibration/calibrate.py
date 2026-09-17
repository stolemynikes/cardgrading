#!/usr/bin/env python3
"""Phase 5: calibration harness.

Not an automatic calibrator — a measurement tool. Point it at a manifest of
cards with known PSA grades, and it runs the same pipeline grade.py uses on
each one, then reports how far the predicted grades were from the actual
ones. Use the per-category error to decide which thresholds in
calibration/thresholds.json need adjusting, edit them by hand, and re-run.

Manifest format (JSON array):
[
  {
    "name": "charizard_base_set",
    "front": "cards/charizard_front.jpg",
    "back": "cards/charizard_back.jpg",
    "actual_grade": {
      "overall": 9,
      "centering": 9,
      "corners_edges": 8,
      "surface": 9
    }
  }
]

Every key in actual_grade is optional — only the categories you provide get
scored. Image paths are resolved relative to
the manifest file's own directory.

Usage:
    python calibration/calibrate.py manifest.json
        [--output-dir output/calibration] [--thresholds calibration/thresholds.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from grade import grade_card  # noqa: E402
from pipeline import detect, scoring  # noqa: E402

CATEGORIES = ["overall", "centering", "corners_edges", "surface"]


@dataclass
class CardEntry:
    name: str
    front: Path
    back: Path
    actual_grade: dict


def load_manifest(path: Path) -> list[CardEntry]:
    raw = json.loads(path.read_text())
    base = path.parent
    entries = []
    for item in raw:
        entries.append(
            CardEntry(
                name=item["name"],
                front=base / item["front"],
                back=base / item["back"],
                actual_grade=item.get("actual_grade", {}),
            )
        )
    return entries


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run the pre-grader against cards with known PSA grades.")
    parser.add_argument("manifest", type=Path, help="Path to a JSON manifest of card entries")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "output" / "calibration", help="Directory for per-card debug output"
    )
    parser.add_argument(
        "--thresholds", type=Path, default=REPO_ROOT / "calibration" / "thresholds.json", help="Path to thresholds.json"
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit scoring.py's grade-combination weights from this batch's complete cases "
        "(needs actual overall + centering/corners_edges/surface predictions) and save to thresholds.json",
    )
    return parser.parse_args(argv)


def predicted_grades(report: dict) -> dict:
    """Extract predicted grades in the same shape as manifest actual_grade."""
    estimate = report.get("grade_estimate")
    if estimate is None:
        return {}
    return {
        "overall": estimate["overall_grade_rounded"],
        "centering": estimate["centering_grade"],
        "corners_edges": estimate["corners_edges_grade"],
        "surface": estimate["surface_grade"],
    }


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No cards scored.")
        return

    print("[summary] mean absolute error, worst-case delta, sample size (per category)")
    for cat in CATEGORIES:
        cat_deltas = [row["deltas"][cat] for row in rows if cat in row["deltas"]]
        if not cat_deltas:
            continue
        mae = sum(abs(d) for d in cat_deltas) / len(cat_deltas)
        worst = max(cat_deltas, key=abs)
        print(f"  {cat:14} MAE={mae:.2f}  worst={worst:+d}  n={len(cat_deltas)}")


def build_fit_rows(rows: list[dict]) -> list[dict]:
    """Complete cases only: an actual overall grade, plus predicted centering/corners_edges/surface."""
    fit_rows = []
    for row in rows:
        predicted, actual = row["predicted"], row["actual"]
        if "overall" not in actual:
            continue
        if any(predicted.get(cat) is None for cat in ("centering", "corners_edges", "surface")):
            continue
        fit_rows.append(
            {
                "centering": predicted["centering"],
                "corners_edges": predicted["corners_edges"],
                "surface": predicted["surface"],
                "overall_actual": actual["overall"],
            }
        )
    return fit_rows


def fit_and_save(rows: list[dict], thresholds: dict, thresholds_path: Path) -> None:
    fit_rows = build_fit_rows(rows)
    print(
        f"\n[fit] {len(fit_rows)} of {len(rows)} cards have complete data for fitting "
        "(need an actual overall grade + centering/corners_edges/surface predictions — the last "
        "usually means --surface photos and a working vision review for every card)"
    )

    if len(fit_rows) < scoring.MIN_CALIBRATION_SAMPLES:
        print(
            f"  not enough data yet — need at least {scoring.MIN_CALIBRATION_SAMPLES}, "
            f"have {len(fit_rows)}. Keeping the existing scoring config."
        )
        return

    # The *previous* scoring config's error on this same set (heuristic, or an
    # earlier fit), for a fair before/after comparison.
    prev_errors = [
        abs(row["predicted"]["overall"] - row["actual"]["overall"])
        for row in rows
        if row["predicted"].get("overall") is not None
        and "overall" in row["actual"]
        and all(row["predicted"].get(cat) is not None for cat in ("centering", "corners_edges", "surface"))
    ]
    prev_mae = sum(prev_errors) / len(prev_errors) if prev_errors else None

    fitted = scoring.fit_linear_weights(fit_rows)
    if fitted is None:
        print("  fit failed unexpectedly — no changes made.")
        return

    print(f"  fitted weights: { {k: round(v, 3) for k, v in fitted['weights'].items()} }")
    print(f"  intercept: {fitted['intercept']:.3f}")
    loo_str = f"  |  leave-one-out MAE: {fitted['loo_mae']:.2f}" if fitted["loo_mae"] is not None else (
        f"  (leave-one-out needs >= {scoring.MIN_SAMPLES_FOR_LOO} samples)"
    )
    print(f"  train MAE: {fitted['train_mae']:.2f}{loo_str}")
    if prev_mae is not None:
        print(f"  previous scoring config's MAE on this same set: {prev_mae:.2f}")
        if fitted["loo_mae"] is not None and fitted["loo_mae"] > prev_mae:
            print(
                "  NOTE: leave-one-out MAE is worse than the previous config — this may be "
                "overfitting at this sample size. Consider gathering more cards before trusting it."
            )

    thresholds["scoring"] = {
        "fitted_weights": {"weights": fitted["weights"], "intercept": fitted["intercept"]},
        "fit_metadata": {
            "n_samples": fitted["n_samples"],
            "train_mae": fitted["train_mae"],
            "loo_mae": fitted["loo_mae"],
            "fitted_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    thresholds_path.write_text(json.dumps(thresholds, indent=2))
    print(f"  saved to {thresholds_path}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    thresholds = detect.load_thresholds(args.thresholds)
    entries = load_manifest(args.manifest)

    rows = []
    for entry in entries:
        card_dir = args.output_dir / entry.name
        try:
            report = grade_card(entry.front, entry.back, thresholds, card_dir, verbose=False)
        except (FileNotFoundError, ValueError) as e:
            print(f"[{entry.name}] FAILED: {e}")
            continue

        predicted = predicted_grades(report)
        deltas = {
            cat: predicted[cat] - entry.actual_grade[cat]
            for cat in entry.actual_grade
            if predicted.get(cat) is not None
        }
        rows.append({"name": entry.name, "predicted": predicted, "actual": entry.actual_grade, "deltas": deltas})

        status = "ok" if report["centering"] is not None else "CAPTURE QUALITY FAILED"
        delta_str = ", ".join(f"{cat} Δ{d:+d}" for cat, d in deltas.items())
        print(f"[{entry.name}] {status}  {delta_str or '(no actual grades to compare)'}")

    print()
    print_summary(rows)

    if args.fit:
        fit_and_save(rows, thresholds, args.thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "calibration_results.json"
    results_path.write_text(json.dumps(rows, indent=2))
    print(f"\nFull results saved to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
