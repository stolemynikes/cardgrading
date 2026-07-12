"""Stage 5: grade assembly — combine sub-grades PSA-style into an overall estimate.

PSA doesn't publish an exact formula, but empirically the weakest sub-grade
dominates: a card rarely grades more than about a point above its worst
category. Two ways to get the combining weights:

1. A hand-tuned heuristic (below) — the default until real data exists.
2. Weights fit by ordinary least squares against your own known-grade cards
   (see fit_linear_weights, used by calibration/calibrate.py --fit). Once
   thresholds.json has a "scoring.fitted_weights" entry, assemble_grade uses
   it instead of the heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CALIBRATION_SAMPLES = 6
MIN_SAMPLES_FOR_LOO = 10
FEATURES = ["centering", "corners_edges", "surface", "min_sub_grade"]


@dataclass
class GradeEstimate:
    centering_grade: int | None  # None if borders were unmeasurable (borderless/full-art)
    corners_edges_grade: int
    surface_grade: int | None  # None if vision review didn't run
    overall_grade: float
    overall_grade_rounded: int
    note: str

    def to_dict(self) -> dict:
        return {
            "centering_grade": self.centering_grade,
            "corners_edges_grade": self.corners_edges_grade,
            "surface_grade": self.surface_grade,
            "overall_grade": round(self.overall_grade, 2),
            "overall_grade_rounded": self.overall_grade_rounded,
            "note": self.note,
        }


def _heuristic_overall(sub_grades: list[int]) -> float:
    min_grade = min(sub_grades)
    avg_grade = sum(sub_grades) / len(sub_grades)
    # Weight toward the worst category, but cap how far the average can pull
    # the estimate above it — PSA overall grades rarely exceed the weakest
    # sub-grade by more than about a point.
    overall = 0.7 * min_grade + 0.3 * avg_grade
    return min(overall, min_grade + 1)


def _fitted_overall(fitted: dict, centering: int, corners_edges: int, surface: int) -> float:
    w = fitted["weights"]
    min_sub = min(centering, corners_edges, surface)
    return (
        w["centering"] * centering
        + w["corners_edges"] * corners_edges
        + w["surface"] * surface
        + w["min_sub_grade"] * min_sub
        + fitted["intercept"]
    )


def assemble_grade(
    centering_grade: int | None, corners_edges_grade: int, surface_grade: int | None, thresholds: dict
) -> GradeEstimate:
    fitted = thresholds.get("scoring", {}).get("fitted_weights")

    if fitted is not None and centering_grade is not None and surface_grade is not None:
        overall = _fitted_overall(fitted, centering_grade, corners_edges_grade, surface_grade)
        meta = thresholds["scoring"]["fit_metadata"]
        note = (
            f"overall estimate from weights fit on {meta['n_samples']} calibrated cards "
            f"(train MAE {meta['train_mae']:.2f}"
            + (f", leave-one-out MAE {meta['loo_mae']:.2f})" if meta.get("loo_mae") is not None else ")")
        )
    else:
        sub_grades = [corners_edges_grade]
        missing = []
        if centering_grade is not None:
            sub_grades.append(centering_grade)
        else:
            missing.append("centering (borders unmeasurable — borderless/full-art card or low-contrast capture)")
        if surface_grade is not None:
            sub_grades.append(surface_grade)
        else:
            missing.append("surface (no vision review)")
        if missing:
            note = f"overall estimate excludes {'; '.join(missing)} — less reliable"
        else:
            note = "all sub-grades included; overall estimate is indicative, not definitive"
        overall = _heuristic_overall(sub_grades)

    overall = max(1.0, min(10.0, overall))

    return GradeEstimate(
        centering_grade=centering_grade,
        corners_edges_grade=corners_edges_grade,
        surface_grade=surface_grade,
        overall_grade=overall,
        overall_grade_rounded=round(overall),
        note=note,
    )


def fit_linear_weights(rows: list[dict]) -> dict | None:
    """Ordinary least squares fit of overall grade from sub-grades + their minimum.

    Each row must have "centering", "corners_edges", "surface", and
    "overall_actual" (all numeric) — filter out incomplete cases before
    calling this. Returns None if there isn't enough data to fit reliably;
    the caller should keep using the hand-tuned heuristic in that case.

    Includes min(sub_grades) as an explicit feature so a linear model can
    still capture some of the "weakest category dominates" nonlinearity that
    a plain weighted average can't express.
    """
    if len(rows) < MIN_CALIBRATION_SAMPLES:
        return None

    design = np.array(
        [
            [r["centering"], r["corners_edges"], r["surface"], min(r["centering"], r["corners_edges"], r["surface"]), 1.0]
            for r in rows
        ],
        dtype=float,
    )
    targets = np.array([r["overall_actual"] for r in rows], dtype=float)

    coeffs, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    predictions = design @ coeffs
    train_mae = float(np.mean(np.abs(predictions - targets)))

    loo_mae = _leave_one_out_mae(design, targets) if len(rows) >= MIN_SAMPLES_FOR_LOO else None

    return {
        "weights": {name: float(coeffs[i]) for i, name in enumerate(FEATURES)},
        "intercept": float(coeffs[4]),
        "n_samples": len(rows),
        "train_mae": train_mae,
        "loo_mae": loo_mae,
    }


def _leave_one_out_mae(design: np.ndarray, targets: np.ndarray) -> float:
    """Refit with each sample held out in turn — a much more honest error
    estimate than in-sample MAE, which overfits badly at small N."""
    n = len(targets)
    errors = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coeffs, _, _, _ = np.linalg.lstsq(design[mask], targets[mask], rcond=None)
        errors.append(abs(design[i] @ coeffs - targets[i]))
    return float(np.mean(errors))
