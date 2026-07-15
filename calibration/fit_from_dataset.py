#!/usr/bin/env python3
"""Fit grade-assembly weights from a public dataset of graded-card photos.

Dataset: pacoalberola/Poke-Grader-Dataset-Images-PSA on HuggingFace — 3,156
eBay/PWCC listing photos of slabbed graded Pokemon cards. The ONLY trusted
label per image is the professional overall grade (the CSV's per-attribute
sub-scores are the dataset author's own model outputs: 0.1-precision values
even for graders whose real labels are 0.5-quantized — synthetic). BGS/CGC
grades are normalized to the PSA scale via the dataset's psa_equivalent
column.

Per image, this script produces OUR OWN sub-estimates and pairs them with the
real overall grade:
- card extracted from the slab photo with the standard detection stage; images
  failing the geometry gates are skipped (slab-not-card detections),
- centering measured by the standard pixel stage on the extracted card,
- corners / edges / surface estimated by the vision flat judgment
  (llm/vision.judge_flat — Gemini free tier works; ~1 call per usable card).

Rows are appended to a JSON file as they complete (interruptible/resumable),
then fed to scoring.fit_linear_weights with leave-one-out validation.

Usage:
    GEMINI_API_KEY=... python calibration/fit_from_dataset.py --rows 150 --fit

The run paces itself to ~9 requests/minute for the vision free tier.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm import vision  # noqa: E402
from pipeline import centering, detect, scoring  # noqa: E402

DATASET = "pacoalberola/Poke-Grader-Dataset-Images-PSA"
CSV_URL = f"https://huggingface.co/datasets/pacoalberola/grades_psa_equivalent/resolve/main/grades.csv"
IMAGE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/images/{{filename}}"
VISION_PACING_SECONDS = 6.5  # ~9/min, under the 10 RPM free-tier ceiling


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=150, help="Target number of complete fit rows")
    p.add_argument("--attempts-per-band", type=int, default=60, help="Max candidates tried per whole-grade band")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "calibration" / "dataset_cache")
    p.add_argument("--rows-file", type=Path, default=REPO_ROOT / "calibration" / "dataset_fit_rows.json")
    p.add_argument("--thresholds", type=Path, default=REPO_ROOT / "calibration" / "thresholds.json")
    p.add_argument("--fit", action="store_true", help="Fit and save weights to thresholds.json when done")
    p.add_argument("--measure-only", action="store_true", help="Skip vision calls (geometry/centering stats only)")
    return p.parse_args(argv)


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "card-pre-grader-calibration"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        print(f"    download failed: {e}", flush=True)
        return False
    if len(data) < 1000:  # HF returns tiny "Not Found" bodies for missing files
        return False
    dest.write_bytes(data)
    return True


def load_candidates(cache: Path) -> list[dict]:
    csv_path = cache / "grades.csv"
    if not download(CSV_URL, csv_path):
        raise SystemExit("could not download the dataset CSV")
    rows = list(csv.DictReader(open(csv_path)))
    usable = [r for r in rows if r.get("psa_equivalent")]
    print(f"dataset: {len(usable)} labeled images", flush=True)
    return usable


def stratified_order(candidates: list[dict], per_band: int) -> list[dict]:
    """Round-robin across whole-grade bands so every grade contributes."""
    bands: dict[int, list[dict]] = defaultdict(list)
    rng = random.Random(42)
    for r in candidates:
        bands[round(float(r["psa_equivalent"]))].append(r)
    for band in bands.values():
        rng.shuffle(band)
    order = []
    for i in range(per_band):
        for grade in sorted(bands):
            if i < len(bands[grade]):
                order.append(bands[grade][i])
    return order


def measure_card(image_path: Path, thresholds: dict, cache: Path) -> tuple[Path, int] | None:
    """Extract the card and measure centering. None = unusable image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    result = detect.detect_and_normalize(img, thresholds)
    if result.warped is None or result.hard_failures:
        return None
    cent = centering.measure_centering(result.warped, result.warped, thresholds)
    if cent.front_grade is None:
        return None
    warp_path = cache / f"{image_path.stem}.warp.png"
    cv2.imwrite(str(warp_path), result.warped)
    return warp_path, cent.front_grade


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    thresholds = detect.load_thresholds(args.thresholds)

    rows: list[dict] = []
    done_files: set[str] = set()
    if args.rows_file.exists():
        rows = json.loads(args.rows_file.read_text())
        done_files = {r["filename"] for r in rows}
        print(f"resuming: {len(rows)} rows already collected", flush=True)

    candidates = stratified_order(load_candidates(args.cache), args.attempts_per_band)
    stats = defaultdict(int)

    for cand in candidates:
        if len(rows) >= args.rows:
            break
        fname = cand["filename"]
        if fname in done_files:
            continue
        stats["attempted"] += 1
        label = float(cand["psa_equivalent"])
        img_path = args.cache / fname
        if not download(IMAGE_URL.format(filename=fname), img_path):
            stats["download_failed"] += 1
            continue

        measured = measure_card(img_path, thresholds, args.cache)
        if measured is None:
            stats["geometry_or_centering_failed"] += 1
            print(f"  [{len(rows)}/{args.rows}] {fname} (psa {label}): skipped (detect/centering)", flush=True)
            continue
        warp_path, centering_grade = measured

        if args.measure_only:
            stats["measured"] += 1
            print(f"  [{len(rows)}/{args.rows}] {fname} (psa {label}): centering g{centering_grade} (measure-only)", flush=True)
            continue

        try:
            judgment, model = vision.judge_flat(warp_path, "front")
        except vision.VisionUnavailable as e:
            msg = str(e)
            print(f"  vision unavailable: {msg[:160]}", flush=True)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("STOPPING: vision quota exhausted — rerun later to resume.", flush=True)
                break
            stats["vision_failed"] += 1
            continue
        finally:
            time.sleep(VISION_PACING_SECONDS)

        row = {
            "filename": fname,
            "source": cand.get("source", ""),
            "centering": centering_grade,
            "corners_edges": min(judgment.corners_grade, judgment.edges_grade),
            "surface": judgment.surface_grade,
            "overall_actual": label,
            "vision_model": model,
        }
        rows.append(row)
        done_files.add(fname)
        args.rows_file.write_text(json.dumps(rows, indent=1))
        print(
            f"  [{len(rows)}/{args.rows}] {fname} (psa {label}): centering g{centering_grade} "
            f"corners/edges g{row['corners_edges']} surface g{row['surface']}",
            flush=True,
        )

    print(f"\ncollection done: {dict(stats)}  rows={len(rows)}", flush=True)

    if args.fit and rows:
        fit_rows = [
            {k: r[k] for k in ("centering", "corners_edges", "surface", "overall_actual")} for r in rows
        ]
        fitted = scoring.fit_linear_weights(fit_rows)
        if fitted is None:
            print("not enough rows to fit")
            return 1
        print(f"fitted weights: { {k: round(v, 3) for k, v in fitted['weights'].items()} }")
        print(f"intercept: {fitted['intercept']:.3f}")
        print(f"train MAE: {fitted['train_mae']:.2f}  LOO MAE: {fitted['loo_mae']}")
        thresholds["scoring"] = {
            "fitted_weights": {"weights": fitted["weights"], "intercept": fitted["intercept"]},
            "fit_metadata": {
                "n_samples": fitted["n_samples"],
                "train_mae": fitted["train_mae"],
                "loo_mae": fitted["loo_mae"],
                "fitted_at": datetime.now(timezone.utc).isoformat(),
                "source": f"huggingface:{DATASET} (real overall grades; our centering + vision corners/edges/surface)",
            },
        }
        args.thresholds.write_text(json.dumps(thresholds, indent=2))
        print(f"saved to {args.thresholds}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
