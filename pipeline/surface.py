"""Stage 4: surface — raking-light defect visibility map (scratches, print lines).

Ships as a visualizer only: the combined defect map and original crop are
meant to be handed to a vision model for judgment (Phase 4), since reliably
telling a real scratch/print-line defect apart from holo foil sparkle needs
more context than a fixed pixel threshold can capture. `defect_area_pct`
here is a rough indicator to sanity-check the LLM call against, not a grade.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SurfaceResult:
    defect_area_pct: float
    holo_area_pct: float
    blob_count: int
    defect_map: np.ndarray
    annotated: np.ndarray

    def to_dict(self) -> dict:
        return {
            "defect_area_pct": round(self.defect_area_pct, 3),
            "holo_area_pct": round(self.holo_area_pct, 3),
            "blob_count": self.blob_count,
            "note": "indicative only, not a definitive surface grade — needs vision-model review",
        }


def _normalize_robust(values: np.ndarray, high_percentile: float = 99.5) -> np.ndarray:
    """Scale to 0-255 by clipping to a high percentile rather than the true max.

    A plain min/max normalize is wrecked by a single outlier pixel — and this
    pipeline reliably produces one: the perspective-warp seam at the image's
    physical boundary (sub-pixel corner-detection error) has a far higher
    Laplacian/DoG response than any real surface defect, so a literal max
    would crush every genuine scratch toward zero.
    """
    hi = float(np.percentile(values, high_percentile))
    if hi <= 0:
        return np.zeros_like(values, dtype=np.uint8)
    return (np.clip(values, 0, hi) / hi * 255).astype(np.uint8)


def _defect_visibility_map(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """High-pass (Laplacian) catches scratches; difference-of-Gaussians catches
    print lines. Raking light makes both show up as sharp local intensity
    changes against an otherwise flat surface."""
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=cfg["laplacian_ksize"]))
    lap_norm = _normalize_robust(lap)

    blur1 = cv2.GaussianBlur(gray, (0, 0), cfg["dog_sigma1"])
    blur2 = cv2.GaussianBlur(gray, (0, 0), cfg["dog_sigma2"])
    dog = np.abs(blur1.astype(np.float32) - blur2.astype(np.float32))
    dog_norm = _normalize_robust(dog)

    return cv2.max(lap_norm, dog_norm)


def _local_variance(channel: np.ndarray, window: int) -> np.ndarray:
    ch = channel.astype(np.float32)
    mean = cv2.blur(ch, (window, window))
    mean_sq = cv2.blur(ch * ch, (window, window))
    return mean_sq - mean * mean


def _holo_mask(crop_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Flag likely holo-foil regions so iridescent sparkle doesn't get counted
    as a surface defect.

    The signal isn't "high saturation" — plenty of ordinary Pokemon card
    borders/print colors are just as saturated as foil. What's distinctive
    about holo foil is that its hue/saturation shifts at a small spatial
    scale (that's what "iridescent" means), whereas a printed solid color is
    locally uniform. So both checks here measure *local variance* — of the
    saturation channel, and of grayscale intensity — rather than an absolute
    level.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat_var = _local_variance(hsv[:, :, 1], 9)
    sat_mask = (sat_var >= cfg["holo_saturation_variance_threshold"]).astype(np.uint8) * 255

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    local_var = _local_variance(gray, 9)
    var_mask = (local_var >= cfg["holo_local_variance_threshold"]).astype(np.uint8) * 255

    combined = cv2.bitwise_or(sat_mask, var_mask)
    d = cfg["holo_mask_dilate_px"]
    return cv2.dilate(combined, np.ones((d, d), np.uint8))


def analyze_surface(crop_bgr: np.ndarray, thresholds: dict) -> SurfaceResult:
    cfg = thresholds["surface"]
    m = cfg["physical_edge_margin_px"]
    if m > 0:
        crop_bgr = crop_bgr[m:-m, m:-m]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    visibility = _defect_visibility_map(gray, cfg)
    holo_mask = _holo_mask(crop_bgr, cfg)

    visibility_masked = visibility.copy()
    visibility_masked[holo_mask > 0] = 0

    _, defect_mask = cv2.threshold(visibility_masked, cfg["defect_score_threshold"], 255, cv2.THRESH_BINARY)
    defect_mask = defect_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    min_area = cfg["min_defect_blob_area_px"]
    kept_mask = np.zeros_like(defect_mask)
    blob_count = 0
    defect_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            kept_mask[labels == i] = 255
            blob_count += 1
            defect_area += area

    total_area = gray.shape[0] * gray.shape[1]
    holo_area = int(np.count_nonzero(holo_mask))
    non_holo_area = max(1, total_area - holo_area)
    defect_area_pct = 100.0 * defect_area / non_holo_area
    holo_area_pct = 100.0 * holo_area / total_area

    annotated = crop_bgr.copy()
    holo_contours, _ = cv2.findContours(holo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, holo_contours, -1, (0, 215, 255), 1)  # amber = masked-out holo region
    defect_contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, defect_contours, -1, (0, 0, 255), 1)  # red = flagged defect

    defect_map = cv2.applyColorMap(kept_mask, cv2.COLORMAP_HOT)

    return SurfaceResult(defect_area_pct, holo_area_pct, blob_count, defect_map, annotated)
