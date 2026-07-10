"""Stage 3: corners & edges — whitening detection via a CLAHE + blue-channel
isolation + adaptive-threshold filter stack.

Operates on the same perspective-corrected canonical images produced by
pipeline.detect (front/back), so no extra capture is needed for this stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RegionResult:
    name: str
    whitening_pct: float
    blob_count: int
    grade: int

    def to_dict(self) -> dict:
        return {
            "whitening_pct": round(self.whitening_pct, 3),
            "blob_count": self.blob_count,
            "grade": self.grade,
        }


@dataclass
class SideResult:
    corners: dict[str, RegionResult] = field(default_factory=dict)
    edges: dict[str, RegionResult] = field(default_factory=dict)
    grade: int = 10

    def to_dict(self) -> dict:
        return {
            "corners": {k: v.to_dict() for k, v in self.corners.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "grade": self.grade,
        }


@dataclass
class CornersEdgesResult:
    front: SideResult
    back: SideResult
    overall_grade: int

    def to_dict(self) -> dict:
        return {
            "front": self.front.to_dict(),
            "back": self.back.to_dict(),
            "overall_grade": self.overall_grade,
        }


@dataclass
class BorderWidths:
    left: float
    right: float
    top: float
    bottom: float


def _region_sizes(borders: BorderWidths, cfg: dict) -> dict:
    """Size each corner/edge crop from the card's own measured border widths.

    A fixed pixel crop size would routinely cross from the border into the
    inner artwork/text panel on cards with a narrower border (or a side
    that's already off-center), and that border/panel color transition
    itself then gets misread as a whitening defect. Sizing from the
    per-card border measurement (done in pipeline.centering) keeps every
    crop inside the border, whatever its width.
    """
    margin = cfg["border_margin_frac"]
    min_px = cfg["min_region_px"]
    max_corner = cfg["corner_crop_px"]
    max_edge = cfg["edge_strip_px"]

    def clamp(value: float, max_px: int) -> int:
        return int(max(min_px, min(value * margin, max_px)))

    return {
        "corners": {
            "top_left": clamp(min(borders.left, borders.top), max_corner),
            "top_right": clamp(min(borders.right, borders.top), max_corner),
            "bottom_right": clamp(min(borders.right, borders.bottom), max_corner),
            "bottom_left": clamp(min(borders.left, borders.bottom), max_corner),
        },
        "edges": {
            "top": clamp(borders.top, max_edge),
            "right": clamp(borders.right, max_edge),
            "bottom": clamp(borders.bottom, max_edge),
            "left": clamp(borders.left, max_edge),
        },
    }


def _crop_regions(image: np.ndarray, sizes: dict, edge_margin: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Crop the corner squares and edge strips.

    Each crop is inset by `edge_margin` on whichever side(s) touch the
    physical card boundary. Perspective correction can't localize the card's
    true corners to sub-pixel precision, so the outermost couple of pixels
    are a blended interpolation seam rather than card content — left
    un-trimmed, that seam reads as a solid line of "whitening" on every
    photo, real defect or not.
    """
    h, w = image.shape[:2]
    cs, es = sizes["corners"], sizes["edges"]
    m = edge_margin

    corners = {
        "top_left": image[m:cs["top_left"], m:cs["top_left"]],
        "top_right": image[m:cs["top_right"], w - cs["top_right"]:w - m],
        "bottom_right": image[h - cs["bottom_right"]:h - m, w - cs["bottom_right"]:w - m],
        "bottom_left": image[h - cs["bottom_left"]:h - m, m:cs["bottom_left"]],
    }
    # Edge strips run between their two adjoining corners so whitening pixels
    # never get double-counted in both a corner crop and an edge crop.
    edges = {
        "top": image[m:es["top"], cs["top_left"]:w - cs["top_right"]],
        "right": image[cs["top_right"]:h - cs["bottom_right"], w - es["right"]:w - m],
        "bottom": image[h - es["bottom"]:h - m, cs["bottom_left"]:w - cs["bottom_right"]],
        "left": image[cs["top_left"]:h - cs["bottom_left"], m:es["left"]],
    }
    return corners, edges


def _whitening_visibility_map(crop_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Combine contrast-boosted grayscale with blue-channel isolation.

    Whitening (exposed white cardstock from a chip or crease) shows up as a
    local brightness spike. Isolating the blue channel gives the strongest
    contrast against a dark-blue Pokemon card back; CLAHE on plain grayscale
    keeps the same filter useful on lighter/more varied front borders.
    """
    tile = cfg["clahe_tile_grid"]
    clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip_limit"], tileGridSize=(tile, tile))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_enhanced = clahe.apply(gray)
    blue_enhanced = clahe.apply(crop_bgr[:, :, 0])
    return cv2.addWeighted(gray_enhanced, 0.5, blue_enhanced, 0.5, 0)


def _whitening_mask(visibility_map: np.ndarray, cfg: dict) -> np.ndarray:
    block_size = cfg["adaptive_block_size"] | 1  # adaptiveThreshold requires an odd block size
    # Negative C flips "threshold = mean - C" into "mean + margin", i.e. flag
    # pixels that are locally brighter than their neighborhood, not dimmer.
    return cv2.adaptiveThreshold(
        visibility_map,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        -cfg["adaptive_c"],
    )


def _grade_from_whitening(pct: float, grade_bands: list[dict]) -> int:
    for tier in grade_bands:
        if pct <= tier["max_whitening_pct"]:
            return tier["grade"]
    return max(1, grade_bands[-1]["grade"] - 2)


def analyze_region(name: str, crop_bgr: np.ndarray, cfg: dict) -> tuple[RegionResult, np.ndarray]:
    """Score one corner/edge crop. Returns the metric plus the kept blob mask for debug overlay."""
    visibility = _whitening_visibility_map(crop_bgr, cfg)
    mask = _whitening_mask(visibility, cfg)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = cfg["min_blob_area_px"]
    kept_mask = np.zeros_like(mask)
    blob_count = 0
    whitening_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            kept_mask[labels == i] = 255
            blob_count += 1
            whitening_area += area

    total_area = mask.shape[0] * mask.shape[1]
    whitening_pct = 100.0 * whitening_area / total_area if total_area else 0.0
    grade = _grade_from_whitening(whitening_pct, cfg["grade_bands"])
    return RegionResult(name, whitening_pct, blob_count, grade), kept_mask


def draw_region_overlay(crop_bgr: np.ndarray, mask: np.ndarray, region: RegionResult) -> np.ndarray:
    overlay = crop_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    label = f"{region.name}: {region.whitening_pct:.2f}% g{region.grade}"
    cv2.putText(overlay, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return overlay


def analyze_side(image: np.ndarray, borders: BorderWidths, cfg: dict) -> tuple[SideResult, dict[str, np.ndarray]]:
    """Analyze all 4 corners + 4 edges of one side. Returns the result plus debug overlay crops."""
    sizes = _region_sizes(borders, cfg)
    corner_crops, edge_crops = _crop_regions(image, sizes, cfg["physical_edge_margin_px"])
    overlays: dict[str, np.ndarray] = {}

    corner_results: dict[str, RegionResult] = {}
    for name, crop in corner_crops.items():
        result, mask = analyze_region(name, crop, cfg)
        corner_results[name] = result
        overlays[f"corner_{name}"] = draw_region_overlay(crop, mask, result)

    edge_results: dict[str, RegionResult] = {}
    for name, crop in edge_crops.items():
        result, mask = analyze_region(name, crop, cfg)
        edge_results[name] = result
        overlays[f"edge_{name}"] = draw_region_overlay(crop, mask, result)

    grade = min(r.grade for r in [*corner_results.values(), *edge_results.values()])
    return SideResult(corner_results, edge_results, grade), overlays


def analyze_corners_edges(
    front: np.ndarray,
    back: np.ndarray,
    front_borders: BorderWidths,
    back_borders: BorderWidths,
    thresholds: dict,
) -> tuple[CornersEdgesResult, dict[str, dict[str, np.ndarray]]]:
    cfg = thresholds["corners_edges"]
    front_result, front_overlays = analyze_side(front, front_borders, cfg)
    back_result, back_overlays = analyze_side(back, back_borders, cfg)
    overall_grade = min(front_result.grade, back_result.grade)
    result = CornersEdgesResult(front_result, back_result, overall_grade)
    return result, {"front": front_overlays, "back": back_overlays}
