"""Stage 3: corners & edges — whitening detection.

A CLAHE + blue-channel + adaptive-threshold stack finds locally bright
structure, and an absolute lightness/saturation gate decides whether that
structure is actually exposed cardstock. Both are needed: local contrast
alone fires on scan noise across flat cardstock, and an absolute test alone
misses the faint edges of a real chip.

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
    # False when the crop isn't uniform border — full-art cards, or a crop
    # sized from a guessed border width that ran into artwork. The numbers
    # above are then measuring paint, not cardstock, and must not be graded.
    measurable: bool = True
    uniformity: float = 1.0
    # Why it was refused, in the report's own words. "n/a" on its own reads
    # the same whether the card has no border to measure or the capture was
    # too dim to find one, and those want different things done about them.
    reason: str | None = None
    # Where this region sits on the card, [x, y, w, h] in 0..1 of the
    # canonical warp — so the report can draw it back onto the card rather
    # than only showing a crop with no sense of place.
    box: list[float] | None = None
    # How much of this region is physically deformed, read from the
    # photometric relief. None when there was no relief to read — a
    # single-image capture, or a side with no rotation scans.
    relief_wear_pct: float | None = None

    def to_dict(self) -> dict:
        return {
            "whitening_pct": round(self.whitening_pct, 3),
            "relief_wear_pct": None if self.relief_wear_pct is None else round(self.relief_wear_pct, 3),
            "blob_count": self.blob_count,
            # A refused region has no grade. It used to serialize the grade it
            # would have had, gated only by `measurable` — which reads as a
            # perfect 10 to anything consuming the raw report without knowing
            # to check the flag alongside it.
            "grade": self.grade if self.measurable else None,
            "measurable": self.measurable,
            "uniformity": round(self.uniformity, 3),
            "reason": None if self.measurable else self.reason,
            "box": self.box,
        }


@dataclass
class SideResult:
    corners: dict[str, RegionResult] = field(default_factory=dict)
    edges: dict[str, RegionResult] = field(default_factory=dict)
    grade: int | None = 10

    # Corners and edges are measured with the same filter stack and combined
    # into one sub-grade for scoring, but they are separate attributes to a
    # grader's eye — a card with four clean corners and one chipped edge
    # tells a different story than the reverse. Reported separately so the
    # report can break the sub-grade out per attribute.
    @property
    def corners_grade(self) -> int | None:
        return min((r.grade for r in self.corners.values() if r.measurable), default=None)

    @property
    def edges_grade(self) -> int | None:
        return min((r.grade for r in self.edges.values() if r.measurable), default=None)

    def _refusal(self, regions: dict) -> str | None:
        """Why this group has no grade — the reason its regions gave."""
        if not regions or any(r.measurable for r in regions.values()):
            return None
        reasons = [r.reason for r in regions.values() if r.reason]
        return reasons[0] if reasons else "no region on this side could be measured"

    def to_dict(self) -> dict:
        return {
            "corners": {k: v.to_dict() for k, v in self.corners.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "corners_grade": self.corners_grade,
            "edges_grade": self.edges_grade,
            "corners_reason": self._refusal(self.corners),
            "edges_reason": self._refusal(self.edges),
            "grade": self.grade,
        }


@dataclass
class CornersEdgesResult:
    front: SideResult
    back: SideResult
    overall_grade: int | None  # None when neither side had a measurable border

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


def region_boxes(sizes: dict, shape: tuple, edge_margin: int) -> dict[str, list[float]]:
    """Where each crop sits on the card, as fractions of the canonical warp.

    Mirrors `_crop_regions` exactly — same slices, expressed as [x, y, w, h]
    in 0..1 — so a report can draw each region back onto the card it came
    from. Fractions rather than pixels because the report renders the card at
    whatever size the page gives it.
    """
    h, w = shape[:2]
    cs, es = sizes["corners"], sizes["edges"]
    m = edge_margin

    def box(x0, y0, x1, y1):
        return [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]

    return {
        "corner_top_left": box(m, m, cs["top_left"], cs["top_left"]),
        "corner_top_right": box(w - cs["top_right"], m, w - m, cs["top_right"]),
        "corner_bottom_right": box(w - cs["bottom_right"], h - cs["bottom_right"], w - m, h - m),
        "corner_bottom_left": box(m, h - cs["bottom_left"], cs["bottom_left"], h - m),
        "edge_top": box(cs["top_left"], m, w - cs["top_right"], es["top"]),
        "edge_right": box(w - es["right"], cs["top_right"], w - m, h - cs["bottom_right"]),
        "edge_bottom": box(cs["bottom_left"], h - es["bottom"], w - cs["bottom_right"], h - m),
        "edge_left": box(m, cs["top_left"], es["left"], h - cs["bottom_left"]),
    }


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


def _absolute_whitening_gate(crop_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Require candidate pixels to actually look like exposed cardstock.

    Whitening is a physical thing: the printed layer has been rubbed through
    and the white core underneath shows. That core is both *lighter* and
    *less saturated* than the border around it, whatever colour that border
    is — which makes those two properties, measured against the crop's own
    median, a card-agnostic test.

    Without this gate the stage was adaptive-threshold only, i.e. "brighter
    than its neighbours" — and on flat cardstock carrying ordinary scan noise
    that is true nearly everywhere. Measured on a real Base Set scan it
    reported 25-36% whitening on all sixteen regions of a card a grading
    service had passed with zero corner defects. Synthetic test cards never
    caught it because their borders are noiseless gradients.

    A genuinely white-bordered card gates almost everything out, which is the
    honest result: whitening on a white border isn't visible to this method
    either.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)

    median_saturation = float(np.median(saturation))
    median_value = float(np.median(value))

    # Desaturation is the load-bearing test. Requiring the pixel to be
    # markedly *lighter* than the border fails on light borders — a Base Set
    # yellow already sits around value 230 and exposed cardstock only reaches
    # ~244, so a "much brighter" rule rejects every real chip on the front of
    # the card. What always holds is that the core is far less saturated than
    # the ink over it, and never darker than it.
    duller = saturation <= median_saturation - cfg["min_saturation_below_median"]
    not_darker = value >= median_value - cfg["max_value_below_median"]
    return ((duller & not_darker).astype(np.uint8)) * 255


def _whitening_mask(visibility_map: np.ndarray, crop_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    block_size = cfg["adaptive_block_size"] | 1  # adaptiveThreshold requires an odd block size
    # Negative C flips "threshold = mean - C" into "mean + margin", i.e. flag
    # pixels that are locally brighter than their neighborhood, not dimmer.
    local = cv2.adaptiveThreshold(
        visibility_map,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        -cfg["adaptive_c"],
    )
    # Local contrast finds the edges of a chip; the absolute gate decides
    # whether there is a chip there at all. Both have to agree.
    return cv2.bitwise_and(local, _absolute_whitening_gate(crop_bgr, cfg))


def _grade_from_pct(pct: float, grade_bands: list[dict]) -> int:
    """Walk a band table. Two tables use this — whitening and relief wear —
    so the ceiling key is read under either name rather than the tables being
    forced to share one that is wrong for one of them."""
    for tier in grade_bands:
        ceiling = tier.get("max_whitening_pct", tier.get("max_pct"))
        if ceiling is not None and pct <= ceiling:
            return tier["grade"]
    return max(1, grade_bands[-1]["grade"] - 2)



def _border_uniformity(crop_bgr: np.ndarray) -> float:
    """How close this crop is to a single flat colour.

    A printed border is near-uniform; artwork is not. Returns the fraction of
    pixels within a modest distance of the crop's own median colour, so it is
    scale- and hue-independent.

    This is the guard centering has had all along and this stage did not. On
    a full-art card there is no border to crop, so `grade.py` falls back to a
    4%-of-width guess and the window lands on artwork — pale, desaturated
    paint that is indistinguishable from exposed cardstock to the whitening
    test. Measured on a real full-art card the stage reported four dinged
    corners on a side the grading service passed with zero.
    """
    if crop_bgr.size == 0:
        return 0.0
    # Measured tile by tile, not over the whole crop. Against one global
    # median a long strip is penalised for its length: a border that changes
    # colour along the side is locally as flat as a corner square, but half
    # of it sits far from the strip's single median. Measured on a real card
    # with a patterned border, the four edge strips scored 0.36-0.39 whole
    # and 0.60-0.71 tiled — refused by a 0.55 floor purely for being long,
    # while the corner squares cut from the same border passed at 0.63-0.72.
    #
    # The median tile is the summary rather than the mean, so one tile
    # landing on a logo or a foil stamp doesn't condemn the side.
    tiles = [_local_uniformity(tile) for tile in _square_tiles(crop_bgr)]
    return float(np.median(tiles)) if tiles else 0.0


def _local_uniformity(crop_bgr: np.ndarray) -> float:
    flat = crop_bgr.reshape(-1, 3).astype(np.int16)
    if flat.size == 0:
        return 0.0
    median = np.median(flat, axis=0)
    distance = np.abs(flat - median).sum(axis=1)
    return float(np.mean(distance < 60))


def _square_tiles(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """Cut a crop into roughly square tiles along its longer axis.

    Square, because the corner squares are the shape this gate was calibrated
    on and they behave; a strip is just a row of them laid end to end.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return []
    side = min(h, w)
    count = max(1, round(max(h, w) / side))
    if count == 1:
        return [crop_bgr]
    step = max(h, w) // count
    tiles = []
    for i in range(count):
        start = i * step
        end = max(h, w) if i == count - 1 else start + step
        tiles.append(crop_bgr[:, start:end] if w > h else crop_bgr[start:end, :])
    return [t for t in tiles if t.size]


def relief_wear_pct(relief_crop: np.ndarray | None, cfg: dict) -> float | None:
    """How much of this corner or edge is physically deformed.

    Read from the photometric relief, which is shape with the albedo already
    divided out — so unlike the whitening map it does not care what colour the
    border is, and does not need the border to be uniform.

    Both of those matter. Whitening is found as a local brightness spike
    against a darker border, which works on a classic dark-bordered card and
    finds nothing at all on a silver-bordered modern one: measured on a card
    whitened deliberately along two edges, the whitening map read 0.0% on
    every region of both the clean and the damaged capture. The same regions
    read 0.001% clean against 1.209% damaged in relief.
    """
    if relief_crop is None or relief_crop.size == 0:
        return None
    deviation = np.abs(relief_crop.astype(np.int16) - 128)
    threshold = cfg.get("relief_wear_threshold", 26)
    return float(100.0 * (deviation > threshold).mean())


def _relief_wear_bands(cfg: dict, is_corner: bool) -> list | None:
    """The band table for this kind of region.

    Corners and edges get their own, because the perspective-warp seam leaves
    false relief around the card's boundary and a corner crop — small, and
    touching two seams — carries far more of it than a long edge crop does.
    """
    bands = cfg.get("relief_wear_bands")
    if isinstance(bands, dict):
        return bands.get("corners" if is_corner else "edges")
    return bands


def analyze_region(
    name: str,
    crop_bgr: np.ndarray,
    cfg: dict,
    relief_crop: np.ndarray | None = None,
    is_corner: bool = True,
) -> tuple[RegionResult, np.ndarray]:
    """Score one corner/edge crop. Returns the metric plus the kept blob mask for debug overlay."""
    uniformity = _border_uniformity(crop_bgr)
    wear_pct = relief_wear_pct(relief_crop, cfg)
    bands = _relief_wear_bands(cfg, is_corner)
    wear_grade = _grade_from_pct(wear_pct, bands) if wear_pct is not None and bands else None

    if uniformity < cfg.get("min_border_uniformity", 0.55):
        floor = cfg.get("min_border_uniformity", 0.55)
        if wear_grade is not None:
            # Relief needs no border at all, so a crop the whitening map can't
            # read is still measurable here. This is not a nicety: the gate
            # was being tripped *by the damage* — a deliberately whitened edge
            # pushed its own uniformity from above the floor to 0.54 and was
            # refused, so the card's worst edge reported "can't measure"
            # instead of a bad grade.
            return (
                RegionResult(
                    name, 0.0, 0, wear_grade, uniformity=uniformity, relief_wear_pct=wear_pct,
                    reason=(
                        f"graded from surface relief only — this crop isn't uniform border "
                        f"({uniformity:.2f} against a {floor:.2f} floor), so whitening measured "
                        "off it would be measured off artwork"
                    ),
                ),
                np.zeros(crop_bgr.shape[:2], np.uint8),
            )
        # Refused rather than graded: reporting a number measured off artwork
        # is worse than reporting that this region can't be measured from
        # this capture.
        return (
            RegionResult(
                name,
                0.0,
                0,
                10,
                measurable=False,
                uniformity=uniformity,
                reason=(
                    f"this crop isn't uniform border ({uniformity:.2f} against a {floor:.2f} floor) — "
                    "a borderless or full-art card, or a crop sized from a border the detector "
                    "couldn't find. Whitening measured off artwork isn't a measurement of the card."
                ),
            ),
            np.zeros(crop_bgr.shape[:2], np.uint8),
        )

    visibility = _whitening_visibility_map(crop_bgr, cfg)
    mask = _whitening_mask(visibility, crop_bgr, cfg)

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
    grade = _grade_from_pct(whitening_pct, cfg["grade_bands"])
    # Two independent readings of the same corner, and the worse one wins.
    # They fail in opposite directions: whitening is blind to a light border,
    # relief is blind to a stain that hasn't deformed anything.
    if wear_grade is not None:
        grade = min(grade, wear_grade)
    return (
        RegionResult(name, whitening_pct, blob_count, grade, uniformity=uniformity, relief_wear_pct=wear_pct),
        kept_mask,
    )


def draw_region_overlay(crop_bgr: np.ndarray, mask: np.ndarray, region: RegionResult) -> np.ndarray:
    """Outline the detected whitening on the crop.

    Deliberately unlabelled. The region name and its numbers used to be
    burned into the pixels, which was fine when these were debug files in an
    output directory — but the report now shows these crops as the evidence
    behind each DINGS entry, at thumbnail size and again full-screen, and
    baked-in text can't be sized, translated, or read by a screen reader.
    The caller has `region` and renders those figures as real text.
    """
    overlay = crop_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    return overlay


def analyze_side(
    image: np.ndarray, borders: BorderWidths, cfg: dict, relief: np.ndarray | None = None
) -> tuple[SideResult, dict[str, np.ndarray]]:
    """Analyze all 4 corners + 4 edges of one side. Returns the result plus debug overlay crops.

    `relief` is the measured Card Vision render for this side, when there is
    one. Cropped identically to the image, it gives every region a second,
    colour-blind reading — see `relief_wear_pct`.
    """
    sizes = _region_sizes(borders, cfg)
    corner_crops, edge_crops = _crop_regions(image, sizes, cfg["physical_edge_margin_px"])
    relief_corners, relief_edges = (
        _crop_regions(relief, sizes, cfg["physical_edge_margin_px"])
        if relief is not None and relief.shape[:2] == image.shape[:2]
        else ({}, {})
    )
    boxes = region_boxes(sizes, image.shape, cfg["physical_edge_margin_px"])
    overlays: dict[str, np.ndarray] = {}

    corner_results: dict[str, RegionResult] = {}
    for name, crop in corner_crops.items():
        result, mask = analyze_region(name, crop, cfg, relief_corners.get(name), is_corner=True)
        result.box = boxes.get(f"corner_{name}")
        corner_results[name] = result
        overlays[f"corner_{name}"] = draw_region_overlay(crop, mask, result)

    edge_results: dict[str, RegionResult] = {}
    for name, crop in edge_crops.items():
        result, mask = analyze_region(name, crop, cfg, relief_edges.get(name), is_corner=False)
        result.box = boxes.get(f"edge_{name}")
        edge_results[name] = result
        overlays[f"edge_{name}"] = draw_region_overlay(crop, mask, result)

    # A refused region contributes nothing. If every region on this side was
    # refused (a full-art card captured without a measurable border), the
    # side has no corners/edges grade at all rather than a default 10.
    measured = [r for r in [*corner_results.values(), *edge_results.values()] if r.measurable]
    grade = min((r.grade for r in measured), default=None)
    return SideResult(corner_results, edge_results, grade), overlays


def analyze_corners_edges(
    front: np.ndarray,
    back: np.ndarray,
    front_borders: BorderWidths,
    back_borders: BorderWidths,
    thresholds: dict,
    front_relief: np.ndarray | None = None,
    back_relief: np.ndarray | None = None,
) -> tuple[CornersEdgesResult, dict[str, dict[str, np.ndarray]]]:
    cfg = thresholds["corners_edges"]
    front_result, front_overlays = analyze_side(front, front_borders, cfg, front_relief)
    back_result, back_overlays = analyze_side(back, back_borders, cfg, back_relief)
    # None means "refused", not "perfect" — a side with nothing measurable
    # must drop out of the combination rather than pull it toward 10.
    sides = [g for g in (front_result.grade, back_result.grade) if g is not None]
    overall_grade = min(sides) if sides else None
    result = CornersEdgesResult(front_result, back_result, overall_grade)
    return result, {"front": front_overlays, "back": back_overlays}
