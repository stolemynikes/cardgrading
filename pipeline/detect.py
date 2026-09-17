"""Stage 1: card contour detection, perspective correction, capture quality gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# Gates whose failure means the detected geometry itself is wrong — the warp
# isn't a usable image of the card, so downstream measurements would be
# meaningless, not merely less reliable. These block grading (retake). The
# remaining gates (resolution, glare, uneven_lighting) are image-quality
# warnings: grading proceeds and the caller surfaces them as caveats.
# Validated against real misdetections: a skewed quad shows up as tilt, and a
# wrong-but-rectangular quad (the minAreaRect fallback around card+background,
# where tilt is 0 by construction) shows up as aspect_ratio.
HARD_GATE_NAMES = frozenset({"card_detection", "tilt", "aspect_ratio"})


@dataclass
class QualityGate:
    name: str
    passed: bool
    detail: str
    value: float | None = None

    def __post_init__(self) -> None:
        # OpenCV/numpy ops hand back np.bool_/np.float64, which json.dumps rejects.
        self.passed = bool(self.passed)
        if self.value is not None:
            self.value = float(self.value)

    @property
    def hard(self) -> bool:
        return self.name in HARD_GATE_NAMES


@dataclass
class DetectResult:
    ok: bool
    warped: np.ndarray | None
    gates: list[QualityGate]
    contour: np.ndarray | None = None

    @property
    def failures(self) -> list[QualityGate]:
        return [g for g in self.gates if not g.passed]

    @property
    def hard_failures(self) -> list[QualityGate]:
        return [g for g in self.gates if not g.passed and g.hard]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail, "value": g.value, "hard": g.hard}
                for g in self.gates
            ],
        }


def load_thresholds(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _estimate_background_color(image: np.ndarray, patch: int = 40) -> np.ndarray:
    """Sample the four image corners, which are assumed to be background (matte, not card)."""
    h, w = image.shape[:2]
    patch = min(patch, h // 4, w // 4)
    corners = [
        image[0:patch, 0:patch],
        image[0:patch, w - patch:w],
        image[h - patch:h, 0:patch],
        image[h - patch:h, w - patch:w],
    ]
    samples = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return samples.mean(axis=0)


# Candidate-quad generation and scoring, tuned against real misdetections:
# on a busy/textured background, Otsu picks a threshold low enough that the
# card merges with background patches into one giant blob — the old
# "largest contour wins" rule then boxed card+background together. Sweeping
# a few higher thresholds re-isolates the card, and scoring every candidate
# by card-likeness picks it out.
CARD_ASPECT = 63.0 / 88.0
MIN_CANDIDATE_AREA_FRAC = 0.05
THRESHOLD_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5)


def _quads_from_contour(contour: np.ndarray) -> list[np.ndarray]:
    """Both quad interpretations of a contour: a polygonal approximation
    (follows perspective-skewed edges) and its minimum-area bounding box
    (robust to rounded corners and nibbled edges). They compete on score."""
    quads = []
    peri = cv2.arcLength(contour, True)
    for eps_frac in (0.02, 0.03, 0.015, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
        if len(approx) == 4:
            quads.append(order_points(approx.reshape(4, 2).astype(np.float32)))
            break
    rect = cv2.minAreaRect(contour)
    quads.append(order_points(cv2.boxPoints(rect).astype(np.float32)))
    return quads


def _card_likeness_penalty(quad: np.ndarray, contour: np.ndarray, area_frac: float) -> float:
    """Lower is more card-like. Aspect ratio dominates (it's the most
    discriminating signal between a card and a boxed background region),
    rectangularity and solidity refine, and a small size bonus breaks ties
    toward larger regions (the card fills most of the guide-box crop)."""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    ratio = min(w, h) / max(max(w, h), 1e-9)
    aspect_dev = abs(ratio - CARD_ASPECT) / CARD_ASPECT
    tilt = max(corner_angle_deviations(quad)) / 90.0
    quad_area = cv2.contourArea(quad.astype(np.int32))
    solidity = cv2.contourArea(contour) / max(quad_area, 1e-9)
    return aspect_dev * 3.0 + tilt * 1.0 + (1.0 - min(solidity, 1.0)) * 1.0 - min(area_frac, 0.5) * 0.2


def find_card_contour(image: np.ndarray) -> np.ndarray | None:
    """Find the card's corner quad against a matte background.

    Thresholds on per-pixel color distance from the background sample rather
    than absolute brightness — a plain brightness split assumes the card is
    brighter than the background, which is false for dark Pokemon card backs
    and would instead pick out just the bright inner text/art panel.

    Binarizes at several thresholds (Otsu and multiples of it), collects
    candidate quads from every sufficiently-large contour at each, and
    returns the most card-like candidate rather than blindly trusting the
    largest contour at the Otsu split.
    """
    bg_color = _estimate_background_color(image)
    diff = image.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(dist_u8, (5, 5), 0)
    image_area = image.shape[0] * image.shape[1]

    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_quad = None
    best_penalty = np.inf
    for mult in THRESHOLD_MULTIPLIERS:
        tval = otsu_val * mult
        if tval > 250:
            continue
        _, thresh = cv2.threshold(blurred, tval, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area_frac = cv2.contourArea(contour) / image_area
            if area_frac < MIN_CANDIDATE_AREA_FRAC:
                continue
            for quad in _quads_from_contour(contour):
                penalty = _card_likeness_penalty(quad, contour, area_frac)
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_quad = quad

    return best_quad


# warpPerspective offers no area-averaging mode: INTER_LINEAR reads a 2x2
# neighbourhood wherever it lands, so it can only properly average a 2x
# reduction. Past that it point-samples and the detail in between is aliased
# rather than averaged.
#
# Measured through this function on a flat grey carrying paper-fibre noise
# and a 150lpi-style screen (card interior only, excluding the warp's border
# pixels), fine-detail standard deviation came out:
#
#     downscale    warp as-is    with this pre-pass    ideal (1/N)
#         4x          13.69             4.56              4.83
#         8x          14.24             2.04              2.41
#
# Without the pre-pass it plateaus at ~14 whatever the input resolution — an
# 8x oversampled scan reached the detectors no cleaner than a 2x one, and
# marginally worse. That surviving texture is exactly what the whitening and
# surface stages respond to, so it is manufactured defect signal.
#
# The floor is just above 1.0 so the common case benefits: a 1200dpi scan
# reduced to the 605dpi canonical is a ~2x downscale, and at a 2.0 floor it
# missed the pre-pass by 0.0008.
MIN_PREFILTER_DOWNSCALE = 1.1


def _prefilter_for_downscale(
    image: np.ndarray, corners: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Area-average the source down to roughly the warp's output scale.

    Leaves the remaining reduction under 2x, which is the range INTER_LINEAR
    handles correctly. Returns the (possibly unchanged) image and the corner
    coordinates rescaled to match it.
    """
    width, height = size
    tl, tr, br, bl = corners.astype(np.float64)
    source_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    source_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    factor = min(source_w / max(width, 1), source_h / max(height, 1))
    if factor < MIN_PREFILTER_DOWNSCALE:
        return image, corners

    # Scale so the card spans roughly the output size, leaving the warp a
    # ~1:1 job. Deliberately not an integer factor: truncating 8.0 to 7 left
    # a 1.14x residual for INTER_LINEAR to botch, and INTER_AREA handles a
    # fractional reduction perfectly well.
    #
    # `factor` is the *smaller* of the two axis ratios, so a perspective-
    # skewed quad is under-reduced rather than over-reduced — any remaining
    # work is a downscale the warp can do, never an upscale that would blur.
    h, w = image.shape[:2]
    new_w = max(1, int(round(w / factor)))
    new_h = max(1, int(round(h / factor)))
    reduced = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Scale by what the resize actually did, not by 1/factor — rounding makes
    # those differ, and a fraction of a pixel of corner error is a fraction of
    # a millimetre of centering error.
    scaled = corners.astype(np.float32) * np.array([new_w / w, new_h / h], dtype=np.float32)
    return reduced, scaled


def perspective_correct(image: np.ndarray, corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image, corners = _prefilter_for_downscale(image, corners, size)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def corner_angle_deviations(corners: np.ndarray) -> list[float]:
    """Deviation from 90 degrees at each of the 4 corners (order: tl,tr,br,bl)."""
    deviations = []
    n = len(corners)
    for i in range(n):
        prev_pt, curr_pt, next_pt = corners[i - 1], corners[i], corners[(i + 1) % n]
        v1, v2 = prev_pt - curr_pt, next_pt - curr_pt
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        deviations.append(abs(angle - 90.0))
    return deviations


def check_resolution(image: np.ndarray, cfg: dict) -> QualityGate:
    shortest_side = min(image.shape[0], image.shape[1])
    passed = shortest_side >= cfg["min_input_shortest_side_px"]
    return QualityGate(
        "resolution", passed, f"shortest side={shortest_side}px", float(shortest_side)
    )


def check_tilt(corners: np.ndarray, cfg: dict) -> QualityGate:
    max_dev = max(corner_angle_deviations(corners))
    passed = max_dev <= cfg["max_corner_angle_deviation_deg"]
    return QualityGate("tilt", passed, f"max corner angle deviation={max_dev:.2f} deg", max_dev)


def check_aspect_ratio(corners: np.ndarray, cfg: dict, tilt_ok: bool = False) -> QualityGate:
    """Check the detected quad's side-length ratio against the card's 63:88.

    Measured on the pre-warp corner quad, NOT the warped image —
    perspective_correct always outputs the canonical WxH, so the warped
    image's aspect ratio is a constant and can't tell a card from a
    misdetected background rectangle. The quad's own geometry can: a
    minAreaRect fallback that boxed card+background together (which passes
    the tilt gate trivially, since boxPoints corners are exactly 90 deg)
    shows up here as a wildly wrong width:height ratio.

    When the tilt gate passed (clean right-angle corners), moderate aspect
    deviation is plausibly perspective foreshortening from a slightly
    off-overhead camera — which perspective_correct fixes — so a relaxed
    tolerance applies. Measured misdetections stay blocked either way: the
    card+background boxes came in at 15-28% deviation, far past the relaxed
    bound, while a genuinely off-angle capture of a real card sits in the
    3-8% range.
    """
    tl, tr, br, bl = corners
    quad_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    quad_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    expected_w, expected_h = cfg["card_aspect_ratio"]
    expected_ratio = expected_w / expected_h
    actual_ratio = quad_w / max(quad_h, 1e-9)
    deviation_pct = 100.0 * abs(actual_ratio - expected_ratio) / expected_ratio
    tolerance = cfg["aspect_ratio_tolerance_pct"]
    if tilt_ok:
        tolerance = cfg.get("aspect_ratio_tolerance_upright_pct", tolerance)
    passed = deviation_pct <= tolerance
    return QualityGate(
        "aspect_ratio",
        passed,
        f"actual={actual_ratio:.4f} expected={expected_ratio:.4f} deviation={deviation_pct:.2f}%",
        deviation_pct,
    )


def detect_glare(warped: np.ndarray, cfg: dict) -> QualityGate:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blown = (gray >= cfg["blown_out_pixel_value"]).astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(blown, connectivity=8)
    max_cluster = max((stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)), default=0)
    total_blown_pct = 100.0 * np.count_nonzero(blown) / blown.size
    passed = (
        max_cluster <= cfg["max_blown_out_cluster_area_px"]
        and total_blown_pct <= cfg["max_blown_out_area_pct"]
    )
    return QualityGate(
        "glare",
        passed,
        f"largest blown-out cluster={max_cluster}px, total blown area={total_blown_pct:.2f}%",
        total_blown_pct,
    )


def detect_uneven_lighting(warped: np.ndarray, cfg: dict) -> QualityGate:
    """Check brightness evenness across the card's outer border ring.

    Sampling the whole card face would confuse printed color contrast (e.g. a
    colored frame around a white text panel) with actual lighting unevenness.
    The border ring is close to a single uniform color on virtually every
    card, so brightness variation there is attributable to the light source.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    n = cfg["grid_size"]
    ring_h = max(1, int(h * cfg["ring_fraction"]))
    ring_w = max(1, int(w * cfg["ring_fraction"]))

    means = []
    for i in range(n):
        x0, x1 = int(w * i / n), int(w * (i + 1) / n)
        means.append(gray[0:ring_h, x0:x1].mean())
        means.append(gray[h - ring_h:h, x0:x1].mean())
    for i in range(n):
        y0, y1 = int(h * i / n), int(h * (i + 1) / n)
        means.append(gray[y0:y1, 0:ring_w].mean())
        means.append(gray[y0:y1, w - ring_w:w].mean())

    means_arr = np.array(means)
    gradient = float(means_arr.max() - means_arr.min())
    passed = gradient <= cfg["max_brightness_gradient"]
    return QualityGate("uneven_lighting", passed, f"border-ring brightness gradient={gradient:.1f}", gradient)


def align_for_surface(image: np.ndarray, thresholds: dict) -> DetectResult:
    """Lightweight alignment for the angled raking-light surface shot.

    Skips the tilt/uneven-lighting/glare gates used by detect_and_normalize:
    those assume a flat, evenly-lit overhead capture, but the surface shot
    is deliberately angled with directional raking light to make scratches
    and print lines visible via shadow — enforcing those gates here would
    reject a correctly-captured photo by design. Only resolution and
    successful card detection are hard requirements to proceed.
    """
    cap_cfg = thresholds["capture"]
    gates: list[QualityGate] = [check_resolution(image, cap_cfg)]

    corners = find_card_contour(image)
    if corners is None:
        gates.append(QualityGate("card_detection", False, "no card contour found"))
        return DetectResult(ok=False, warped=None, gates=gates, contour=None)
    gates.append(QualityGate("card_detection", True, "card contour found"))

    size = (cap_cfg["canonical_width_px"], cap_cfg["canonical_height_px"])
    warped = perspective_correct(image, corners, size)

    ok = all(g.passed for g in gates)
    return DetectResult(ok=ok, warped=warped, gates=gates, contour=corners)


# The canonical warp is a measurement surface: 1500x2100 across a 63mm card,
# ~605 dpi, and every threshold in thresholds.json is calibrated against it.
# It is not a viewing surface. A 1200dpi scan carries about 2.5x that detail
# per axis, and zooming into the canonical warp only interpolates what was
# already thrown away — so a second warp is kept at the capture's own scale
# for inspection. Nothing is ever measured from it.
MAX_DETAIL_LONG_EDGE_PX = 6000


def detail_warp(image: np.ndarray, corners: np.ndarray | None, thresholds: dict) -> np.ndarray | None:
    """Perspective-correct the card at (roughly) the capture's own scale.

    Returns None when there's nothing to gain — a capture at or below the
    canonical resolution would only be upscaled, which adds bytes and no
    detail.
    """
    if corners is None:
        return None
    cap_cfg = thresholds["capture"]
    canonical_w, canonical_h = cap_cfg["canonical_width_px"], cap_cfg["canonical_height_px"]

    quad = corners.astype(np.float64)
    top = float(np.linalg.norm(quad[1] - quad[0]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    width = (top + bottom) / 2.0
    height = (left + right) / 2.0
    if width <= 0 or height <= 0:
        return None

    # Hold the canonical aspect rather than the measured one: the card is a
    # known shape, and letting a pixel or two of corner error stretch the
    # detail view would make it disagree with the overlay drawn on top of it.
    scale = min(width / canonical_w, height / canonical_h)
    scale = min(scale, MAX_DETAIL_LONG_EDGE_PX / canonical_h)
    if scale <= 1.05:
        return None
    size = (int(round(canonical_w * scale)), int(round(canonical_h * scale)))
    return perspective_correct(image, corners, size)


def detect_and_normalize(image: np.ndarray, thresholds: dict) -> DetectResult:
    """Run Stage 1: find the card, perspective-correct it, and check capture quality."""
    cap_cfg = thresholds["capture"]
    gates: list[QualityGate] = [check_resolution(image, cap_cfg)]

    corners = find_card_contour(image)
    if corners is None:
        gates.append(QualityGate("card_detection", False, "no card contour found"))
        return DetectResult(ok=False, warped=None, gates=gates, contour=None)
    gates.append(QualityGate("card_detection", True, "card contour found"))
    tilt_gate = check_tilt(corners, cap_cfg)
    gates.append(tilt_gate)
    gates.append(check_aspect_ratio(corners, cap_cfg, tilt_ok=tilt_gate.passed))

    size = (cap_cfg["canonical_width_px"], cap_cfg["canonical_height_px"])
    warped = perspective_correct(image, corners, size)

    gates.append(detect_glare(warped, cap_cfg["glare"]))
    gates.append(detect_uneven_lighting(warped, cap_cfg["uneven_lighting"]))

    ok = all(g.passed for g in gates)
    return DetectResult(ok=ok, warped=warped, gates=gates, contour=corners)
