"""Stage 2: centering — border-width measurement and PSA-style tolerance grading.

Assumes the input image has already been perspective-corrected to the canonical
size (pipeline.detect.detect_and_normalize), so the physical card edge sits
exactly at the image boundary. Centering is then just: find where the outer
printed border/frame ends and the inner artwork/text panel begins, on each
side, and compare distances from the card edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AxisCentering:
    side_a: str  # "left" or "top"
    side_b: str  # "right" or "bottom"
    side_a_px: float
    side_b_px: float
    side_a_pct: float
    side_b_pct: float
    ratio_str: str
    grade: float
    # False when this axis's two boundaries couldn't be found confidently —
    # the px/pct/grade values above are then argmax-of-noise and must not be
    # shown as measurements.
    measurable: bool = True
    # What the published table alone would give, before PSA's 5% front
    # leeway. Reported next to the grade so the allowance is never invisible.
    strict_grade: float | None = None
    leeway_applied: bool = False
    # How much this border's width varies along the side, in pixels. A
    # straight cut holds steady; a card cut out of square does not, and PSA
    # grades "the most off-center part of the card", not the average of it.
    variation_px: float = 0.0
    # Why the axis was refused, in the report's own words. "unmeasurable"
    # alone reads the same whether the card has no border to find or the
    # detector found something that isn't a border, and those want different
    # things done about them — the second is what the manual adjustment is for.
    reason: str | None = None


# How far the detected border boundary may wander along a side before the
# axis stops being a measurement, as a multiple of the narrower border.
#
# A printed border edge is a straight line parallel to the card edge, so on a
# card whose border the detector really found, this is near zero: synthetic
# cards with straight printed borders measure 0.00-0.03, including one cut
# deliberately off-centre. On a real silver-bordered card where the detector
# was locking onto internal artwork instead, the same figure ran 0.74 to 3.06
# — the boundary moving by three times the width of the border it claimed to
# have found.
#
# That card is why this exists. Scanned twice, it measured 19/81 on one pass
# and 67/33 on the other, and reported grade 5 and grade 8 for one physical
# card whose centering had not changed. Both numbers came with a variation
# figure that said the samples disagreed; nothing acted on it.
#
# Set well above the noise of a real straight border and well below a
# boundary that isn't one. A genuinely skewed cut — the case variation_px was
# added to catch — should still measure and still grade: it wanders, but by a
# fraction of the border, not by multiples of it.
MAX_BOUNDARY_WANDER = 0.5


def _refuse_if_boundary_wanders(axis: AxisCentering, max_wander: float) -> None:
    """Drop an axis whose "border" isn't straight enough to be one."""
    if not axis.measurable:
        return
    narrower = min(axis.side_a_px, axis.side_b_px)
    if narrower <= 0:
        return
    wander = axis.variation_px / narrower
    if wander <= max_wander:
        return
    axis.measurable = False
    axis.reason = (
        f"the {axis.side_a}/{axis.side_b} boundary wanders {axis.variation_px:.0f}px along the side, "
        f"{wander:.1f}x the {narrower:.0f}px border it would be measuring — a printed border edge is "
        "straight, so this is tracking artwork rather than the border. Place the boundaries by hand "
        "to grade this axis."
    )


def axis_dict(a: AxisCentering) -> dict:
    return {
        "side_a": a.side_a,
        "side_b": a.side_b,
        "side_a_px": round(a.side_a_px, 1),
        "side_b_px": round(a.side_b_px, 1),
        "side_a_pct": round(a.side_a_pct, 1),
        "side_b_pct": round(a.side_b_pct, 1),
        "ratio": a.ratio_str,
        # PSA and every third-party tool print the larger share first
        # ("65/35"); we lead with left/right because which side the card is
        # shifted toward is half the information. Both, then.
        "ratio_conventional": f"{max(a.side_a_pct, a.side_b_pct):.0f}/{min(a.side_a_pct, a.side_b_pct):.0f}",
        "grade": a.grade,
        "strict_grade": a.strict_grade,
        "leeway_applied": a.leeway_applied,
        "variation_px": round(a.variation_px, 1),
        "measurable": a.measurable,
        "reason": a.reason,
    }


@dataclass
class CenteringResult:
    front_horizontal: AxisCentering
    front_vertical: AxisCentering
    front_grade: int | None  # None when the front's borders couldn't be measured
    back_horizontal: AxisCentering
    back_vertical: AxisCentering
    back_grade: int | None
    overall_grade: int | None
    front_measurable: bool = True
    back_measurable: bool = True
    front_confidence: dict | None = None  # per-boundary peak confidences, for debugging
    back_confidence: dict | None = None

    def to_dict(self) -> dict:
        return {
            "front": {
                "horizontal": axis_dict(self.front_horizontal),
                "vertical": axis_dict(self.front_vertical),
                "grade": self.front_grade,
                "measurable": self.front_measurable,
                "boundary_confidence": self.front_confidence,
            },
            "back": {
                "horizontal": axis_dict(self.back_horizontal),
                "vertical": axis_dict(self.back_vertical),
                "grade": self.back_grade,
                "measurable": self.back_measurable,
                "boundary_confidence": self.back_confidence,
            },
            "overall_grade": self.overall_grade,
        }


def _edge_profile(band: np.ndarray, canny_low: int, canny_high: int, axis: int) -> np.ndarray:
    """Sum of Canny edge response along `axis` (0=per-row, 1=per-column)."""
    edges = cv2.Canny(band, canny_low, canny_high)
    return edges.sum(axis=axis).astype(np.float64)


# The perspective warp leaves a strong straight artifact line within the first
# few pixels of every image edge (interpolation/background bleed). It scores
# near-perfect boundary confidence — a card edge is the straightest line in
# the image — and once produced a fake "60/40 grade 10" from a top=3px/
# bottom=2px "border".
#
# 2% of the dimension, not the 0.6% this started at. Measured on real card
# scans the seam runs out to ~1.4% (21px on a 1500px card), and every real
# border is far wider than 2%: a Base Set yellow border is ~4.7%, a Pokemon
# back border ~4%. The old margin let the seam win outright whenever the true
# boundary was faint.
def _edge_exclusion_px(dim: int) -> int:
    return max(4, int(dim * 0.02))


def _first_prominent_edge(window: np.ndarray, peak_fraction: float) -> int | None:
    """Index of the first edge reaching `peak_fraction` of the window's best.

    Not the strongest edge — the *first* strong one. Scanning inward from the
    card edge, the border/panel boundary is the first thing you cross; an
    argmax happily skips past it to something louder further in. Measured on
    a real Base Set scan the right-hand border read 175px (11.7%) instead of
    66px, because an interior frame line beat the true boundary, giving a
    71/29 ratio for a card the grading service measured at 52/48.
    """
    if window.size == 0:
        return None
    peak = float(window.max())
    if peak <= 0:
        return None
    hits = np.where(window >= peak * peak_fraction)[0]
    return int(hits[0]) if hits.size else None


def _offset_from_start(profile: np.ndarray, search_px: int, start_at: int = 1,
                       peak_fraction: float = 0.7) -> int:
    """Distance from index 0 to the first prominent edge within search_px."""
    search_px = min(search_px, len(profile) - 1)
    window = profile[start_at:search_px]
    index = _first_prominent_edge(window, peak_fraction)
    return search_px if index is None else index + start_at


def _offset_from_end(profile: np.ndarray, search_px: int, start_at: int = 1,
                     peak_fraction: float = 0.7) -> int:
    """Distance from the last index to the first prominent edge, scanning inward."""
    n = len(profile)
    search_px = min(search_px, n - 1)
    window = profile[n - search_px:n - start_at]
    if window.size == 0:
        return search_px
    # Reversed so "first" means first from the card's edge, not from index 0.
    index = _first_prominent_edge(window[::-1], peak_fraction)
    if index is None:
        return search_px
    return index + start_at


# A real border→panel boundary is a straight line spanning the whole sampled
# band, so its strongest profile column reaches a large fraction of the
# theoretical maximum (band_height * 255). Scattered art texture on a
# borderless/full-art card peaks far lower, and a boundary lost to blur or
# low contrast peaks near zero. Measured on synthetic scenes at canonical
# resolution: sharp/lightly-blurred bordered cards score 0.52-0.76,
# borderless art ~0.15-0.17, dim or heavily blurred captures ~0.
DEFAULT_MIN_BOUNDARY_CONFIDENCE = 0.35

# Second-chance bar for illumination-normalized re-measurement. Gamma+CLAHE
# recovers boundaries from underexposed captures (a dim bordered card's
# confidence goes 0.0 -> ~0.5) but also inflates art texture (borderless
# scenes rise to ~0.44), so the normalized pass needs a stricter bar than
# the raw pass — 0.5 sits between those two measured outcomes.
DEFAULT_NORMALIZED_MIN_CONFIDENCE = 0.5


def _normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Gamma-correct toward mid-gray, then CLAHE for local contrast — makes
    a border/panel boundary in an underexposed capture visible to Canny."""
    mean = gray.mean()
    if mean > 0:
        gamma = np.log(0.5) / np.log(np.clip(mean / 255.0, 0.05, 0.95))
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        gray = lut[gray]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _peak_confidence(profile: np.ndarray, window: slice, band_px: int) -> float:
    """Strongest edge in the window as a fraction of a perfect straight edge."""
    vals = profile[window]
    if vals.size == 0:
        return 0.0
    return float(vals.max()) / (band_px * 255.0)


# PSA measures centering as "the percent of difference at the most off-center
# part of the card" — a point, not an average. A card cut square has a border
# the same width all the way down a side and any sample point gives the same
# answer; a card cut slightly out of square does not, and sampling only the
# middle quietly reports the card at its best. So each side is sampled at
# several points along its length and the worst sample is the one that counts.
# Five, not three, and not for the obvious reason. More sample points do find
# the worst part of a skewed border — but the bigger effect is on confidence:
# a shorter band contains less of the slant, so the Canny peak stays sharp.
# Measured on a card whose border wanders 20px down its side, peak confidence
# went 0.17 (one band, refused) -> 0.23 (three, refused) -> 0.39 (five,
# measured). Narrower sampling rescues skewed cards from being declared
# unmeasurable, which is the failure this was meant to fix in the first place.
DEFAULT_SAMPLE_BANDS = 5
# Bands stay clear of the corners, where the rounded cut and any corner wear
# put a false boundary inside the search window.
SAMPLE_SPAN = (0.18, 0.82)


def _sample_bands(extent: int, count: int) -> list[tuple[int, int]]:
    """`count` windows spread along a side, between SAMPLE_SPAN of it.

    A count of 1 reproduces the single central band this used to sample, so
    the behaviour is recoverable from config alone.
    """
    if count <= 1:
        return [(int(extent * 0.35), int(extent * 0.65))]
    lo, hi = SAMPLE_SPAN
    width = (hi - lo) / count
    bands = []
    for i in range(count):
        start = int(extent * (lo + i * width))
        end = int(extent * (lo + (i + 1) * width))
        if end - start >= 2:
            bands.append((start, end))
    return bands or [(int(extent * 0.35), int(extent * 0.65))]


def _axis_samples(gray: np.ndarray, cfg: dict, axis: str) -> list[dict]:
    """Boundary offsets and confidences at each sample point along one axis.

    `axis` is "horizontal" (left/right boundaries, sampled down the card) or
    "vertical" (top/bottom, sampled across it).
    """
    h, w = gray.shape
    margin = cfg["search_margin_pct"] / 100.0
    peak_fraction = cfg.get("boundary_peak_fraction", 0.7)
    count = int(cfg.get("sample_bands", DEFAULT_SAMPLE_BANDS))

    if axis == "horizontal":
        extent, span, names = h, w, ("left", "right")
    else:
        extent, span, names = w, h, ("top", "bottom")
    search_px = int(span * margin)
    exclusion = _edge_exclusion_px(span)

    samples = []
    for start, end in _sample_bands(extent, count):
        band = gray[start:end, :] if axis == "horizontal" else gray[:, start:end]
        profile = _edge_profile(band, cfg["canny_low"], cfg["canny_high"], axis=0 if axis == "horizontal" else 1)
        band_thickness = end - start
        n = len(profile)
        samples.append(
            {
                names[0]: float(_offset_from_start(profile, search_px, exclusion, peak_fraction)),
                names[1]: float(_offset_from_end(profile, search_px, exclusion, peak_fraction)),
                "confidence": {
                    names[0]: _peak_confidence(profile, slice(exclusion, search_px), band_thickness),
                    names[1]: _peak_confidence(profile, slice(n - search_px, n - exclusion), band_thickness),
                },
            }
        )
    return samples


def _worst_sample(samples: list[dict], names: tuple[str, str], min_confidence: float) -> tuple[dict, float, float]:
    """Pick the sample where the card reads most off-centre.

    Only samples whose boundaries were both found confidently are eligible —
    taking the maximum over noisy picks would reliably select the noise.
    With none eligible, the central sample is returned so the caller still
    has numbers to show its confidence gate.

    Returns (sample, variation of side a, variation of side b), where the
    variations are how much each border width moved across the eligible
    samples: a straight cut holds steady, a skewed one doesn't.
    """
    a, b = names
    eligible = [
        s for s in samples if s["confidence"][a] >= min_confidence and s["confidence"][b] >= min_confidence
    ]
    pool = eligible or samples

    def worse_pct(sample: dict) -> float:
        total = sample[a] + sample[b]
        return max(sample[a], sample[b]) / total * 100.0 if total > 0 else 50.0

    variation_a = max(s[a] for s in pool) - min(s[a] for s in pool)
    variation_b = max(s[b] for s in pool) - min(s[b] for s in pool)
    return (max(pool, key=worse_pct) if eligible else pool[len(pool) // 2]), variation_a, variation_b


def _measure_borders(
    image: np.ndarray, cfg: dict, normalize: bool = False
) -> tuple[tuple[float, float, float, float], dict[str, float], dict[str, float]]:
    """Return ((left_px, right_px, top_px, bottom_px), confidence, variation).

    The offsets are always computed (argmax of the edge profile picks
    *something* in every image); the confidence dict is what says whether
    each pick is a real border boundary or just the strongest noise. The
    variation dict says how much each border moved between sample points.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if normalize:
        gray = _normalize_illumination(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    min_conf = cfg.get("min_boundary_confidence", DEFAULT_MIN_BOUNDARY_CONFIDENCE)

    horizontal, var_left, var_right = _worst_sample(
        _axis_samples(gray, cfg, "horizontal"), ("left", "right"), min_conf
    )
    vertical, var_top, var_bottom = _worst_sample(
        _axis_samples(gray, cfg, "vertical"), ("top", "bottom"), min_conf
    )

    confidence = {**horizontal["confidence"], **vertical["confidence"]}
    variation = {"left": var_left, "right": var_right, "top": var_top, "bottom": var_bottom}
    widths = (horizontal["left"], horizontal["right"], vertical["top"], vertical["bottom"])
    return widths, confidence, variation


def _tidy_grade(grade: float) -> float | int:
    """Half-grade tables carry floats; whole grades still serialize as ints.

    TAG grades centering on a half-point ladder (10, 9, 8.5, 8, 7.5...), so
    grades are floats throughout. Emitting `7.0` where every previous report
    said `7` would churn every stored report and every test for nothing.
    """
    return int(grade) if float(grade).is_integer() else float(grade)


def _grade_from_ratio(worse_pct: float, tolerances: list[dict], leeway_points: float = 0.0,
                      leeway_min_grade: float = 7.0) -> float:
    """tolerances: ascending max_ratio per grade, best (10) first.

    `leeway_points` implements PSA's published allowance: "A 5% leeway is
    given to the front centering minimum standards for cards which grade PSA
    7 or better." It widens each tier at or above `leeway_min_grade` and is
    not applied below that, nor to backs. It is a real published rule, and
    ignoring it under-graded every card sitting just past a line — a measured
    65.06% against a 65 limit is a PSA 8 under the leeway, not a 7.
    """
    for tier in tolerances:
        limit = tier["max_ratio"]
        if leeway_points and tier["grade"] >= leeway_min_grade:
            limit += leeway_points
        if worse_pct <= limit:
            return float(tier["grade"])
    return float(max(1, tolerances[-1]["grade"] - 2))


def _axis_centering(side_a_name: str, side_b_name: str, px_a: float, px_b: float, tolerances: list[dict],
                    leeway_points: float = 0.0, leeway_min_grade: float = 7.0) -> AxisCentering:
    total = px_a + px_b if (px_a + px_b) > 0 else 1.0
    pct_a = 100.0 * px_a / total
    pct_b = 100.0 * px_b / total
    worse = max(pct_a, pct_b)
    strict = _grade_from_ratio(worse, tolerances)
    grade = _grade_from_ratio(worse, tolerances, leeway_points, leeway_min_grade)
    # Ordered side_a/side_b — left/right, top/bottom — not largest-first.
    # Largest-first threw away *which* side the card is shifted toward, which
    # is half the information: it rendered "55/45" for a card sitting 44.5%
    # top / 55.5% bottom, indistinguishable from the opposite miscut and not
    # comparable to how any grading service reports the same measurement.
    ratio_str = f"{pct_a:.0f}/{pct_b:.0f}"
    axis = AxisCentering(
        side_a_name, side_b_name, px_a, px_b, pct_a, pct_b, ratio_str, _tidy_grade(grade)
    )
    axis.strict_grade = _tidy_grade(strict)
    axis.leeway_applied = grade > strict
    return axis


def draw_overlay(image: np.ndarray, axis_h: AxisCentering, axis_v: AxisCentering) -> np.ndarray:
    """Draw the detected border boundary lines over a copy of the image.

    Lines are drawn only for measurable axes — an unmeasurable axis's
    positions are argmax-of-noise that would look authoritative but aren't.

    Lines only, no text. The ratios and grades used to be drawn into the
    pixels, which was right when this was a debug file in an output
    directory; the report now renders those same figures as real text
    directly beneath the overlay, so a burned-in copy was duplicated,
    illegible at thumbnail size, and invisible to a screen reader.
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    color = (0, 255, 0)
    thickness = 3
    if axis_h.measurable:
        left, right = int(axis_h.side_a_px), int(axis_h.side_b_px)
        cv2.line(overlay, (left, 0), (left, h), color, thickness)
        cv2.line(overlay, (w - right, 0), (w - right, h), color, thickness)
    if axis_v.measurable:
        top, bottom = int(axis_v.side_a_px), int(axis_v.side_b_px)
        cv2.line(overlay, (0, top), (w, top), color, thickness)
        cv2.line(overlay, (0, h - bottom), (w, h - bottom), color, thickness)
    return overlay


# A Canny edge is the right instrument for a printed border with a crisp
# inner edge — a Base Set yellow frame against artwork. It is the wrong one
# for a Pokemon card back, where the blue frame meets blue artwork: measured
# on two real backs the edge response scored 0.22-0.35 there and correctly
# refused, leaving unmeasurable an axis both grading services report.
#
# Colour is no better on that card. Sampled across the border and well into
# the swirl, the colour never departs from the border's own by more than
# noise until ~19% of the card width — and that departure is a bright
# highlight in the artwork, not the border's edge. A first attempt at this
# keyed on colour and produced ratios that looked plausible only because
# both sides landed equally deep in the artwork.
#
# What actually separates them is texture. The border is flat printed ink;
# the artwork is not. On all three bordered sides tested, per-column texture
# sits at a low plateau across the border and steps up by roughly 2x at its
# inner edge — at 4-5% of the card width, which is where a real border ends.
def _texture_boundary(image: np.ndarray, cfg: dict) -> tuple[tuple[float, float, float, float], dict[str, float]]:
    """Border widths from where flat printed border gives way to textured art.

    Returns the same (widths, confidence) shape as `_measure_borders`.
    Confidence is how far the texture beyond the boundary clears the
    threshold, so a card with no border at all — full-art, textured to the
    edge — has no plateau to rise from and scores zero rather than inventing
    a boundary.
    """
    rise = cfg.get("texture_rise_factor", 1.8)
    floor = cfg.get("texture_rise_floor", 6.0)
    hold = cfg.get("texture_hold_px", 120)
    smooth = cfg.get("texture_smoothing_sigma", 6.0)
    h, w = image.shape[:2]

    def profile(axis: str) -> np.ndarray:
        # 60% of the card, not centering's 30%: this boundary is defined by a
        # statistic, and a statistic wants samples.
        if axis == "h":
            band = cv2.cvtColor(image[int(h * 0.20):int(h * 0.80)], cv2.COLOR_BGR2GRAY)
            values = band.astype(np.float32).std(axis=0)
        else:
            band = cv2.cvtColor(image[:, int(w * 0.20):int(w * 0.80)], cv2.COLOR_BGR2GRAY)
            values = band.astype(np.float32).std(axis=1)
        return cv2.GaussianBlur(values.reshape(1, -1), (0, 0), smooth).ravel()

    def scan(values: np.ndarray, dim: int) -> tuple[float, float]:
        exclusion = _edge_exclusion_px(dim)
        limit = int(dim * 0.25)
        baseline = float(np.median(values[exclusion:exclusion + 20]))
        threshold = max(baseline * rise, baseline + floor)
        hits = np.where(values[exclusion:limit] > threshold)[0]
        if not hits.size:
            return float(limit), 0.0
        offset = int(hits[0]) + exclusion
        following = values[offset:offset + hold]
        if not following.size:
            return float(offset), 0.0
        margin = threshold - baseline
        confidence = float(min(1.0, max(0.0, (float(np.median(following)) - baseline) / max(margin, 1e-6))))
        return float(offset), confidence

    columns, rows = profile("h"), profile("v")
    left, c_left = scan(columns, w)
    right, c_right = scan(columns[::-1], w)
    top, c_top = scan(rows, h)
    bottom, c_bottom = scan(rows[::-1], h)
    return (left, right, top, bottom), {"left": c_left, "right": c_right, "top": c_top, "bottom": c_bottom}


def _measure_side(
    image: np.ndarray, border_cfg: dict
) -> tuple[tuple[float, float, float, float], dict[str, float], dict[str, float], bool, bool]:
    """Measure one side's borders, with an illumination-normalized retry.

    Raw measurement first; boundaries below the confidence floor get a
    second chance on a gamma+CLAHE-normalized image against a stricter bar
    (normalization recovers dim captures but also inflates art texture).

    Measurability is decided PER AXIS — horizontal needs left+right,
    vertical needs top+bottom. A real capture measured V confidently while
    H was genuinely invisible (a soft-focus shot of a card back, whose
    blue-frame→blue-swirl left/right transition is the lowest-contrast
    boundary on the card); an all-four rule threw the good axis away.

    Returns (widths, confidences, variations, h_measurable, v_measurable).
    All three are per-boundary, from whichever pass confirmed that axis (raw
    preferred); unconfirmed axes keep their raw numbers for debugging.
    """
    min_conf = border_cfg.get("min_boundary_confidence", DEFAULT_MIN_BOUNDARY_CONFIDENCE)
    norm_bar = border_cfg.get("normalized_min_boundary_confidence", DEFAULT_NORMALIZED_MIN_CONFIDENCE)

    (l, r, t, b), conf, var = _measure_borders(image, border_cfg)
    h_ok = conf["left"] >= min_conf and conf["right"] >= min_conf
    v_ok = conf["top"] >= min_conf and conf["bottom"] >= min_conf

    if not (h_ok and v_ok):
        (l_n, r_n, t_n, b_n), conf_n, var_n = _measure_borders(image, border_cfg, normalize=True)
        if not h_ok and conf_n["left"] >= norm_bar and conf_n["right"] >= norm_bar:
            l, r = l_n, r_n
            conf = {**conf, "left": conf_n["left"], "right": conf_n["right"]}
            var = {**var, "left": var_n["left"], "right": var_n["right"]}
            h_ok = True
        if not v_ok and conf_n["top"] >= norm_bar and conf_n["bottom"] >= norm_bar:
            t, b = t_n, b_n
            conf = {**conf, "top": conf_n["top"], "bottom": conf_n["bottom"]}
            var = {**var, "top": var_n["top"], "bottom": var_n["bottom"]}
            v_ok = True

    # Last resort before giving up on an axis: the texture boundary. Only
    # consulted once the edge detector has failed both its passes, and it has
    # to clear a high bar — a card with no border produces no plateau to rise
    # from and scores zero, which is what keeps full-art fronts refused.
    if not (h_ok and v_ok):
        (l_c, r_c, t_c, b_c), conf_c = _texture_boundary(image, border_cfg)
        colour_bar = border_cfg.get("min_texture_boundary_confidence", 0.9)
        if not h_ok and conf_c["left"] >= colour_bar and conf_c["right"] >= colour_bar:
            l, r = l_c, r_c
            conf = {**conf, "left": conf_c["left"], "right": conf_c["right"]}
            # The texture fallback reads a single profile, so it has no spread
            # to report. Zero, rather than a stale number from the edge pass.
            var = {**var, "left": 0.0, "right": 0.0}
            h_ok = True
        if not v_ok and conf_c["top"] >= colour_bar and conf_c["bottom"] >= colour_bar:
            t, b = t_c, b_c
            conf = {**conf, "top": conf_c["top"], "bottom": conf_c["bottom"]}
            var = {**var, "top": 0.0, "bottom": 0.0}
            v_ok = True

    return (l, r, t, b), conf, var, h_ok, v_ok


def measure_centering(front: np.ndarray, back: np.ndarray, thresholds: dict) -> CenteringResult:
    cfg = thresholds["centering"]
    border_cfg = cfg["border_detect"]

    # An axis whose boundaries can't be found confidently is unmeasurable —
    # a borderless/full-art card, or a capture too blurry/dim to show that
    # boundary. Reporting its ratio anyway would be fake precision from
    # argmax-of-noise (a real full-art card produced a confident-looking
    # "89/11 grade 3" before this check existed). Measurable axes still
    # grade: a side's grade is the min over its measurable axes, None only
    # when neither axis measures.
    (fl, fr, ft, fb), front_conf, front_var, f_h_ok, f_v_ok = _measure_side(front, border_cfg)
    (bl, br, bt, bb), back_conf, back_var, b_h_ok, b_v_ok = _measure_side(back, border_cfg)

    # PSA's leeway is published for the front only, and only from grade 7 up.
    leeway = float(cfg.get("front_leeway_points", 0.0))
    leeway_min = float(cfg.get("leeway_min_grade", 7.0))

    max_wander = float(cfg.get("max_boundary_wander", MAX_BOUNDARY_WANDER))

    def side(hpx, vpx, h_ok, v_ok, tolerances, variation, leeway_points):
        axis_h = _axis_centering("left", "right", *hpx, tolerances, leeway_points, leeway_min)
        axis_h.measurable = h_ok
        axis_h.variation_px = max(variation["left"], variation["right"])
        axis_v = _axis_centering("top", "bottom", *vpx, tolerances, leeway_points, leeway_min)
        axis_v.measurable = v_ok
        axis_v.variation_px = max(variation["top"], variation["bottom"])
        for axis in (axis_h, axis_v):
            _refuse_if_boundary_wanders(axis, max_wander)
        grades = [a.grade for a in (axis_h, axis_v) if a.measurable]
        return axis_h, axis_v, (min(grades) if grades else None)

    front_h, front_v, front_grade = side(
        (fl, fr), (ft, fb), f_h_ok, f_v_ok, cfg["front_tolerances"], front_var, leeway
    )
    back_h, back_v, back_grade = side(
        (bl, br), (bt, bb), b_h_ok, b_v_ok, cfg["back_tolerances"], back_var, 0.0
    )

    measured = [g for g in (front_grade, back_grade) if g is not None]
    overall_grade = min(measured) if measured else None

    return CenteringResult(
        front_horizontal=front_h,
        front_vertical=front_v,
        front_grade=front_grade,
        back_horizontal=back_h,
        back_vertical=back_v,
        back_grade=back_grade,
        overall_grade=overall_grade,
        # Side-level flag means "this side produced a grade at all" — i.e.
        # at least one axis measured. Per-axis truth lives on the axes.
        front_measurable=front_grade is not None,
        back_measurable=back_grade is not None,
        front_confidence={k: round(v, 3) for k, v in front_conf.items()},
        back_confidence={k: round(v, 3) for k, v in back_conf.items()},
    )


# ---------------------------------------------------------------------------
# Manual override.
#
# The detector refuses an axis it can't find confidently, which is the right
# default — but it has a third failure mode that no confidence gate catches:
# finding *an* edge, confidently enough to pass, that isn't the border. Modern
# cards stack an artwork boundary, an inner frame band and a thin rule inside
# a few millimetres, and the first strong edge inward is not always the one a
# grader measures to.
#
# So the boundaries can be placed by hand. A hand-placed boundary is always
# measurable: the confidence score describes how sure the *detector* was, and
# once a human has said where the edge is that question is moot.
# ---------------------------------------------------------------------------

MANUAL_SIDES = ("left", "right", "top", "bottom")


def split_override(override: dict) -> tuple[dict, dict]:
    """Pull (border widths, card-edge insets) out of one side's payload.

    Accepts the bare four widths as well, so the first version of the API —
    which had no concept of a movable card edge — keeps working.
    """
    if "borders" in override:
        borders = override["borders"]
        edges = override.get("edges") or {name: 0.0 for name in MANUAL_SIDES}
    else:
        borders = override
        edges = {name: 0.0 for name in MANUAL_SIDES}
    return borders, edges


def draw_manual_overlay(image: np.ndarray, borders: dict, edges: dict) -> np.ndarray:
    """Draw both boundary pairs: the card edge and the border/artwork line.

    Two colours because they answer different questions — green is what the
    grade is computed from, amber is where the card was taken to start. Amber
    is omitted at zero inset, which is the normal case and would otherwise
    paint a line along every image border on every report.
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    inner = (0, 255, 0)
    outer = (0, 170, 255)
    for name, inset in edges.items():
        position = int(round(float(inset)))
        if position <= 0:
            continue
        if name == "left":
            cv2.line(overlay, (position, 0), (position, h), outer, 3)
        elif name == "right":
            cv2.line(overlay, (w - position, 0), (w - position, h), outer, 3)
        elif name == "top":
            cv2.line(overlay, (0, position), (w, position), outer, 3)
        else:
            cv2.line(overlay, (0, h - position), (w, h - position), outer, 3)

    left = int(round(float(edges["left"]) + float(borders["left"])))
    right = int(round(float(edges["right"]) + float(borders["right"])))
    top = int(round(float(edges["top"]) + float(borders["top"])))
    bottom = int(round(float(edges["bottom"]) + float(borders["bottom"])))
    cv2.line(overlay, (left, 0), (left, h), inner, 3)
    cv2.line(overlay, (w - right, 0), (w - right, h), inner, 3)
    cv2.line(overlay, (0, top), (w, top), inner, 3)
    cv2.line(overlay, (0, h - bottom), (w, h - bottom), inner, 3)
    return overlay


def manual_axes(
    borders: dict, tolerances: list[dict], leeway_points: float = 0.0, leeway_min_grade: float = 7.0
) -> tuple[AxisCentering, AxisCentering]:
    """Build the two axes for one side from hand-placed border widths.

    `borders` holds the four widths in canonical-warp pixels, measured from
    the card edge inward — the same quantity `side_a_px`/`side_b_px` report.

    The leeway arguments have to be passed through here too: a measurement is
    a measurement however it was arrived at, and a hand-placed boundary that
    graded differently from the identical detected one would be a bug the
    user could see.
    """
    axis_h = _axis_centering(
        "left", "right", float(borders["left"]), float(borders["right"]), tolerances, leeway_points, leeway_min_grade
    )
    axis_v = _axis_centering(
        "top", "bottom", float(borders["top"]), float(borders["bottom"]), tolerances, leeway_points, leeway_min_grade
    )
    return axis_h, axis_v


def _numbers(values: dict, label: str) -> tuple[dict | None, str | None]:
    if not isinstance(values, dict):
        return None, f"{label} must be an object with left, right, top and bottom"
    out = {}
    for name in MANUAL_SIDES:
        if name not in values:
            return None, f"missing {label}: {name}"
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            return None, f"{label} {name!r} is not a number"
        if not (value >= 0):  # also rejects NaN, which compares false to everything
            return None, f"{label} {name!r} must be zero or more"
        out[name] = value
    return out, None


def validate_manual_borders(borders: dict, width: int, height: int, edges: dict | None = None) -> str | None:
    """None if these widths describe a possible card, else why they don't.

    `edges` optionally says where the card's own edge sits, as an inset from
    each side of the warp. It's normally zero — the warp is *defined* by the
    detected corners, so the card edge is the image edge by construction —
    but corner detection can be a pixel or two out, and at this scale that
    moves a border width by more than the gap between two grades.
    """
    values, problem = _numbers(borders, "border width")
    if problem is not None:
        return problem
    insets = {name: 0.0 for name in MANUAL_SIDES}
    if edges is not None:
        insets, problem = _numbers(edges, "card edge inset")
        if problem is not None:
            return problem

    # Opposite boundaries that meet or cross would put the artwork panel at
    # zero or negative width, which is not a card.
    for axis, span, a, b in (("left and right", width, "left", "right"), ("top and bottom", height, "top", "bottom")):
        if insets[a] + values[a] + insets[b] + values[b] >= span:
            return f"{axis} boundaries overlap — together they cover the whole card"
    return None


def regrade_with_manual_borders(centering: dict, overrides: dict, thresholds: dict) -> dict:
    """Rebuild a centering block from hand-placed border widths.

    `overrides` maps "front" and/or "back" to `{"borders": {...}}` and
    optionally `{"edges": {...}}` — the four border widths, and where the
    card's own edge sits if it isn't the image edge. A side that isn't
    overridden is carried through untouched, so one side can be corrected
    without disturbing the other.
    """
    cfg = thresholds["centering"]
    updated = {key: dict(value) if isinstance(value, dict) else value for key, value in centering.items()}

    for side, tolerance_key in (("front", "front_tolerances"), ("back", "back_tolerances")):
        override = overrides.get(side)
        if not override:
            continue
        borders, edges = split_override(override)
        # PSA's leeway is a front-only rule, exactly as in measure_centering.
        leeway = float(cfg.get("front_leeway_points", 0.0)) if side == "front" else 0.0
        axis_h, axis_v = manual_axes(
            borders, cfg[tolerance_key], leeway, float(cfg.get("leeway_min_grade", 7.0))
        )
        updated[side] = {
            "horizontal": {**axis_dict(axis_h), "manual": True},
            "vertical": {**axis_dict(axis_v), "manual": True},
            "grade": min(axis_h.grade, axis_v.grade),
            "measurable": True,
            "manual": True,
            # Where the card edge was taken to be, so a later reader can see
            # that a non-zero inset is why these widths don't match the warp.
            "card_edge_px": {name: round(float(edges[name]), 1) for name in MANUAL_SIDES},
            # The detector's confidences described boundaries that are no
            # longer the ones being shown, so they'd be actively misleading.
            "boundary_confidence": None,
        }

    grades = [updated[side].get("grade") for side in ("front", "back") if isinstance(updated.get(side), dict)]
    measured = [g for g in grades if g is not None]
    updated["overall_grade"] = min(measured) if measured else None
    return updated


# ---------------------------------------------------------------------------
# What the other graders would say.
#
# Every serious centering tool reports PSA, BGS, CGC and TAG side by side off
# one measurement, and they diverge more than you'd expect: the same card can
# be a PSA 8 and a BGS 7, because BGS grades both axes against the front table
# and tightens the back far harder. Only PSA drives our grade — the rest are
# reference, and each one carries where its table came from, because these are
# third-party transcriptions rather than primary sources.
# ---------------------------------------------------------------------------


def grade_against(worse_front_pct: float | None, worse_back_pct: float | None, grader: dict) -> dict:
    """One grader's verdict on an already-measured card.

    A missing side doesn't vote — an unmeasurable back shouldn't silently
    become a perfect one.
    """
    leeway = float(grader.get("front_leeway_points", 0.0))
    leeway_min = float(grader.get("leeway_min_grade", 7.0))
    grades = []
    front_grade = back_grade = None
    if worse_front_pct is not None and grader.get("front"):
        front_grade = _tidy_grade(_grade_from_ratio(worse_front_pct, grader["front"], leeway, leeway_min))
        grades.append(front_grade)
    if worse_back_pct is not None and grader.get("back"):
        back_grade = _tidy_grade(_grade_from_ratio(worse_back_pct, grader["back"]))
        grades.append(back_grade)
    return {
        "label": grader.get("label", "?"),
        "front": front_grade,
        "back": back_grade,
        "grade": min(grades) if grades else None,
        "source": grader.get("source", ""),
    }


def _worst_measured_pct(side: dict | None) -> float | None:
    """The most off-centre measurable axis of one side, as a percentage."""
    if not isinstance(side, dict):
        return None
    worst = None
    for axis_key in ("horizontal", "vertical"):
        axis = side.get(axis_key)
        if not isinstance(axis, dict) or axis.get("measurable") is False:
            continue
        pct = max(float(axis.get("side_a_pct", 50.0)), float(axis.get("side_b_pct", 50.0)))
        worst = pct if worst is None else max(worst, pct)
    return worst


def compare_graders(centering: dict, thresholds: dict) -> dict:
    """Run one measured card past every configured grader's table."""
    graders = thresholds.get("centering", {}).get("graders") or {}
    front_pct = _worst_measured_pct(centering.get("front"))
    back_pct = _worst_measured_pct(centering.get("back"))
    return {key: grade_against(front_pct, back_pct, grader) for key, grader in graders.items()}
