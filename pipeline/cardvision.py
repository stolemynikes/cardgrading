"""Card Vision: a grayscale relief render of the card's physical surface.

The point is to strip *print* and leave *geometry* — scratches, dents,
creases and edge lifting are shape, artwork is not. Two ways to get there,
in descending order of honesty:

1. Photometric stereo (`photometric_card_vision`). Several captures of the
   same card from a fixed camera with the light coming from different
   directions solve for a per-pixel surface normal. Print has no effect on
   the normal, so the render genuinely shows only relief. A flatbed scanner
   is the easiest rig for this: its lamp is fixed relative to the scan axis,
   so rotating the card 90 degrees on the glass between scans rotates the
   light in the card's frame. Four scans = four light directions, already
   registered to each other by the canonical warp.

2. Single-image approximation (`single_image_card_vision`). One ordinary
   capture, high-pass filtered to drop the low-frequency albedo, then
   attenuated wherever the *color* is also changing — a printed line moves
   chroma, a scratch in the laminate moves only luminance. Useful, but it
   cannot fully separate fine print detail from real geometry, and it must
   be labeled as an approximation wherever it is shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Scanner lamps sit close to the glass, so the light arrives steeply rather
# than raking. 70 degrees of elevation is a reasonable default for a CCD
# flatbed; a phone with a lamp off to one side is far shallower.
DEFAULT_ELEVATION_DEG = 70.0

# Virtual light used to shade the solved normal map. Fixed at a diagonal so
# that neither purely horizontal nor purely vertical scratches disappear
# into the shading direction.
RENDER_AZIMUTH_DEG = 315.0

# The azimuths the *measured* render is swept over, in degrees.
#
# A virtual light casts no shadow along a defect that runs parallel to it, so
# a single fixed azimuth has a blind direction. Measured on one damaged card,
# sweeping the light over the same solved normals moved the longest defect
# from 58px to 281px depending only on where the light was put — and the
# longest defect is a grading input. A grade that depends on an arbitrary
# choice of light direction is not a measurement.
#
# Four lights at right angles, kept per pixel at their worst, is the same
# thing a grader does with a real card: turn it under the lamp and keep what
# you saw. Nothing can then hide from every light at once.
MEASUREMENT_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
RENDER_ELEVATION_DEG = 35.0


@dataclass
class CardVisionResult:
    """`relief` is the image meant for the transparency slider: mid-gray where
    the surface is flat, bright/dark where it is not."""

    relief: np.ndarray  # uint8 grayscale, canonical size
    method: str  # "photometric_stereo" | "single_image"
    light_count: int
    normal_map: np.ndarray | None = None  # uint8 BGR debug visualization
    albedo: np.ndarray | None = None  # uint8 grayscale, photometric only
    roughness_pct: float = 0.0  # share of the card whose relief exceeds the flat band
    # Why the photometric solve wasn't used, when it wasn't. "single_image"
    # alone can't distinguish "no rotation scans were attached" from "four
    # were attached and one of them failed", and those are entirely different
    # problems — one is a capture that never happened, the other is a capture
    # that did and was rejected without saying so.
    fallback_reason: str | None = None
    # Scans that were attached but had no detectable card in them, named so
    # the operator knows which one to redo. Separate from fallback_reason
    # because a dropped scan doesn't necessarily cost the solve: four scans
    # carry one spare, and three non-collinear directions still solve. It is
    # still worth saying — a solve running on three lights instead of four
    # has less to work with, and nothing else in the report would show it.
    dropped_scans: list[str] | None = None
    # The rotation measured for each scan, in degrees counter-clockwise from
    # the reference orientation. Recorded because it is the one solve input
    # that used to be asserted rather than measured, and a wrong one produces
    # a plausible render of a badly damaged card rather than an error.
    rotations_deg: list[int] | None = None
    # Per-frame alignment diagnostics. Recorded because a misaligned set does
    # not fail — it renders the card's own print embossed twice and calls it
    # damage, and from the render alone there is no telling whether the fit
    # was refused, accepted and wrong, or never needed.
    registration: list[dict] | None = None
    # The detected card quad from each rotation scan, for the dimensions
    # stage. Not serialized — these are measurement inputs, not report data.
    scan_quads: list | None = None
    # The single-light measured render, for the corners and edges.
    #
    # Sweeping the light maximises noise along with signal, and near the card's
    # physical boundary the noise is the perspective-warp seam, which is most
    # of what a corner crop contains. Measured: sweeping took an undamaged
    # card's top-left corner from 0.00% wear to 11.89%, which would have graded
    # a clean card's corners a 6. In the card's interior the signal dominates
    # and the sweep is the better measurement; at the edges it is not, so the
    # two stages read different renders and each says why.
    edge_relief: np.ndarray | None = None
    # The same relief rendered at unit gain. `relief` is the picture, and its
    # gain is a viewing preference; this is what gets measured. Without the
    # split, turning the render up to taste doubled the card's measured defect
    # area and cost it a surface grade — a cosmetic setting must not move a
    # measurement.
    measurement_relief: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "light_count": self.light_count,
            "roughness_pct": round(self.roughness_pct, 3),
            "fallback_reason": self.fallback_reason,
            "dropped_scans": self.dropped_scans,
            "rotations_deg": self.rotations_deg,
            "registration": self.registration,
            "note": (
                "surface normals solved from multiple light directions — print does not affect this render"
                if self.method == "photometric_stereo"
                else "approximation from a single capture — fine print detail can still leak into this render"
            ),
        }


# ---------------------------------------------------------------- geometry


def light_vectors(azimuths_deg: list[float], elevation_deg: float) -> np.ndarray:
    """Unit light directions (N,3) in the card's frame, +Z out of the card.

    Azimuth 0 puts the light along +X (card's right), increasing
    counter-clockwise, which matches the image coordinate convention used
    below once the Y axis is flipped.
    """
    elev = np.radians(elevation_deg)
    az = np.radians(np.asarray(azimuths_deg, dtype=np.float64))
    return np.stack(
        [np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.full_like(az, np.sin(elev))],
        axis=1,
    )


def scanner_azimuths(count: int, lamp_azimuth_deg: float, clockwise: bool) -> list[float]:
    """Light azimuths in the *card's* frame for a rotate-on-the-glass scan set.

    The scanner's lamp never moves; the card does. Rotating the card
    clockwise by 90 degrees moves the lamp counter-clockwise by 90 degrees
    relative to the card, hence the sign flip.
    """
    step = 360.0 / count
    sign = 1.0 if clockwise else -1.0
    return [(lamp_azimuth_deg + sign * step * k) % 360.0 for k in range(count)]


# ------------------------------------------------------------ registration


def _registration_features(gray: np.ndarray) -> np.ndarray:
    """What the alignment should be matched on: structure common to the set.

    The frames in a photometric set differ in *shading* by design — that
    difference is the entire signal being solved for — so matching raw
    brightness asks the fit to align the one thing that is not supposed to
    align. Print edges are identical in all of them, and a high-pass is what
    is left when the shading is removed.
    """
    high_pass = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 8.0)
    return high_pass


# What a registration correction is allowed to be. An affine fit has six
# degrees of freedom and will spend them on whatever shading survives the
# high-pass, so a fit outside these bounds is the fit drifting rather than a
# card that moved, and applying it would be worse than leaving it alone.
#
# The first version of these was guesswork — "detection error is a fraction of
# a percent" — and it was wrong. Measured across one real four-scan set, the
# detector found the same card at widths of 2875, 2896, 2980 and 2993 pixels:
# a spread of 4.1%. The fits ECC proposed to correct that were right (1.040
# and 1.033) and the guard refused both for exceeding 3%, which left two of
# four frames unregistered and the card's own print embossed twice.
#
# So these are sized against what a real set does, with headroom, while
# staying far from the range where a fit has clearly gone wrong. Shear stays
# tight: the frames are rectified images of a flat card, and a fit that wants
# to skew one is chasing brightness rather than geometry.
MAX_REGISTRATION_SCALE_DRIFT = 0.08
MAX_REGISTRATION_SHIFT_FRAC = 0.06
MAX_REGISTRATION_SHEAR = 0.03


def _plausible_registration(warp: np.ndarray, shape: tuple) -> bool:
    linear = warp[:, :2]
    scale_x = float(np.hypot(linear[0, 0], linear[1, 0]))
    scale_y = float(np.hypot(linear[0, 1], linear[1, 1]))
    if not (1 - MAX_REGISTRATION_SCALE_DRIFT <= scale_x <= 1 + MAX_REGISTRATION_SCALE_DRIFT):
        return False
    if not (1 - MAX_REGISTRATION_SCALE_DRIFT <= scale_y <= 1 + MAX_REGISTRATION_SCALE_DRIFT):
        return False
    # Columns of a rigid-plus-scale warp stay perpendicular; shear means the
    # fit has started deforming the card to chase brightness.
    shear = abs(float(np.dot(linear[:, 0], linear[:, 1])) / max(scale_x * scale_y, 1e-6))
    if shear > MAX_REGISTRATION_SHEAR:
        return False
    height, width = shape[:2]
    limit = MAX_REGISTRATION_SHIFT_FRAC * max(height, width)
    return abs(float(warp[0, 2])) <= limit and abs(float(warp[1, 2])) <= limit


def _geometric_residual(ref_features, img_features, warp, shape) -> float:
    """How far apart the two frames still are after the fit, in grey levels.

    Measured on the high-pass, so it answers "do the print edges line up"
    rather than "are these the same brightness" — the frames are not supposed
    to be the same brightness.
    """
    height, width = shape[:2]
    moved = cv2.warpAffine(
        img_features, warp, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    margin = max(40, int(0.05 * min(height, width)))
    a = ref_features[margin:-margin, margin:-margin]
    b = moved[margin:-margin, margin:-margin]
    return float(np.abs(a - b).mean())


def _scale_about_centre(factor: float, shape: tuple) -> np.ndarray:
    height, width = shape[:2]
    centre_x, centre_y = width / 2.0, height / 2.0
    return np.array(
        [[factor, 0.0, centre_x - factor * centre_x], [0.0, factor, centre_y - factor * centre_y]],
        dtype=np.float32,
    )


# Candidate scales for the seed search. The detector's own spread across a
# real four-scan set was 4.1%, so the search has to cover more than that or it
# hands ECC a starting point it cannot walk from.
SEED_SCALES = tuple(round(1.0 + step * 0.01, 2) for step in range(-6, 7))


def _correlate(ref_features: np.ndarray, img_features: np.ndarray) -> tuple[tuple[float, float], float]:
    """Windowed phase correlation. Windowed because an unwindowed transform
    reads the frame's own border as the strongest feature and locks onto it."""
    window = cv2.createHanningWindow(ref_features.shape[::-1], cv2.CV_32F)
    try:
        shift, response = cv2.phaseCorrelate(
            ref_features.astype(np.float64), img_features.astype(np.float64), window.astype(np.float64)
        )
    except cv2.error:
        return (0.0, 0.0), -1.0
    return (float(shift[0]), float(shift[1])), float(response)


def _estimate_seed(ref_features: np.ndarray, img_features: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """A starting warp good enough for ECC to refine from, found globally.

    ECC is a local optimiser and the high-pass it matches on is deliberately
    spiky, so it has to be started close. Translation alone is not close
    enough: the detector's spread across a real set was 4.1% of scale, which
    is eighty pixels of displacement at the card's edges, and from identity
    ECC simply never found it.

    So the scale is searched too — coarsely, by trying a handful of candidates
    and keeping whichever correlates best. One dimension, at quarter size,
    over a dozen values: cheap, and it turns a problem ECC cannot solve into
    one it only has to polish.
    """
    small_ref = cv2.resize(ref_features, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_img = cv2.resize(img_features, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    best_warp = np.eye(2, 3, dtype=np.float32)
    best_response = -1.0
    for factor in SEED_SCALES:
        if factor == 1.0:
            candidate = small_img
        else:
            candidate = cv2.warpAffine(
                small_img,
                _scale_about_centre(factor, small_img.shape),
                small_img.shape[::-1],
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        (dx, dy), response = _correlate(small_ref, candidate)
        if response > best_response:
            best_response = response
            warp = _scale_about_centre(factor, ref_features.shape)
            warp[0, 2] += dx / scale
            warp[1, 2] += dy / scale
            best_warp = warp
    return best_warp


def register_to_reference(reference: np.ndarray, image: np.ndarray, report: dict | None = None) -> np.ndarray:
    """Refine `image` onto `reference` with an affine fit.

    The canonical warp already puts every capture in the same frame, so this
    only has to clean up corner-detection error — but photometric stereo is
    very sensitive to exactly that: a two-pixel shift turns every
    high-contrast print edge into a fake ridge in the normal map.

    Affine rather than rigid, because the error to correct is not rigid. Each
    frame is warped from its own detection of the card, and those quads differ
    in *size* as well as position — measured on a real set, 0.55% of the
    card's width, which is eight pixels. Rotation-and-translation cannot
    express that, so it left the card edge standing in five separate places
    within fourteen pixels of the real one.

    Coarse-to-fine: a quarter-size pass to catch the gross offset, then a
    half-size pass to refine it. The linear part of an affine warp is
    scale-invariant and only the translation column needs rescaling.
    """
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ref_features = _registration_features(ref_gray)
    img_features = _registration_features(img_gray)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    warp = np.eye(2, 3, dtype=np.float32)
    # Seed the translation globally before refining. ECC is a local optimiser
    # and the high-pass it matches on is deliberately spiky, which leaves it a
    # capture range of about ten pixels — measured, it corrected a 10px offset
    # and silently did nothing to a 15px one. Phase correlation has no such
    # limit: it finds the shift from the whole image at once, and only the
    # scale and rotation are left for ECC to work out from close range.
    warp = _estimate_seed(ref_features, img_features)
    seed = warp.copy()

    fitted = False
    for scale in (0.25, 0.5):
        small_ref = cv2.resize(ref_features, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_img = cv2.resize(img_features, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Only the translation column is resolution-dependent; the linear
        # part of an affine warp means the same thing at any scale.
        attempt = warp.copy()
        attempt[0, 2] *= scale
        attempt[1, 2] *= scale
        try:
            _, attempt = cv2.findTransformECC(
                small_ref, small_img, attempt, cv2.MOTION_AFFINE, criteria, None, 5
            )
        except cv2.error:
            # ECC diverges on a pair with too little shared structure. Keep
            # whatever the coarser pass found; an unregistered frame is
            # better than a wrongly-registered one, and the canonical warp
            # already had it approximately right.
            break
        attempt[0, 2] /= scale
        attempt[1, 2] /= scale
        if not _plausible_registration(attempt, ref_gray.shape):
            if report is not None:
                report["rejected_at"] = scale
                report["rejected_warp"] = [round(float(v), 4) for v in attempt.ravel()]
            break
        warp = attempt
        fitted = True

    if report is not None:
        report["seed"] = {
            "scale": round(float(np.hypot(seed[0, 0], seed[1, 0])), 5),
            "shift_px": [round(float(seed[0, 2]), 1), round(float(seed[1, 2]), 1)],
        }
        report["fitted"] = fitted
        report["shift_px"] = [round(float(warp[0, 2]), 1), round(float(warp[1, 2]), 1)]
        report["scale"] = [
            round(float(np.hypot(warp[0, 0], warp[1, 0])), 5),
            round(float(np.hypot(warp[0, 1], warp[1, 1])), 5),
        ]
        report["residual"] = round(_geometric_residual(ref_features, img_features, warp, ref_gray.shape), 3)

    if not fitted:
        return image

    h, w = ref_gray.shape[:2]
    return cv2.warpAffine(
        image, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )


# ------------------------------------------------------- photometric stereo


def solve_normals(grays: list[np.ndarray], lights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares Lambertian solve: I_k = albedo * (n . l_k) for every pixel.

    Returns (normals HxWx3 float32, albedo HxW float32). With 3 lights this
    is exact; with 4+ it is overdetermined, which is what makes the four-scan
    flatbed set robust to the one direction where a scratch happens to be
    invisible.
    """
    stack = np.stack([g.astype(np.float32) / 255.0 for g in grays], axis=0)  # (N,H,W)
    n_lights, h, w = stack.shape
    observations = stack.reshape(n_lights, h * w)

    # g = albedo * n, recovered by the pseudo-inverse of the light matrix.
    pinv = np.linalg.pinv(lights.astype(np.float32))  # (3,N)
    g = pinv @ observations  # (3, H*W)

    albedo = np.linalg.norm(g, axis=0)
    safe = np.maximum(albedo, 1e-6)
    normals = (g / safe).T.reshape(h, w, 3).astype(np.float32)

    # Pixels with no usable signal (deep shadow, clipped highlight) get a
    # flat normal rather than a random direction from dividing by noise.
    flat = albedo.reshape(h, w) < 1e-3
    normals[flat] = (0.0, 0.0, 1.0)

    return normals, albedo.reshape(h, w).astype(np.float32)


# Floor on the autoscale divisor. Below this much shading deviation the card
# is flat to within sensor noise, and dividing by the noise would stretch it
# across the full range and render a clean card as a field of fake defects.
# 0.02 sits an order of magnitude above scanner read noise and well below the
# ~0.05 deviation a real scratch produces, so on an ordinary card this is the
# operating point and the autoscale only engages on genuinely rough surfaces.
MIN_RELIEF_DEVIATION = 0.02


# How many noise sigmas a deviation must clear before it counts as relief.
NOISE_THRESHOLD_SIGMAS = 3.0
# MAD -> sigma for normally distributed noise.
MAD_TO_SIGMA = 1.4826


# Where the noise floor is read from: the quietest tenth of the card, tile by
# tile. The whole-card median assumed "a card is mostly flat", which is only
# true if you don't count the print — measured on a real scan the whole-card
# estimate put the floor at 0.222 while the quietest tiles sat at 0.109, so
# the threshold was measuring printed detail and erasing everything under it.
# A scratch six grey levels deep is 0.024 of deviation: it never had a chance.
NOISE_FLOOR_PERCENTILE = 10.0
NOISE_TILE_PX = 64


def _noise_sigma(deviation: np.ndarray) -> float:
    """Robust noise estimate, read from the flattest part of the card.

    Measured per tile and taken at a low percentile, because the quantity
    wanted is how much the card's *undisturbed* surface varies — and the
    median over the whole card is dominated by print, not by the flat stock
    between it. Using the median rather than the standard deviation within a
    tile keeps a defect crossing that tile from inflating its own floor.

    Measured against a synthetic groove on a real scan, moving to this
    estimate multiplied the rendered signal by 5x at 25 grey levels of depth
    and by 14x at 12 — the difference between a visible scratch and nothing.
    """
    if deviation.size == 0:
        return 0.0
    tile = NOISE_TILE_PX
    height, width = deviation.shape[:2]
    mads = []
    for y in range(0, height - tile + 1, tile):
        for x in range(0, width - tile + 1, tile):
            patch = deviation[y : y + tile, x : x + tile]
            mads.append(float(np.median(np.abs(patch - np.median(patch)))))
    if not mads:
        mads = [float(np.median(np.abs(deviation - np.median(deviation))))]
    return MAD_TO_SIGMA * float(np.percentile(np.asarray(mads), NOISE_FLOOR_PERCENTILE))


# The shading deviation that maps to the top of the output range when the
# render is being *measured* rather than looked at. An absolute number, in the
# same 0-1 units as the shading deviation itself.
#
# It has to be absolute. The displayed render normalizes by the card's own
# 99.5th percentile, which is right for a picture — every card fills the
# range — and catastrophic for a measurement, because it makes every number
# relative to that card's own worst feature. Measured on one card scanned
# clean and then again after being creased and scratched, the damaged card's
# normalizer came out 2.55x the clean one's (0.703 vs 0.276), so its ordinary
# ink was scaled down by that much and it measured *less* defect area than
# when it was undamaged: surface grade 6 damaged against 3 clean. Adding
# damage to a card improved its score.
#
# The value is calibrated against that one pair and should be treated as
# provisional. What it is not is a free parameter for taste: changing it
# changes every surface number, which is exactly the coupling that made the
# autoscale wrong.
MEASUREMENT_REFERENCE_DEVIATION = 0.8


def _autoscale(
    deviation: np.ndarray, gain: float, headroom: float = 110.0, reference: float | None = None
) -> np.ndarray:
    """Map a signed deviation field onto a mid-gray-centered 8-bit image.

    Two steps, in this order. First soft-threshold at a few sigmas of the
    measured noise: anything smaller is not evidence of shape, and skipping
    this makes the second step actively harmful — a clean card's render
    would be the sensor's own noise stretched to full contrast, which reads
    as a surface covered in defects.

    Then scale. `reference` is the deviation that maps to the top of the
    range: pass one to measure, leave it out to display.

    Left out, it is the 99.5th percentile of what survived the threshold —
    a high percentile rather than the maximum, because relief always has a
    few extreme outliers (the warp seam at the card's physical boundary, a
    dust speck) and normalizing by those would crush every real defect
    toward invisibility. Same reasoning as `surface._normalize_robust`.

    That autoscale is for looking at, never for measuring. It makes the
    output relative to the card in front of it, so two cards can't be
    compared and a card can't even be compared with itself before and after
    damage — see MEASUREMENT_REFERENCE_DEVIATION.
    """
    sigma = _noise_sigma(deviation)
    magnitude = np.maximum(np.abs(deviation) - NOISE_THRESHOLD_SIGMAS * sigma, 0.0)
    shrunk = np.sign(deviation) * magnitude

    if reference is None:
        reference = float(np.percentile(magnitude, 99.5))

    # A soft knee rather than a hard clip. Linear in the middle — tanh(x) is
    # x for small x, so ordinary relief is unchanged — and compressing at the
    # ends, where the old mapping drove every strong print edge to pure black
    # or white. That flattening is what made the render read as hard outlines
    # instead of a surface: once two features both clip, nothing distinguishes
    # a deep gouge from a printed rule.
    scale = gain / max(reference, MIN_RELIEF_DEVIATION)
    return np.clip(128.0 + headroom * np.tanh(scale * shrunk), 0, 255).astype(np.uint8)


def shade_normals(
    normals: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    gain: float,
    reference: float | None = None,
) -> np.ndarray:
    """Render the normal map under a virtual raking light, albedo discarded.

    Discarding albedo is the whole point: the output shows shape only, so
    artwork disappears and a scratch that is invisible against busy print
    becomes a plain bright or dark line.
    """
    light = light_vectors([azimuth_deg], elevation_deg)[0].astype(np.float32)
    shading = normals @ light
    # A perfectly flat card shades to sin(elevation); center on that so flat
    # regions land on mid-gray and only deviations carry signal.
    flat_level = float(np.sin(np.radians(elevation_deg)))
    return _autoscale(shading - flat_level, gain, reference=reference)


# Where the print is, and how far past its edges to reach. Ink sits proud of
# the cardstock, so a solved normal map contains the artwork as real relief —
# measured on one card, 87% of the pixels the surface stage flagged as defects
# lay on printed ink. Photometric stereo removes albedo; it does not remove
# the physical thickness of the ink laid on top.
INK_GRADIENT_PERCENTILE = 88.0
INK_MASK_DILATE_PX = 9


def ink_mask(albedo: np.ndarray, cfg: dict) -> np.ndarray:
    """The printed areas, found in the solved albedo.

    Albedo is the card's own colour with the shading divided out, so its
    gradient is exactly where ink starts and stops — and that is where ink
    relief lives. Dilated, because the raised edge of a printed line extends
    past the colour boundary that produced it.
    """
    gray = albedo if albedo.ndim == 2 else cv2.cvtColor(albedo, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.0)
    cutoff = float(np.percentile(gradient, cfg.get("ink_gradient_percentile", INK_GRADIENT_PERCENTILE)))
    mask = (gradient > cutoff).astype(np.uint8)
    dilate = int(cfg.get("ink_mask_dilate_px", INK_MASK_DILATE_PX))
    return cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))


def suppress_ink(
    relief: np.ndarray, albedo: np.ndarray, cfg: dict, percentile: float | None = None
) -> np.ndarray:
    """Flatten the relief wherever the card is printed.

    Applied to the measured render and never to the displayed one. The
    picture should show the ink — it is most of what makes a relief render
    legible as a card — but counting it as damage is what made an undamaged
    card grade 3 while the same card, creased and scratched, graded 6.

    The cost is real and worth stating: a scratch running through printed
    artwork is suppressed along with the print, so this trades sensitivity
    inside the art for the ability to compare two cards at all. Measured on
    one card clean and then damaged, it moved the pair from 6/6 (with the
    clean card reading *worse*) to 10 clean against 9 damaged.

    `percentile` overrides how much is called ink, because the same signal
    means different things in different places. In the card's interior,
    relief that follows an albedo edge is print. At the border it is just as
    likely to be *whitening* — exposed cardstock is a colour change as well
    as a physical one — so suppressing as hard there deletes the defect being
    looked for. Measured on a deliberately whitened edge, tightening the
    interior setting from 88 to 84 took that edge from 1.21% wear to 0.15%
    and handed the card back a clean edges grade.
    """
    if albedo is None:
        return relief
    if percentile is not None:
        cfg = {**cfg, "ink_gradient_percentile": percentile}
    out = relief.copy()
    out[ink_mask(albedo, cfg) > 0] = 128
    return out


def normal_map_visualization(normals: np.ndarray) -> np.ndarray:
    """Standard tangent-space normal-map colors, for eyeballing the solve."""
    encoded = (normals * 0.5 + 0.5) * 255.0
    rgb = np.clip(encoded, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def photometric_card_vision(
    images_bgr: list[np.ndarray],
    azimuths_deg: list[float],
    cfg: dict,
    registration_reference: np.ndarray | None = None,
) -> CardVisionResult:
    """Full photometric-stereo Card Vision from 3+ differently-lit captures.

    `registration_reference` is the frame everything is aligned onto, and so
    the frame the render comes out in. Pass the flat capture's own warp and
    the relief lands in the same frame as the image the report cross-fades it
    against; measured without it, the two sat three pixels and half a percent
    of scale apart, which is visible on a slider. It contributes no light of
    its own — only its geometry.
    """
    if len(images_bgr) < 3:
        raise ValueError("photometric stereo needs at least 3 differently-lit captures")
    if len(images_bgr) != len(azimuths_deg):
        raise ValueError("one light azimuth is required per capture")

    reference = images_bgr[0] if registration_reference is None else registration_reference
    registration: list[dict] = []
    aligned = []
    for img in images_bgr:
        frame_report: dict = {}
        aligned.append(register_to_reference(reference, img, frame_report))
        registration.append(frame_report)
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in aligned]

    lights = light_vectors(azimuths_deg, cfg.get("light_elevation_deg", DEFAULT_ELEVATION_DEG))
    normals, albedo = solve_normals(grays, lights)

    # A little smoothing on the normals kills per-pixel sensor noise without
    # touching a scratch, which is many pixels long.
    normals = cv2.GaussianBlur(normals, (0, 0), cfg.get("normal_smoothing_sigma", 1.0))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)

    def render(gain: float, reference: float | None = None) -> np.ndarray:
        return blank_warp_seam(
            shade_normals(normals, RENDER_AZIMUTH_DEG, RENDER_ELEVATION_DEG, gain, reference), cfg
        )

    def swept(reference: float) -> np.ndarray:
        """The card under every light at once, kept at its worst per pixel."""
        worst = None
        for azimuth in cfg.get("measurement_azimuths_deg", MEASUREMENT_AZIMUTHS_DEG):
            lit = shade_normals(normals, azimuth, RENDER_ELEVATION_DEG, 1.0, reference)
            deviation = np.abs(lit.astype(np.int16) - 128)
            worst = deviation if worst is None else np.maximum(worst, deviation)
        return blank_warp_seam(np.clip(128 + worst, 0, 255).astype(np.uint8), cfg)

    # The picture autoscales so every card fills the range and is worth
    # looking at. The measurement does not: it is rendered at unit gain
    # against a fixed absolute reference, so a given physical relief maps to
    # the same number on every card and the thresholds in thresholds.json
    # mean one fixed thing. Sharing one autoscaled render between the two —
    # which is what this did — made every surface number relative to the
    # card's own worst feature.
    # The picture is lit from one direction, because that is what a lit surface
    # looks like and a render averaged over four lights reads flat. The
    # measurement is swept, because a fixed light has a blind direction and the
    # grade must not depend on which one was picked.
    relief = render(cfg.get("relief_gain", 1.0))
    albedo_u8 = np.clip(albedo * 255.0, 0, 255).astype(np.uint8)
    reference = cfg.get("measurement_reference_deviation", MEASUREMENT_REFERENCE_DEVIATION)
    measurement = _blank_border(suppress_ink(swept(reference), albedo_u8, cfg), cfg)
    edge_measurement = suppress_ink(
        render(1.0, reference), albedo_u8, cfg, cfg.get("ink_gradient_percentile_edges")
    )

    return CardVisionResult(
        relief=relief,
        method="photometric_stereo",
        light_count=len(images_bgr),
        normal_map=normal_map_visualization(normals),
        albedo=albedo_u8,
        roughness_pct=_roughness_pct(measurement, cfg),
        registration=registration,
        measurement_relief=measurement,
        edge_relief=edge_measurement,
    )


# --------------------------------------------------- single-image fallback


def single_image_card_vision(image_bgr: np.ndarray, cfg: dict) -> CardVisionResult:
    """Approximate Card Vision from one ordinary capture.

    Two filters do the work. A high-pass drops the slowly-varying part of
    the image, which is where flat color fields and lighting gradients live.
    Then a chroma-gradient weight attenuates what is left wherever hue is
    also changing: ink boundaries move color, a scratch through clear
    laminate moves brightness only. What survives both is *mostly* geometry.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[:, :, 0]

    base = cv2.GaussianBlur(luminance, (0, 0), cfg.get("albedo_sigma", 12.0))
    residual = luminance - base

    chroma = lab[:, :, 1:3]
    chroma_gradient = np.zeros(luminance.shape, dtype=np.float32)
    for c in range(2):
        gx = cv2.Sobel(chroma[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(chroma[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        chroma_gradient += gx * gx + gy * gy
    chroma_gradient = np.sqrt(chroma_gradient)
    chroma_gradient = cv2.GaussianBlur(chroma_gradient, (0, 0), 1.5)

    k = cfg.get("chroma_suppression", 24.0)
    print_weight = 1.0 / (1.0 + (chroma_gradient / k) ** 2)

    # /255 puts the residual on the same 0-1 footing as the photometric
    # shading deviation, so one autoscale serves both paths.
    deviation = residual * print_weight / 255.0
    gain = cfg.get("single_image_gain", 1.0)
    relief_u8 = blank_warp_seam(_autoscale(deviation, gain), cfg)
    measurement = blank_warp_seam(
        _autoscale(
            deviation, 1.0, reference=cfg.get("measurement_reference_deviation", MEASUREMENT_REFERENCE_DEVIATION)
        ),
        cfg,
    )

    return CardVisionResult(
        relief=relief_u8,
        method="single_image",
        light_count=1,
        roughness_pct=_roughness_pct(measurement, cfg),
        measurement_relief=measurement,
    )


# How far in from the card's edge the surface measurement starts, in
# millimetres. This is the boundary between two attributes, not a filter.
#
# The surface stage measures the card's face; the corners and edges stage
# measures its boundary. That is how the grading services divide their own
# sub-grades, and measuring the same strip in both double-counts it.
#
# It also happens to be where the measurement is worst. Straightening the card
# leaves a ridge of false relief hugging its boundary: on a card with no
# damage at all, defect density inside this strip ran a hundred times the
# density of the card's interior (0.144% against 0.001%), and the three marks
# left on that card after print suppression all sat at x=0.7-0.8mm — every one
# of them an artifact, together costing a clean card a grade.
#
# Nothing is lost by it. A crease at a corner is still found, and still caps
# the card, through the corners and edges stage — measured on a real creased
# corner, 24.2% wear against 0.00% on the same corner undamaged.
SURFACE_EDGE_MARGIN_MM = 1.5


def _blank_border(relief: np.ndarray, cfg: dict) -> np.ndarray:
    """Flatten the strip the corners and edges stage owns."""
    margin_mm = cfg.get("surface_edge_margin_mm", SURFACE_EDGE_MARGIN_MM)
    if margin_mm <= 0:
        return relief
    from pipeline.dimensions import NOMINAL_WIDTH_MM

    px_per_mm = relief.shape[1] / NOMINAL_WIDTH_MM
    margin = int(round(margin_mm * px_per_mm))
    if margin <= 0:
        return relief
    out = relief.copy()
    out[:margin, :] = 128
    out[-margin:, :] = 128
    out[:, :margin] = 128
    out[:, -margin:] = 128
    return out


def blank_warp_seam(relief: np.ndarray, cfg: dict) -> np.ndarray:
    """Flatten the band at the card's physical boundary.

    `perspective_correct` maps the detected quad onto the canonical rectangle,
    and corner detection is not sub-pixel perfect — so the outermost pixels
    are an interpolated blend of card edge and background rather than card.
    In a relief render that seam becomes a bright line right around the card:
    measured on a real render its deviation was 76 at the second column,
    falling to 8 by the eighth.

    Set to mid-gray rather than cropped, so the render still lines up pixel
    for pixel with everything else in the canonical frame. Mid-gray is this
    render's "flat", which is the honest thing to say about a band where
    there is no measurement.
    """
    margin = int(cfg.get("physical_edge_margin_px", 0))
    if margin <= 0:
        return relief
    trimmed = relief.copy()
    trimmed[:margin, :] = 128
    trimmed[-margin:, :] = 128
    trimmed[:, :margin] = 128
    trimmed[:, -margin:] = 128
    return trimmed


def _roughness_pct(relief: np.ndarray, cfg: dict) -> float:
    """Share of the card whose relief leaves the flat band around mid-gray.

    An indicator to sanity-check a vision judgment against, not a grade —
    a heavily embossed or textured card legitimately scores high here.
    """
    band = cfg.get("flat_band", 18)
    deviation = np.abs(relief.astype(np.int16) - 128)
    return float(100.0 * np.count_nonzero(deviation > band) / deviation.size)


def card_vision(
    images_bgr: list[np.ndarray], azimuths_deg: list[float] | None, cfg: dict
) -> CardVisionResult:
    """Photometric stereo when there are enough lit captures, else the
    single-image approximation. Callers pass whatever they have."""
    if len(images_bgr) >= 3 and azimuths_deg:
        return photometric_card_vision(images_bgr, azimuths_deg, cfg)
    return single_image_card_vision(images_bgr[0], cfg)
