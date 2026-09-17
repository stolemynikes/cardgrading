#!/usr/bin/env python3
"""Draw a fictional trading card, and light it from several directions.

Used only to populate the demo report, so the UI can be judged against
something that looks like a real card instead of a grey rectangle. The card
is invented — original name, original artwork, original back design — because
the point is the *format*, and reproducing a real card's art to preview a
layout would be both unnecessary and someone else's property.

Two things make it a useful demo rather than a picture:

1. It carries a **height field** as well as colour. A scratch, a dinged
   corner and some edge wear exist as geometry, not as painted-on marks, so
   photometric stereo has something real to recover and Card Vision shows
   what it would show on a real scan.
2. The front is deliberately printed off-centre, which is what most cards
   actually fail on, so the centering stage and the DINGS list have real
   content instead of a clean sweep of 10s.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 63x88mm at 600dpi. The number matters: `seed_demo` tells the pipeline this
# DPI, and the dimensions stage measures the card against the real 63x88mm
# nominal — so a mismatch here shows up as a fake miscut in the demo report.
SCAN_DPI = 600.0
CARD_W, CARD_H = 1488, 2078
MARGIN = 150  # background around the card — detect.py samples it for contrast

FONT_DIR = Path("/System/Library/Fonts/Supplemental")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def _rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill=None, outline=None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _holo_wash(img: Image.Image, box, seed: int = 3) -> None:
    """A faint iridescent shimmer over the art window.

    Not decoration: holo is the hardest thing for surface detection to tell
    apart from damage, and `surface._holo_mask` keys off *local variance* of
    saturation rather than absolute saturation — so the shimmer has to vary
    at a small spatial scale while staying subtle enough that the artwork
    still reads. Composited at low alpha rather than drawn opaque, which is
    the difference between foil and confetti.
    """
    x0, y0, x1, y1 = box
    layer = Image.new("RGB", (x1 - x0, y1 - y0), (0, 0, 0))
    ld = ImageDraw.Draw(layer)
    rng = np.random.default_rng(seed)
    for _ in range(900):
        cx, cy = int(rng.integers(0, x1 - x0)), int(rng.integers(0, y1 - y0))
        r = int(rng.integers(3, 13))
        colour = [(150, 110, 230), (70, 170, 230), (220, 150, 90), (120, 220, 180)][int(rng.integers(0, 4))]
        ld.ellipse((cx - r, cy - r, cx + r, cy + r), fill=colour)
    layer = layer.filter(ImageFilter.GaussianBlur(2.5))
    base = img.crop(box)
    img.paste(Image.blend(base, Image.blend(base, layer, 0.5), 0.34), box)


def draw_front(offset_x: int = 34, offset_y: int = 6) -> np.ndarray:
    """The card face, laid out like a 1999-era Base Set card.

    The anatomy is the reference — evolution badge, name/HP row, framed art
    window, species bar, power block, attack rows with energy costs, the
    weakness/resistance/retreat strip, flavour box, footer. The species,
    artwork and every line of text are invented: the layout is what this has
    to reproduce for the report to be judged against something realistic.

    `offset_x`/`offset_y` shift the printed panel inside the border, which is
    how a real miscut presents — the border stays the same size, the print
    moves.
    """
    img = Image.new("RGB", (CARD_W, CARD_H), (222, 178, 42))
    d = ImageDraw.Draw(img)

    # Border: a gradient, because a flat fill makes the uneven-lighting gate
    # behave unrealistically.
    for y in range(CARD_H):
        t = y / CARD_H
        d.line([(0, y), (CARD_W, y)], fill=(int(232 - 26 * t), int(190 - 30 * t), int(60 - 14 * t)))

    panel = (72 + offset_x, 68 + offset_y, CARD_W - 72 + offset_x, CARD_H - 88 + offset_y)
    px0, py0, px1, py1 = panel
    for y in range(py0, py1):
        t = (y - py0) / (py1 - py0)
        d.line([(px0, y), (px1, y)], fill=(int(238 - 16 * t), int(214 - 26 * t), int(180 - 30 * t)))
    d.rectangle(panel, outline=(176, 138, 30), width=4)
    inner_w = px1 - px0

    tiny = _font("Arial Bold Italic.ttf", 30)
    d.text((px0 + 190, py0 + 26), "Evolves from Cindling", font=tiny, fill=(40, 34, 28))
    d.text((px1 - 430, py0 + 26), "Put Emberlisk on the Stage 1 card", font=_font("Arial.ttf", 26), fill=(70, 62, 52))

    # Evolution badge: dark diamond in a gold frame, top-left.
    badge = (px0 + 30, py0 + 64, px0 + 170, py0 + 204)
    d.rectangle(badge, fill=(60, 52, 44), outline=(196, 158, 44), width=5)
    d.text((px0 + 38, py0 + 24), "STAGE 1", font=_font("Arial Bold.ttf", 26), fill=(40, 34, 28))
    d.ellipse((px0 + 62, py0 + 96, px0 + 138, py0 + 172), fill=(214, 116, 48))

    d.text((px0 + 200, py0 + 74), "Emberlisk", font=_font("Arial Bold.ttf", 92), fill=(26, 24, 22))
    d.text((px1 - 330, py0 + 86), "90 HP", font=_font("Arial Bold.ttf", 64), fill=(190, 44, 38))
    d.ellipse((px1 - 130, py0 + 78, px1 - 46, py0 + 162), fill=(214, 96, 42), outline=(140, 58, 20), width=5)

    # Art window
    art = (px0 + 46, py0 + 222, px1 - 46, py0 + 222 + int(inner_w * 0.64))
    ax0, ay0, ax1, ay1 = art
    for y in range(ay0, ay1):
        t = (y - ay0) / (ay1 - ay0)
        d.line([(ax0, y), (ax1, y)], fill=(int(40 + 130 * t), int(32 + 52 * t), int(70 - 16 * t)))
    _holo_wash(img, (ax0 + 8, ay0 + 8, ax1 - 8, ay1 - 8))
    cx, cy = (ax0 + ax1) // 2, int(ay0 + (ay1 - ay0) * 0.62)
    for radius, colour in ((250, (226, 122, 44)), (170, (244, 172, 58)), (96, (252, 224, 126))):
        d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour)
    d.polygon([(cx, cy - 370), (cx + 126, cy - 60), (cx - 126, cy - 60)], fill=(250, 204, 92))
    d.rectangle(art, outline=(196, 158, 44), width=8)
    d.rectangle((ax0 - 6, ay0 - 6, ax1 + 6, ay1 + 6), outline=(150, 116, 30), width=3)

    # Species bar
    bar = (px0 + 120, ay1 + 20, px1 - 120, ay1 + 76)
    d.rounded_rectangle(bar, radius=8, fill=(226, 206, 156), outline=(150, 116, 30), width=3)
    d.text((bar[0] + 26, bar[1] + 10), 'Ember Card. Length: 3\' 2", Weight: 61 lbs.',
           font=_font("Arial Bold Italic.ttf", 30), fill=(40, 34, 28))

    y = bar[3] + 26
    d.text((px0 + 46, y), "Card Power: Slow Burn", font=_font("Arial Bold.ttf", 44), fill=(48, 62, 150))
    y += 56
    for line in ("Once during your turn, you may turn all Energy attached",
                 "to Emberlisk into Fire Energy for the rest of the turn."):
        d.text((px0 + 46, y), line, font=_font("Arial.ttf", 32), fill=(46, 40, 34))
        y += 42

    y += 14
    for cost, name, desc, damage in (
        (2, "Cinder Flick", "Flip a coin. If heads, the Defending card is Burned.", "30"),
        (3, "Kindle Storm", "Discard an Energy attached to this card to use this attack.", "70"),
    ):
        d.line([(px0 + 40, y), (px1 - 40, y)], fill=(176, 154, 108), width=3)
        y += 16
        for i in range(cost):
            ex = px0 + 52 + (i % 2) * 56
            ey = y + (i // 2) * 56
            d.ellipse((ex, ey, ex + 48, ey + 48), fill=(214, 96, 42), outline=(140, 58, 20), width=3)
        d.text((px0 + 180, y + 6), name, font=_font("Arial Bold.ttf", 50), fill=(26, 24, 22))
        d.text((px0 + 180, y + 62), desc, font=_font("Arial.ttf", 30), fill=(72, 64, 56))
        d.text((px1 - 150, y + 14), damage, font=_font("Arial Bold.ttf", 62), fill=(26, 24, 22))
        y += 124

    d.line([(px0 + 40, y), (px1 - 40, y)], fill=(176, 154, 108), width=3)
    y += 14
    label = _font("Arial Bold.ttf", 26)
    for text, x in (("weakness", px0 + 70), ("resistance", px0 + 440), ("retreat cost", px1 - 330)):
        d.text((x, y), text, font=label, fill=(120, 46, 46))
    y += 36
    d.ellipse((px0 + 92, y, px0 + 136, y + 44), fill=(70, 130, 200))
    d.ellipse((px0 + 470, y, px0 + 514, y + 44), fill=(150, 120, 90))
    d.text((px0 + 524, y + 4), "-30", font=_font("Arial Bold.ttf", 34), fill=(46, 40, 34))
    for i in range(2):
        d.ellipse((px1 - 300 + i * 60, y, px1 - 256 + i * 60, y + 44), fill=(196, 196, 196), outline=(120, 120, 120), width=3)

    # Flavour box and footer are anchored to the bottom of the panel rather
    # than flowed after the attacks — the attack block's height varies with
    # its text, and letting it push these down ran them off the card.
    flavour = (px0 + 46, py1 - 168, px1 - 46, py1 - 72)
    d.rectangle(flavour, outline=(176, 154, 108), width=3)
    d.text((flavour[0] + 20, flavour[1] + 12), "Sheds embers when it sleeps. Trainers keep it",
           font=_font("Arial Italic.ttf", 30), fill=(60, 52, 44))
    d.text((flavour[0] + 20, flavour[1] + 50), "well away from anything dry.    LV. 32   #17",
           font=_font("Arial Italic.ttf", 30), fill=(60, 52, 44))

    d.text((px0 + 46, py1 - 54), "Illus. Card Pre-Grader", font=_font("Arial Italic.ttf", 26), fill=(110, 100, 88))
    d.text((px0 + 400, py1 - 52), "(c) 2026 Card Pre-Grader - demo card, not a real release",
           font=_font("Arial.ttf", 20), fill=(126, 116, 104))
    d.text((px1 - 190, py1 - 54), "017/120", font=_font("Arial Bold.ttf", 30), fill=(80, 72, 64))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_back() -> np.ndarray:
    """An original back design — same role as a real one (dark, centred
    emblem, uniform border) without copying anybody's."""
    img = Image.new("RGB", (CARD_W, CARD_H), (26, 38, 92))
    d = ImageDraw.Draw(img)
    for y in range(CARD_H):
        t = y / CARD_H
        d.line([(0, y), (CARD_W, y)], fill=(int(30 + 26 * t), int(44 + 30 * t), int(104 + 34 * t)))

    _rounded(d, (66, 66, CARD_W - 66, CARD_H - 66), 16, outline=(214, 178, 60), width=8)

    cx, cy = CARD_W // 2, CARD_H // 2
    for radius, colour in ((470, (20, 30, 76)), (400, (216, 180, 62)), (330, (28, 40, 96))):
        d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour)
    d.polygon([(cx, cy - 250), (cx + 216, cy + 150), (cx - 216, cy + 150)], fill=(216, 180, 62))
    d.ellipse((cx - 90, cy - 30, cx + 90, cy + 150), fill=(28, 40, 96))
    d.text((cx - 250, cy + 430), "PRE-GRADER", font=_font("Arial Bold.ttf", 92), fill=(216, 180, 62))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def height_field(front: bool) -> np.ndarray:
    """Physical relief, independent of what's printed. Negative digs in.

    These are the three defect shapes the pipeline is built to find, so the
    demo exercises the real code path rather than a happy one.
    """
    height = np.zeros((CARD_H, CARD_W), np.float32)

    if front:
        cv2.line(height, (300, 1500), (1150, 620), -1.0, 5)        # scratch across the art
        cv2.line(height, (820, 300), (1180, 470), -0.6, 3)         # lighter scuff
        cv2.circle(height, (CARD_W - 60, CARD_H - 58), 78, -0.8, -1)  # dinged bottom-right corner
    else:
        cv2.circle(height, (54, 52), 92, -0.9, -1)                 # dinged top-left corner
        cv2.circle(height, (CARD_W - 50, 60), 70, -0.7, -1)
        height[:, :26] -= 0.55                                     # worn left edge
        height[CARD_H - 30:, :] -= 0.5                             # worn bottom edge

    return cv2.GaussianBlur(height, (0, 0), 2.0)


def apply_wear(albedo: np.ndarray, front: bool) -> np.ndarray:
    """Paint the whitening that goes with the physical damage.

    Chipping isn't only a change of shape — it tears the printed layer and
    exposes white cardstock underneath, and *that* is what Stage 3 measures.
    Height alone gives a card that looks damaged under Card Vision and still
    grades a clean 10 on corners and edges, which is not how a real card
    behaves. Kept co-located with `height_field` so the two agree.
    """
    worn = albedo.copy()
    rng = np.random.default_rng(7)

    def chip(centre, radius, density=0.55):
        y0, y1 = max(0, centre[1] - radius), min(CARD_H, centre[1] + radius)
        x0, x1 = max(0, centre[0] - radius), min(CARD_W, centre[0] + radius)
        patch = worn[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (yy - centre[1]) ** 2 + (xx - centre[0]) ** 2 <= radius ** 2
        speckle = rng.random(patch.shape[:2]) < density
        mask = inside & speckle
        patch[mask] = (238, 240, 242)

    def wear_strip(y_slice, x_slice, density=0.4):
        patch = worn[y_slice, x_slice]
        mask = rng.random(patch.shape[:2]) < density
        patch[mask] = (236, 238, 240)

    # Tuned so the demo lands on a believable mid-grade card: one clearly
    # dinged corner on the front, more wear on the back, light edge rub. A
    # heavier hand grades it a 3, which fills the report with damage but
    # stops being representative of what you'd actually be scanning.
    # Scaled against the *crop*, not the card: Stage 3 sizes each corner
    # window from the measured border width, which on this card is only tens
    # of pixels — so a chip that looks tiny next to a 1488px card still fills
    # a large share of the window it's judged in.
    if front:
        chip((CARD_W - 18, CARD_H - 16), 9, 0.20)
    else:
        chip((14, 13), 11, 0.22)
        chip((CARD_W - 13, 15), 8, 0.16)
        wear_strip(slice(0, CARD_H), slice(0, 4), 0.04)

    return worn


def _normals(height: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(height)
    normals = np.dstack([-gx, -gy, np.ones_like(height)])
    return normals / np.linalg.norm(normals, axis=2, keepdims=True)


def light(albedo: np.ndarray, height: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Shade the card as if lit from one direction, and drop it on a dark
    background with a margin — the same thing a scan produces."""
    elev, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    vector = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)], np.float32)
    shading = np.clip(_normals(height) @ vector, 0, None)[:, :, None]
    # Renormalise so a flat card sits at its own albedo whatever the
    # elevation is — otherwise every scan in the set has a different overall
    # exposure and the solve reads that as shape.
    lit = np.clip(albedo.astype(np.float32) * shading / np.sin(elev), 0, 255).astype(np.uint8)

    canvas = np.full((CARD_H + 2 * MARGIN, CARD_W + 2 * MARGIN, 3), 24, np.uint8)
    canvas[MARGIN:MARGIN + CARD_H, MARGIN:MARGIN + CARD_W] = lit
    return canvas


def rotate_on_glass(image: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Turn the card clockwise on the scanner glass, as the capture protocol
    for photometric stereo describes."""
    for _ in range(quarter_turns % 4):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image
