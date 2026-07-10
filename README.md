# Card Pre-Grader

A local Python CLI that takes phone photos of a Pokémon card and produces a PSA-style
pre-grade estimate: sub-scores for centering, corners/edges, and surface condition,
plus an overall grade estimate. Built for personal use — not a commercial product,
not a replacement for an actual grading service.

**Status: feature-complete, not yet validated against real card photos.** Everything
described below has been built and unit/integration-tested with synthetic images.
Real-world accuracy is unknown until it's run against actual phone photos of actual
cards — see "What's not proven yet" below.

## What this is (and isn't)

- A **command-line tool**, not an app. There's no GUI, no web server, no way for a
  phone to connect directly to this — you take photos with your phone, get them onto
  the machine running this tool (AirDrop, cable, cloud sync — whatever), and run the
  CLI against those image files.
- A **hybrid** system: deterministic computer vision (OpenCV) does centering/corner/
  edge measurement and surface defect visualization; a Claude vision call does the one
  judgment call that's genuinely hard to get right algorithmically (telling a real
  surface defect apart from holo-foil sparkle or normal print detail). The LLM step is
  explicitly labeled "indicative, not definitive" everywhere it shows up in the report.
- Calibrated against **guesses**, not real PSA data, as of this writing. All the
  tolerance tables and thresholds in `calibration/thresholds.json` are reasonable
  starting points, not verified against known-grade cards yet.

## How it works — pipeline overview

Every stage operates on **perspective-corrected canonical images** (1500×2100 px,
matching a card's real 63:88mm aspect ratio) so that all downstream measurements are
in a consistent coordinate system regardless of how the original photo was framed.

### Stage 1 — Detect & normalize (`pipeline/detect.py`)
Finds the card's contour against the dark matte background and perspective-corrects
it to the canonical size. Runs a set of capture-quality gates and rejects the photo
(asks for a retake) if any fail:
- **Resolution** — image too small/cropped
- **Tilt** — camera too far off-perpendicular (corner angles deviate too far from 90°)
- **Aspect ratio** — sanity-checks the corrected image against the real 63:88 card ratio
- **Glare** — blown-out highlight clusters on the card face
- **Uneven lighting** — brightness gradient across the card's *border ring specifically*
  (not the whole face — printed color contrast between the border and inner panel
  would otherwise look identical to a lighting problem)

Card detection thresholds on **color distance from the sampled background**, not
absolute brightness — a plain brightness split would fail on dark Pokémon card backs,
which are often *darker* than a "dark matte" background, not brighter.

There's a second, lighter entry point, `align_for_surface()`, used only for the angled
raking-light surface shot — it skips the tilt/glare/uneven-lighting gates, since that
shot is *deliberately* angled and unevenly lit by design (that's what makes raking
light work), and those gates would reject a correctly-captured photo.

### Stage 2 — Centering (`pipeline/centering.py`)
Pure geometry: finds the boundary between the card's printed border and its inner
artwork/text panel on all four sides (via Canny edge detection along sampled bands),
computes left/right and top/bottom ratios, and grades them against PSA-style tolerance
tables (55/45 → grade 10, 60/40 → grade 9, etc., with a looser table for the back).
Produces an overlay image showing exactly where it thinks the border/panel boundary is.

### Stage 3 — Corners & edges (`pipeline/corners_edges.py`)
Crops the four corners and four edge strips, and runs a filter stack (CLAHE contrast
boost + blue-channel isolation + adaptive threshold) to turn "whitening" — chipped
corners/edges revealing white cardstock — into countable blobs. Crop sizes are derived
from Stage 2's *measured* border widths (not fixed pixel sizes), so the analysis window
never accidentally crosses from the border into the inner panel — which would otherwise
misread the normal border/panel color transition as a huge fake defect on every card.

### Stage 4 — Surface (`pipeline/surface.py`)
Uses a second, angled photo lit with raking light to reveal surface texture. Runs a
Laplacian (high-pass, catches scratches) + difference-of-Gaussians (catches print
lines) filter stack, with a holo-foil mask that excludes iridescent regions — detected
by *local variance* of saturation (the flicker of foil), not absolute saturation, since
ordinary vividly-colored card borders are often just as saturated as actual foil.

This stage produces a **visualizer only** — a `defect_area_pct` figure and an annotated
defect map — not a grade by itself. Telling a real scratch apart from holo shimmer or
normal print linework reliably needs actual visual judgment, which is Stage 4.5.

### Stage 4.5 — Vision-model judgment (`llm/vision.py`)
Sends the raking-light crop plus the algorithmic defect map to Claude (currently
`claude-opus-4-8`) with PSA-style surface grading standards in the prompt, and asks for
a structured judgment (grade estimate, confidence, list of defects found, whether holo
was present). Runs automatically whenever `--surface` photos are provided and Claude
API credentials are available (`ANTHROPIC_API_KEY` env var, or an `ant auth login`
profile); if no credentials are configured, this step is skipped with a note in the
report and the rest of the pipeline still runs normally.

### Stage 5 — Grade assembly (`pipeline/scoring.py`)
Combines the three sub-grades (centering, corners/edges, surface) into one overall
estimate. Two ways this combination can happen:
- **Hand-tuned heuristic** (the default until real data exists): weighted toward the
  worst sub-grade, capped so the average can't pull the estimate more than ~1 point
  above the weakest category — approximating PSA's real-world "weakest link dominates"
  behavior.
- **Fitted weights**: once you've run `calibration/calibrate.py --fit` against enough
  known-PSA-grade cards, the combination weights are replaced with an ordinary-least-
  squares fit against your own data (see Calibration below). `assemble_grade()` uses
  fitted weights automatically when they exist in `thresholds.json`, and falls back to
  the heuristic otherwise — nothing needs to change in how you invoke `grade.py`.

## Setup

Requires Python 3.11+. The system Python on this Mac is 3.9 and Homebrew's newer
Python had a broken system-libexpat link as of this writing — the working setup uses
[`uv`](https://github.com/astral-sh/uv) to provision an isolated interpreter:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

Then either `source .venv/bin/activate` or call `.venv/bin/python` directly.

## Usage

```bash
# Front/back flat overhead shots only (centering + corners/edges)
.venv/bin/python grade.py front.jpg back.jpg

# Include angled raking-light shots for surface analysis + vision judgment
.venv/bin/python grade.py front.jpg back.jpg --surface front_angled.jpg back_angled.jpg
```

Output goes to `output/<front-filename>_<timestamp>/` by default (`--output-dir` to
change it): aligned images, centering overlay, corner/edge blob overlays, surface
defect maps, and a `report.json` with every stage's full detail plus the final
`grade_estimate`.

### Capture protocol (matters a lot — read before shooting)

- Card on a **dark matte background**
- Phone **directly overhead on a tripod/stand** — not handheld; camera shake and
  inconsistent framing will trip Stage 1's quality gates
- **Diffuse light from two sides** for the flat shot — no direct lamp, it causes glare
- **Two shots per side**: one flat overhead (measurements), one angled with raking
  light (surface defects) — four photos total per card if you want the surface stage

## Calibration

`calibration/calibrate.py` batch-runs the pipeline against a JSON manifest of cards
with **known** PSA grades and reports how far off the predictions were:

```bash
.venv/bin/python calibration/calibrate.py manifest.json
.venv/bin/python calibration/calibrate.py manifest.json --fit   # also fit scoring.py's weights
```

Manifest format:
```json
[
  {
    "name": "charizard_base_set",
    "front": "cards/charizard_front.jpg",
    "back": "cards/charizard_back.jpg",
    "front_angled": "cards/charizard_front_angled.jpg",
    "back_angled": "cards/charizard_back_angled.jpg",
    "actual_grade": { "overall": 9, "centering": 9, "corners_edges": 8, "surface": 9 }
  }
]
```

`--fit` needs at least 6 cards with complete data (an actual overall grade, plus
predicted centering/corners_edges/surface — the last requires `--surface` photos and
working API credentials for every card in the batch) before it'll touch
`thresholds.json`; below that it reports how many more you need and leaves the
existing config alone. Every stage's per-card debug output (aligned images, overlays,
defect maps) gets written under `output/calibration/<card-name>/` so you can look at
exactly why a prediction was off.

## Project structure

```
grade.py                    CLI entrypoint; grade_card() does the actual orchestration
pipeline/
  detect.py                 Stage 1 — contour detection, perspective correction, quality gates
  centering.py               Stage 2 — border measurement, PSA tolerance grading
  corners_edges.py           Stage 3 — whitening detection
  surface.py                 Stage 4 — scratch/print-line defect visualization
  scoring.py                 Stage 5 — grade assembly (heuristic or fitted weights)
llm/
  vision.py                  Stage 4.5 — Claude vision call for surface judgment
calibration/
  thresholds.json             All tunable values — nothing is hardcoded in the pipeline modules
  calibrate.py                Batch calibration harness (measure + --fit)
output/                       Reports and debug images land here (gitignored-style scratch dir)
```

## What's not proven yet

- **Nothing has been run against a real phone photo of a real card.** Every test so
  far used synthetically generated images. Real cards will have real-world lighting
  variance, sensor noise, actual print details, and actual physical wear that synthetic
  test images can't replicate.
- **All threshold values are guesses** — whitening grade bands, centering tolerances,
  CLAHE/adaptive-threshold parameters, holo detection thresholds. They came from
  reasoning about the problem, not from measuring real cards. `calibrate.py` exists to
  fix this once real known-grade cards are run through it.
- **The vision judgment prompt is untested against a live API call** in this
  environment (no credentials were available during development) — the structured
  output parsing and prompt quality are verified against the SDK's contract, not
  against an actual Claude response.
- **The overall grade formula** (heuristic weighting) is an approximation of how PSA
  combines sub-grades, not a verified formula — expect it to need adjustment once
  `--fit` has real data to work with.

## Possible future work

- **Multi-light photometric stereo** for the surface stage: multiple raking-light
  shots from a fixed camera position, each lit from a different direction, combined to
  recover actual surface shape (not just contrast) — the technique TAG Grading's
  "Photometric Stereoscopic Imaging" is based on. Would directly separate physical
  defects from holo shimmer/print color rather than relying on contrast heuristics.
  Bigger lift than anything built so far — new capture rig, new CV module, protocol
  changes — sketched out but not started.
- **Learning from TAG's public DIG reports** (per-card defect data + grades, published
  at `tagd.co/CERT#`) as a larger calibration dataset than manually grading your own
  cards — needs a scale-conversion (TAG's 1000-point score → this tool's 1-10) and a
  read of TAG's terms before doing any bulk collection.
- A local web upload flow (phone → browser → this tool) if the AirDrop-and-run-the-CLI
  workflow gets tedious.
