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
Finds the card's contour against the background and perspective-corrects it to the
canonical size. Runs a set of capture-quality gates, split into two severities:

**Hard gates** (geometry is wrong — grading would be meaningless, so it blocks and
asks for a retake):
- **Card detection** — no card-like contour found at all
- **Tilt** — the detected quad's corner angles deviate too far from 90°
- **Aspect ratio** — the detected quad's side-length ratio is too far from the real
  63:88 card ratio (measured on the *pre-warp* corner quad — the warped image is
  always canonical-sized, so measuring it would make this gate a constant)

**Soft gates** (image quality is degraded — grading proceeds, and the report shows a
"grading might be worse because of" warning):
- **Resolution** — image below the minimum short side
- **Glare** — blown-out highlight clusters on the card face
- **Uneven lighting** — brightness gradient across the card's *border ring specifically*
  (not the whole face — printed color contrast between the border and inner panel
  would otherwise look identical to a lighting problem)

Tilt and aspect are complementary: a skewed misdetection fails tilt, while a
wrong-but-rectangular one (the minimum-area-rect fallback boxing card + background
together, whose corners are exactly 90° by construction) fails aspect.

Card detection thresholds on **color distance from the sampled background**, not
absolute brightness — a plain brightness split would fail on dark Pokémon card backs,
which are often *darker* than a "dark matte" background, not brighter. It binarizes
at several thresholds (Otsu and multiples of it), collects candidate quads from every
sizeable contour, and picks the most card-like by aspect ratio, rectangularity,
solidity, and size — on busy/textured backgrounds a single Otsu split tends to merge
the card and background into one blob, which the old "largest contour wins" rule
would then grade as if it were the card.

### Stage 1.5 — Dimensions (`pipeline/dimensions.py`)
Measures the card's real physical size in millimetres against the 63×88mm nominal,
catching miscuts, diamond cuts and trimming — defects of the card itself rather than
of its condition, which is why a card outside tolerance is capped at grade 8 no
matter how clean its surface is.

This needs **absolute scale**, so it only runs on a capture whose scale is known: a
flatbed scan with `--dpi` passed. A phone photo has no scale (the distance to the
card is unknown), so the attribute reports as unmeasurable rather than guessing.

Every scan of the front is an independent measurement, and where a rotation set
exists all of them are used: the **median** is the figure and the **spread** is
reported beside it. This is not caution for its own sake. Measured on a real flatbed
the same card came out 2.9% different — 1.8mm on a 63mm card — depending only on
whether it was lying portrait or landscape on the glass, against a tolerance of
0.75mm. Resting the verdict on whichever scan happened to be the flat capture made
one card read "2.13mm miscut" in one run and "within tolerance" in the next.

When the spread is wider than the tolerance being applied, **no miscut verdict is
given at all** — `within_tolerance` is None rather than True or False. A measurement
that disagrees with itself by more than the thing it's being judged against cannot
settle the question, and saying so is the honest answer. "Can't tell" is also not a
defect, so it produces no DING.

### Stage 1.6 — Card Vision (`pipeline/cardvision.py`)
A grayscale render of the card's *physical relief* with print stripped out —
scratches, dents, creases and edge lifting are shape; artwork is not. The report
cross-fades it against the normal capture on a transparency slider. Two ways to
produce it:

**Photometric stereo** (the real thing). Several captures from a fixed camera with
the light arriving from a different direction each time solve for a per-pixel
surface normal. Print does not affect a normal, so the render genuinely contains
only geometry. A flatbed is the easiest rig for this: its lamp is fixed relative to
the scan axis, so **rotating the card 90° on the glass between scans rotates the
light in the card's frame**. Four scans = four light directions, and the canonical
warp has already registered them to each other (an ECC pass cleans up the residual
sub-pixel error, which matters — a two-pixel misalignment turns every print edge
into a fake ridge in the normal map).

Scans go in any order, turned either way. Each scan's rotation is **measured from
the image**, not declared: orientation from the detected quad, then upright versus
upside-down by correlating against the flat capture — the one thing geometry can't
settle, since both are portrait. The measured rotation gives the light azimuth
directly, because a card turned clockwise by θ moves the lamp clockwise by θ in the
card's frame, and θ is exactly the counter-clockwise rotation needed to stand that
scan upright.

It is derived rather than declared because declaring it fails silently. File order
and "which way did I turn it" are both easy to get wrong, and a wrong one doesn't
error — it solves every normal against the wrong light and produces a plausible
render of a badly damaged card. `--rotation` survives as the fallback for a scan the
correlation can't resolve, and `--lamp-azimuth` states where the lamp lights from.

The four warps are **registered to each other** before the solve, and this matters
more than it sounds: a two-pixel shift turns every high-contrast print edge into a
fake ridge. The fit is affine, not rigid — measured across one real set the detector
found the same card at widths spanning 4.1%, which rotation-and-translation cannot
express — matched on high-pass features so it doesn't chase the shading the solve
exists to measure, and seeded globally (phase correlation plus a coarse scale
search) because ECC's own capture range is about ten pixels and the offsets are
larger than that. A fit outside plausible bounds is declined rather than applied.

The normalised warps are kept with the report (`<side>_rotation_<k>.png`), so the
solve can be re-run without the card going back on the glass.

**Single-image approximation** (the fallback, and what the phone flow gets). One
ordinary capture, high-pass filtered to drop the low-frequency albedo, then
attenuated wherever *color* is also changing — an ink boundary moves chroma, a
scratch through clear laminate moves only brightness. Useful, but it cannot fully
separate fine print detail from real geometry, and the report labels which of the
two methods produced what you're looking at, because that difference changes how you
read a mark.

Both paths soft-threshold at the measured noise floor before autoscaling. Without
that, a clean card renders as its own noise stretched to full contrast, which reads
as a surface covered in defects.

The floor is read from the **flattest tenth of the card**, tile by tile, not from the
median of the whole thing. "A card is mostly flat" is only true if you don't count
the print: measured on a real scan the whole-card estimate put the floor at 0.222
while the quietest tiles sat at 0.109, and a scratch six grey levels deep is 0.024 —
so the threshold was measuring printed detail and erasing everything beneath it.
Moving to the tiled estimate multiplied the rendered signal by 5× at 25 grey levels
of depth and 14× at 12.

Note what photometric stereo does *not* remove. It removes albedo — colour,
brightness, foil shimmer — so a dark colour can no longer be mistaken for a dent. It
does not remove the physical topography of the ink, because printed ink genuinely
has thickness. Print shows up in these renders, correctly, as relief.

Rendering and measurement are separate. `relief` carries the configured
`relief_gain` and is a picture; `measurement_relief` is the same solve at unit gain
and is what gets measured. They were the same image until turning the gain up for
looks doubled a card's measured defect area and cost it a surface grade.

#### Does your scanner even light off-axis?

Photometric stereo only works if the lamp reaches the card at an angle. CCD flatbeds
do; **CIS** flatbeds (Canon LiDe and most cheap USB-powered units) put an LED strip
nearly flush against the glass, and flat light means rotating the card changes
nothing. Rather than guess, measure it — two scans, one of them with the card turned
180°:

```bash
.venv/bin/python calibration/check_photometric.py scan_0.png scan_180.png --output flip.png
```

A half-turn reverses the light relative to the card while leaving optics, focus and
the card identical, so any pixel whose brightness *flips* between the two is being
shaded by geometry. The script reports that flipped fraction and says whether the
solve is worth running. A "too flat" verdict doesn't make the scanner useless — it's
still the better capture for centering, corners/edges and dimensions; it just means
that scanner can't support a surface grade.

### Stage 2 — Centering (`pipeline/centering.py`)
Pure geometry: finds the boundary between the card's printed border and its inner
artwork/text panel on all four sides (via Canny edge detection along sampled bands),
computes left/right and top/bottom ratios, and grades them against PSA-style tolerance
tables (55/45 → grade 10, 60/40 → grade 9, etc., with a looser table for the back).
Produces an overlay image showing exactly where it thinks the border/panel boundary is.

Each boundary detection carries a **confidence** (how much of the sampled band the
strongest edge spans — a real border boundary is a straight line across the whole
band; noise isn't). A side with any low-confidence boundary is reported as
**unmeasurable** rather than graded: borderless/full-art cards have no border to
measure, and a too-dim or blurry capture can hide a real one. The overall grade then
uses the measurable side(s), or excludes centering entirely. Before this check, a
real full-art promo produced a confident-looking "89/11 grade 3" from pure noise —
which also collapsed the Stage 3 crop sizes to meaningless 12×12 patches.

#### Measuring at the worst point, and placing it by hand

PSA grades "the percent of difference at the most off-center part of the card" — a
point, not an average. Each side is sampled at five bands along its length and the
worst sample counts, with low-confidence samples excluded so noise can't win the
vote. The bigger effect turned out to be on *confidence*: a shorter band contains
less of a slanted border's slant, so the Canny peak stays sharp. On a card whose
border wanders 20px down its side, peak confidence went 0.17 (one band, refused) →
0.23 (three, refused) → 0.39 (five, measured).

The detector has a third failure mode that no confidence gate catches: finding *an*
edge, confidently enough to pass, that isn't the border. Modern cards stack an
artwork boundary, an inner frame band and a thin rule within a few millimetres. So
the boundaries can be **placed by hand** — eight lines over a zoomable full-screen
view of the warp, a card edge and a border boundary per side, with a loupe. A
hand-placed boundary is always measurable: the confidence score describes how sure
the *detector* was, and once a person has said where the edge is that question is
moot. Saved corrections rewrite the stored report and re-run corners and edges with
them, since those crops are sized from the border widths.

#### Tolerance tables

PSA's are transcribed from the published standards, including the **5% front leeway**
for cards grading 7 or better ("A 5% leeway is given to the front centering minimum
standards"), and the back tolerance that stays at 90/10 from Mint 9 all the way down —
only Gem Mint 10 tightens, to 75/25. Backs are cut far less precisely than fronts and
nobody looks at them, so the leniency is real; assuming it tightened progressively,
as the front does, graded an 88/12 back as a 7 instead of a 9.

Every other service's table (BGS, CGC, SGC, TAG, ACE) is reported alongside for
reference, each carrying a note of where it came from — all but PSA's are
third-party transcriptions rather than primary sources, which is exactly how the
back tolerances were wrong here before.

### Stage 3 — Corners & edges (`pipeline/corners_edges.py`)
Crops the four corners and four edge strips, and runs a filter stack (CLAHE contrast
boost + blue-channel isolation + adaptive threshold) to turn "whitening" — chipped
corners/edges revealing white cardstock — into countable blobs. Crop sizes are derived
from Stage 2's *measured* border widths (not fixed pixel sizes), so the analysis window
never accidentally crosses from the border into the inner panel — which would otherwise
misread the normal border/panel color transition as a huge fake defect on every card.

### Stage 4 — Surface (`pipeline/surface.py`)
Grades surface defects — scratches, dents, creases, print lines — from whichever
signal the capture provides, deterministically and offline. Defects are found by
thresholding, counted, and scored against tolerance bands, with a separate cap on
grade for a single long scratch (a hairline can cover almost no area and still be
the first thing a grader sees).

What the signal is decides how much weight it carries:

| Source | What it is | Graded? |
|---|---|---|
| `photometric_relief` | solved surface normals | yes — print and foil are absent from a normal map |
| `single_image_relief` | one-capture Card Vision approximation | no — print demonstrably leaks in |

This used to require a vision model, and for the raking-light photo it used to start
from that was the right call: a fixed threshold genuinely cannot separate a scratch
from holo sparkle.
**Photometric stereo removes the premise.** A surface-normal map contains no albedo,
so foil, artwork and print lines are gone before anything is measured, and a
threshold against that signal is a measurement rather than a guess.

### Stage 4.5 — Card identification (`llm/vision.py`)
The only model call left in the pipeline, and it never touches a grade. It answers
what the card is (name/set/number, full-art, holo), sanity-checks the capture pair
itself (same side shot twice, front/back swapped, mismatched cards), and drives the
market-price lookup in `market.py`.

Two providers, chosen by whichever key is configured — `ANTHROPIC_API_KEY` (or an
`ant auth login` profile) for Claude, else `GEMINI_API_KEY` for Gemini's free tier.
**With neither, the card still grades.** Every sub-grade is measured by `pipeline/`;
skipping identification costs a name and the market line, nothing else.

One part of that sanity check is duplicated offline in `grade.py`, because it's the
one failure that's both easy to hit and completely silent: uploading the same side
twice produces a full, confident report in which every "back" number was measured on
the front and scored against PSA's looser back table. `check_capture_pair()` hashes
the two files and correlates their thumbnails, and the report carries the result under
`capture_pair`. It warns — it never refuses. Grading one side against both tolerance
tables is a legitimate thing to do deliberately.

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

The report leads with a **score out of 1000** rather than the 1–10 grade. It's the
same estimate without the rounding — `assemble_grade()` works in floats throughout —
so two cards that both land on 9 can still be told apart. It's this tool's own
number, derived from this tool's own sub-grades, and is not any grading company's
scale or comparable to one.

Sub-grades are also reported **per side and per attribute** — front/back × centering,
corners, edges, surface — since front and back are separate surfaces with separate
wear, and corners and edges fail in different ways.

### Stage 6 — DINGS (`pipeline/dings.py`)
"Defects Identified of Notable Grade Significance": the handful of findings that
actually set the number, ranked worst-first and pulled to the top of the report — the
worst corner and worst edge on each side, the axis that capped centering, the defects
the vision model called out, and a bad cut. Nothing here is a new measurement; it all
appears in the detail sections too. A region grading a clean 10 is never listed, an
unmeasurable centering axis is not a defect (that's a borderless card, and the
centering section already says so), and corners and edges rank separately so a card
with four clean corners and one chipped edge still surfaces the edge.

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

# Surface relief from rotations — camera and light fixed, card turned 90 degrees
.venv/bin/python grade.py front.jpg back.jpg \
    --photometric-front r0.jpg r90.jpg r180.jpg r270.jpg \
    --photometric-back  b0.jpg b90.jpg b180.jpg b270.jpg

# Flatbed scans: --dpi makes dimensions measurable (miscut/trim detection)
.venv/bin/python grade.py front.tif back.tif --dpi 1200

# Full Card Vision: 4 scans per side, card turned 90 degrees on the glass each time
.venv/bin/python grade.py front.tif back.tif --dpi 1200 \
    --photometric-front f0.tif f90.tif f180.tif f270.tif \
    --photometric-back  b0.tif b90.tif b180.tif b270.tif
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
- **For surface relief, rotate the card, not the light.** Take 3–6 shots of the same
  side with the camera and the light both fixed, turning the *card* 90° between each.
  That's photometric stereo: the light arrives from a different direction in the
  card's own frame each time, which is enough to solve a per-pixel surface normal.
  A tripod and one desk lamp are the whole rig — a flatbed just does it for free,
  since its lamp is fixed relative to the scan axis.
- **Don't merge the rotations yourself.** The app combines them; averaging the frames
  first cancels out the very shading the solve depends on.

#### Scanning instead of shooting

A flatbed beats a phone for everything except surface. It's orthographic (no
perspective, no lens distortion), evenly lit, and repeatable — which is what makes
centering measurable to the precision the tolerance tables assume, and what makes
fitted calibration weights fit card variance instead of capture variance.

- **1200 dpi is plenty; higher is fine but no longer wasteful.** The canonical warp
  is 1500px across a 63mm card, i.e. ~605 dpi effective, so nothing above that reaches
  a detector. `perspective_correct` area-averages the source down before warping —
  `warpPerspective` cannot, since INTER_LINEAR reads a 2×2 neighbourhood and so only
  averages a 2× reduction, point-sampling anything beyond. Measured through that
  function, fine texture surviving an 8× downscale went from ~14 standard deviation to
  ~2 once the pre-pass was added; without it an oversampled scan reached the whitening
  and surface stages no cleaner than a modest one.
- **TIFF or PNG**, never JPEG — ringing at the border/panel boundary feeds straight
  into the centering edge detector.
- **Turn off** auto-crop, auto-color/exposure, unsharp mask, descreen and dust
  removal. All of them move or invent edges.
- **Don't auto-crop**: leave margin around the card. `detect.py` samples the
  background color to threshold the contour, and no margin means no background.
- **Put a matte card behind it** rather than relying on the white lid — a white lid
  against a white card border is a low color distance.
- Clean the glass. At 1200 dpi a dust speck is a multi-pixel blob on the border.
- Glossy cards pressed to glass can produce **Newton's rings**; if you see rainbow
  interference banding, the holo-variance mask will read it as foil.

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
    "actual_grade": { "overall": 9, "centering": 9, "corners_edges": 8, "surface": 9 }
  }
]
```

`--fit` needs at least 6 cards with complete data (an actual overall grade, plus
predicted centering/corners_edges/surface — the last requires a photometric rotation
set for every card in the batch) before it'll touch
`thresholds.json`; below that it reports how many more you need and leaves the
existing config alone. Every stage's per-card debug output (aligned images, overlays,
defect maps) gets written under `output/calibration/<card-name>/` so you can look at
exactly why a prediction was off.

## Webapp (`webapp/`)

A FastAPI + vanilla-JS app that wraps the pipeline so the whole flow — photograph,
grade, retake — happens on a phone:

```bash
.venv/bin/python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
# then for phone access (getUserMedia needs HTTPS):
cloudflared tunnel --url http://localhost:8000
```

- **Guided live-camera capture**: a card-shaped guide box, step sequence
  (front → back), 4K camera request, and a
  photo-picker fallback when no camera is available.
- **Live pre-checks** on the viewfinder (coarse, client-side, never block the
  shutter): too dark, no card detected (red), too little color, glare, too far
  away (yellow); green guide border when the frame looks good. Readings are
  debounced over several ticks so borderline scenes don't flicker.
- **Hard vs soft gate handling**: geometry failures bounce back to the failing
  step with a retake message; quality failures grade anyway with a red
  "grading might be worse because of" banner at the top of the report.
- **No accounts**: uploads live in a per-job temp dir deleted in a `finally`;
  stale temp dirs are swept on startup. The vision judgment (Stage 4.5) is
  optional — the app works fully without API credentials.
- **Finished reports are kept** under `reports/<job-id>/`, and bounded there. A
  photometric report is ~140MB, most of it the full-scale detail warps and the
  stored rotation scans, which exist only to be zoomed into. After each save,
  `store.prune_report_images()` drops those from all but the ten newest reports
  (`DEFAULT_KEEP_FULL_IMAGES`). Nothing that was measured is lost: an old report
  still opens, still shows every overlay, and still re-grades from hand-placed
  borders — only the zoom falls back to the canonical warp.
- Server binds to localhost only; remote access is via Tailscale or a Cloudflare
  tunnel, never `0.0.0.0`.

## Tests

```bash
.venv/bin/python -m pytest tests/          # pipeline: gates, scoring, detection
cd webapp/tests && npm install && npm test # frontend: jsdom regression suite
```

## Project structure

```
grade.py                    CLI entrypoint; grade_card() does the actual orchestration
market.py                   Raw-card market price lookup (pokemontcg.io) for identified cards
pipeline/
  detect.py                 Stage 1 — contour detection, perspective correction, quality gates
  dimensions.py              Stage 1.5 — physical size in mm, miscut/trim detection (scans only)
  cardvision.py              Stage 1.6 — relief render: photometric stereo, or a 1-shot approximation
  centering.py               Stage 2 — border measurement, PSA tolerance grading
  corners_edges.py           Stage 3 — whitening detection
  surface.py                 Stage 4 — defect detection and deterministic surface grading
  scoring.py                 Stage 5 — grade assembly (heuristic or fitted weights) + score
  dings.py                   Stage 6 — ranking the defects that actually set the grade
llm/
  vision.py                  Card identification only (Claude or Gemini) — never a grade
webapp/
  main.py                    FastAPI app (upload validation, job endpoints)
  jobs.py                    In-memory job queue, temp-dir lifecycle, progress messages
  static/                    Single-page frontend (no build step)
  tests/                     jsdom regression suite for the frontend
calibration/
  thresholds.json             All tunable values — nothing is hardcoded in the pipeline modules
  calibrate.py                Batch calibration harness (measure + --fit)
  check_photometric.py        Two-scan test of whether a scanner lights the card off-axis
tests/                        pytest suite for the pipeline
output/                       Reports and debug images land here (gitignored)
```

## What's not proven yet

- **Real-photo experience is thin.** The pipeline has been exercised against real
  phone photos of real cards (which drove the candidate-scored detection and the
  hard/soft gate split), but only a handful, all on an unhelpfully busy background.
  No card with a known professional grade has been run end-to-end yet.
- **Most threshold values are guesses** — whitening grade bands, centering tolerances,
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

- **Calibration against professionally graded cards — attempted, negative result
  (2026-07).** TAG DIG scraping was ruled out first: their ToS §5.2(j) explicitly
  prohibits automated collection. Instead, `calibration/fit_from_dataset.py` collected
  150 fit rows from a public HF dataset of slabbed-card listing photos (real PSA-scale
  overall grades; grade bands 1–10 all represented; our own centering measurement +
  vision flat judgments as features). Verdict: **do not ship the fitted weights.**
  Leave-one-out MAE plateaued at 2.3 grades for every feature combination. Root cause
  is input resolution, not the method: cards extracted from slab photos are ~300px
  wide, and at that size the vision judgments cluster at 7–9 regardless of true grade
  (a PSA 2 and a PSA 9 both "look fine"), while pixel centering is pure noise
  (corr −0.04, n=150). Weights fit on those inputs would also transfer wrongly to the
  app's full-resolution captures. What WOULD work: the same harness pointed at
  high-resolution labeled images — most practically, photographing your own
  professionally graded cards through the app (`calibration/calibrate.py --fit`).
  The collected rows are kept in `calibration/dataset_fit_rows.json` for reuse.
- **A blur pre-check / gate.** Motion blur silently degrades corner and surface
  analysis, and nothing currently warns about it. A cheap Laplacian-based metric was
  prototyped but didn't discriminate on available data: smooth-but-sharp card art
  scored *lower* than blurry-but-textured scenes, and there were no matched
  sharp/blurry captures from the actual phone camera to calibrate against. Needs a
  small set of deliberate sharp-vs-shaky captures of the same card before it can
  ship — a mis-calibrated warning is worse than none.
- **A duplicate-side check** — warn when the front and back captures look like the
  same photo (easy user slip in the step flow).
- **`ImageCapture.takePhoto()`** would give full-sensor stills instead of video-stream
  frames, but it isn't supported on iOS Safari, which is the primary test device —
  the 4K stream request is the practical ceiling there for now.
