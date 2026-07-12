"""Vision-model calls for indicative surface grading (Claude or Gemini).

Stage 4's algorithmic defect map (pipeline/surface.py) can't reliably tell a
real scratch/print-line defect apart from holo foil sparkle or ordinary print
detail — that needs actual judgment about what a Pokemon card is supposed to
look like. This sends the raking-light crop plus the defect map to a vision
model with PSA-style surface standards and asks for a structured judgment.

Provider selection (no configuration beyond the API key itself):
- ANTHROPIC_API_KEY set -> Claude
- else GEMINI_API_KEY (or GOOGLE_API_KEY) set -> Gemini (free tier works)
- else -> a Claude attempt is still made (the anthropic SDK can resolve
  credentials from an `ant auth login` profile without an env var); if that
  fails too, VisionUnavailable is raised and the caller records the review
  as skipped.

This sub-score is always indicative, never definitive — the caller is
responsible for labeling it as such in the report.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

CLAUDE_MODEL = "claude-opus-4-8"
# Stable alias that tracks the newest flash model — pinned previews
# (e.g. gemini-3-flash-preview) get retired and would start 404ing.
GEMINI_MODEL = "gemini-flash-latest"


class VisionUnavailable(Exception):
    """The vision review couldn't run (no credentials, network error, rate
    limit, unparseable response). Never fatal: callers treat it as "review
    skipped" and the rest of the report still stands."""


SURFACE_GRADING_STANDARDS = """You are assisting with pre-grading a Pokemon trading card's surface \
condition, using PSA's 1-10 surface/print-quality standards as a reference:

- 10 (Gem Mint): flawless surface, no scratches, print lines, indentations, or surface wear visible \
even under close inspection.
- 9 (Mint): a very minor, hard-to-see surface flaw — a faint scratch or tiny print imperfection.
- 8-7 (Near Mint-Mint / Near Mint): light scratching or minor print lines visible under normal light, \
not distracting.
- 6-5 (Excellent-Mint / Excellent): noticeable scratches, print lines, or surface wear, clearly visible.
- 4 and below: significant surface damage — heavy scratching, creasing, indentations, or print defects.

You will be shown two images of the same card region, captured with raking (angled) light to reveal \
surface texture:
1. The original color crop.
2. A defect visibility map, where an algorithmic filter (Laplacian + difference-of-Gaussians) has \
highlighted areas of high local contrast in red/hot colors. This map is NOISY — holo foil sparkle, \
normal print linework, and text edges all show up here too, not just real defects. Use the defect map \
only as a hint of where to look, then judge from the original crop whether what's there is an actual \
physical defect (scratch, crease, indentation, print line, whitening) versus holo foil pattern, normal \
card artwork, or text.

Ignore any region that is clearly holographic foil (rainbow, iridescent sparkle pattern) — that is not \
a defect."""


class SurfaceJudgment(BaseModel):
    surface_grade: int = Field(ge=1, le=10, description="PSA-style surface sub-grade estimate, 1-10")
    confidence: str = Field(description="low, medium, or high confidence, given the photo quality")
    defects_found: list[str] = Field(description="Short description of each real physical defect found; empty if none")
    holo_regions_ignored: bool = Field(description="Whether holo-foil regions were present and excluded from judgment")
    reasoning: str = Field(description="Brief explanation of the grade, 2-4 sentences")


FLAT_GRADING_STANDARDS = """You are assisting with pre-grading a Pokemon trading card from a single \
flat, evenly-lit, perspective-corrected photo of one whole side, using PSA's 1-10 standards as a \
reference (10 = Gem Mint, flawless; 9 = one very minor flaw; 8-7 = light wear visible under normal \
light; 6-5 = clearly noticeable wear; 4 and below = significant damage).

Judge three things independently from what is actually visible:
1. Corners — sharpness vs. rounding/whitening/dings at each of the four corners.
2. Edges — whitening, chipping, or roughness along the four edges.
3. Surface — scratches, print lines, indentations, creases, staining. IMPORTANT: a flat evenly-lit \
photo hides shallow scratches and print lines that only show under angled (raking) light, so treat \
your surface estimate as an upper bound and say so in the reasoning; lower your stated confidence if \
lighting or focus limits what you can see.

Do not penalize holographic foil patterns, normal print texture, or artwork elements as defects. If \
the photo is too dark, blurry, or small to judge a category, grade it conservatively and mark \
confidence low."""


class FlatJudgment(BaseModel):
    corners_grade: int = Field(ge=1, le=10, description="PSA-style corners sub-grade estimate from this photo")
    edges_grade: int = Field(ge=1, le=10, description="PSA-style edges sub-grade estimate from this photo")
    surface_grade: int = Field(ge=1, le=10, description="PSA-style surface estimate — an upper bound, since flat lighting hides shallow defects")
    confidence: str = Field(description="low, medium, or high confidence, given the photo quality")
    defects_found: list[str] = Field(description="Short description of each visible defect; empty if none")
    reasoning: str = Field(description="Brief explanation of the grades, 2-4 sentences")


def _encode_image(path: Path) -> tuple[str, str]:
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


# Both providers take the same shape of request: a system prompt, an
# alternating sequence of text and image-Path items, and a Pydantic schema
# for the structured response.
def _judge_claude(system: str, items: list, schema: type[BaseModel]) -> BaseModel:
    client = anthropic.Anthropic()

    content = []
    for item in items:
        if isinstance(item, Path):
            data, media = _encode_image(item)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        else:
            content.append({"type": "text", "text": item})

    try:
        response = client.messages.parse(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        )
    except TypeError as e:
        # The SDK raises a plain TypeError (not an AnthropicError subclass) when
        # it can't resolve any credentials.
        raise VisionUnavailable(f"could not authenticate with the Claude API: {e}") from e
    except anthropic.AnthropicError as e:
        raise VisionUnavailable(f"Claude API error: {e}") from e

    judgment = response.parsed_output
    if judgment is None:
        raise VisionUnavailable("Claude did not return a parseable judgment")
    return judgment


def _judge_gemini(system: str, items: list, schema: type[BaseModel]) -> BaseModel:
    # Imported lazily: google-genai is only needed when a Gemini key is
    # actually configured, so a missing/broken install can't take down the
    # no-AI code path.
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise VisionUnavailable(f"google-genai SDK not installed: {e}") from e

    client = genai.Client()

    contents = []
    for item in items:
        if isinstance(item, Path):
            mime = "image/png" if item.suffix.lower() == ".png" else "image/jpeg"
            contents.append(types.Part.from_bytes(data=item.read_bytes(), mime_type=mime))
        else:
            contents.append(item)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
    except Exception as e:  # genai errors (auth, 429 rate limit, network) share no useful base with ours
        raise VisionUnavailable(f"Gemini API error: {e}") from e

    judgment = response.parsed
    if not isinstance(judgment, schema):
        raise VisionUnavailable("Gemini did not return a parseable judgment")
    return judgment


def _call_structured(system: str, items: list, schema: type[BaseModel]) -> tuple[BaseModel, str]:
    """Dispatch to whichever provider has credentials. Returns (judgment, model)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _judge_claude(system, items, schema), CLAUDE_MODEL
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return _judge_gemini(system, items, schema), GEMINI_MODEL
    # No env keys — the anthropic SDK may still find an `ant auth login`
    # profile; let it try, and normalize the failure if it can't.
    return _judge_claude(system, items, schema), CLAUDE_MODEL


def judge_surface(crop_path: Path, defect_map_path: Path) -> tuple[SurfaceJudgment, str]:
    """Judge surface condition from the raking-light crop + defect map.

    Returns (judgment, model_name) — the model name goes into the report so
    it's clear which provider produced which judgment when comparing runs.
    Raises VisionUnavailable on any failure; vision review is optional, so
    callers should catch it and treat it as "review skipped", not a fatal
    error for the rest of the report.
    """
    return _call_structured(
        SURFACE_GRADING_STANDARDS,
        [
            "Original crop (raking light):",
            crop_path,
            "Algorithmic defect visibility map (red/hot = high local contrast, NOT necessarily a real defect):",
            defect_map_path,
            "Judge this card region's surface condition.",
        ],
        SurfaceJudgment,
    )


def judge_flat(aligned_path: Path, side_label: str) -> tuple[FlatJudgment, str]:
    """Judge corners/edges/surface from a flat perspective-corrected capture.

    Complements the deterministic pipeline on the same photo: an opinion on
    corners/edges wear that doesn't depend on the whitening thresholds, and
    an upper-bound surface estimate when no raking-light shots were taken.
    Same error contract as judge_surface.
    """
    return _call_structured(
        FLAT_GRADING_STANDARDS,
        [
            f"Perspective-corrected flat photo of the card's {side_label}:",
            aligned_path,
            "Judge this side's corners, edges, and (as far as visible) surface condition.",
        ],
        FlatJudgment,
    )
