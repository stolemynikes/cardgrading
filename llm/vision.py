"""Claude API vision calls for indicative surface grading.

Stage 4's algorithmic defect map (pipeline/surface.py) can't reliably tell a
real scratch/print-line defect apart from holo foil sparkle or ordinary print
detail — that needs actual judgment about what a Pokemon card is supposed to
look like. This sends the raking-light crop plus the defect map to Claude
with PSA-style surface standards and asks for a structured judgment.

This sub-score is always indicative, never definitive — the caller is
responsible for labeling it as such in the report.
"""

from __future__ import annotations

import base64
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-4-8"

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


def _encode_image(path: Path) -> tuple[str, str]:
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


def judge_surface(crop_path: Path, defect_map_path: Path) -> SurfaceJudgment:
    """Ask Claude to judge surface condition from the original crop + defect map.

    Raises anthropic.AnthropicError (or a subclass) on any failure — missing
    credentials, network error, rate limit, etc. Surface vision review is
    optional, so callers should catch this and treat it as "review skipped",
    not a fatal error for the rest of the report.
    """
    client = anthropic.Anthropic()

    crop_data, crop_media = _encode_image(crop_path)
    map_data, map_media = _encode_image(defect_map_path)

    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=1024,
            system=SURFACE_GRADING_STANDARDS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Original crop (raking light):"},
                        {"type": "image", "source": {"type": "base64", "media_type": crop_media, "data": crop_data}},
                        {
                            "type": "text",
                            "text": "Algorithmic defect visibility map (red/hot = high local contrast, "
                            "NOT necessarily a real defect):",
                        },
                        {"type": "image", "source": {"type": "base64", "media_type": map_media, "data": map_data}},
                        {"type": "text", "text": "Judge this card region's surface condition."},
                    ],
                }
            ],
            output_format=SurfaceJudgment,
        )
    except TypeError as e:
        # The SDK raises a plain TypeError (not an AnthropicError subclass) when
        # it can't resolve any credentials — normalize it so callers only need
        # to catch anthropic.AnthropicError to detect "review unavailable".
        raise anthropic.AnthropicError(f"could not authenticate with the Claude API: {e}") from e

    judgment = response.parsed_output
    if judgment is None:
        raise anthropic.AnthropicError("model did not return a parseable surface judgment")
    return judgment
