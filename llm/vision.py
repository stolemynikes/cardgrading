"""Card identification from two aligned photos (Claude or Gemini).

This is the *only* thing in the pipeline that needs a model. Every grade —
centering, corners, edges, surface, dimensions — is measured deterministically
by `pipeline/`, so a card can be graded end to end with no API key and no
network. What a model still does better than pixels is say which card it is,
and that needs world knowledge rather than measurement.

Surface judgment used to live here too, because on a raking-light photo a
threshold can't tell a scratch from holo sparkle. Photometric stereo removed
the premise: a surface-normal map has no albedo in it, so `pipeline/surface.py`
now grades that signal directly and reproducibly.

Provider selection (no configuration beyond the API key itself):
- ANTHROPIC_API_KEY set -> Claude
- else GEMINI_API_KEY (or GOOGLE_API_KEY) set -> Gemini (free tier works)
- else -> a Claude attempt is still made (the anthropic SDK can resolve
  credentials from an `ant auth login` profile without an env var); if that
  fails too, VisionUnavailable is raised and the caller records the
  identification as skipped.

Identification is always optional. A skipped call costs the report a card
name and the market lookup, never a grade.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

CLAUDE_MODEL = "claude-opus-4-8"
# Pinned models, tried in order; the next is tried on quota/availability
# errors. Deliberately NOT the "gemini-flash-latest" alias: free-tier daily
# quota is per-model, and the alias tracks the newest flash — which can be a
# just-released model with a tiny preview quota (it resolved to a 20
# requests/DAY model once, and every vision stage silently died mid-testing).
# Established models carry the real free tier (hundreds/day), and the
# fallback lives in a separate quota bucket.
GEMINI_MODELS = ("gemini-2.5-flash", "gemini-3.1-flash-lite")


class VisionUnavailable(Exception):
    """The vision review couldn't run (no credentials, network error, rate
    limit, unparseable response). Never fatal: callers treat it as "review
    skipped" and the rest of the report still stands."""


IDENTIFY_STANDARDS = """You are identifying a trading card from two perspective-corrected photos \
that are supposed to be the FRONT and the BACK of the same physical card, in that order.

Identify the card as precisely as you can (name, set, collector number). Also assess:
- Which side each photo actually shows — users sometimes shoot the same side twice or swap the order.
- Whether the two photos plausibly belong to the same physical card (matching game, era, wear).
- Whether the card is a full-art/borderless style (artwork running to the card edges, no plain \
printed border) — this matters because border-based centering measurement doesn't apply to such cards.
- Whether the card has holographic foil.

If you can't identify the exact card, give your best guess and say confidence is low. Never invent \
a collector number you can't see."""


class CardIdentification(BaseModel):
    card_name: str = Field(description="Card name, e.g. 'Gothitelle'; best guess if unsure")
    set_name: str = Field(description="Set name, e.g. 'SVP Black Star Promos'; empty string if unknown")
    collector_number: str = Field(description="Collector number as printed, e.g. '211'; empty string if not visible")
    game: str = Field(description="Which game: pokemon, magic, yugioh, sports, other")
    is_full_art: bool = Field(description="True if the artwork runs to the card edges (borderless/full-art style)")
    is_holo: bool = Field(description="True if the card has holographic foil")
    front_image_side: str = Field(description="Which side the FIRST photo actually shows: front, back, or unclear")
    back_image_side: str = Field(description="Which side the SECOND photo actually shows: front, back, or unclear")
    looks_like_same_card: bool = Field(description="Whether the two photos plausibly show the same physical card")
    confidence: str = Field(description="low, medium, or high confidence in the identification")


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


def _judge_gemini(system: str, items: list, schema: type[BaseModel]) -> tuple[BaseModel, str]:
    """Returns (judgment, model_used) — models are tried in GEMINI_MODELS
    order, moving on when one is quota-exhausted or unavailable."""
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

    last_error: Exception | None = None
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except Exception as e:  # genai errors (auth, 429 rate limit, network) share no useful base with ours
            last_error = e
            continue
        judgment = response.parsed
        if isinstance(judgment, schema):
            return judgment, model
        last_error = VisionUnavailable(f"{model} did not return a parseable judgment")

    raise VisionUnavailable(f"Gemini API error: {last_error}") from last_error


def _call_structured(system: str, items: list, schema: type[BaseModel]) -> tuple[BaseModel, str]:
    """Dispatch to whichever provider has credentials. Returns (judgment, model)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _judge_claude(system, items, schema), CLAUDE_MODEL
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return _judge_gemini(system, items, schema)
    # No env keys — the anthropic SDK may still find an `ant auth login`
    # profile; let it try, and normalize the failure if it can't.
    return _judge_claude(system, items, schema), CLAUDE_MODEL


def identify_card(front_aligned: Path, back_aligned: Path) -> tuple[CardIdentification, str]:
    """Identify the card from the two aligned captures, in one call.

    Also sanity-checks the capture pair itself: which side each photo really
    shows (catches shooting the same side twice / swapping front and back)
    and whether the pair plausibly belongs to one physical card. Same error
    contract as judge_surface.
    """
    return _call_structured(
        IDENTIFY_STANDARDS,
        [
            "Photo 1 — supposed to be the card's FRONT:",
            front_aligned,
            "Photo 2 — supposed to be the card's BACK:",
            back_aligned,
            "Identify the card and assess the photo pair.",
        ],
        CardIdentification,
    )
