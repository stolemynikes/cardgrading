"""Vision provider selection and failure normalization (no network calls).

judge_surface picks a provider from configured credentials — Claude when
ANTHROPIC_API_KEY is set, Gemini when only GEMINI_API_KEY/GOOGLE_API_KEY is —
and every provider failure (missing key, rate limit, unparseable output)
must surface as VisionUnavailable so callers can treat the review as
"skipped" rather than crashing the grade.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from llm import vision

JUDGMENT = vision.SurfaceJudgment(
    surface_grade=8,
    confidence="medium",
    defects_found=["light scratch near the top edge"],
    holo_regions_ignored=False,
    reasoning="Minor surface wear visible under raking light.",
)

CROP = Path("crop.png")
DEFECT_MAP = Path("map.png")


def clear_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestProviderSelection:
    def test_anthropic_key_selects_claude(self, monkeypatch):
        clear_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch.object(vision, "_judge_claude", return_value=JUDGMENT) as claude:
            judgment, model = vision.judge_surface(CROP, DEFECT_MAP)
        claude.assert_called_once()
        assert judgment is JUDGMENT
        assert model == vision.CLAUDE_MODEL

    def test_gemini_key_selects_gemini(self, monkeypatch):
        clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with patch.object(vision, "_judge_gemini", return_value=JUDGMENT) as gemini:
            judgment, model = vision.judge_surface(CROP, DEFECT_MAP)
        gemini.assert_called_once()
        assert model == vision.GEMINI_MODEL

    def test_google_api_key_alias_also_selects_gemini(self, monkeypatch):
        clear_keys(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        with patch.object(vision, "_judge_gemini", return_value=JUDGMENT):
            _, model = vision.judge_surface(CROP, DEFECT_MAP)
        assert model == vision.GEMINI_MODEL

    def test_anthropic_wins_when_both_keys_present(self, monkeypatch):
        clear_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with patch.object(vision, "_judge_claude", return_value=JUDGMENT) as claude, \
             patch.object(vision, "_judge_gemini") as gemini:
            _, model = vision.judge_surface(CROP, DEFECT_MAP)
        claude.assert_called_once()
        gemini.assert_not_called()
        assert model == vision.CLAUDE_MODEL

    def test_no_keys_attempts_claude_profile_fallback(self, monkeypatch):
        # No env keys: the anthropic SDK may still resolve an `ant auth login`
        # profile, so Claude gets one attempt before giving up.
        clear_keys(monkeypatch)
        with patch.object(vision, "_judge_claude", side_effect=vision.VisionUnavailable("no creds")):
            with pytest.raises(vision.VisionUnavailable):
                vision.judge_surface(CROP, DEFECT_MAP)


class TestFlatJudgment:
    FLAT = vision.FlatJudgment(
        corners_grade=8,
        edges_grade=7,
        surface_grade=9,
        confidence="medium",
        defects_found=[],
        reasoning="Light edge wear visible.",
    )

    def test_judge_flat_uses_same_provider_selection(self, monkeypatch):
        clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with patch.object(vision, "_judge_gemini", return_value=self.FLAT) as gemini:
            judgment, model = vision.judge_flat(Path("aligned.png"), "front")
        gemini.assert_called_once()
        assert model == vision.GEMINI_MODEL
        assert judgment.corners_grade == 8

    def test_judge_flat_passes_flat_standards_and_schema(self, monkeypatch):
        clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with patch.object(vision, "_judge_gemini", return_value=self.FLAT) as gemini:
            vision.judge_flat(Path("aligned.png"), "back")
        system, items, schema = gemini.call_args[0]
        assert "flat" in system.lower()
        assert schema is vision.FlatJudgment
        assert any(isinstance(i, Path) for i in items)
        assert any("back" in i for i in items if isinstance(i, str))


class TestIdentifyCard:
    IDENT = vision.CardIdentification(
        card_name="Gothitelle",
        set_name="SVP Black Star Promos",
        collector_number="211",
        game="pokemon",
        is_full_art=True,
        is_holo=True,
        front_image_side="front",
        back_image_side="back",
        looks_like_same_card=True,
        confidence="high",
    )

    def test_identify_uses_provider_selection(self, monkeypatch):
        clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with patch.object(vision, "_judge_gemini", return_value=self.IDENT) as gemini:
            ident, model = vision.identify_card(Path("front.png"), Path("back.png"))
        gemini.assert_called_once()
        assert model == vision.GEMINI_MODEL
        assert ident.is_full_art is True

    def test_identify_sends_both_images_and_schema(self, monkeypatch):
        clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with patch.object(vision, "_judge_gemini", return_value=self.IDENT) as gemini:
            vision.identify_card(Path("front.png"), Path("back.png"))
        system, items, schema = gemini.call_args[0]
        assert schema is vision.CardIdentification
        assert sum(1 for i in items if isinstance(i, Path)) == 2
        assert "FRONT" in " ".join(i for i in items if isinstance(i, str))


class TestFailureNormalization:
    def test_gemini_error_surfaces_as_vision_unavailable(self, monkeypatch, tmp_path):
        # A real (non-mocked) _judge_gemini call with a bogus key must come
        # back as VisionUnavailable — auth errors, 429 rate limits, and
        # network failures all take this path.
        clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "definitely-not-a-real-key")
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal header; never reaches decoding
        with pytest.raises(vision.VisionUnavailable):
            vision.judge_surface(crop, crop)

    def test_claude_missing_credentials_surfaces_as_vision_unavailable(self, monkeypatch, tmp_path):
        clear_keys(monkeypatch)
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(vision.VisionUnavailable):
            vision.judge_surface(crop, crop)
