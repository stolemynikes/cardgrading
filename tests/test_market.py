"""Market price lookup: best-effort, never raises, None on any failure."""

from unittest.mock import patch

import market

CARD_RESPONSE = [
    {
        "name": "Pangoro",
        "number": "111",
        "set": {"name": "Astral Radiance"},
        "images": {"small": "https://images.pokemontcg.io/swsh10/111.png"},
        "tcgplayer": {"prices": {"normal": {"market": 0.15}, "reverseHolofoil": {"market": 0.42}}},
        "cardmarket": {"prices": {"trendPrice": 0.11}},
    }
]


class TestLookupPrices:
    def test_happy_path_extracts_prices(self):
        with patch.object(market, "_query", return_value=CARD_RESPONSE) as q:
            result = market.lookup_prices("Pangoro", "111", "Astral Radiance")
        assert result is not None
        assert result["matched_name"] == "Pangoro"
        assert result["matched_set"] == "Astral Radiance"
        assert result["prices"]["tcgplayer_normal"] == 0.15
        assert result["prices"]["tcgplayer_reverseHolofoil"] == 0.42
        assert result["prices"]["cardmarket_trend"] == 0.11
        assert result["source"] == "pokemontcg.io"
        # first query should be the precise name+number one
        assert "number:111" in q.call_args_list[0][0][0]["q"]

    def test_falls_back_to_name_only_query(self):
        # number-specific query finds nothing; name-only does
        with patch.object(market, "_query", side_effect=[[], CARD_RESPONSE]):
            result = market.lookup_prices("Pangoro", "999")
        assert result is not None
        assert result["matched_name"] == "Pangoro"

    def test_no_match_returns_none(self):
        with patch.object(market, "_query", return_value=[]):
            assert market.lookup_prices("Cardthatdoesnotexist") is None

    def test_network_error_returns_none(self):
        with patch.object(market, "_query", side_effect=OSError("no network")):
            assert market.lookup_prices("Pangoro", "111") is None

    def test_empty_name_returns_none_without_calling_api(self):
        with patch.object(market, "_query") as q:
            assert market.lookup_prices("") is None
        q.assert_not_called()

    def test_missing_price_blocks_tolerated(self):
        bare = [{"name": "Pangoro", "number": "111", "set": {"name": "X"}}]
        with patch.object(market, "_query", return_value=bare):
            result = market.lookup_prices("Pangoro")
        assert result is not None
        assert result["prices"] == {}

    def test_printed_number_format_normalized(self):
        # Cards print "193/264" but the API's number field is "193" — the
        # precise query must use the normalized form.
        with patch.object(market, "_query", return_value=CARD_RESPONSE) as q:
            market.lookup_prices("Latias", "193/264")
        assert "number:193" in q.call_args_list[0][0][0]["q"]
        assert "193/264" not in q.call_args_list[0][0][0]["q"]

    def test_failed_precise_query_still_tries_fallback(self):
        # A malformed first query (API 400) must not kill the name-only
        # fallback — this exact bug shipped once. The precise query fails
        # on every retry attempt; only then does the name-only query run.
        failures = [OSError("400")] * market.ATTEMPTS_PER_QUERY
        with patch.object(market, "_query", side_effect=[*failures, CARD_RESPONSE]) as q:
            result = market.lookup_prices("Latias", "193/264")
        assert result is not None
        assert result["matched_name"] == "Pangoro"  # fixture card, proves a later query ran
        assert 'number:' not in q.call_args_list[-1][0][0]["q"], "the successful query was the name-only fallback"

    def test_exact_name_match_beats_substring_match(self):
        # Real failure: querying "Latias" returned "Mega Latias ex" first
        # (substring match + newest-release ordering), pricing a $0.30 card
        # at $92. The exact-name result must win regardless of order.
        mega_first = [
            {"name": "Mega Latias ex", "number": "1", "set": {"name": "Newest Set"},
             "tcgplayer": {"prices": {"holofoil": {"market": 92.83}}}},
            {"name": "Latias", "number": "193", "set": {"name": "Fusion Strike"},
             "tcgplayer": {"prices": {"normal": {"market": 0.18}}}},
        ]
        with patch.object(market, "_query", return_value=mega_first):
            result = market.lookup_prices("Latias")
        assert result["matched_name"] == "Latias"
        assert result["prices"]["tcgplayer_normal"] == 0.18

    def test_set_name_used_as_middle_fallback(self):
        with patch.object(market, "_query", side_effect=[[], [], CARD_RESPONSE, ]) as q:
            # number query fails to match, set query fails to match, name-only hits
            market.lookup_prices("Pangoro", "999", "Astral Radiance")
        assert 'set.name:"Astral Radiance"' in q.call_args_list[1][0][0]["q"]

    def test_transient_failure_retried(self):
        # One cold-cache timeout then success on the same query — the real
        # pokemontcg.io behavior that made lookups silently fail in full
        # pipeline runs while passing in isolated (warm-cache) tests.
        with patch.object(market, "_query", side_effect=[OSError("timeout"), CARD_RESPONSE]) as q:
            result = market.lookup_prices("Latias", "193")
        assert result is not None
        assert q.call_count == 2
        assert q.call_args_list[0][0][0]["q"] == q.call_args_list[1][0][0]["q"], "retry reused the same query"
