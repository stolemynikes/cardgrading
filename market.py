"""Market price lookup for an identified card via the pokemontcg.io API.

Purely optional context: pre-grading exists to answer "is this card worth
submitting", so once the vision stage has identified the card, this fetches
the raw (ungraded) market price. Free API; an optional POKEMONTCG_API_KEY
env var raises the rate limits. Any failure — network, no match, schema
surprise — returns None and the report simply omits the value line.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

API_URL = "https://api.pokemontcg.io/v2/cards"
# pokemontcg.io is slow on cold queries (5-15s is common before their CDN
# cache warms) — a short timeout made the lookup silently fail most of the
# time in real runs while working in isolated tests against a warm cache.
TIMEOUT_SECONDS = 15
ATTEMPTS_PER_QUERY = 2


def _query(params: dict) -> list[dict]:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "card-pre-grader"})
    api_key = os.environ.get("POKEMONTCG_API_KEY")
    if api_key:
        request.add_header("X-Api-Key", api_key)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
        return json.load(resp).get("data", [])


def _extract_prices(card: dict) -> dict:
    prices = {}
    tcgplayer = (card.get("tcgplayer") or {}).get("prices") or {}
    # tcgplayer prices are keyed by finish (normal, holofoil, reverseHolofoil…)
    for finish, values in tcgplayer.items():
        market = (values or {}).get("market")
        if market is not None:
            prices[f"tcgplayer_{finish}"] = market
    cardmarket = (card.get("cardmarket") or {}).get("prices") or {}
    if cardmarket.get("trendPrice") is not None:
        prices["cardmarket_trend"] = cardmarket["trendPrice"]
    return prices


def _normalize_number(collector_number: str) -> str:
    """Cards print numbers like '193/264'; the API's number field is just
    '193'. Keep alphanumeric promo numbers ('SWSH284') as-is."""
    return collector_number.split("/")[0].strip()


def lookup_prices(card_name: str, collector_number: str = "", set_name: str = "") -> dict | None:
    """Best-effort market prices for the identified card. None on any failure."""
    if not card_name:
        return None
    # Most-precise query first, loosening step by step — the vision model's
    # number/set guesses are less reliable than the name.
    queries = []
    number = _normalize_number(collector_number)
    if number:
        queries.append(f'name:"{card_name}" number:{number}')
    if set_name:
        queries.append(f'name:"{card_name}" set.name:"{set_name}"')
    queries.append(f'name:"{card_name}"')

    for q in queries:
        # Failures are per-query: a malformed precise query must not kill
        # the fallbacks (this exact bug shipped once — the model returned
        # "193/264" and the whole lookup silently died).
        cards = None
        for _ in range(ATTEMPTS_PER_QUERY):
            try:
                cards = _query({"q": q, "pageSize": 10, "orderBy": "-set.releaseDate"})
                break
            except Exception:
                continue
        if cards:
            # The API's name matching is substring-ish: querying "Latias"
            # returns "Mega Latias ex" too, and release-date ordering put a
            # $92 card first for a $0.30 one. An exact name match wins;
            # only fall back to the API's first result if none exists.
            exact = [c for c in cards if (c.get("name") or "").lower() == card_name.lower()]
            card = exact[0] if exact else cards[0]
            return {
                "matched_name": card.get("name", card_name),
                "matched_set": (card.get("set") or {}).get("name", ""),
                "matched_number": card.get("number", ""),
                "prices": _extract_prices(card),
                "image_url": (card.get("images") or {}).get("small", ""),
                "source": "pokemontcg.io",
            }
    return None
