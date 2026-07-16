#!/usr/bin/env python3
"""Collect grade-labeled card images via PSA's official Public API.

Uses the documented endpoints (https://www.psacard.com/publicapi):
- GET /cert/GetByCertNumber/{cert}   -> category, subject, grade
- GET /cert/GetImagesByCertNumber/{cert} -> PSA's own scans of that card

Authentication: a bearer token generated while signed in at
psacard.com/publicapi, supplied via the PSA_API_TOKEN env var. Free
accounts get ~100 calls/day; this script spends them carefully, records
every cert it has ever tried (never re-spends a call), and stops cleanly
when the API starts refusing.

Cert discovery: the API is lookup-only, so certs are found by walking
outward from known seed certs. PSA grades submissions in batches, so
neighbors of a Pokemon cert are usually more Pokemon cards from the same
submission. Seed with any cert numbers you know (your own graded cards
are ideal); one verified seed ships as a default.

Output: calibration/psa_cache/manifest.json — one entry per kept cert
(number, grade, subject, downloaded front/back image paths), ready for a
measurement + fit pass.

Usage:
    PSA_API_TOKEN=... python calibration/collect_psa.py --budget 95
    PSA_API_TOKEN=... python calibration/collect_psa.py --seeds 136840848,12345678
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_BASE = "https://api.psacard.com/publicapi"
# Cert read off a slab label in the public Poke-Grader dataset (2023 CLF
# Snorlax, PSA 10) — every collected Pokemon cert becomes a future seed.
DEFAULT_SEEDS = [136840848]
POKEMON_MARKERS = ("POKEMON", "POKÉMON")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=int, default=95, help="Max API calls to spend this run (free tier: 100/day)")
    p.add_argument("--seeds", type=str, default="", help="Comma-separated cert numbers to walk from (adds to stored seeds)")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "calibration" / "psa_cache")
    p.add_argument("--pace", type=float, default=1.5, help="Seconds between API calls")
    return p.parse_args(argv)


class Api:
    def __init__(self, token: str, pace: float):
        self.token = token
        self.pace = pace
        self.calls = 0

    def get(self, path: str) -> dict | None:
        """One API call. None = request refused/failed (auth, limit, 5xx)."""
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            headers={"authorization": f"bearer {self.token}", "User-Agent": "card-pre-grader-calibration"},
        )
        self.calls += 1
        time.sleep(self.pace)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            print(f"    API {e.code} on {path}", flush=True)
            if e.code in (401, 403, 429, 500):
                raise QuotaOrAuthError(f"HTTP {e.code}")
            return None
        except Exception as e:
            print(f"    API error on {path}: {e}", flush=True)
            return None


class QuotaOrAuthError(Exception):
    pass


def parse_grade(cert: dict) -> float | None:
    """CardGrade comes as strings like 'MINT 9', 'GEM MT 10', 'NM-MT 8'."""
    raw = (cert.get("CardGrade") or "").strip()
    for token in reversed(raw.replace("-", " ").split()):
        try:
            value = float(token)
        except ValueError:
            continue
        if 1 <= value <= 10:
            return value
    return None


def is_pokemon(cert: dict) -> bool:
    hay = " ".join(str(cert.get(k, "")) for k in ("Brand", "Category", "Subject")).upper()
    return any(m in hay for m in POKEMON_MARKERS)


def walk_order(seeds: list[int], tried: set[int]) -> list[int]:
    """Alternate outward from every seed: s+1, s-1, s+2, s-2, ..."""
    order = []
    for radius in range(0, 2000):
        for seed in seeds:
            for cert in (seed + radius, seed - radius) if radius else (seed,):
                if cert > 0 and cert not in tried:
                    order.append(cert)
                    tried = tried | {cert}
    # de-dup preserving order
    seen: set[int] = set()
    return [c for c in order if not (c in seen or seen.add(c))]


def download_image(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "card-pre-grader-calibration"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"    image download failed: {e}", flush=True)
        return False


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    token = os.environ.get("PSA_API_TOKEN")
    if not token:
        print("PSA_API_TOKEN not set — generate one at psacard.com/publicapi while signed in.")
        return 1

    args.cache.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache / "manifest.json"
    attempted_path = args.cache / "attempted.json"
    manifest: list[dict] = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    attempted: set[int] = set(json.loads(attempted_path.read_text())) if attempted_path.exists() else set()

    seeds = list(DEFAULT_SEEDS)
    seeds += [int(s) for s in args.seeds.split(",") if s.strip()]
    # every Pokemon cert we've already collected seeds future walks
    seeds += [int(m["cert"]) for m in manifest]
    seeds = sorted(set(seeds))
    print(f"{len(manifest)} cards collected so far; {len(attempted)} certs attempted; {len(seeds)} seeds", flush=True)

    api = Api(token, args.pace)
    kept = 0

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=1))
        attempted_path.write_text(json.dumps(sorted(attempted)))

    try:
        for cert_no in walk_order(seeds, attempted):
            if api.calls >= args.budget:
                print("budget reached", flush=True)
                break
            attempted.add(cert_no)
            data = api.get(f"/cert/GetByCertNumber/{cert_no}")
            if not data or not (data.get("PSACert") or {}).get("CertNumber"):
                continue
            cert = data["PSACert"]
            grade = parse_grade(cert)
            if not is_pokemon(cert) or grade is None:
                print(f"  {cert_no}: skip ({cert.get('Brand', '?')[:30]} / {cert.get('CardGrade', '?')})", flush=True)
                continue
            if api.calls >= args.budget:
                break
            images = api.get(f"/cert/GetImagesByCertNumber/{cert_no}") or []
            fronts = [i for i in images if i.get("IsFrontImage")]
            backs = [i for i in images if not i.get("IsFrontImage")]
            if not fronts:
                print(f"  {cert_no}: {cert.get('Subject','?')[:30]} psa {grade} — no images", flush=True)
                continue
            front_path = args.cache / f"{cert_no}_front.jpg"
            back_path = args.cache / f"{cert_no}_back.jpg"
            if not download_image(fronts[0].get("ImageURL", ""), front_path):
                continue
            has_back = bool(backs) and download_image(backs[0].get("ImageURL", ""), back_path)
            manifest.append(
                {
                    "cert": cert_no,
                    "grade": grade,
                    "subject": cert.get("Subject", ""),
                    "year": cert.get("Year", ""),
                    "brand": cert.get("Brand", ""),
                    "variety": cert.get("Variety", ""),
                    "front": front_path.name,
                    "back": back_path.name if has_back else None,
                }
            )
            kept += 1
            save()
            print(f"  {cert_no}: KEPT {cert.get('Subject','?')[:30]} psa {grade} (front{'+back' if has_back else ''})", flush=True)
    except QuotaOrAuthError as e:
        print(f"stopping: API refused ({e}) — daily limit or bad token. Progress is saved; rerun tomorrow.", flush=True)
    finally:
        save()

    print(f"\nrun done: {api.calls} API calls, {kept} new cards, {len(manifest)} total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
