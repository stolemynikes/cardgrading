"""Which way the card was turned between photometric scans.

This is the one capture parameter that fails silently. The scans themselves
look identical whichever way the card went round; only the mapping from frame
index to light direction changes, and getting it wrong solves every normal
against the wrong light — a plausible-looking render, lit from the wrong side,
with the two middle frames' directions swapped.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pipeline import cardvision
from webapp import main


class TestAzimuths:
    def test_the_two_directions_disagree(self):
        cw = cardvision.scanner_azimuths(4, 90.0, clockwise=True)
        ccw = cardvision.scanner_azimuths(4, 90.0, clockwise=False)
        assert cw != ccw

    def test_only_the_middle_frames_swap(self):
        """Which is why it's so easy to miss: the first and third-turn frames
        agree, so half the set looks right either way."""
        cw = [a % 360 for a in cardvision.scanner_azimuths(4, 90.0, clockwise=True)]
        ccw = [a % 360 for a in cardvision.scanner_azimuths(4, 90.0, clockwise=False)]
        assert cw[0] == ccw[0]
        assert cw[2] == ccw[2]
        assert cw[1] != ccw[1] and cw[3] != ccw[3]

    def test_both_span_all_four_directions(self):
        for clockwise in (True, False):
            azimuths = {a % 360 for a in cardvision.scanner_azimuths(4, 90.0, clockwise=clockwise)}
            assert len(azimuths) == 4, "four turns have to give four distinct light directions"


def _png(colour=200) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((80, 60, 3), colour, np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path / "reports")
    with TestClient(main.app) as test_client:
        yield test_client


def _files():
    return [("front", ("f.png", _png(), "image/png")), ("back", ("b.png", _png(190), "image/png"))]


class TestEndpoint:
    def test_counter_clockwise_is_accepted(self, client):
        res = client.post("/api/grade", files=_files(), data={"rotation": "ccw"})
        assert res.status_code == 200

    def test_clockwise_is_accepted(self, client):
        res = client.post("/api/grade", files=_files(), data={"rotation": "cw"})
        assert res.status_code == 200

    def test_an_unknown_direction_is_refused_rather_than_defaulted(self, client):
        """Defaulting would solve the scans against the wrong light and
        return a confident, wrong answer."""
        res = client.post("/api/grade", files=_files(), data={"rotation": "widdershins"})
        assert res.status_code == 400
        assert "cw" in res.json()["detail"]

    def test_omitting_it_still_works(self, client):
        """Nearly every capture has no rotation set at all — the field only
        matters when photometric scans are attached."""
        assert client.post("/api/grade", files=_files()).status_code == 200


class TestUiDefault:
    def test_the_page_offers_both_directions(self):
        html = (main.STATIC_DIR / "index.html").read_text()
        assert 'id="rotation-input"' in html
        assert 'value="ccw"' in html and 'value="cw"' in html

    def test_the_hint_no_longer_hardcodes_clockwise(self):
        """It used to instruct one direction while the solve assumed it —
        which was fine until someone turned the card the other way."""
        html = (main.STATIC_DIR / "index.html").read_text()
        hint = html[html.index("3&ndash;6 shots") : html.index("3&ndash;6 shots") + 300]
        assert "clockwise each time" not in hint
