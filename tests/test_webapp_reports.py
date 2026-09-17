"""The HTTP surface for saved reports: listing, fetching, permalinks, delete.

These go through the real app rather than calling `store` directly — the
point of the endpoints is the id validation and the response shape the
frontend depends on, and neither lives in the store.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp import main, store

REPORT_ID = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path / "reports")
    with TestClient(main.app) as test_client:
        yield test_client


def _save(tmp_path, name: str = "Charizard", report_id: str = REPORT_ID) -> None:
    store.save_report(
        tmp_path / "reports",
        report_id,
        {
            "card_id": {"card_name": name, "set_name": "Base Set"},
            "grade_estimate": {"overall_grade_rounded": 9, "score": 910},
        },
        {},
    )


def test_empty_listing(client):
    assert client.get("/api/reports").json() == {"reports": []}


def test_listing_shows_saved_reports(client, tmp_path):
    _save(tmp_path)
    reports = client.get("/api/reports").json()["reports"]
    assert len(reports) == 1
    assert reports[0]["card_name"] == "Charizard"
    assert reports[0]["report_id"] == REPORT_ID


def test_fetching_a_report_matches_the_job_response_shape(client, tmp_path):
    """The frontend renders a revisited report through the same path as a
    fresh one, so the two responses have to agree on their keys."""
    _save(tmp_path)
    body = client.get(f"/api/report/{REPORT_ID}").json()
    assert body["status"] == "done"
    assert body["report_id"] == REPORT_ID
    assert "report" in body and "images" in body


def test_unknown_report_is_404(client):
    assert client.get("/api/report/" + "a" * 32).status_code == 404


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "a" * 31, "A" * 32])
def test_malformed_ids_are_404_not_500(client, bad_id):
    assert client.get(f"/api/report/{bad_id}").status_code == 404
    assert client.get(f"/r/{bad_id}").status_code == 404


def test_permalink_serves_the_app_shell(client, tmp_path):
    """The deep link returns the SPA, which reads the id back out of the URL —
    there's no second template to drift from the live report's rendering."""
    _save(tmp_path)
    response = client.get(f"/r/{REPORT_ID}")
    assert response.status_code == 200
    assert "<title>Card Pre-Grader</title>" in response.text


def test_permalink_for_a_wellformed_but_unknown_id_still_serves_the_app(client):
    """The id is valid, so the app loads and reports 'not found' itself —
    which reads better than a bare 404 page and keeps the nav intact."""
    assert client.get("/r/" + "b" * 32).status_code == 200


def test_delete_removes_it_from_the_listing(client, tmp_path):
    _save(tmp_path)
    assert client.delete(f"/api/report/{REPORT_ID}").json() == {"deleted": REPORT_ID}
    assert client.get("/api/reports").json()["reports"] == []
    assert client.delete(f"/api/report/{REPORT_ID}").status_code == 404


def test_listing_limit_is_clamped(client, tmp_path):
    for i in range(3):
        _save(tmp_path, f"Card {i}", report_id=f"{i:032x}")
    assert len(client.get("/api/reports?limit=2").json()["reports"]) == 2
    # Absurd limits must not turn into an absurd read.
    assert len(client.get("/api/reports?limit=100000").json()["reports"]) == 3
    assert len(client.get("/api/reports?limit=0").json()["reports"]) == 1
