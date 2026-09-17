"""Report store — the on-disk half of "reports are saved under a UUID"."""

from __future__ import annotations

import json

import pytest

from webapp import store

REPORT_ID = "0123456789abcdef0123456789abcdef"
OTHER_ID = "fedcba9876543210fedcba9876543210"


def _report(name: str = "Charizard", grade: int = 9, score: int = 910) -> dict:
    return {
        "card_id": {"card_name": name, "set_name": "Base Set"},
        "grade_estimate": {"overall_grade_rounded": grade, "score": score},
        "centering": {"front": {}, "back": {}},
    }


def _image(tmp_path, name: str) -> "object":
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    return path


def test_save_then_load_round_trips(tmp_path):
    images = {"front_aligned": _image(tmp_path, "a.png"), "front_card_vision": _image(tmp_path, "b.png")}
    store.save_report(tmp_path / "reports", REPORT_ID, _report(), images)

    loaded = store.load_report(tmp_path / "reports", REPORT_ID)
    assert loaded is not None
    assert loaded["report"]["card_id"]["card_name"] == "Charizard"
    assert set(loaded["images"]) == {"front_aligned", "front_card_vision"}
    assert loaded["images"]["front_aligned"].startswith("data:image/png;base64,")


def test_missing_report_is_none_not_an_error(tmp_path):
    assert store.load_report(tmp_path / "reports", OTHER_ID) is None


@pytest.mark.parametrize(
    "bad_id",
    ["../etc/passwd", "..", "not-a-uuid", "0123456789ABCDEF0123456789ABCDEF", "0123456789abcdef", ""],
)
def test_malformed_ids_are_rejected(tmp_path, bad_id):
    """Report ids become directory names, so anything that isn't a uuid4 hex
    must never reach the filesystem."""
    assert not store.is_valid_report_id(bad_id)
    assert store.report_dir(tmp_path, bad_id) is None
    assert store.load_report(tmp_path, bad_id) is None
    assert store.delete_report(tmp_path, bad_id) is False
    with pytest.raises(ValueError):
        store.save_report(tmp_path, bad_id, _report(), {})


def test_listing_is_newest_first_and_summary_only(tmp_path):
    reports_dir = tmp_path / "reports"
    store.save_report(reports_dir, REPORT_ID, _report("Pikachu", 8, 820), {})
    store.save_report(reports_dir, OTHER_ID, _report("Blastoise", 10, 990), {})
    # created_at has one-second resolution, so force a distinguishable order
    # rather than depending on how fast the two writes above ran.
    meta_path = reports_dir / REPORT_ID / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["created_at"] = "2020-01-01T00:00:00+00:00"
    meta_path.write_text(json.dumps(meta))

    listed = store.list_reports(reports_dir)
    assert [r["card_name"] for r in listed] == ["Blastoise", "Pikachu"]
    assert listed[0]["score"] == 990
    # A summary must not drag the whole report (and its images) along.
    assert "centering" not in listed[0]


def test_listing_ignores_junk_directories(tmp_path):
    reports_dir = tmp_path / "reports"
    store.save_report(reports_dir, REPORT_ID, _report(), {})
    (reports_dir / "not-a-report").mkdir()
    (reports_dir / ("deadbeef" * 3)).mkdir()  # 24 hex chars — right alphabet, wrong length
    (reports_dir / OTHER_ID).mkdir()  # a directory with no meta.json
    assert [r["report_id"] for r in store.list_reports(reports_dir)] == [REPORT_ID]


def test_listing_survives_a_corrupt_meta_file(tmp_path):
    reports_dir = tmp_path / "reports"
    store.save_report(reports_dir, REPORT_ID, _report(), {})
    store.save_report(reports_dir, OTHER_ID, _report("Mewtwo"), {})
    (reports_dir / REPORT_ID / "meta.json").write_text("{ truncated")
    assert [r["report_id"] for r in store.list_reports(reports_dir)] == [OTHER_ID]


def test_missing_listing_directory_is_empty_not_an_error(tmp_path):
    assert store.list_reports(tmp_path / "never-created") == []


def test_saving_twice_replaces_rather_than_merges(tmp_path):
    reports_dir = tmp_path / "reports"
    store.save_report(reports_dir, REPORT_ID, _report(), {"front_aligned": _image(tmp_path, "a.png")})
    store.save_report(reports_dir, REPORT_ID, _report("Mew"), {})
    loaded = store.load_report(reports_dir, REPORT_ID)
    assert loaded["report"]["card_id"]["card_name"] == "Mew"
    assert loaded["images"] == {}


def test_no_staging_directory_is_left_behind(tmp_path):
    """The write stages then renames, so a half-written report can't be read
    as a real one — and the staging directory must not linger either."""
    reports_dir = tmp_path / "reports"
    store.save_report(reports_dir, REPORT_ID, _report(), {})
    assert [c.name for c in reports_dir.iterdir()] == [REPORT_ID]


def test_delete_removes_the_report(tmp_path):
    reports_dir = tmp_path / "reports"
    store.save_report(reports_dir, REPORT_ID, _report(), {})
    assert store.delete_report(reports_dir, REPORT_ID) is True
    assert store.load_report(reports_dir, REPORT_ID) is None
    assert store.delete_report(reports_dir, REPORT_ID) is False


def test_summary_tolerates_a_report_with_no_card_id(tmp_path):
    """A skipped vision step (no API key) leaves card_id null — that's the
    common case on this setup, not an edge case."""
    reports_dir = tmp_path / "reports"
    store.save_report(reports_dir, REPORT_ID, {"card_id": None, "grade_estimate": None}, {})
    summary = store.list_reports(reports_dir)[0]
    assert summary["card_name"] is None
    assert summary["grade"] is None
