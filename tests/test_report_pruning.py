"""Bounding the reports directory.

Reports are meant to still be there next week, so nothing deletes them. What
the store does instead is age out the images that exist only to be zoomed
into — the full-scale detail warps and the stored rotation scans — which are
most of a report's 140MB and none of its measurements.

The behaviour that matters is the split: what was measured survives forever,
what was only for looking at does not.
"""

from __future__ import annotations

import json

import pytest

from webapp import store

MEASURED_KEYS = {"front_aligned", "back_aligned", "front_centering_overlay", "front_card_vision"}
VIEW_ONLY_KEYS = {"front_detail", "back_detail", "front_rotation_0", "front_rotation_3"}


def _id(n: int) -> str:
    return f"{n:032x}"


def _report(grade: int = 9) -> dict:
    return {
        "card_id": {"card_name": "Charizard", "set_name": "Base Set"},
        "grade_estimate": {"overall_grade_rounded": grade, "score": 910},
        "centering": {"front": {"grade": grade}},
    }


def _images(tmp_path, keys):
    made = {}
    for key in keys:
        path = tmp_path / f"{key}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + key.encode())
        made[key] = path
    return made


def _save(reports_dir, tmp_path, n: int, created_at: str) -> str:
    report_id = _id(n)
    store.save_report(reports_dir, report_id, _report(), _images(tmp_path, MEASURED_KEYS | VIEW_ONLY_KEYS))
    # save_report stamps "now"; the ordering under test is by created_at, so
    # it has to be settable rather than raced.
    meta_path = reports_dir / report_id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["created_at"] = created_at
    meta_path.write_text(json.dumps(meta))
    return report_id


@pytest.fixture
def reports(tmp_path):
    reports_dir = tmp_path / "reports"
    ids = [_save(reports_dir, tmp_path, n, f"2026-09-{n + 1:02d}T10:00:00+00:00") for n in range(5)]
    return reports_dir, ids


def _stored_keys(reports_dir, report_id) -> set[str]:
    return {p.stem for p in (reports_dir / report_id / "images").glob("*.png")}


def test_the_newest_reports_keep_everything(reports):
    reports_dir, ids = reports
    store.prune_report_images(reports_dir, keep_full=2)
    for report_id in ids[-2:]:
        assert _stored_keys(reports_dir, report_id) == MEASURED_KEYS | VIEW_ONLY_KEYS


def test_older_reports_lose_only_the_view_only_images(reports):
    reports_dir, ids = reports
    store.prune_report_images(reports_dir, keep_full=2)
    for report_id in ids[:-2]:
        assert _stored_keys(reports_dir, report_id) == MEASURED_KEYS


def test_a_pruned_report_still_loads_and_still_has_its_measurements(reports):
    reports_dir, ids = reports
    store.prune_report_images(reports_dir, keep_full=2)
    loaded = store.load_report(reports_dir, ids[0])
    assert loaded is not None
    assert loaded["report"]["grade_estimate"]["overall_grade_rounded"] == 9
    assert loaded["images_pruned"] is True
    assert "front_aligned" in loaded["images"]
    assert "front_detail" not in loaded["images"]


def test_pruning_is_reported_and_idempotent(reports):
    reports_dir, ids = reports
    first = store.prune_report_images(reports_dir, keep_full=2)
    assert set(first) == set(ids[:-2])
    # Running again must not re-walk work it already did, and must not touch
    # the reports it deliberately left alone.
    assert store.prune_report_images(reports_dir, keep_full=2) == []
    for report_id in ids[-2:]:
        assert _stored_keys(reports_dir, report_id) == MEASURED_KEYS | VIEW_ONLY_KEYS


def test_the_flag_survives_an_edit(reports):
    """Re-grading a pruned report from hand-placed borders rewrites its
    report.json and meta.json. That must not claim the deleted warps are
    back — the flag describes the disk, not the edit."""
    reports_dir, ids = reports
    store.prune_report_images(reports_dir, keep_full=2)
    summary = store.update_report(reports_dir, ids[0], _report(grade=6))
    assert summary.images_pruned is True
    assert summary.grade == 6
    assert store.load_report(reports_dir, ids[0])["images_pruned"] is True


def test_created_at_ordering_decides_not_directory_order(reports):
    """The newest report is the newest by timestamp. Relying on iteration
    order would prune whichever one the filesystem happened to list last."""
    reports_dir, tmp_ids = reports
    # Make the oldest-written report the newest-dated one.
    meta_path = reports_dir / tmp_ids[0] / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["created_at"] = "2026-12-31T10:00:00+00:00"
    meta_path.write_text(json.dumps(meta))

    store.prune_report_images(reports_dir, keep_full=1)
    assert _stored_keys(reports_dir, tmp_ids[0]) == MEASURED_KEYS | VIEW_ONLY_KEYS
    assert _stored_keys(reports_dir, tmp_ids[4]) == MEASURED_KEYS


def test_a_report_with_no_meta_is_left_alone(reports, tmp_path):
    """No meta.json means a half-written or foreign directory: it can't be
    ordered, and pruning by guessed age loses the wrong report's images."""
    reports_dir, ids = reports
    (reports_dir / ids[0] / "meta.json").unlink()
    store.prune_report_images(reports_dir, keep_full=0)
    assert _stored_keys(reports_dir, ids[0]) == MEASURED_KEYS | VIEW_ONLY_KEYS


def test_keep_full_zero_prunes_everything(reports):
    reports_dir, ids = reports
    assert set(store.prune_report_images(reports_dir, keep_full=0)) == set(ids)
    for report_id in ids:
        assert _stored_keys(reports_dir, report_id) == MEASURED_KEYS


def test_a_negative_keep_is_a_programming_error_not_a_wipe(reports):
    reports_dir, ids = reports
    with pytest.raises(ValueError):
        store.prune_report_images(reports_dir, keep_full=-1)
    assert _stored_keys(reports_dir, ids[0]) == MEASURED_KEYS | VIEW_ONLY_KEYS


def test_missing_directory_is_not_an_error(tmp_path):
    assert store.prune_report_images(tmp_path / "nothing-here") == []


def test_a_finished_job_prunes_after_saving(tmp_path, monkeypatch):
    """The wiring, not the pruning. `prune_report_images` is only useful if
    something calls it, and a feature reachable from nowhere is how the
    reports directory grew to 266MB unnoticed in the first place."""
    import asyncio

    from webapp import jobs

    calls = []
    monkeypatch.setattr(jobs.store, "prune_report_images", lambda base: calls.append(base) or [])
    monkeypatch.setattr(jobs, "grade_card", lambda *a, **k: _report())

    job_id = jobs.create_job()
    job_root = tmp_path / "job"
    (job_root / "output").mkdir(parents=True)
    reports_dir = tmp_path / "reports"

    asyncio.run(
        jobs.run_job(
            job_id, job_root / "front", job_root / "back", {}, job_root / "output",
            job_root, reports_dir=reports_dir,
        )
    )

    assert jobs.get_job(job_id).status == "done"
    assert calls == [reports_dir]


def test_a_pruning_failure_does_not_cost_the_report(tmp_path, monkeypatch):
    """Pruning is housekeeping. A card the user just waited two minutes for
    must survive it going wrong."""
    import asyncio

    from webapp import jobs

    def boom(base):
        raise OSError("disk went away")

    monkeypatch.setattr(jobs.store, "prune_report_images", boom)
    monkeypatch.setattr(jobs, "grade_card", lambda *a, **k: _report())

    job_id = jobs.create_job()
    job_root = tmp_path / "job"
    (job_root / "output").mkdir(parents=True)

    asyncio.run(
        jobs.run_job(
            job_id, job_root / "front", job_root / "back", {}, job_root / "output",
            job_root, reports_dir=tmp_path / "reports",
        )
    )

    job = jobs.get_job(job_id)
    assert job.status == "done"
    assert job.result["report_id"] == job_id
