"""RunWorkspace: 실행별 산출물 격리와 manifest 기록."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from guitarvideo2tab.workspace import (
    LATEST_LINK,
    RunWorkspace,
    list_runs,
    slugify,
)

TS = datetime(2026, 7, 25, 23, 45, 12)


def test_slugify_keeps_hangul_and_strips_punctuation() -> None:
    assert slugify("02-02. 주이름 찬양 _ Verse 1.mp4") == "02-02-주이름-찬양-Verse-1"
    assert slugify("/path/to/My Song!!.mp4") == "My-Song"
    assert slugify("....") == "run"


def test_slugify_truncates_long_names() -> None:
    assert len(slugify("x" * 200)) <= 48


def test_create_makes_run_scoped_directories(tmp_path: Path) -> None:
    ws = RunWorkspace.create(tmp_path, "clip.mp4", timestamp=TS)

    assert ws.run_id == "20260725-234512_clip"
    assert ws.root == tmp_path / ws.run_id
    assert ws.input_dir.is_dir()
    assert ws.stages_dir.is_dir()
    assert ws.output_dir.is_dir()
    assert ws.manifest_path.exists()


def test_manifest_records_source_and_timestamp(tmp_path: Path) -> None:
    ws = RunWorkspace.create(tmp_path, "clip.mp4", timestamp=TS)
    manifest = json.loads(ws.manifest_path.read_text())

    assert manifest["run_id"] == ws.run_id
    assert manifest["source"] == "clip.mp4"
    assert manifest["created_at"].startswith("2026-07-25T23:45:12")


def test_same_second_runs_do_not_collide(tmp_path: Path) -> None:
    first = RunWorkspace.create(tmp_path, "clip.mp4", timestamp=TS)
    second = RunWorkspace.create(tmp_path, "clip.mp4", timestamp=TS)

    assert first.root != second.root
    assert second.run_id.endswith("-2")
    assert first.manifest_path.exists() and second.manifest_path.exists()


def test_latest_symlink_points_at_newest_run(tmp_path: Path) -> None:
    RunWorkspace.create(tmp_path, "old.mp4", timestamp=datetime(2026, 7, 25, 1, 0, 0))
    newest = RunWorkspace.create(tmp_path, "new.mp4", timestamp=TS)

    link = tmp_path / LATEST_LINK
    assert link.is_symlink()
    assert link.resolve() == newest.root.resolve()


def test_stage_and_output_paths_are_scoped(tmp_path: Path) -> None:
    ws = RunWorkspace.create(tmp_path, "clip.mp4", timestamp=TS)

    assert ws.stage("04_midi_events.json").parent == ws.stages_dir
    assert ws.output("tab.mid").parent == ws.output_dir
    assert ws.relative(ws.output("tab.mid")) == "output/tab.mid"


def test_record_stage_accumulates_entries(tmp_path: Path) -> None:
    ws = RunWorkspace.create(tmp_path, "clip.mp4", timestamp=TS)

    ws.record_stage("01_download", count=None, elapsed_sec=1.2345, path=ws.stage("v.mp4"))
    ws.record_stage("04_transcribe", count=352, elapsed_sec=9.0)
    ws.save_manifest()

    stages = json.loads(ws.manifest_path.read_text())["stages"]
    assert [s["stage"] for s in stages] == ["01_download", "04_transcribe"]
    assert stages[0]["elapsed_sec"] == 1.234 or stages[0]["elapsed_sec"] == 1.235
    assert stages[0]["path"] == "stages/v.mp4"
    assert stages[1]["count"] == 352


def test_record_output_and_totals(tmp_path: Path) -> None:
    ws = RunWorkspace.create(tmp_path, "clip.mp4", timestamp=TS)

    ws.record_output("midi", ws.output("tab.mid"))
    ws.record_output("musicxml", ws.output("tab.musicxml"))
    ws.record_totals(measures=21, notes_written=147)
    ws.set_config(subdivision=16)
    ws.save_manifest()

    manifest = json.loads(ws.manifest_path.read_text())
    assert manifest["outputs"] == {
        "midi": "output/tab.mid",
        "musicxml": "output/tab.musicxml",
    }
    assert manifest["totals"] == {"measures": 21, "notes_written": 147}
    assert manifest["config"]["subdivision"] == 16


def test_open_reads_existing_manifest(tmp_path: Path) -> None:
    created = RunWorkspace.create(tmp_path, "clip.mp4", timestamp=TS)
    created.record_totals(measures=3)
    created.save_manifest()

    reopened = RunWorkspace.open(created.root)
    assert reopened.run_id == created.run_id
    assert reopened.manifest["totals"]["measures"] == 3


def test_list_runs_returns_newest_first_excluding_latest_link(tmp_path: Path) -> None:
    RunWorkspace.create(tmp_path, "a.mp4", timestamp=datetime(2026, 7, 25, 1, 0, 0))
    RunWorkspace.create(tmp_path, "b.mp4", timestamp=datetime(2026, 7, 25, 2, 0, 0))

    runs = list_runs(tmp_path)
    assert len(runs) == 2
    assert runs[0].name.startswith("20260725-020000")
    assert all(r.name != LATEST_LINK for r in runs)


def test_list_runs_on_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "nope") == []
