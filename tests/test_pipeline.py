"""End-to-end Pipeline orchestration tests (모킹).

각 외부 모듈은 단위 테스트에서 이미 검증되었으므로, 여기서는
파이프라인 단계 호출 순서 · 데이터 전달 contract · 워크스페이스 기록을 확인한다.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from guitarvideo2tab import pipeline as pipeline_mod
from guitarvideo2tab.models import (
    FretboardFrame,
    FretPosition,
    HandKeypoints,
    MidiEvent,
    NoteEvent,
)
from guitarvideo2tab.pipeline import Pipeline
from guitarvideo2tab.workspace import RunWorkspace

TS = datetime(2026, 7, 25, 23, 45, 12)


@pytest.fixture
def workspace(tmp_path: Path) -> RunWorkspace:
    return RunWorkspace.create(tmp_path / "runs", "clip.mp4", timestamp=TS)


def _stub_pipeline_dependencies(monkeypatch, recorder: dict) -> None:
    """파이프라인 단계들을 결정론적 스텁으로 교체한다."""

    def stub_download(source, output_dir):
        recorder["download"] = (source, output_dir)
        path = Path(output_dir) / "video.mp4"
        path.write_bytes(b"fake-video")
        return path

    def stub_split(video_path, output_dir):
        recorder["split"] = (video_path, output_dir)
        audio, video_only = Path(output_dir) / "audio.wav", Path(output_dir) / "video-only.mp4"
        audio.write_bytes(b"fake-audio")
        video_only.write_bytes(b"fake-video-only")
        return audio, video_only

    def stub_stem(audio_path, output_dir):
        recorder["stem"] = (audio_path, output_dir)
        path = Path(output_dir) / "guitar.wav"
        path.write_bytes(b"fake-guitar")
        return path

    monkeypatch.setattr(pipeline_mod, "download_video", stub_download)
    monkeypatch.setattr(pipeline_mod, "split_audio_video", stub_split)
    monkeypatch.setattr(pipeline_mod, "separate_guitar_stem", stub_stem)

    # 이 코드베이스는 string=1 이 최저음현(tuning[0]=40) 이다.
    # 40 + 3 = 43 → (string 1, fret 3) 과 pitch 가 일치하는 음.
    midi_event = MidiEvent(pitch=43, start_time=0.0, end_time=0.5, velocity=80)
    fret_position = FretPosition(timestamp=0.1, string=1, fret=3, confidence=0.95)

    class StubTranscriber:
        def __init__(self, *a, **kw):
            recorder["transcriber_init"] = (a, kw)

        def transcribe(self, audio_path):
            recorder["transcribe"] = audio_path
            return [midi_event]

    class StubAudioClassifier:
        def __init__(self, **kw):
            recorder["audio_classifier_init"] = kw

        def classify(self, midi_events, audio_path):
            recorder["audio_classify"] = (len(midi_events), audio_path)
            return []

    class StubFretboardDetector:
        def __init__(self, **kw):
            recorder["fretboard_init"] = kw

        def detect(self, video_path):
            recorder["fretboard_detect"] = video_path
            return [FretboardFrame(timestamp=0.0, homography=None, corners=None, visible=False)]

    class StubHandTracker:
        def __init__(self, **kw):
            recorder["hands_init"] = kw

        def track(self, video_path):
            recorder["hands_track"] = video_path
            return [HandKeypoints(timestamp=0.0, left_hand=None, right_hand=None)]

    class StubFretEstimator:
        def __init__(self, *a, **kw):
            recorder["fret_estimator_init"] = (a, kw)

        def estimate(self, hands, fretboards):
            recorder["fret_estimate"] = (len(hands), len(fretboards))
            return [fret_position]

    class StubVisionClassifier:
        def __init__(self, **kw):
            recorder["vision_classifier_init"] = kw

        def classify(self, hands):
            recorder["vision_classify"] = len(hands)
            return []

    class StubLateFusion:
        def __init__(self, *a, **kw):
            recorder["fusion_init"] = (a, kw)

        def fuse(self, midi_events, audio_techs, fret_positions, vision_techs):
            recorder["fuse"] = (
                len(midi_events), len(audio_techs), len(fret_positions), len(vision_techs)
            )
            return [NoteEvent(midi_event=midi_event, string=1, fret=3, technique=None)]

    monkeypatch.setattr(pipeline_mod, "BasicPitchTranscriber", StubTranscriber)
    monkeypatch.setattr(pipeline_mod, "TARTTechniqueClassifier", StubAudioClassifier)
    monkeypatch.setattr(pipeline_mod, "FretboardDetector", StubFretboardDetector)
    monkeypatch.setattr(pipeline_mod, "HandTracker", StubHandTracker)
    monkeypatch.setattr(pipeline_mod, "FretEstimator", StubFretEstimator)
    monkeypatch.setattr(pipeline_mod, "VisionTechniqueClassifier", StubVisionClassifier)
    monkeypatch.setattr(pipeline_mod, "LateFusion", StubLateFusion)


def test_pipeline_run_orchestrates_all_stages_in_order(workspace, monkeypatch):
    recorder: dict = {}
    _stub_pipeline_dependencies(monkeypatch, recorder)

    outputs = Pipeline(workspace=workspace, tempo_bpm=120.0).run("https://example.com/clip")

    stages = workspace.stages_dir
    assert recorder["download"] == ("https://example.com/clip", stages)
    assert recorder["split"][0] == stages / "video.mp4"
    assert recorder["stem"][0] == stages / "audio.wav"
    assert recorder["transcribe"] == stages / "guitar.wav"
    assert recorder["audio_classify"] == (1, stages / "guitar.wav")
    assert recorder["fretboard_detect"] == stages / "video-only.mp4"
    assert recorder["hands_track"] == stages / "video-only.mp4"
    assert recorder["fret_estimate"] == (1, 1)
    assert recorder["vision_classify"] == 1
    assert recorder["fuse"] == (1, 0, 1, 0)
    assert set(outputs) == {"midi", "musicxml"}


def test_pipeline_writes_both_output_formats(workspace, monkeypatch):
    _stub_pipeline_dependencies(monkeypatch, {})

    outputs = Pipeline(workspace=workspace, tempo_bpm=120.0).run("clip.mp4")

    assert outputs["midi"] == workspace.output("tab.mid")
    assert outputs["musicxml"] == workspace.output("tab.musicxml")
    assert outputs["midi"].exists() and outputs["midi"].stat().st_size > 0
    assert outputs["musicxml"].exists() and outputs["musicxml"].stat().st_size > 0


def test_pipeline_dumps_intermediates_into_stages_dir(workspace, monkeypatch):
    _stub_pipeline_dependencies(monkeypatch, {})

    Pipeline(workspace=workspace, tempo_bpm=120.0).run("clip.mp4")

    names = {p.name for p in workspace.stages_dir.glob("*.json")}
    assert names == {
        "04_midi_events.json",
        "05_audio_techniques.json",
        "06_fretboards.json",
        "07_hands.json",
        "08_fret_positions.json",
        "09_vision_techniques.json",
        "10_notes_fused.json",
    }


def test_pipeline_skips_json_dumps_when_disabled(workspace, monkeypatch):
    _stub_pipeline_dependencies(monkeypatch, {})

    Pipeline(workspace=workspace, tempo_bpm=120.0, save_intermediates=False).run("clip.mp4")

    assert list(workspace.stages_dir.glob("*.json")) == []
    # 미디어 산출물은 save_intermediates 와 무관하게 남는다
    assert (workspace.stages_dir / "guitar.wav").exists()


def test_manifest_records_stages_outputs_and_totals(workspace, monkeypatch):
    _stub_pipeline_dependencies(monkeypatch, {})

    Pipeline(workspace=workspace, tempo_bpm=120.0).run("clip.mp4")
    manifest = json.loads(workspace.manifest_path.read_text())

    stage_names = [s["stage"] for s in manifest["stages"]]
    assert stage_names[:3] == ["01_download", "02_split", "03_stem"]
    assert "11_midi" in stage_names and "12_musicxml" in stage_names
    assert manifest["outputs"] == {
        "midi": "output/tab.mid",
        "musicxml": "output/tab.musicxml",
    }
    totals = manifest["totals"]
    assert totals["midi_events"] == 1
    assert totals["notes_resolved"] == 1
    assert totals["tempo_bpm"] == 120.0
    assert totals["measures"] >= 1
    # pitch 43 = 최저음현 3프렛 → 운지가 pitch 와 일치하므로 100%
    assert totals["fingering_match_ratio"] == 1.0
    assert manifest["config"]["subdivision"] == 16


def test_pipeline_forwards_weights_to_classifiers(workspace, monkeypatch, tmp_path):
    recorder: dict = {}
    _stub_pipeline_dependencies(monkeypatch, recorder)

    audio_w, vision_w = tmp_path / "tart.pt", tmp_path / "tcn.pt"
    fret_w, hands_asset = tmp_path / "yolo.pt", tmp_path / "hands.task"

    Pipeline(
        workspace=workspace,
        tempo_bpm=120.0,
        audio_weights=audio_w,
        vision_weights=vision_w,
        fretboard_weights=fret_w,
        hands_model_asset=hands_asset,
    ).run("clip.mp4")

    assert recorder["audio_classifier_init"]["weights_path"] == audio_w
    assert recorder["vision_classifier_init"]["weights_path"] == vision_w
    assert recorder["fretboard_init"]["weights_path"] == fret_w
    assert recorder["hands_init"]["model_asset_path"] == hands_asset


def test_dump_failure_does_not_abort_pipeline(workspace, monkeypatch):
    pipeline = Pipeline(workspace=workspace, tempo_bpm=120.0)

    def exploding_write(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", exploding_write)

    with pytest.warns(RuntimeWarning):
        assert pipeline._dump("04_midi_events", [1, 2, 3]) is None


def test_two_runs_do_not_share_artifacts(tmp_path, monkeypatch):
    _stub_pipeline_dependencies(monkeypatch, {})
    runs_dir = tmp_path / "runs"

    first = RunWorkspace.create(runs_dir, "a.mp4", timestamp=datetime(2026, 7, 25, 1, 0, 0))
    second = RunWorkspace.create(runs_dir, "b.mp4", timestamp=datetime(2026, 7, 25, 2, 0, 0))
    Pipeline(workspace=first, tempo_bpm=120.0).run("a.mp4")
    Pipeline(workspace=second, tempo_bpm=90.0).run("b.mp4")

    assert first.root != second.root
    assert first.output("tab.mid").exists() and second.output("tab.mid").exists()
    assert json.loads(first.manifest_path.read_text())["totals"]["tempo_bpm"] == 120.0
    assert json.loads(second.manifest_path.read_text())["totals"]["tempo_bpm"] == 90.0
