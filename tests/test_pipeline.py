"""End-to-end Pipeline orchestration tests (모킹).

각 외부 모듈은 단위 테스트에서 이미 검증되었으므로, 여기서는
파이프라인 단계 호출 순서와 데이터 전달 contract만 확인한다.
"""
from __future__ import annotations

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


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path / "work"


def _stub_pipeline_dependencies(monkeypatch, recorder: dict) -> None:
    """파이프라인 단계들을 결정론적 스텁으로 교체한다.

    recorder 에는 단계별 호출 인자가 누적된다.
    """

    def stub_download(source, output_dir):
        recorder["download"] = (source, output_dir)
        return Path(output_dir) / "video.mp4"

    def stub_split(video_path, output_dir):
        recorder["split"] = (video_path, output_dir)
        return (Path(output_dir) / "audio.wav", Path(output_dir) / "video-only.mp4")

    def stub_stem(audio_path, output_dir):
        recorder["stem"] = (audio_path, output_dir)
        return Path(output_dir) / "guitar.wav"

    monkeypatch.setattr(pipeline_mod, "download_video", stub_download)
    monkeypatch.setattr(pipeline_mod, "split_audio_video", stub_split)
    monkeypatch.setattr(pipeline_mod, "separate_guitar_stem", stub_stem)

    midi_event = MidiEvent(pitch=64, start_time=0.0, end_time=0.5, velocity=80)
    fret_position = FretPosition(timestamp=0.1, string=2, fret=5, confidence=0.95)

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
            return []  # weights_path=None 일 때 동작 모사

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
                len(midi_events),
                len(audio_techs),
                len(fret_positions),
                len(vision_techs),
            )
            return [NoteEvent(midi_event=midi_event, string=2, fret=5, technique=None)]

    class StubTabWriter:
        def __init__(self, *a, **kw):
            recorder["writer_init"] = (a, kw)

        def write_gpx(self, notes, output_path):
            recorder["write_gpx"] = (len(notes), output_path)
            Path(output_path).write_bytes(b"fake-gpx")
            return output_path

    monkeypatch.setattr(pipeline_mod, "BasicPitchTranscriber", StubTranscriber)
    monkeypatch.setattr(pipeline_mod, "TARTTechniqueClassifier", StubAudioClassifier)
    monkeypatch.setattr(pipeline_mod, "FretboardDetector", StubFretboardDetector)
    monkeypatch.setattr(pipeline_mod, "HandTracker", StubHandTracker)
    monkeypatch.setattr(pipeline_mod, "FretEstimator", StubFretEstimator)
    monkeypatch.setattr(pipeline_mod, "VisionTechniqueClassifier", StubVisionClassifier)
    monkeypatch.setattr(pipeline_mod, "LateFusion", StubLateFusion)
    monkeypatch.setattr(pipeline_mod, "TabWriter", StubTabWriter)


def test_pipeline_run_orchestrates_all_stages_in_order(workdir, tmp_path, monkeypatch):
    recorder: dict = {}
    _stub_pipeline_dependencies(monkeypatch, recorder)

    output = tmp_path / "out.gpx"
    pipeline = Pipeline(workdir=workdir)
    result = pipeline.run("https://example.com/clip", output)

    assert result == output
    assert output.read_bytes() == b"fake-gpx"

    # 모든 단계가 정확한 입력으로 호출되었는지 확인
    assert recorder["download"] == ("https://example.com/clip", workdir)
    assert recorder["split"][0] == workdir / "video.mp4"
    assert recorder["stem"][0] == workdir / "audio.wav"
    assert recorder["transcribe"] == workdir / "guitar.wav"
    assert recorder["audio_classify"] == (1, workdir / "guitar.wav")
    assert recorder["fretboard_detect"] == workdir / "video-only.mp4"
    assert recorder["hands_track"] == workdir / "video-only.mp4"
    assert recorder["fret_estimate"] == (1, 1)
    assert recorder["vision_classify"] == 1
    assert recorder["fuse"] == (1, 0, 1, 0)
    assert recorder["write_gpx"] == (1, output)


def test_pipeline_creates_workdir_if_missing(tmp_path, monkeypatch):
    recorder: dict = {}
    _stub_pipeline_dependencies(monkeypatch, recorder)

    workdir = tmp_path / "deep" / "nested" / "work"
    assert not workdir.exists()

    Pipeline(workdir=workdir).run("input.mp4", tmp_path / "out.gpx")

    assert workdir.is_dir()


def test_pipeline_forwards_weights_to_classifiers(workdir, tmp_path, monkeypatch):
    recorder: dict = {}
    _stub_pipeline_dependencies(monkeypatch, recorder)

    audio_w = tmp_path / "tart.pt"
    vision_w = tmp_path / "tcn.pt"
    fret_w = tmp_path / "yolo.pt"
    hands_asset = tmp_path / "hands.task"

    Pipeline(
        workdir=workdir,
        audio_weights=audio_w,
        vision_weights=vision_w,
        fretboard_weights=fret_w,
        hands_model_asset=hands_asset,
    ).run("input.mp4", tmp_path / "out.gpx")

    assert recorder["audio_classifier_init"]["weights_path"] == audio_w
    assert recorder["vision_classifier_init"]["weights_path"] == vision_w
    assert recorder["fretboard_init"]["weights_path"] == fret_w
    assert recorder["hands_init"]["model_asset_path"] == hands_asset


def test_pipeline_records_intermediate_paths(workdir, tmp_path, monkeypatch):
    recorder: dict = {}
    _stub_pipeline_dependencies(monkeypatch, recorder)

    pipeline = Pipeline(workdir=workdir)
    pipeline.run("input.mp4", tmp_path / "out.gpx")

    paths = pipeline._intermediate_paths
    assert paths["video"] == workdir / "video.mp4"
    assert paths["audio_wav"] == workdir / "audio.wav"
    assert paths["video_only"] == workdir / "video-only.mp4"
    assert paths["guitar_wav"] == workdir / "guitar.wav"
