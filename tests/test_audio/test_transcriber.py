"""Unit tests for BasicPitchTranscriber (mock-based)."""
from __future__ import annotations

from pathlib import Path

import pytest

from guitarvideo2tab.audio.transcriber import BasicPitchTranscriber
from guitarvideo2tab.models import MidiEvent, PitchContour


def _make_fake_predict(note_events, captured: dict | None = None):
    def fake_predict(audio_path, **kwargs):
        if captured is not None:
            captured["audio_path"] = audio_path
            captured["kwargs"] = kwargs
        return None, None, list(note_events)

    return fake_predict


def test_transcribe_preserves_pitch_contour_with_bend(monkeypatch):
    bends = [0.0, 0.25, 0.5, 0.25]
    note_events = [(1.0, 1.5, 64, 0.8, bends)]
    monkeypatch.setattr(
        "guitarvideo2tab.audio.transcriber.predict",
        _make_fake_predict(note_events),
    )

    events = BasicPitchTranscriber().transcribe(Path("dummy.wav"))

    assert len(events) == 1
    evt = events[0]
    assert isinstance(evt, MidiEvent)
    assert evt.pitch == 64
    assert evt.start_time == pytest.approx(1.0)
    assert evt.end_time == pytest.approx(1.5)
    assert evt.pitch_contour is not None
    assert isinstance(evt.pitch_contour, PitchContour)
    assert len(evt.pitch_contour.time_pitch_curve) == len(bends)
    assert evt.pitch_contour.bend_semitones == pytest.approx(0.5)
    # First sample should match start_time, pitch + bend[0]
    t0, p0 = evt.pitch_contour.time_pitch_curve[0]
    assert t0 == pytest.approx(1.0)
    assert p0 == pytest.approx(64.0)


def test_transcribe_returns_none_contour_when_no_bend(monkeypatch):
    note_events = [
        (0.0, 0.5, 60, 0.5, []),
        (1.0, 1.5, 62, 0.5, None),
    ]
    monkeypatch.setattr(
        "guitarvideo2tab.audio.transcriber.predict",
        _make_fake_predict(note_events),
    )

    events = BasicPitchTranscriber().transcribe(Path("dummy.wav"))

    assert len(events) == 2
    assert events[0].pitch_contour is None
    assert events[1].pitch_contour is None


def test_transcribe_clamps_velocity(monkeypatch):
    note_events = [
        (0.0, 0.1, 60, 2.0, []),
        (0.2, 0.3, 61, -0.1, []),
        (0.4, 0.5, 62, 0.5, []),
    ]
    monkeypatch.setattr(
        "guitarvideo2tab.audio.transcriber.predict",
        _make_fake_predict(note_events),
    )

    events = BasicPitchTranscriber().transcribe(Path("dummy.wav"))

    assert events[0].velocity == 127
    assert events[1].velocity == 0
    assert events[2].velocity == round(0.5 * 127)


def test_transcribe_passes_instance_thresholds(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "guitarvideo2tab.audio.transcriber.predict",
        _make_fake_predict([], captured=captured),
    )

    transcriber = BasicPitchTranscriber(onset_threshold=0.4, frame_threshold=0.2)
    transcriber.transcribe(Path("foo.wav"))

    assert captured["kwargs"]["onset_threshold"] == 0.4
    assert captured["kwargs"]["frame_threshold"] == 0.2
    assert captured["kwargs"]["multiple_pitch_bends"] is True
    assert captured["audio_path"] == "foo.wav"


def test_transcribe_zero_pads_note_id(monkeypatch):
    note_events = [
        (0.0, 0.1, 60, 0.5, [0.1]),
        (0.2, 0.3, 61, 0.5, [0.2]),
        (0.4, 0.5, 62, 0.5, [0.3]),
    ]
    monkeypatch.setattr(
        "guitarvideo2tab.audio.transcriber.predict",
        _make_fake_predict(note_events),
    )

    events = BasicPitchTranscriber().transcribe(Path("dummy.wav"))

    assert events[0].pitch_contour.note_id == "000000"
    assert events[1].pitch_contour.note_id == "000001"
    assert events[2].pitch_contour.note_id == "000002"
