"""Tests for guitarvideo2tab.models dataclasses."""
from __future__ import annotations

import pytest

from guitarvideo2tab.models import (
    FretboardFrame,
    FretPosition,
    HandKeypoints,
    MidiEvent,
    NoteEvent,
    PitchContour,
    TechniqueAnnotation,
)


class TestPitchContour:
    def test_required_fields_only(self):
        contour = PitchContour(note_id="n1", time_pitch_curve=[(0.0, 60.0), (0.1, 60.5)])
        assert contour.note_id == "n1"
        assert contour.time_pitch_curve == [(0.0, 60.0), (0.1, 60.5)]
        assert contour.bend_semitones == 0.0

    def test_bend_semitones_override(self):
        contour = PitchContour(
            note_id="n2",
            time_pitch_curve=[(0.0, 64.0)],
            bend_semitones=1.5,
        )
        assert contour.bend_semitones == 1.5


class TestMidiEvent:
    def test_required_fields_default_pitch_contour_none(self):
        event = MidiEvent(pitch=60, start_time=0.0, end_time=0.5, velocity=80)
        assert event.pitch == 60
        assert event.start_time == 0.0
        assert event.end_time == 0.5
        assert event.velocity == 80
        assert event.pitch_contour is None

    def test_pitch_contour_injection_preserved(self):
        contour = PitchContour(note_id="n1", time_pitch_curve=[(0.0, 60.0)])
        event = MidiEvent(
            pitch=60,
            start_time=0.0,
            end_time=0.5,
            velocity=80,
            pitch_contour=contour,
        )
        assert event.pitch_contour is contour


class TestTechniqueAnnotation:
    @pytest.mark.parametrize(
        "label",
        [
            "bend",
            "slide",
            "hammer-on",
            "pull-off",
            "vibrato",
            "palm-mute",
            "tapping",
            "sweep-picking",
            "alternate-picking",
            "legato",
        ],
    )
    def test_all_technique_labels_accepted(self, label):
        annotation = TechniqueAnnotation(technique=label, confidence=0.9, source="audio")
        assert annotation.technique == label

    @pytest.mark.parametrize("source", ["audio", "vision", "fusion"])
    def test_all_modality_sources_accepted(self, source):
        annotation = TechniqueAnnotation(technique="bend", confidence=0.5, source=source)
        assert annotation.source == source

    def test_params_default_factory_isolates_instances(self):
        a = TechniqueAnnotation(technique="bend", confidence=0.9, source="audio")
        b = TechniqueAnnotation(technique="slide", confidence=0.8, source="vision")
        a.params["k"] = 1
        assert a.params is not b.params
        assert b.params == {}


class TestHandKeypoints:
    def test_none_hands_allowed(self):
        kp = HandKeypoints(timestamp=0.5, left_hand=None, right_hand=None)
        assert kp.left_hand is None
        assert kp.right_hand is None

    def test_21_keypoints_per_hand(self):
        left = [(float(i), float(i)) for i in range(21)]
        right = [(float(i) + 1, float(i) + 1) for i in range(21)]
        kp = HandKeypoints(timestamp=1.0, left_hand=left, right_hand=right)
        assert len(kp.left_hand) == 21
        assert len(kp.right_hand) == 21


class TestFretboardFrame:
    def test_visible_default_true(self):
        frame = FretboardFrame(timestamp=0.0, homography=None, corners=None)
        assert frame.visible is True

    def test_homography_and_corners_payload(self):
        homography = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        frame = FretboardFrame(
            timestamp=2.0,
            homography=homography,
            corners=corners,
            visible=False,
        )
        assert frame.homography == homography
        assert frame.corners == corners
        assert frame.visible is False


class TestFretPosition:
    def test_all_fields_required(self):
        pos = FretPosition(timestamp=0.0, string=2, fret=5, confidence=0.95)
        assert pos.timestamp == 0.0
        assert pos.string == 2
        assert pos.fret == 5
        assert pos.confidence == 0.95

    def test_equality(self):
        a = FretPosition(timestamp=0.0, string=2, fret=5, confidence=0.95)
        b = FretPosition(timestamp=0.0, string=2, fret=5, confidence=0.95)
        c = FretPosition(timestamp=0.1, string=2, fret=5, confidence=0.95)
        assert a == b
        assert a != c


class TestNoteEvent:
    def test_default_technique_none(self):
        midi = MidiEvent(pitch=60, start_time=0.0, end_time=0.5, velocity=80)
        note = NoteEvent(midi_event=midi, string=1, fret=3)
        assert note.midi_event is midi
        assert note.string == 1
        assert note.fret == 3
        assert note.technique is None

    def test_composes_midi_and_technique(self):
        midi = MidiEvent(pitch=64, start_time=0.0, end_time=0.5, velocity=90)
        annotation = TechniqueAnnotation(technique="bend", confidence=0.9, source="fusion")
        note = NoteEvent(midi_event=midi, string=2, fret=7, technique=annotation)
        assert note.technique is annotation
        assert note.midi_event.pitch == 64
