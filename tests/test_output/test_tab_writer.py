"""Tests for TabWriter (NoteEvent → Guitar Pro file)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import guitarpro

from guitarvideo2tab.models import (
    MidiEvent,
    NoteEvent,
    PitchContour,
    TechniqueAnnotation,
)
from guitarvideo2tab.output.tab_writer import TabWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _midi(pitch: int = 60) -> MidiEvent:
    return MidiEvent(pitch=pitch, start_time=0.0, end_time=0.5, velocity=80)


def _note(
    string: int,
    fret: int,
    technique: TechniqueAnnotation | None = None,
    pitch_contour: PitchContour | None = None,
) -> NoteEvent:
    midi = _midi()
    if pitch_contour is not None:
        midi = MidiEvent(
            pitch=60,
            start_time=0.0,
            end_time=0.5,
            velocity=80,
            pitch_contour=pitch_contour,
        )
    return NoteEvent(midi_event=midi, string=string, fret=fret, technique=technique)


# ---------------------------------------------------------------------------
# Test 1: empty list → Song with standard guitar track is written
# ---------------------------------------------------------------------------

def test_empty_notes_writes_song_with_standard_track(tmp_path: Path) -> None:
    output = tmp_path / "out.gp5"
    writer = TabWriter()

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write") as mock_write:
        result = writer.write_gp5([], output)

    mock_write.assert_called_once()
    song_arg, path_arg = mock_write.call_args.args
    assert isinstance(song_arg, guitarpro.Song)
    assert path_arg == str(output)
    assert result == output

    # One track with EADGBE tuning strings
    assert len(song_arg.tracks) == 1
    track = song_arg.tracks[0]
    assert len(track.strings) == 6
    # Standard EADGBE MIDI values: 40, 45, 50, 55, 59, 64
    midi_values = [s.value for s in track.strings]
    assert midi_values == list(writer.tuning)


# ---------------------------------------------------------------------------
# Test 2: 3 notes (different strings/frets) → 3 Notes in the Song's track
# ---------------------------------------------------------------------------

def test_three_notes_create_three_beats(tmp_path: Path) -> None:
    notes = [
        _note(string=1, fret=0),
        _note(string=2, fret=5),
        _note(string=3, fret=7),
    ]
    writer = TabWriter()
    output = tmp_path / "out.gpx"

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write") as mock_write:
        result = writer.write_gpx(notes, output)

    assert result == output
    song_arg, _ = mock_write.call_args.args

    voice = song_arg.tracks[0].measures[0].voices[0]
    all_notes = [n for beat in voice.beats for n in beat.notes]
    assert len(all_notes) == 3

    # Verify fret values preserved
    fret_values = [n.value for n in all_notes]
    assert fret_values == [0, 5, 7]

    # Verify string values preserved
    string_values = [n.string for n in all_notes]
    assert string_values == [1, 2, 3]


# ---------------------------------------------------------------------------
# Test 3: one note with bend TechniqueAnnotation + pitch_contour → effect.bend set
# ---------------------------------------------------------------------------

def test_bend_technique_with_pitch_contour_sets_bend_effect(tmp_path: Path) -> None:
    contour = PitchContour(
        note_id="n0",
        time_pitch_curve=[(0.0, 0.0), (0.1, 0.5), (0.2, 1.0), (0.3, 1.0)],
        bend_semitones=1.0,
    )
    ann = TechniqueAnnotation(technique="bend", confidence=0.9, source="audio")
    note_event = _note(string=1, fret=7, technique=ann, pitch_contour=contour)

    writer = TabWriter()
    output = tmp_path / "out.gp5"

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write") as mock_write:
        writer.write_gp5([note_event], output)

    song_arg, _ = mock_write.call_args.args
    voice = song_arg.tracks[0].measures[0].voices[0]
    gp_note = voice.beats[0].notes[0]

    assert gp_note.effect.bend is not None
    assert len(gp_note.effect.bend.points) == 4  # one per curve point


# ---------------------------------------------------------------------------
# Test 4: NoteEvent with string=-1 is skipped
# ---------------------------------------------------------------------------

def test_unresolved_string_minus1_is_skipped(tmp_path: Path) -> None:
    notes = [
        _note(string=-1, fret=5),   # should be skipped
        _note(string=1, fret=3),    # should be included
        _note(string=2, fret=-1),   # should be skipped
    ]
    writer = TabWriter()
    output = tmp_path / "out.gp5"

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write") as mock_write:
        result = writer.write_gp5(notes, output)

    assert result == output
    song_arg, _ = mock_write.call_args.args
    voice = song_arg.tracks[0].measures[0].voices[0]
    all_notes = [n for beat in voice.beats for n in beat.notes]
    assert len(all_notes) == 1
    assert all_notes[0].value == 3
    assert all_notes[0].string == 1


# ---------------------------------------------------------------------------
# Test 5: returned path equals input path (both methods)
# ---------------------------------------------------------------------------

def test_returned_path_equals_input_path(tmp_path: Path) -> None:
    output_gpx = tmp_path / "song.gpx"
    output_gp5 = tmp_path / "song.gp5"
    writer = TabWriter()

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write"):
        assert writer.write_gpx([], output_gpx) == output_gpx
        assert writer.write_gp5([], output_gp5) == output_gp5
