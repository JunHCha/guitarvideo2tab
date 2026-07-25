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

def _note(
    string: int,
    fret: int,
    technique: TechniqueAnnotation | None = None,
    pitch_contour: PitchContour | None = None,
    start: float = 0.0,
    end: float = 0.5,
) -> NoteEvent:
    midi = MidiEvent(
        pitch=60,
        start_time=start,
        end_time=end,
        velocity=80,
        pitch_contour=pitch_contour,
    )
    return NoteEvent(midi_event=midi, string=string, fret=fret, technique=technique)


def _sounding_beats(song: guitarpro.Song) -> list[guitarpro.Beat]:
    """쉼표를 제외한 실제 발음 beat 만 모든 마디에서 모은다."""
    return [
        beat
        for measure in song.tracks[0].measures
        for beat in measure.voices[0].beats
        if beat.notes
    ]


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
# Test 2: 동시 발음 3음 → 하나의 화음 Beat (3개의 연속 Beat 가 아님)
# ---------------------------------------------------------------------------

def test_simultaneous_notes_become_single_chord_beat(tmp_path: Path) -> None:
    notes = [
        _note(string=1, fret=0),
        _note(string=2, fret=5),
        _note(string=3, fret=7),
    ]
    writer = TabWriter(tempo_bpm=120.0)
    output = tmp_path / "out.gpx"

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write") as mock_write:
        result = writer.write_gpx(notes, output)

    assert result == output
    song_arg, _ = mock_write.call_args.args

    sounding = _sounding_beats(song_arg)
    assert len(sounding) == 1, "동시 발음은 하나의 화음 Beat 여야 한다"
    assert [n.value for n in sounding[0].notes] == [0, 5, 7]
    assert [n.string for n in sounding[0].notes] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Test 2b: 시간차를 둔 음들은 각각의 Beat 이며 마디를 넘어 배치된다
# ---------------------------------------------------------------------------

def test_sequential_notes_span_multiple_measures(tmp_path: Path) -> None:
    # 120BPM 4/4 → 한 마디 2초. 0.0/0.5초는 1마디, 2.0초는 2마디.
    notes = [
        _note(string=1, fret=1, start=0.0, end=0.5),
        _note(string=2, fret=2, start=0.5, end=1.0),
        _note(string=3, fret=3, start=2.0, end=2.5),
    ]
    writer = TabWriter(tempo_bpm=120.0)
    output = tmp_path / "out.gp5"

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write") as mock_write:
        writer.write_gp5(notes, output)

    song_arg, _ = mock_write.call_args.args

    assert len(song_arg.tracks[0].measures) == 2
    assert len(song_arg.measureHeaders) == 2

    sounding = _sounding_beats(song_arg)
    assert len(sounding) == 3
    assert [n.value for beat in sounding for n in beat.notes] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Test 2c: 각 마디의 beat 길이 합이 정확히 한 마디여야 한다
# ---------------------------------------------------------------------------

def test_each_measure_is_rhythmically_complete(tmp_path: Path) -> None:
    notes = [
        _note(string=1, fret=1, start=0.0, end=0.3),
        _note(string=2, fret=2, start=1.25, end=1.5),
        _note(string=3, fret=3, start=3.0, end=3.5),
    ]
    writer = TabWriter(tempo_bpm=120.0)
    output = tmp_path / "out.gp5"

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write") as mock_write:
        writer.write_gp5(notes, output)

    song_arg, _ = mock_write.call_args.args

    for measure in song_arg.tracks[0].measures:
        total = sum(beat.duration.time for beat in measure.voices[0].beats)
        assert total == measure.header.length, (
            f"마디 {measure.header.number} 길이 불일치: {total} != {measure.header.length}"
        )


# ---------------------------------------------------------------------------
# Test 2d: tempo 가 Song 에 반영된다
# ---------------------------------------------------------------------------

def test_tempo_is_written_to_song(tmp_path: Path) -> None:
    writer = TabWriter(tempo_bpm=96.0)

    with patch("guitarvideo2tab.output.tab_writer.guitarpro.write") as mock_write:
        writer.write_gp5([_note(string=1, fret=0)], tmp_path / "out.gp5")

    song_arg, _ = mock_write.call_args.args
    assert song_arg.tempo == 96


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
