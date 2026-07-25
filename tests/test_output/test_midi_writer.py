"""MIDI 출력: 양자화된 틱이 그대로 반영되는지 실제 파일로 검증한다."""
from __future__ import annotations

from pathlib import Path

import mido

from guitarvideo2tab.models import MidiEvent, NoteEvent
from guitarvideo2tab.output.midi_writer import write_midi
from guitarvideo2tab.output.rhythm import QUARTER_TICKS, RhythmGrid


def _note(start: float, end: float, pitch: int = 60, string: int = 1, fret: int = 0,
          velocity: int = 80) -> NoteEvent:
    return NoteEvent(
        midi_event=MidiEvent(pitch=pitch, start_time=start, end_time=end, velocity=velocity),
        string=string,
        fret=fret,
    )


def _grid(**kwargs) -> RhythmGrid:
    return RhythmGrid(tempo_bpm=120.0, **kwargs)


def test_written_midi_can_be_reopened(tmp_path: Path) -> None:
    out = write_midi([_note(0.0, 0.5)], tmp_path / "tab.mid", _grid())

    assert out.exists()
    midi = mido.MidiFile(str(out))
    assert midi.ticks_per_beat == QUARTER_TICKS


def test_tempo_and_time_signature_are_written(tmp_path: Path) -> None:
    out = write_midi([_note(0.0, 0.5)], tmp_path / "tab.mid", _grid(numerator=3, denominator=4))
    midi = mido.MidiFile(str(out))

    tempos = [m for t in midi.tracks for m in t if m.type == "set_tempo"]
    signatures = [m for t in midi.tracks for m in t if m.type == "time_signature"]
    assert mido.tempo2bpm(tempos[0].tempo) == 120.0
    assert (signatures[0].numerator, signatures[0].denominator) == (3, 4)


def test_note_lands_on_quantized_tick(tmp_path: Path) -> None:
    # 120BPM 에서 두 번째 박은 정확히 960틱
    out = write_midi([_note(0.5, 1.0, pitch=64)], tmp_path / "tab.mid", _grid())
    midi = mido.MidiFile(str(out))

    absolute, cursor = [], 0
    for message in midi.tracks[0]:
        cursor += message.time
        if message.type == "note_on":
            absolute.append((cursor, message.note))

    assert absolute == [(QUARTER_TICKS, 64)]


def test_note_duration_matches_quantized_length(tmp_path: Path) -> None:
    out = write_midi([_note(0.0, 0.5, pitch=60)], tmp_path / "tab.mid", _grid())
    midi = mido.MidiFile(str(out))

    cursor, on_tick, off_tick = 0, None, None
    for message in midi.tracks[0]:
        cursor += message.time
        if message.type == "note_on":
            on_tick = cursor
        elif message.type == "note_off":
            off_tick = cursor
    assert off_tick - on_tick == QUARTER_TICKS


def test_chord_notes_share_the_same_onset(tmp_path: Path) -> None:
    notes = [
        _note(0.0, 0.5, pitch=52, string=1, fret=0),
        _note(0.0, 0.5, pitch=57, string=2, fret=0),
        _note(0.0, 0.5, pitch=62, string=3, fret=0),
    ]
    out = write_midi(notes, tmp_path / "tab.mid", _grid())
    midi = mido.MidiFile(str(out))

    cursor, onsets = 0, []
    for message in midi.tracks[0]:
        cursor += message.time
        if message.type == "note_on":
            onsets.append((cursor, message.note))

    assert len(onsets) == 3
    assert {tick for tick, _ in onsets} == {0}
    assert sorted(pitch for _, pitch in onsets) == [52, 57, 62]


def test_rests_produce_no_notes(tmp_path: Path) -> None:
    out = write_midi([], tmp_path / "tab.mid", _grid())
    midi = mido.MidiFile(str(out))

    assert [m for t in midi.tracks for m in t if m.type == "note_on"] == []


def test_velocity_is_clamped_into_midi_range(tmp_path: Path) -> None:
    out = write_midi([_note(0.0, 0.5, velocity=999)], tmp_path / "tab.mid", _grid())
    midi = mido.MidiFile(str(out))

    velocities = [m.velocity for t in midi.tracks for m in t if m.type == "note_on"]
    assert velocities and all(1 <= v <= 127 for v in velocities)
