"""양자화된 리듬을 표준 MIDI 파일로 출력한다.

``rhythm.py`` 가 만든 틱 격자를 그대로 쓴다. ``QUARTER_TICKS`` 를 MIDI 의
ticks_per_beat 으로 사용하므로 악보(MusicXML)와 MIDI 의 리듬이 정확히 일치한다.

원본(양자화 전) Basic Pitch 이벤트는 ``stages/*_midi_events.json`` 에 남으므로,
여기서 나오는 ``.mid`` 는 "악보와 같은 리듬"을 듣기 위한 것이다.
"""
from __future__ import annotations

from pathlib import Path

import mido

from ..models import NoteEvent
from .rhythm import QUARTER_TICKS, QuantizedBeat, RhythmGrid, quantize_notes

DEFAULT_PROGRAM = 25  # General MIDI: Acoustic Guitar (steel)
DEFAULT_VELOCITY = 80


def _beat_events(
    beats: list[QuantizedBeat], measure_ticks: int
) -> list[tuple[int, int, int, int]]:
    """``(start_tick, end_tick, pitch, velocity)`` 목록을 절대 틱으로 만든다."""
    events: list[tuple[int, int, int, int]] = []
    for beat in beats:
        if beat.is_rest:
            continue
        start = beat.measure_index * measure_ticks + beat.offset_ticks
        end = start + beat.duration_ticks
        for note in beat.notes:
            velocity = max(1, min(127, note.midi_event.velocity or DEFAULT_VELOCITY))
            events.append((start, end, note.midi_event.pitch, velocity))
    return events


def write_midi(
    notes: list[NoteEvent],
    output_path: Path,
    grid: RhythmGrid,
    program: int = DEFAULT_PROGRAM,
    track_name: str = "Guitar",
) -> Path:
    """NoteEvent 목록을 양자화하여 ``.mid`` 로 쓴다."""
    beats = quantize_notes(notes, grid)

    midi = mido.MidiFile(ticks_per_beat=QUARTER_TICKS)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(grid.tempo_bpm), time=0)
    )
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=grid.numerator,
            denominator=grid.denominator,
            time=0,
        )
    )
    track.append(mido.Message("program_change", program=program, time=0))

    # note_on/note_off 를 절대 틱으로 모은 뒤 델타로 변환한다.
    timeline: list[tuple[int, int, int, int]] = []
    for start, end, pitch, velocity in _beat_events(beats, grid.measure_ticks):
        timeline.append((start, 1, pitch, velocity))  # note_on
        timeline.append((end, 0, pitch, 0))           # note_off
    # 같은 틱에서는 note_off 를 먼저 처리해야 같은 음 재타건이 끊기지 않는다.
    timeline.sort(key=lambda e: (e[0], e[1], e[2]))

    cursor = 0
    for tick, kind, pitch, velocity in timeline:
        delta = tick - cursor
        cursor = tick
        track.append(
            mido.Message(
                "note_on" if kind else "note_off",
                note=int(pitch),
                velocity=int(velocity),
                time=int(delta),
            )
        )

    track.append(mido.MetaMessage("end_of_track", time=0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(output_path))
    return output_path
