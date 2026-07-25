"""Tests for rhythm quantization (tempo → grid → measures)."""
from __future__ import annotations

from guitarvideo2tab.models import MidiEvent, NoteEvent
from guitarvideo2tab.output.rhythm import (
    QUARTER_TICKS,
    RhythmGrid,
    build_grid,
    duration_span,
    estimate_tempo_from_events,
    quantize_notes,
    ticks_to_duration,
)

# 120 BPM / 4-4 기준 상수
#   4분음표 = 0.5초 = 960틱, 한 마디 = 2.0초 = 3840틱, 16분 격자 = 240틱
_MEASURE_TICKS = 3840
_QUARTER = 960


def _note(start: float, end: float, pitch: int = 60, string: int = 1, fret: int = 0) -> NoteEvent:
    return NoteEvent(
        midi_event=MidiEvent(pitch=pitch, start_time=start, end_time=end, velocity=80),
        string=string,
        fret=fret,
    )


def _grid(**kwargs) -> RhythmGrid:
    return RhythmGrid(tempo_bpm=120.0, **kwargs)


# ---------------------------------------------------------------------------
# RhythmGrid 기본 산술
# ---------------------------------------------------------------------------


def test_grid_derived_values_at_120bpm() -> None:
    grid = _grid()
    assert grid.ticks_per_second == 1920.0
    assert grid.grid_ticks == 240        # 16분음표
    assert grid.measure_ticks == _MEASURE_TICKS
    assert QUARTER_TICKS == 960


def test_grid_measure_ticks_respects_time_signature() -> None:
    assert _grid(numerator=3, denominator=4).measure_ticks == 3 * 960   # 3/4
    assert _grid(numerator=6, denominator=8).measure_ticks == 6 * 480   # 6/8


def test_to_ticks_snaps_to_grid() -> None:
    grid = _grid()
    assert grid.to_ticks(0.0) == 0
    assert grid.to_ticks(0.5) == _QUARTER          # 정확히 4분음표
    assert grid.to_ticks(0.51) == _QUARTER         # 살짝 늦어도 같은 칸으로 스냅
    assert grid.to_ticks(0.49) == _QUARTER         # 살짝 빨라도 마찬가지
    assert grid.to_ticks(2.0) == _MEASURE_TICKS    # 두 번째 마디 첫 박


def test_to_ticks_never_negative() -> None:
    grid = _grid(origin_sec=1.0)
    assert grid.to_ticks(0.0) == 0


def test_to_ticks_honours_origin_offset() -> None:
    grid = _grid(origin_sec=1.0)
    assert grid.to_ticks(1.0) == 0
    assert grid.to_ticks(1.5) == _QUARTER


# ---------------------------------------------------------------------------
# 틱 → Guitar Pro Duration
# ---------------------------------------------------------------------------


def test_ticks_to_duration_exact_values() -> None:
    assert ticks_to_duration(3840) == (1, False)    # 온음표
    assert ticks_to_duration(1920) == (2, False)    # 2분음표
    assert ticks_to_duration(960) == (4, False)     # 4분음표
    assert ticks_to_duration(480) == (8, False)     # 8분음표
    assert ticks_to_duration(240) == (16, False)    # 16분음표


def test_ticks_to_duration_dotted_values() -> None:
    assert ticks_to_duration(1440) == (4, True)     # 점4분음표
    assert ticks_to_duration(720) == (8, True)      # 점8분음표
    assert ticks_to_duration(2880) == (2, True)     # 점2분음표


def test_ticks_to_duration_rounds_down_when_unrepresentable() -> None:
    # 1200틱은 단일 Duration 으로 표현 불가 → 960(4분음표)로 내림
    assert ticks_to_duration(1200) == (4, False)
    assert duration_span(1200) == 960


def test_ticks_to_duration_floors_at_minimum() -> None:
    assert ticks_to_duration(1) == (64, False)
    assert duration_span(1) == 60


# ---------------------------------------------------------------------------
# Tempo 추정
# ---------------------------------------------------------------------------


def test_estimate_tempo_reads_quarter_notes_as_120bpm() -> None:
    events = [MidiEvent(60, t, t + 0.4, 80) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert estimate_tempo_from_events(events) == 120.0


def test_estimate_tempo_folds_eighth_notes_up() -> None:
    # 0.25초 간격을 4분음표로 보면 240BPM → 범위 밖이므로 8분음표로 해석해 120BPM
    events = [MidiEvent(60, t * 0.25, t * 0.25 + 0.2, 80) for t in range(8)]
    assert estimate_tempo_from_events(events) == 120.0


def test_estimate_tempo_defaults_when_no_intervals() -> None:
    assert estimate_tempo_from_events([]) == 120.0
    assert estimate_tempo_from_events([MidiEvent(60, 0.0, 0.5, 80)]) == 120.0


def test_build_grid_honours_explicit_tempo() -> None:
    grid = build_grid([], tempo_bpm=90.0)
    assert grid.tempo_bpm == 90.0
    assert grid.origin_sec == 0.0


# ---------------------------------------------------------------------------
# 양자화 · 화음 그룹핑
# ---------------------------------------------------------------------------


def test_simultaneous_notes_become_one_chord_beat() -> None:
    """같은 시각의 3개 음은 3개 beat 가 아니라 1개 화음 beat 여야 한다."""
    notes = [
        _note(0.0, 0.5, string=1, fret=0),
        _note(0.0, 0.5, string=2, fret=2),
        _note(0.0, 0.5, string=3, fret=2),
    ]
    beats = quantize_notes(notes, _grid())

    sounding = [b for b in beats if not b.is_rest]
    assert len(sounding) == 1
    assert len(sounding[0].notes) == 3
    assert [n.string for n in sounding[0].notes] == [1, 2, 3]


def test_sequential_notes_become_separate_beats() -> None:
    notes = [_note(t, t + 0.5) for t in (0.0, 0.5, 1.0, 1.5)]
    beats = quantize_notes(notes, _grid())

    sounding = [b for b in beats if not b.is_rest]
    assert len(sounding) == 4
    assert [b.offset_ticks for b in sounding] == [0, 960, 1920, 2880]
    assert all(b.duration_ticks == _QUARTER for b in sounding)
    assert all(b.measure_index == 0 for b in sounding)


def test_notes_are_split_across_measures() -> None:
    # 0.0초=1마디 첫 박, 2.0초=2마디 첫 박
    notes = [_note(0.0, 0.5), _note(2.0, 2.5)]
    beats = quantize_notes(notes, _grid())

    sounding = [b for b in beats if not b.is_rest]
    assert [b.measure_index for b in sounding] == [0, 1]
    assert [b.offset_ticks for b in sounding] == [0, 0]


def test_gaps_are_filled_with_rests() -> None:
    """한 마디에 4분음표 하나만 있으면 나머지 3박은 쉼표로 채워진다."""
    beats = quantize_notes([_note(0.0, 0.5)], _grid())

    assert len(beats) == 2
    assert not beats[0].is_rest
    assert beats[1].is_rest
    assert beats[1].offset_ticks == 960
    assert beats[1].duration_ticks == 2880  # 점2분쉼표


def test_leading_silence_is_filled_with_rest() -> None:
    """첫 음이 2박째에 나오면 앞에 쉼표가 들어간다."""
    beats = quantize_notes([_note(0.5, 1.0)], _grid())

    assert beats[0].is_rest
    assert beats[0].offset_ticks == 0
    assert beats[0].duration_ticks == 960
    assert not beats[1].is_rest
    assert beats[1].offset_ticks == 960


def test_every_measure_sums_to_full_measure() -> None:
    """모든 마디의 beat 길이 합이 정확히 한 마디여야 한다 (악보 유효성의 핵심)."""
    notes = [_note(t, t + 0.3) for t in (0.0, 0.75, 1.5, 2.25, 3.1, 5.0)]
    beats = quantize_notes(notes, _grid())

    totals: dict[int, int] = {}
    for beat in beats:
        totals[beat.measure_index] = totals.get(beat.measure_index, 0) + beat.duration_ticks

    assert totals, "최소 한 마디는 생성되어야 한다"
    for measure_index, total in totals.items():
        assert total == _MEASURE_TICKS, f"마디 {measure_index} 길이 불일치: {total}"


def test_empty_measure_is_filled_with_rests() -> None:
    """음이 없는 중간 마디도 쉼표로 채워져 마디 번호가 연속해야 한다."""
    notes = [_note(0.0, 0.5), _note(4.0, 4.5)]  # 1마디와 3마디 (2마디는 비어 있음)
    beats = quantize_notes(notes, _grid())

    measures = {b.measure_index for b in beats}
    assert measures == {0, 1, 2}
    assert all(b.is_rest for b in beats if b.measure_index == 1)


def test_note_duration_is_clipped_at_measure_boundary() -> None:
    """마디 끝을 넘는 긴 음은 마디 경계에서 잘린다."""
    # 마지막 박(offset 2880)에서 시작하는 2초짜리 음 → 960틱으로 잘림
    beats = quantize_notes([_note(1.5, 3.5)], _grid())

    sounding = [b for b in beats if not b.is_rest]
    assert sounding[0].offset_ticks == 2880
    assert sounding[0].duration_ticks == 960
    assert sounding[0].measure_index == 0


def test_note_duration_is_limited_by_next_onset() -> None:
    """음이 길게 울려도 다음 음이 빨리 나오면 그 간격까지만 차지한다."""
    # 첫 음은 1초(2박) 울리지만 0.5초 뒤 다음 음이 나옴 → 4분음표로 축소
    beats = quantize_notes([_note(0.0, 1.0), _note(0.5, 1.0)], _grid())

    sounding = [b for b in beats if not b.is_rest]
    assert sounding[0].duration_ticks == _QUARTER


def test_empty_input_returns_no_beats() -> None:
    assert quantize_notes([], _grid()) == []
