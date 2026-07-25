"""리듬 해석: tempo 추정 → 양자화 → 마디 분할.

Basic Pitch 가 만든 ``MidiEvent`` 의 절대 시각(초)을 Guitar Pro 의 틱 격자로
옮기는 단계. 이 모듈이 없으면 모든 음이 한 마디에 4분음표로 쌓여서 악보를
읽을 수 없다.

처리 순서::

    1. estimate_tempo   — 오디오(librosa) 또는 onset IOI 로 BPM 추정
    2. quantize         — 절대 시각(초) → 격자에 스냅된 절대 틱
    3. 화음 그룹핑        — 같은 틱에 떨어진 음들을 하나의 Beat 로 묶음
    4. 길이 결정          — 다음 onset 까지의 간격과 실제 발음 길이 중 짧은 쪽
    5. 마디 분할          — measure_ticks 로 나눠 마디 인덱스/오프셋 계산
    6. 쉼표 채우기        — 마디 내 빈 구간과 빈 마디를 쉼표로 메움

Guitar Pro 틱 규약: 4분음표 = 960 틱 (``guitarpro.Duration.quarterTime``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..models import NoteEvent

# guitarpro.Duration.quarterTime 과 동일해야 한다.
QUARTER_TICKS = 960
WHOLE_TICKS = QUARTER_TICKS * 4

DEFAULT_TEMPO = 120.0
DEFAULT_SUBDIVISION = 16  # 16분음표 격자

# BPM 추정 결과를 강제로 밀어넣는 범위. 이 밖으로 나가면 2배/절반으로 접는다.
_BPM_MIN = 60.0
_BPM_MAX = 180.0

# Guitar Pro 가 단일 Duration 으로 표현 가능한 (value, isDotted) → 틱 매핑.
# 내림차순 정렬해두고 greedy 로 가장 큰 표현 가능 길이를 고른다.
_DURATION_TABLE: list[tuple[int, bool, int]] = sorted(
    (
        (value, dotted, (WHOLE_TICKS // value) * 3 // 2 if dotted else WHOLE_TICKS // value)
        for value in (1, 2, 4, 8, 16, 32, 64)
        for dotted in (False, True)
    ),
    key=lambda item: item[2],
    reverse=True,
)

_MIN_DURATION_TICKS = _DURATION_TABLE[-1][2]


@dataclass
class RhythmGrid:
    """양자화 격자 정의.

    Attributes:
        tempo_bpm: 분당 4분음표 수.
        numerator: 박자표 분자 (4/4 의 4).
        denominator: 박자표 분모 (4/4 의 4).
        subdivision: 양자화 해상도. 16 이면 16분음표 격자.
        origin_sec: 첫 다운비트의 시각(초). 이 시각이 틱 0 이 된다.
    """

    tempo_bpm: float = DEFAULT_TEMPO
    numerator: int = 4
    denominator: int = 4
    subdivision: int = DEFAULT_SUBDIVISION
    origin_sec: float = 0.0

    @property
    def ticks_per_second(self) -> float:
        return self.tempo_bpm / 60.0 * QUARTER_TICKS

    @property
    def grid_ticks(self) -> int:
        """격자 한 칸의 틱 수. subdivision=16 이면 240."""
        return max(1, WHOLE_TICKS // self.subdivision)

    @property
    def measure_ticks(self) -> int:
        """한 마디의 틱 수. 4/4 이면 3840."""
        return self.numerator * (WHOLE_TICKS // self.denominator)

    def to_ticks(self, t_sec: float) -> int:
        """절대 시각(초)을 격자에 스냅된 절대 틱으로 변환한다."""
        raw = (t_sec - self.origin_sec) * self.ticks_per_second
        step = self.grid_ticks
        snapped = int(round(raw / step)) * step
        return max(0, snapped)

    def duration_to_ticks(self, seconds: float) -> int:
        """지속 시간(초)을 격자에 스냅된 틱 길이로 변환한다(최소 한 칸)."""
        step = self.grid_ticks
        raw = seconds * self.ticks_per_second
        snapped = int(round(raw / step)) * step
        return max(step, snapped)


@dataclass
class QuantizedBeat:
    """마디 안의 한 박(拍). ``notes`` 가 비어 있으면 쉼표다."""

    measure_index: int
    offset_ticks: int
    duration_ticks: int
    notes: list[NoteEvent] = field(default_factory=list)

    @property
    def is_rest(self) -> bool:
        return not self.notes


def ticks_to_duration(ticks: int) -> tuple[int, bool]:
    """틱 길이를 Guitar Pro Duration ``(value, isDotted)`` 로 내림 변환한다.

    표현 불가능한 길이는 **표현 가능한 가장 큰 길이로 내림**한다. 남는 시간은
    호출자가 쉼표로 메운다. 이렇게 하면 잇단음표/타이 없이도 마디 총합이 맞는다.
    """
    for value, dotted, span in _DURATION_TABLE:
        if span <= ticks:
            return value, dotted
    return _DURATION_TABLE[-1][0], _DURATION_TABLE[-1][1]


def duration_span(ticks: int) -> int:
    """``ticks_to_duration`` 이 실제로 소비하는 틱 수."""
    value, dotted = ticks_to_duration(ticks)
    span = WHOLE_TICKS // value
    return span * 3 // 2 if dotted else span


# ---------------------------------------------------------------------------
# Tempo estimation
# ---------------------------------------------------------------------------


def _fold_into_range(bpm: float) -> float:
    """BPM 을 [60, 180) 범위로 2배/절반 접기."""
    if bpm <= 0:
        return DEFAULT_TEMPO
    while bpm < _BPM_MIN:
        bpm *= 2.0
    while bpm >= _BPM_MAX:
        bpm /= 2.0
    return bpm


def estimate_tempo_from_audio(audio_path: Path) -> tuple[float, float]:
    """librosa beat tracking 으로 (BPM, 첫 다운비트 시각) 을 추정한다.

    librosa 는 basic-pitch 의 전이 의존성이라 항상 설치되어 있지만, 없거나
    분석에 실패하면 ``(DEFAULT_TEMPO, 0.0)`` 으로 폴백한다.
    """
    try:
        import librosa  # type: ignore[import]
    except ImportError:  # pragma: no cover - librosa 는 basic-pitch 전이 의존성
        return DEFAULT_TEMPO, 0.0

    try:
        y, sr = librosa.load(str(audio_path), mono=True)
        tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    except Exception:  # noqa: BLE001 - 분석 실패는 폴백으로 흡수 (진단 보조 기능)
        return DEFAULT_TEMPO, 0.0

    # librosa >= 0.10 은 tempo 를 ndarray 로 돌려준다.
    bpm = float(tempo.item() if hasattr(tempo, "item") else tempo)
    origin = float(beat_times[0]) if len(beat_times) else 0.0
    if not bpm or bpm != bpm:  # 0 또는 NaN
        return DEFAULT_TEMPO, origin
    return _fold_into_range(bpm), origin


def estimate_tempo_from_events(midi_events) -> float:
    """onset 간격(IOI) 최빈값으로 BPM 을 추정한다(오디오 없을 때 폴백).

    최빈 IOI 가 16분/8분/4분음표 중 무엇인지 알 수 없으므로, 각 가정으로
    환산한 BPM 중 [60, 180) 에 드는 값을 택한다.
    """
    onsets = sorted({round(e.start_time, 3) for e in midi_events})
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if b - a > 1e-3]
    if not iois:
        return DEFAULT_TEMPO

    # 10ms 버킷으로 뭉쳐 최빈값을 찾는다.
    buckets: dict[int, int] = {}
    for ioi in iois:
        key = int(round(ioi * 100))
        buckets[key] = buckets.get(key, 0) + 1
    mode_ioi = max(buckets, key=lambda k: buckets[k]) / 100.0
    if mode_ioi <= 0:
        return DEFAULT_TEMPO

    # mode_ioi 가 4분/8분/16분/2분음표라고 가정했을 때의 BPM 후보를 순서대로 시도.
    # 4분음표 가정을 먼저 두어야 균등 간격 연주에서 실제의 절반 BPM 이 나오지 않는다.
    for notes_per_quarter in (1.0, 2.0, 4.0, 0.5):
        bpm = 60.0 / (mode_ioi * notes_per_quarter)
        if _BPM_MIN <= bpm < _BPM_MAX:
            return bpm
    return _fold_into_range(60.0 / mode_ioi)


def build_grid(
    midi_events,
    audio_path: Path | None = None,
    tempo_bpm: float | None = None,
    numerator: int = 4,
    denominator: int = 4,
    subdivision: int = DEFAULT_SUBDIVISION,
) -> RhythmGrid:
    """양자화 격자를 만든다.

    ``tempo_bpm`` 이 주어지면 추정을 건너뛴다. 아니면 ``audio_path`` 로
    librosa 추정을 시도하고, 그것도 없으면 onset IOI 로 폴백한다.
    """
    origin = 0.0
    if tempo_bpm is None:
        if audio_path is not None:
            tempo_bpm, origin = estimate_tempo_from_audio(audio_path)
        else:
            tempo_bpm = estimate_tempo_from_events(midi_events)
    return RhythmGrid(
        tempo_bpm=tempo_bpm,
        numerator=numerator,
        denominator=denominator,
        subdivision=subdivision,
        origin_sec=origin,
    )


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def _dedupe_by_string(chord: list[NoteEvent]) -> list[NoteEvent]:
    """한 현에서는 동시에 한 음만 울릴 수 있으므로 현당 하나만 남긴다.

    같은 격자 칸에 같은 현의 음이 여러 개 몰리면 velocity 가 큰 쪽을 택한다.
    Guitar Pro 는 beat 마다 현 비트마스크를 쓰고 **세팅된 비트 수만큼만** 노트를
    기록하므로, 현이 중복되면 마스크와 노트 수가 어긋나 파일이 깨진다.
    """
    best: dict[int, NoteEvent] = {}
    for note in chord:
        current = best.get(note.string)
        if current is None or note.midi_event.velocity > current.midi_event.velocity:
            best[note.string] = note
    return [best[string] for string in sorted(best)]


def _group_by_onset(notes: list[NoteEvent], grid: RhythmGrid) -> list[tuple[int, list[NoteEvent]]]:
    """같은 격자 칸에 떨어진 음들을 화음으로 묶는다(현 중복 제거)."""
    groups: dict[int, list[NoteEvent]] = {}
    for note in notes:
        tick = grid.to_ticks(note.midi_event.start_time)
        groups.setdefault(tick, []).append(note)
    return [(tick, _dedupe_by_string(chord)) for tick, chord in sorted(groups.items())]


def quantize_notes(notes: list[NoteEvent], grid: RhythmGrid) -> list[QuantizedBeat]:
    """NoteEvent 목록을 마디별 QuantizedBeat 목록으로 변환한다.

    - 같은 격자 칸의 음은 하나의 Beat(화음)로 묶인다.
    - 각 Beat 의 길이는 ``min(다음 onset 까지의 간격, 실제 발음 길이)`` 이며
      마디 경계를 넘지 않도록 잘린다.
    - 빈 구간과 빈 마디는 쉼표 Beat 로 채워진다.
    """
    if not notes:
        return []

    measure_ticks = grid.measure_ticks
    groups = _group_by_onset(notes, grid)

    beats: list[QuantizedBeat] = []
    for idx, (tick, chord) in enumerate(groups):
        next_tick = groups[idx + 1][0] if idx + 1 < len(groups) else None

        sounding = max(
            grid.duration_to_ticks(n.midi_event.end_time - n.midi_event.start_time)
            for n in chord
        )
        gap = (next_tick - tick) if next_tick is not None else sounding
        span = max(grid.grid_ticks, min(sounding, gap))

        # 마디 경계를 넘으면 자른다 (타이는 미지원 — 아래 TODO 참조).
        offset = tick % measure_ticks
        span = min(span, measure_ticks - offset)

        beats.append(
            QuantizedBeat(
                measure_index=tick // measure_ticks,
                offset_ticks=offset,
                duration_ticks=duration_span(span),
                notes=chord,
            )
        )

    return _fill_rests(beats, measure_ticks)


def _fill_rests(beats: list[QuantizedBeat], measure_ticks: int) -> list[QuantizedBeat]:
    """마디 내 빈 구간과 완전히 빈 마디를 쉼표로 메운다."""
    if not beats:
        return []

    by_measure: dict[int, list[QuantizedBeat]] = {}
    for beat in beats:
        by_measure.setdefault(beat.measure_index, []).append(beat)

    filled: list[QuantizedBeat] = []
    for measure_index in range(max(by_measure) + 1):
        measure_beats = sorted(
            by_measure.get(measure_index, []), key=lambda b: b.offset_ticks
        )
        cursor = 0
        for beat in measure_beats:
            if beat.offset_ticks > cursor:
                filled.extend(
                    _rest_beats(measure_index, cursor, beat.offset_ticks - cursor)
                )
            filled.append(beat)
            cursor = beat.offset_ticks + beat.duration_ticks
        if cursor < measure_ticks:
            filled.extend(_rest_beats(measure_index, cursor, measure_ticks - cursor))

    return filled


def _rest_beats(measure_index: int, offset: int, remaining: int) -> list[QuantizedBeat]:
    """``remaining`` 틱을 표현 가능한 쉼표 여러 개로 분해한다."""
    rests: list[QuantizedBeat] = []
    while remaining >= _MIN_DURATION_TICKS:
        span = duration_span(remaining)
        rests.append(
            QuantizedBeat(
                measure_index=measure_index,
                offset_ticks=offset,
                duration_ticks=span,
                notes=[],
            )
        )
        offset += span
        remaining -= span
    return rests
