"""AlphaTab/PyGuitarPro로 NoteEvent → .gpx/.gp5 출력."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import guitarpro

from ..models import NoteEvent
from .rhythm import QuantizedBeat, RhythmGrid, build_grid, quantize_notes, ticks_to_duration

logger = logging.getLogger(__name__)

# BendPoint position range in pyguitarpro is 0-12 (semitones * ~8.33 per semitone)
_BEND_MAX_POSITION = 12
_BEND_SEMITONE_VALUE = 100  # 100 units = 1 semitone in GP bend scale


def _build_measure_headers(count: int, grid: RhythmGrid) -> list[guitarpro.MeasureHeader]:
    """``count`` 개의 마디 헤더를 만들고 start 틱을 누적시킨다.

    Guitar Pro 의 첫 마디는 틱 0 이 아니라 ``Duration.quarterTime`` 에서 시작한다.
    """
    headers: list[guitarpro.MeasureHeader] = []
    start = guitarpro.Duration.quarterTime
    for index in range(count):
        header = guitarpro.MeasureHeader(
            number=index + 1,
            start=start,
            timeSignature=guitarpro.TimeSignature(
                numerator=grid.numerator,
                denominator=guitarpro.Duration(value=grid.denominator),
            ),
        )
        headers.append(header)
        start += header.length
    return headers


def _build_song(
    notes: list[NoteEvent],
    tuning: tuple[int, ...],
    grid: RhythmGrid,
) -> guitarpro.Song:
    """양자화된 리듬을 반영한 guitarpro.Song 을 만든다."""
    strings = [
        guitarpro.GuitarString(number=i + 1, value=midi)
        for i, midi in enumerate(tuning)
    ]

    valid_notes = [n for n in notes if n.string >= 1 and n.fret >= 0]
    skipped = len(notes) - len(valid_notes)
    if skipped:
        logger.debug("Skipping %d NoteEvent(s) with string=-1 or fret=-1", skipped)

    quantized = quantize_notes(valid_notes, grid)
    measure_count = max((b.measure_index for b in quantized), default=-1) + 1
    measure_count = max(measure_count, 1)  # 음이 없어도 빈 마디 하나는 필요

    headers = _build_measure_headers(measure_count, grid)

    song = guitarpro.Song(tempo=int(round(grid.tempo_bpm)), measureHeaders=headers)
    track = guitarpro.Track(song=song, number=1, strings=strings, name="Guitar")
    song.tracks = [track]

    by_measure: dict[int, list[QuantizedBeat]] = {}
    for beat in quantized:
        by_measure.setdefault(beat.measure_index, []).append(beat)

    measures: list[guitarpro.Measure] = []
    for index, header in enumerate(headers):
        measure = guitarpro.Measure(track=track, header=header)
        voice = measure.voices[0]
        voice.beats = [
            _build_beat(voice, qbeat, header.start)
            for qbeat in sorted(by_measure.get(index, []), key=lambda b: b.offset_ticks)
        ] or [_full_measure_rest(voice, grid, header.start)]
        measures.append(measure)
    track.measures = measures

    return song


def _build_beat(
    voice: guitarpro.Voice,
    qbeat: QuantizedBeat,
    measure_start: int,
) -> guitarpro.Beat:
    """QuantizedBeat 하나를 guitarpro.Beat 으로 변환한다(화음은 한 Beat 안의 여러 Note)."""
    value, is_dotted = ticks_to_duration(qbeat.duration_ticks)
    beat = guitarpro.Beat(voice)
    beat.duration = guitarpro.Duration(value=value, isDotted=is_dotted)
    beat.start = measure_start + qbeat.offset_ticks

    if qbeat.is_rest:
        beat.status = guitarpro.BeatStatus.rest
        beat.notes = []
        return beat

    beat.status = guitarpro.BeatStatus.normal
    gp_notes = []
    for note_event in qbeat.notes:
        note = guitarpro.Note(
            beat=beat,
            value=note_event.fret,
            string=note_event.string,
            type=guitarpro.NoteType.normal,
        )
        _apply_technique(note, beat, note_event)
        gp_notes.append(note)
    beat.notes = gp_notes
    return beat


def _full_measure_rest(
    voice: guitarpro.Voice,
    grid: RhythmGrid,
    measure_start: int,
) -> guitarpro.Beat:
    """온쉼표 한 개로 채워진 빈 마디를 만든다."""
    value, is_dotted = ticks_to_duration(grid.measure_ticks)
    beat = guitarpro.Beat(voice)
    beat.duration = guitarpro.Duration(value=value, isDotted=is_dotted)
    beat.start = measure_start
    beat.status = guitarpro.BeatStatus.rest
    beat.notes = []
    return beat


def _apply_technique(
    note: guitarpro.Note,
    beat: guitarpro.Beat,
    note_event: NoteEvent,
) -> None:
    """Map TechniqueAnnotation to pyguitarpro effect fields (best-effort)."""
    ann = note_event.technique
    if ann is None:
        return

    technique = ann.technique

    if technique == "bend":
        pitch_contour = note_event.midi_event.pitch_contour
        if pitch_contour and pitch_contour.time_pitch_curve:
            curve = pitch_contour.time_pitch_curve
            # Normalise time → [0, 12] position range
            times = [t for t, _ in curve]
            t_min, t_max = min(times), max(times)
            t_range = t_max - t_min if t_max > t_min else 1.0

            points: list[guitarpro.BendPoint] = []
            for t, pitch_delta in curve:
                position = int(round(((t - t_min) / t_range) * _BEND_MAX_POSITION))
                value = int(round(pitch_delta * _BEND_SEMITONE_VALUE))
                points.append(guitarpro.BendPoint(position=position, value=value))
            note.effect.bend = guitarpro.BendEffect(
                type=guitarpro.BendType.bend,
                points=points,
            )
        else:
            # Minimal default bend (1 semitone)
            note.effect.bend = guitarpro.BendEffect(
                type=guitarpro.BendType.bend,
                points=[
                    guitarpro.BendPoint(position=0, value=0),
                    guitarpro.BendPoint(position=6, value=_BEND_SEMITONE_VALUE),
                    guitarpro.BendPoint(position=12, value=_BEND_SEMITONE_VALUE),
                ],
            )

    elif technique == "slide":
        note.effect.slides = [guitarpro.SlideType.shiftSlideTo]

    elif technique in ("hammer-on", "pull-off"):
        note.effect.hammer = True

    elif technique == "vibrato":
        note.effect.vibrato = True

    elif technique == "palm-mute":
        note.effect.palmMute = True

    elif technique == "tapping":
        # BeatEffect has slapEffect for tapping; NoteEffect has no isTapping field
        beat.effect.slapEffect = guitarpro.SlapEffect.tapping


def write_song(
    notes: list[NoteEvent],
    output_path: Path,
    tuning: tuple[int, ...],
    grid: RhythmGrid,
) -> Path:
    """Build song and write to output_path via guitarpro.write."""
    song = _build_song(notes, tuning, grid)
    guitarpro.write(song, str(output_path))
    return output_path


@dataclass
class TabWriter:
    """NoteEvent 목록을 Guitar Pro 파일로 직렬화한다.

    ``audio_path`` 를 주면 librosa 로 tempo/다운비트를 추정한다. 주지 않으면
    노트 onset 간격(IOI)으로 추정하며, ``tempo_bpm`` 을 직접 주면 추정을 건너뛴다.
    """

    tuning: tuple[int, ...] = (40, 45, 50, 55, 59, 64)  # 표준 EADGBE (MIDI)
    tempo_bpm: float | None = None
    numerator: int = 4
    denominator: int = 4
    subdivision: int = 16

    def _grid(self, notes: list[NoteEvent], audio_path: Path | None) -> RhythmGrid:
        return build_grid(
            [n.midi_event for n in notes],
            audio_path=audio_path,
            tempo_bpm=self.tempo_bpm,
            numerator=self.numerator,
            denominator=self.denominator,
            subdivision=self.subdivision,
        )

    def write_gpx(
        self,
        notes: list[NoteEvent],
        output_path: Path,
        audio_path: Path | None = None,
    ) -> Path:
        """Write notes to a Guitar Pro file (.gpx format)."""
        return write_song(notes, output_path, self.tuning, self._grid(notes, audio_path))

    def write_gp5(
        self,
        notes: list[NoteEvent],
        output_path: Path,
        audio_path: Path | None = None,
    ) -> Path:
        """Write notes to a Guitar Pro file (.gp5 format)."""
        return write_song(notes, output_path, self.tuning, self._grid(notes, audio_path))
