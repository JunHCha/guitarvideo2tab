"""AlphaTab/PyGuitarPro로 NoteEvent → .gpx/.gp5 출력."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import guitarpro

from ..models import NoteEvent

logger = logging.getLogger(__name__)

# BendPoint position range in pyguitarpro is 0-12 (semitones * ~8.33 per semitone)
_BEND_MAX_POSITION = 12
_BEND_SEMITONE_VALUE = 100  # 100 units = 1 semitone in GP bend scale


def _build_song(notes: list[NoteEvent], tuning: tuple[int, ...]) -> guitarpro.Song:
    """Build a minimal valid guitarpro.Song from NoteEvent list."""
    # --- Strings ---
    strings = [
        guitarpro.GuitarString(number=i + 1, value=midi)
        for i, midi in enumerate(tuning)
    ]

    # --- Measure header (one measure, tempo 120) ---
    header = guitarpro.MeasureHeader(number=1)

    # --- Song skeleton ---
    song = guitarpro.Song(
        tempo=120,
        measureHeaders=[header],
    )

    # --- Track ---
    track = guitarpro.Track(
        song=song,
        number=1,
        strings=strings,
        name="Guitar",
    )
    song.tracks = [track]

    # --- Measure / Voice ---
    measure = guitarpro.Measure(track=track, header=header)
    track.measures = [measure]

    voice = measure.voices[0]

    # --- Populate beats/notes ---
    valid_notes = [
        n for n in notes if n.string >= 1 and n.fret >= 0
    ]
    skipped = len(notes) - len(valid_notes)
    if skipped:
        logger.debug("Skipping %d NoteEvent(s) with string=-1 or fret=-1", skipped)

    if not valid_notes:
        # At least one rest beat so the measure is structurally valid
        rest_beat = guitarpro.Beat(voice, status=guitarpro.BeatStatus.empty)
        voice.beats = [rest_beat]
    else:
        beats = []
        for note_event in valid_notes:
            beat = guitarpro.Beat(voice)
            beat.status = guitarpro.BeatStatus.normal
            beat.duration = guitarpro.Duration(value=4)  # quarter note

            note = guitarpro.Note(
                beat=beat,
                value=note_event.fret,
                string=note_event.string,
                type=guitarpro.NoteType.normal,
            )

            _apply_technique(note, beat, note_event)

            beat.notes = [note]
            beats.append(beat)

        voice.beats = beats

    return song


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
) -> Path:
    """Build song and write to output_path via guitarpro.write."""
    song = _build_song(notes, tuning)
    guitarpro.write(song, str(output_path))
    return output_path


@dataclass
class TabWriter:
    tuning: tuple[int, ...] = (40, 45, 50, 55, 59, 64)  # 표준 EADGBE (MIDI)

    def write_gpx(self, notes: list[NoteEvent], output_path: Path) -> Path:
        """Write notes to a Guitar Pro file (.gpx format)."""
        return write_song(notes, output_path, self.tuning)

    def write_gp5(self, notes: list[NoteEvent], output_path: Path) -> Path:
        """Write notes to a Guitar Pro file (.gp5 format)."""
        return write_song(notes, output_path, self.tuning)
