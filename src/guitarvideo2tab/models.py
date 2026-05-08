"""Core data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TechniqueLabel = Literal[
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
]

ModalitySource = Literal["audio", "vision", "fusion"]


@dataclass
class PitchContour:
    note_id: str
    time_pitch_curve: list[tuple[float, float]]
    bend_semitones: float = 0.0


@dataclass
class MidiEvent:
    pitch: int
    start_time: float
    end_time: float
    velocity: int
    pitch_contour: PitchContour | None = None


@dataclass
class TechniqueAnnotation:
    technique: TechniqueLabel
    confidence: float
    source: ModalitySource
    params: dict = field(default_factory=dict)


@dataclass
class HandKeypoints:
    timestamp: float
    left_hand: list[tuple[float, float]] | None
    right_hand: list[tuple[float, float]] | None


@dataclass
class FretboardFrame:
    timestamp: float
    homography: list[list[float]] | None
    corners: list[tuple[float, float]] | None
    visible: bool = True


@dataclass
class FretPosition:
    timestamp: float
    string: int
    fret: int
    confidence: float


@dataclass
class NoteEvent:
    midi_event: MidiEvent
    string: int
    fret: int
    technique: TechniqueAnnotation | None = None
