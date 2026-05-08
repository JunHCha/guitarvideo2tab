"""Late Fusion: 오디오·비전 결과를 신뢰도 가중 투표로 결합 (ADR-001 D4)."""
from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    FretPosition,
    MidiEvent,
    NoteEvent,
    TechniqueAnnotation,
)

CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5


@dataclass
class LateFusion:
    confidence_high: float = CONFIDENCE_HIGH
    confidence_low: float = CONFIDENCE_LOW

    def fuse(
        self,
        midi_events: list[MidiEvent],
        audio_techniques: list[TechniqueAnnotation],
        fret_positions: list[FretPosition],
        vision_techniques: list[TechniqueAnnotation],
    ) -> list[NoteEvent]:
        raise NotImplementedError(
            "각 MIDI 이벤트의 시간창에 해당하는 fret_position과 두 기법 후보를 매칭. "
            "string/fret: 비전 신뢰도>=HIGH면 비전 우선, 가림이면 오디오 prior. "
            "기법: 신뢰도 가중 투표 (HIGH 일치→확정, 불일치→비전 우선, 비전 실패→오디오)."
        )
