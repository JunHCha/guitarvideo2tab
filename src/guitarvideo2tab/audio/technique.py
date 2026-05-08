"""TART 2단계 MLP 기반 오디오 표현 기법 분류기."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import MidiEvent, TechniqueAnnotation


@dataclass
class TARTTechniqueClassifier:
    weights_path: Path | None = None

    def classify(
        self,
        midi_events: list[MidiEvent],
        audio_path: Path,
    ) -> list[TechniqueAnnotation]:
        raise NotImplementedError(
            "TART 2단계 MLP — 노트별 spectral 특징 + pitch contour를 입력으로 "
            "기법 라벨과 신뢰도를 추출. source='audio'로 표시."
        )
