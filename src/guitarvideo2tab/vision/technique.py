"""TCN/Transformer on hand keypoint 시계열 — 비전 표현 기법 분류기."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import HandKeypoints, TechniqueAnnotation


@dataclass
class VisionTechniqueClassifier:
    weights_path: Path | None = None
    window_ms: int = 300
    model_arch: str = "tcn"

    def classify(self, hands: list[HandKeypoints]) -> list[TechniqueAnnotation]:
        raise NotImplementedError(
            "21 keypoint × 2D × T 시계열을 ±window_ms 슬라이딩 윈도우로 분할 → "
            "TCN/Transformer로 9-class 기법 분류. source='vision'으로 표시. "
            "Mitsou et al.(2023) 데이터셋으로 사전학습."
        )
