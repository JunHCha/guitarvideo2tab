"""운지 손 keypoint를 프렛보드 좌표계로 매핑하여 (string, fret) 추정."""
from __future__ import annotations

from dataclasses import dataclass

from ..models import FretboardFrame, FretPosition, HandKeypoints


@dataclass
class FretEstimator:
    num_strings: int = 6
    num_frets: int = 24

    def estimate(
        self,
        hands: list[HandKeypoints],
        fretboards: list[FretboardFrame],
    ) -> list[FretPosition]:
        raise NotImplementedError(
            "각 프레임에서 운지 손가락 끝점을 호모그래피로 프렛보드 정규 좌표로 변환, "
            "string(1-6) × fret(0-24) 매핑. 가림 프레임은 confidence 낮춤."
        )
