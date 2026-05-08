"""MediaPipe Hands로 좌·우손 21 keypoint 시계열 추출."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import HandKeypoints


@dataclass
class HandTracker:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def track(self, video_path: Path) -> list[HandKeypoints]:
        raise NotImplementedError(
            "MediaPipe Hands solution으로 매 프레임 21 keypoint × 2 손 추출. "
            "추적 실패 프레임의 keypoint는 보간 또는 None 처리."
        )
