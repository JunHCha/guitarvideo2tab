"""YOLOv8-OBB 프렛보드 검출 + 호모그래피 워핑."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import FretboardFrame


@dataclass
class FretboardDetector:
    weights_path: Path | None = None
    confidence_threshold: float = 0.5

    def detect(self, video_path: Path) -> list[FretboardFrame]:
        raise NotImplementedError(
            "YOLOv8-OBB로 매 프레임 프렛보드 4점 OBB 검출 → 호모그래피 행렬 계산. "
            "검출 실패 프레임은 visible=False, homography=None."
        )
