"""AlphaTab/PyGuitarPro로 NoteEvent → .gpx/.gp5 출력."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import NoteEvent


@dataclass
class TabWriter:
    tuning: tuple[int, ...] = (40, 45, 50, 55, 59, 64)  # 표준 EADGBE (MIDI)

    def write_gpx(self, notes: list[NoteEvent], output_path: Path) -> Path:
        raise NotImplementedError(
            "NoteEvent를 PyGuitarPro Track/Beat/Note 객체로 변환. "
            "기법은 bendPoints/slideType/isHammerPullOrigin/vibrato/palmMute/isTapping 필드에 매핑. "
            "PitchContour는 bendPoints 시간-피치 곡선으로 변환."
        )

    def write_gp5(self, notes: list[NoteEvent], output_path: Path) -> Path:
        raise NotImplementedError("write_gpx와 동일 로직, .gp5 포맷으로 직렬화.")
