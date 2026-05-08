"""Demucs 6-stem 모델로 믹스 → 기타 stem 분리."""
from __future__ import annotations

from pathlib import Path


def separate_guitar_stem(audio_path: Path, output_dir: Path) -> Path:
    raise NotImplementedError(
        "demucs htdemucs_6s 모델 호출하여 'guitar' stem만 추출. 출력은 guitar.wav."
    )
