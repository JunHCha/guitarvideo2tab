"""ffmpeg으로 비디오 → (오디오 WAV, 비디오 트랙) 분리. PTS 보존 필수."""
from __future__ import annotations

from pathlib import Path


def split_audio_video(video_path: Path, output_dir: Path) -> tuple[Path, Path]:
    raise NotImplementedError(
        "ffmpeg-python으로 오디오 WAV와 비디오 트랙 분리. "
        "PTS(Presentation Timestamp) 보존 옵션(-copyts) 필수 — 동기화 정확도에 직결."
    )
