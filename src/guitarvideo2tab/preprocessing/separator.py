"""ffmpeg으로 비디오 → (오디오 WAV, 비디오 트랙) 분리. PTS 보존 필수."""
from __future__ import annotations

from pathlib import Path

import ffmpeg


def split_audio_video(video_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """비디오 파일을 오디오 WAV와 비디오 전용 MP4로 분리한다.

    Args:
        video_path: 원본 비디오 파일 경로.
        output_dir: 출력 파일을 저장할 디렉터리.

    Returns:
        (audio_path, video_path_out) — 오디오 WAV와 비디오 전용 MP4 경로.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    audio_path = output_dir / f"{stem}.wav"
    video_path_out = output_dir / f"{stem}.video.mp4"

    src = ffmpeg.input(str(video_path))

    # 오디오 추출 → WAV, PTS 보존
    ffmpeg.output(src, str(audio_path), copyts=None).run(overwrite_output=True)

    # 비디오 전용 트랙 추출 → MP4, 오디오 제거, PTS 보존
    ffmpeg.output(src, str(video_path_out), an=None, copyts=None).run(
        overwrite_output=True
    )

    return audio_path, video_path_out
