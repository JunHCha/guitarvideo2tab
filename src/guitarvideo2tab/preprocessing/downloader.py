"""YouTube/URL 영상 다운로드 (yt-dlp 래퍼)."""
from __future__ import annotations

from pathlib import Path


def download_video(source: str, output_dir: Path) -> Path:
    raise NotImplementedError(
        "yt-dlp으로 영상 다운로드. URL이면 받아서 output_dir에 저장, 로컬 파일이면 그대로 반환."
    )
