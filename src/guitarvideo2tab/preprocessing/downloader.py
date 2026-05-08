"""YouTube/URL 영상 다운로드 (yt-dlp 래퍼)."""
from __future__ import annotations

from pathlib import Path

import yt_dlp


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def download_video(source: str, output_dir: Path) -> Path:
    """URL이면 yt-dlp로 output_dir에 mp4 다운로드, 로컬 파일이면 절대경로 반환."""
    if not _is_url(source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(source)
        return path.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        requested = info.get("requested_downloads") if isinstance(info, dict) else None
        if requested:
            return Path(requested[0]["filepath"])
        return Path(ydl.prepare_filename(info))
