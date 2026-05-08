"""download_video 단위 테스트."""
from __future__ import annotations

from pathlib import Path

import pytest

from guitarvideo2tab.preprocessing import downloader
from guitarvideo2tab.preprocessing.downloader import download_video


def test_local_file_returns_resolved_path(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake mp4 content")

    result = download_video(str(video), tmp_path / "out")

    assert result == video.resolve()


def test_missing_local_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"

    with pytest.raises(FileNotFoundError):
        download_video(str(missing), tmp_path / "out")


class _FakeYDL:
    """yt_dlp.YoutubeDL 대체 페이크 — 옵션과 URL을 캡처한다."""

    captured_opts: dict | None = None
    captured_url: str | None = None
    captured_download: bool | None = None

    def __init__(self, opts: dict) -> None:
        type(self).captured_opts = opts
        self._opts = opts

    def __enter__(self) -> "_FakeYDL":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def extract_info(self, url: str, download: bool = True) -> dict:
        type(self).captured_url = url
        type(self).captured_download = download
        outtmpl = self._opts["outtmpl"]
        filepath = outtmpl.replace("%(id)s", "abc123").replace("%(ext)s", "mp4")
        return {
            "id": "abc123",
            "ext": "mp4",
            "requested_downloads": [{"filepath": filepath}],
        }

    def prepare_filename(self, info: dict) -> str:  # pragma: no cover - 폴백 경로
        outtmpl = self._opts["outtmpl"]
        return outtmpl.replace("%(id)s", info["id"]).replace("%(ext)s", info["ext"])


def test_url_download_invokes_yt_dlp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeYDL.captured_opts = None
    _FakeYDL.captured_url = None
    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", _FakeYDL)

    out_dir = tmp_path / "downloads"
    url = "https://www.youtube.com/watch?v=abc123"

    result = download_video(url, out_dir)

    assert _FakeYDL.captured_url == url
    assert _FakeYDL.captured_opts is not None
    opts = _FakeYDL.captured_opts
    assert "outtmpl" in opts
    assert opts["merge_output_format"] == "mp4"
    assert opts["quiet"] is True
    assert str(out_dir) in opts["outtmpl"]
    assert out_dir.exists()
    assert result == Path(str(out_dir / "abc123.mp4"))


def test_url_download_falls_back_to_prepare_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeYDLNoRequested(_FakeYDL):
        def extract_info(self, url: str, download: bool = True) -> dict:
            type(self).captured_url = url
            return {"id": "xyz789", "ext": "mp4"}

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDLNoRequested)

    out_dir = tmp_path / "dl"
    result = download_video("http://example.com/v.mp4", out_dir)

    assert result.name == "xyz789.mp4"
