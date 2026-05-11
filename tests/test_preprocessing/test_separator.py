"""split_audio_video 단위 테스트."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from guitarvideo2tab.preprocessing import separator
from guitarvideo2tab.preprocessing.separator import (  # noqa: F401
    split_audio_video,
)

# ---------------------------------------------------------------------------
# Fake ffmpeg chain
# ---------------------------------------------------------------------------

class _FakeOutputStream:
    """ffmpeg.output(...)의 페이크 — .run() 호출을 캡처한다."""

    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs
        _FakeOutputStream.calls.append(self)

    def run(self, **run_kwargs: Any) -> None:  # noqa: ANN401
        self.run_kwargs = run_kwargs

    # 클래스 레벨 콜 레코드
    calls: list["_FakeOutputStream"] = []


class _FakeNode:
    """ffmpeg.input(...)이 반환하는 노드 페이크 — .output()을 체이닝한다."""

    def __init__(self, src: str) -> None:
        self.src = src

    def output(self, *args: Any, **kwargs: Any) -> _FakeOutputStream:
        return _FakeOutputStream(args, kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_correct_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """반환값이 예상 경로인지 확인한다."""
    _FakeOutputStream.calls.clear()
    monkeypatch.setattr(separator.ffmpeg, "input", lambda src: _FakeNode(src))
    monkeypatch.setattr(
        separator.ffmpeg,
        "output",
        lambda node, path, **kw: node.output(path, **kw),
    )

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    audio_path, video_path_out = split_audio_video(video, out_dir)

    assert audio_path == out_dir / "clip.wav"
    assert video_path_out == out_dir / "clip.video.mp4"


def test_output_dir_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """output_dir가 자동으로 생성되는지 확인한다."""
    _FakeOutputStream.calls.clear()
    monkeypatch.setattr(separator.ffmpeg, "input", lambda src: _FakeNode(src))
    monkeypatch.setattr(
        separator.ffmpeg,
        "output",
        lambda node, path, **kw: node.output(path, **kw),
    )

    video = tmp_path / "song.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "nested" / "output"

    assert not out_dir.exists()
    split_audio_video(video, out_dir)
    assert out_dir.exists()


def test_copyts_in_output_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """-copyts 플래그(copyts=None)가 ffmpeg.output 호출에 포함되는지 확인한다."""
    _FakeOutputStream.calls.clear()
    monkeypatch.setattr(separator.ffmpeg, "input", lambda src: _FakeNode(src))

    captured: list[dict[str, Any]] = []

    def _fake_output(node: _FakeNode, path: str, **kwargs: Any) -> _FakeOutputStream:
        captured.append(kwargs)
        return node.output(path, **kwargs)

    monkeypatch.setattr(separator.ffmpeg, "output", _fake_output)

    video = tmp_path / "track.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    split_audio_video(video, out_dir)

    # 적어도 한 번의 출력 호출에 copyts=None이 있어야 한다
    assert any("copyts" in kw and kw["copyts"] is None for kw in captured), (
        f"copyts=None not found in any output call; captured kwargs: {captured}"
    )


def test_video_output_has_no_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """비디오 전용 출력에 -an(an=None) 플래그가 있는지 확인한다."""
    _FakeOutputStream.calls.clear()
    monkeypatch.setattr(separator.ffmpeg, "input", lambda src: _FakeNode(src))

    captured: list[dict[str, Any]] = []

    def _fake_output(node: _FakeNode, path: str, **kwargs: Any) -> _FakeOutputStream:
        captured.append({"path": path, "kwargs": kwargs})
        return node.output(path, **kwargs)

    monkeypatch.setattr(separator.ffmpeg, "output", _fake_output)

    video = tmp_path / "guitar.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    split_audio_video(video, out_dir)

    # .video.mp4 출력에는 an=None이 있어야 한다
    video_calls = [c for c in captured if c["path"].endswith(".video.mp4")]
    assert video_calls, "No output call found for .video.mp4"
    assert any("an" in c["kwargs"] and c["kwargs"]["an"] is None for c in video_calls), (
        f"an=None not found in video output call; video_calls: {video_calls}"
    )
