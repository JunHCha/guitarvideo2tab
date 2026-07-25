"""Unit tests for separate_guitar_stem (mock-based, no real model loading)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from guitarvideo2tab.preprocessing import stem as stem_module
from guitarvideo2tab.preprocessing.stem import separate_guitar_stem


class _FakeTensor:
    """Stand-in for a torch.Tensor — only identity matters in these tests."""


def _patch_demucs(monkeypatch, stems: dict, samplerate: int = 44100, save_audio_fn=None):
    """Replace the demucs/torchaudio seams with controllable fakes."""
    monkeypatch.setattr(stem_module, "_run_demucs", lambda _path: (stems, samplerate))
    fn = save_audio_fn if save_audio_fn is not None else MagicMock()
    monkeypatch.setattr(stem_module, "_save_audio", fn)
    return fn


def test_returns_guitar_wav_path(tmp_path, monkeypatch):
    """Returned path should be output_dir / 'guitar.wav'."""
    stems = {"guitar": _FakeTensor(), "drums": _FakeTensor()}
    _patch_demucs(monkeypatch, stems)

    result = separate_guitar_stem(Path("dummy.wav"), tmp_path)

    assert result == tmp_path / "guitar.wav"


def test_save_audio_called_with_guitar_tensor(tmp_path, monkeypatch):
    """_save_audio must receive the guitar tensor specifically."""
    guitar_tensor = _FakeTensor()
    stems = {"guitar": guitar_tensor, "drums": _FakeTensor(), "bass": _FakeTensor()}

    captured: dict = {}

    def fake_save_audio(tensor, path, sr):  # noqa: ANN001
        captured["tensor"] = tensor
        captured["path"] = path
        captured["sr"] = sr

    _patch_demucs(monkeypatch, stems, save_audio_fn=fake_save_audio)

    separate_guitar_stem(Path("dummy.wav"), tmp_path)

    assert captured["tensor"] is guitar_tensor
    assert captured["path"] == str(tmp_path / "guitar.wav")
    assert captured["sr"] == 44100


def test_output_dir_created(tmp_path, monkeypatch):
    """output_dir should be created even when it does not exist beforehand."""
    stems = {"guitar": _FakeTensor()}
    new_dir = tmp_path / "nested" / "output"
    _patch_demucs(monkeypatch, stems)

    separate_guitar_stem(Path("dummy.wav"), new_dir)

    assert new_dir.is_dir()


def test_keyerror_when_guitar_stem_missing(tmp_path, monkeypatch):
    """KeyError must be raised when the model returns no 'guitar' stem."""
    stems = {"drums": _FakeTensor(), "bass": _FakeTensor(), "vocals": _FakeTensor()}
    _patch_demucs(monkeypatch, stems)

    with pytest.raises(KeyError, match="guitar"):
        separate_guitar_stem(Path("dummy.wav"), tmp_path)


def test_keyerror_message_lists_available_stems(tmp_path, monkeypatch):
    """KeyError message should mention the available stem names."""
    stems = {"drums": _FakeTensor(), "piano": _FakeTensor()}
    _patch_demucs(monkeypatch, stems)

    with pytest.raises(KeyError) as exc_info:
        separate_guitar_stem(Path("dummy.wav"), tmp_path)

    error_text = str(exc_info.value)
    assert "drums" in error_text or "piano" in error_text
