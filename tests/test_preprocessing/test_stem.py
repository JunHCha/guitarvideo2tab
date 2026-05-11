"""Unit tests for separate_guitar_stem (mock-based, no real model loading)."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers / fake objects
# ---------------------------------------------------------------------------

class _FakeTensor:
    """Stand-in for a torch.Tensor — only identity matters in these tests."""


class _FakeSeparator:
    """Fake Separator whose separate_audio_file returns a preset stems dict."""

    def __init__(self, stems: dict, *, model: str = "htdemucs_6s"):
        self._stems = stems
        self.model = model
        self.samplerate = 44100

    def separate_audio_file(self, audio_path):  # noqa: ANN001
        return {}, self._stems


def _build_fake_demucs_api(stems: dict, save_audio_fn=None):
    """Return a fake ``demucs.api`` module object with Separator and save_audio."""
    fake_module = types.ModuleType("demucs.api")

    class _Cls:
        def __init__(self, model: str = "htdemucs_6s"):
            self._inner = _FakeSeparator(stems, model=model)
            self.samplerate = self._inner.samplerate

        def separate_audio_file(self, audio_path):  # noqa: ANN001
            return self._inner.separate_audio_file(audio_path)

    fake_module.Separator = _Cls
    fake_module.save_audio = save_audio_fn if save_audio_fn is not None else MagicMock()
    return fake_module


def _inject_fake_demucs_api(monkeypatch, stems: dict, save_audio_fn=None):
    """Inject a fake demucs.api into sys.modules for the duration of a test."""
    fake_api = _build_fake_demucs_api(stems, save_audio_fn)
    monkeypatch.setitem(sys.modules, "demucs.api", fake_api)
    # Also ensure the parent package is present (already installed, but be safe)
    if "demucs" not in sys.modules:
        monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    return fake_api


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_guitar_wav_path(tmp_path, monkeypatch):
    """Returned path should be output_dir / 'guitar.wav'."""
    guitar_tensor = _FakeTensor()
    stems = {"guitar": guitar_tensor, "drums": _FakeTensor()}
    _inject_fake_demucs_api(monkeypatch, stems)

    from guitarvideo2tab.preprocessing.stem import separate_guitar_stem

    result = separate_guitar_stem(Path("dummy.wav"), tmp_path)

    assert result == tmp_path / "guitar.wav"


def test_save_audio_called_with_guitar_tensor(tmp_path, monkeypatch):
    """save_audio must receive the guitar tensor specifically."""
    guitar_tensor = _FakeTensor()
    stems = {"guitar": guitar_tensor, "drums": _FakeTensor(), "bass": _FakeTensor()}

    captured: dict = {}

    def fake_save_audio(tensor, path, sr):  # noqa: ANN001
        captured["tensor"] = tensor
        captured["path"] = path
        captured["sr"] = sr

    _inject_fake_demucs_api(monkeypatch, stems, save_audio_fn=fake_save_audio)

    from guitarvideo2tab.preprocessing.stem import separate_guitar_stem

    separate_guitar_stem(Path("dummy.wav"), tmp_path)

    assert captured["tensor"] is guitar_tensor
    assert captured["path"] == str(tmp_path / "guitar.wav")
    assert captured["sr"] == 44100


def test_output_dir_created(tmp_path, monkeypatch):
    """output_dir should be created even when it does not exist beforehand."""
    stems = {"guitar": _FakeTensor()}
    new_dir = tmp_path / "nested" / "output"
    _inject_fake_demucs_api(monkeypatch, stems)

    from guitarvideo2tab.preprocessing.stem import separate_guitar_stem

    separate_guitar_stem(Path("dummy.wav"), new_dir)

    assert new_dir.is_dir()


def test_keyerror_when_guitar_stem_missing(tmp_path, monkeypatch):
    """KeyError must be raised when the model returns no 'guitar' stem."""
    stems = {"drums": _FakeTensor(), "bass": _FakeTensor(), "vocals": _FakeTensor()}
    _inject_fake_demucs_api(monkeypatch, stems)

    from guitarvideo2tab.preprocessing.stem import separate_guitar_stem

    with pytest.raises(KeyError, match="guitar"):
        separate_guitar_stem(Path("dummy.wav"), tmp_path)


def test_keyerror_message_lists_available_stems(tmp_path, monkeypatch):
    """KeyError message should mention the available stem names."""
    stems = {"drums": _FakeTensor(), "piano": _FakeTensor()}
    _inject_fake_demucs_api(monkeypatch, stems)

    from guitarvideo2tab.preprocessing.stem import separate_guitar_stem

    with pytest.raises(KeyError) as exc_info:
        separate_guitar_stem(Path("dummy.wav"), tmp_path)

    # The string representation of a KeyError wraps the message in quotes
    error_text = str(exc_info.value)
    assert "drums" in error_text or "piano" in error_text
