"""Unit tests for TARTTechniqueClassifier (mock-based, no real model/audio)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

import guitarvideo2tab.audio.technique as technique_mod
from guitarvideo2tab.audio.technique import TARTTechniqueClassifier, _extract_feature
from guitarvideo2tab.models import MidiEvent, TechniqueAnnotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_midi_event(
    pitch: int = 60,
    start_time: float = 0.0,
    end_time: float = 0.5,
    velocity: int = 80,
) -> MidiEvent:
    return MidiEvent(
        pitch=pitch,
        start_time=start_time,
        end_time=end_time,
        velocity=velocity,
    )


def _make_fake_audio(sr: int = 22050, duration: float = 1.0) -> np.ndarray:
    """합성 sine 파를 float32 mono 배열로 반환."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_fake_model(num_classes: int = 10, winning_class: int = 2) -> MagicMock:
    """deterministic logits 를 반환하는 fake torch 모델."""
    logits = torch.zeros(num_classes)
    logits[winning_class] = 10.0  # 확실히 winning_class 가 argmax

    model = MagicMock()
    model.return_value = logits  # model(feature) → logits
    model.eval.return_value = model
    return model


# ---------------------------------------------------------------------------
# Test 1 — weights_path=None → empty list
# ---------------------------------------------------------------------------

def test_classify_no_weights_returns_empty_list():
    """weights_path=None 일 때 classify 는 항상 빈 리스트를 반환해야 한다.

    TART 가중치가 공개되지 않았으므로 이것은 의도된 폴백 동작이다.
    """
    classifier = TARTTechniqueClassifier(weights_path=None)
    event = _make_midi_event()
    result = classifier.classify([event], Path("fake.wav"))
    assert result == []


def test_classify_no_weights_ignores_audio_path():
    """weights_path=None 이면 audio_path 존재 여부와 무관하게 [] 반환."""
    classifier = TARTTechniqueClassifier()
    events = [_make_midi_event(pitch=p) for p in [60, 64, 67]]
    result = classifier.classify(events, Path("nonexistent_file.wav"))
    assert result == []


# ---------------------------------------------------------------------------
# Test 2 — weights_path provided, fake model with deterministic logits
# ---------------------------------------------------------------------------

def test_classify_with_weights_returns_annotations(monkeypatch, tmp_path):
    """weights_path 가 주어지면 MidiEvent 와 1:1 대응하는 TechniqueAnnotation 반환."""
    fake_model = _make_fake_model(winning_class=0)  # "bend"
    fake_weights_file = tmp_path / "fake_weights.pt"
    fake_weights_file.write_bytes(b"fake")  # 실제 내용 불필요 (torch.load 패치)

    # torch.load 는 state_dict 를 반환하도록 패치; model_factory 는 fake_model 반환
    fake_state_dict: dict = {}
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: fake_state_dict)
    fake_model.load_state_dict = MagicMock()

    # 오디오 로딩 패치
    audio_array = _make_fake_audio()
    monkeypatch.setattr(technique_mod, "_load_audio", lambda path: (audio_array, 22050))

    events = [
        _make_midi_event(pitch=60, start_time=0.0, end_time=0.2),
        _make_midi_event(pitch=64, start_time=0.2, end_time=0.4),
        _make_midi_event(pitch=67, start_time=0.4, end_time=0.6),
    ]

    classifier = TARTTechniqueClassifier(
        weights_path=fake_weights_file,
        model_factory=lambda: fake_model,
    )
    result = classifier.classify(events, Path("fake.wav"))

    # 길이가 입력과 일치해야 한다
    assert len(result) == len(events)

    # 각 항목이 TechniqueAnnotation 이고 source=="audio" 여야 한다
    for ann in result:
        assert isinstance(ann, TechniqueAnnotation)
        assert ann.source == "audio"
        assert 0.0 <= ann.confidence <= 1.0
        assert ann.technique == "bend"  # winning_class=0 → "bend"


def test_classify_with_weights_source_is_audio(monkeypatch, tmp_path):
    """반환된 모든 TechniqueAnnotation 의 source 는 'audio' 여야 한다."""
    fake_model = _make_fake_model(winning_class=4)  # "vibrato"
    fake_weights_file = tmp_path / "w.pt"
    fake_weights_file.write_bytes(b"x")

    fake_state_dict: dict = {}
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: fake_state_dict)
    fake_model.load_state_dict = MagicMock()
    audio_array = _make_fake_audio()
    monkeypatch.setattr(technique_mod, "_load_audio", lambda path: (audio_array, 22050))

    events = [_make_midi_event() for _ in range(5)]
    classifier = TARTTechniqueClassifier(
        weights_path=fake_weights_file,
        model_factory=lambda: fake_model,
    )
    result = classifier.classify(events, Path("fake.wav"))

    assert all(ann.source == "audio" for ann in result)
    assert all(ann.technique == "vibrato" for ann in result)


# ---------------------------------------------------------------------------
# Test 3 — monkeypatched audio loading (no real WAV needed)
# ---------------------------------------------------------------------------

def test_classify_with_patched_audio_load(monkeypatch, tmp_path):
    """_load_audio 를 패치하여 합성 배열을 주입 — 실제 WAV 파일 불필요."""
    fake_model = _make_fake_model(winning_class=1)  # "slide"
    fake_weights_file = tmp_path / "w2.pt"
    fake_weights_file.write_bytes(b"y")

    fake_state_dict: dict = {}
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: fake_state_dict)
    fake_model.load_state_dict = MagicMock()

    # 실제 파일 대신 합성 배열 주입
    synthetic_audio = _make_fake_audio(sr=16000, duration=2.0)
    monkeypatch.setattr(technique_mod, "_load_audio", lambda path: (synthetic_audio, 16000))

    events = [
        _make_midi_event(pitch=69, start_time=0.0, end_time=0.3),
        _make_midi_event(pitch=71, start_time=0.5, end_time=0.8),
    ]

    classifier = TARTTechniqueClassifier(
        weights_path=fake_weights_file,
        model_factory=lambda: fake_model,
    )
    result = classifier.classify(events, Path("no_real_file.wav"))

    assert len(result) == 2
    for ann in result:
        assert isinstance(ann, TechniqueAnnotation)
        assert ann.source == "audio"
        assert ann.technique == "slide"


# ---------------------------------------------------------------------------
# Test 4 — _extract_feature returns expected tensor shape
# ---------------------------------------------------------------------------

def test_extract_feature_returns_correct_shape():
    """_extract_feature 는 (1, n_mels, frames) shape 의 Tensor 를 반환해야 한다."""
    audio = _make_fake_audio(sr=22050, duration=1.0)
    tensor = _extract_feature(audio, sr=22050, start_time=0.1, end_time=0.4, n_mels=64)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 1   # batch dim
    assert tensor.shape[1] == 64  # n_mels


def test_extract_feature_zero_length_segment():
    """시작/종료 시각이 동일한 경우 영 텐서(shape 고정)를 반환해야 한다."""
    audio = _make_fake_audio(sr=22050, duration=1.0)
    tensor = _extract_feature(audio, sr=22050, start_time=0.5, end_time=0.5, n_mels=32)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 32, 1)


# ---------------------------------------------------------------------------
# Test 5 — empty midi_events list
# ---------------------------------------------------------------------------

def test_classify_empty_events_returns_empty(monkeypatch, tmp_path):
    """midi_events 가 빈 리스트면 결과도 빈 리스트여야 한다 (weights 있어도)."""
    fake_model = _make_fake_model()
    fake_weights_file = tmp_path / "w3.pt"
    fake_weights_file.write_bytes(b"z")

    fake_state_dict: dict = {}
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: fake_state_dict)
    fake_model.load_state_dict = MagicMock()
    monkeypatch.setattr(technique_mod, "_load_audio", lambda path: (_make_fake_audio(), 22050))

    classifier = TARTTechniqueClassifier(
        weights_path=fake_weights_file,
        model_factory=lambda: fake_model,
    )
    result = classifier.classify([], Path("fake.wav"))
    assert result == []
