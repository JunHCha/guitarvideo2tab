"""Tests for VisionTechniqueClassifier."""
from __future__ import annotations

from pathlib import Path
from typing import get_args
from unittest.mock import MagicMock

import pytest
import torch

from guitarvideo2tab.models import HandKeypoints, TechniqueAnnotation, TechniqueLabel
from guitarvideo2tab.vision.technique import VisionTechniqueClassifier

# All valid technique labels
_LABELS = list(get_args(TechniqueLabel))
_N_CLASSES = len(_LABELS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hands(n: int = 10, dt: float = 0.033) -> list[HandKeypoints]:
    """Return n HandKeypoints with simple synthetic keypoints."""
    kp_21 = [(float(j), float(j)) for j in range(21)]
    return [
        HandKeypoints(
            timestamp=i * dt,
            left_hand=kp_21,
            right_hand=kp_21,
        )
        for i in range(n)
    ]


def _fake_model(label_idx: int) -> MagicMock:
    """Return a mock nn.Module whose __call__ returns fixed logits (B, C)."""
    logits = torch.zeros(1, _N_CLASSES)
    logits[0, label_idx] = 10.0  # argmax → label_idx

    model = MagicMock(spec=torch.nn.Module)
    model.eval.return_value = model
    model.load_state_dict.return_value = None
    model.return_value = logits
    model.__call__ = MagicMock(return_value=logits)
    return model


def _make_state_dict_fixture(tmp_path: Path, label_idx: int, monkeypatch: pytest.MonkeyPatch):
    """Write a fake state-dict file and patch torch.load + model_factory."""
    fake_weights = tmp_path / "model.pt"
    fake_state_dict: dict = {}
    fake_weights.write_bytes(b"")  # sentinel — torch.load is patched

    fake_model = _fake_model(label_idx)

    monkeypatch.setattr(
        "guitarvideo2tab.vision.technique.torch.load",
        lambda path, map_location=None, weights_only=False: fake_state_dict,
    )

    def factory() -> torch.nn.Module:
        return fake_model

    return fake_weights, factory, fake_model


# ---------------------------------------------------------------------------
# Test 1: weights_path=None → []
# ---------------------------------------------------------------------------

def test_no_weights_returns_empty() -> None:
    clf = VisionTechniqueClassifier(weights_path=None)
    hands = _make_hands(5)
    result = clf.classify(hands)
    assert result == [], "Expected empty list when weights_path is None"


# ---------------------------------------------------------------------------
# Test 2: model_factory + state_dict path → list of TechniqueAnnotation
# ---------------------------------------------------------------------------

def test_with_fake_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    label_idx = 3  # e.g. "pull-off"
    expected_label = _LABELS[label_idx]

    fake_weights, factory, _ = _make_state_dict_fixture(tmp_path, label_idx, monkeypatch)

    clf = VisionTechniqueClassifier(
        weights_path=fake_weights,
        model_factory=factory,
        window_ms=300,
    )
    hands = _make_hands(n=20, dt=0.020)  # 20 frames × 20ms = 400ms → 2 windows
    result = clf.classify(hands)

    assert isinstance(result, list)
    assert len(result) >= 1, "Expected at least one annotation"

    for ann in result:
        assert isinstance(ann, TechniqueAnnotation)
        assert ann.source == "vision", f"source must be 'vision', got {ann.source!r}"
        assert ann.technique in _LABELS, f"Unknown label: {ann.technique!r}"
        assert ann.technique == expected_label
        assert 0.0 <= ann.confidence <= 1.0
        assert "window_start" in ann.params
        assert "window_end" in ann.params


# ---------------------------------------------------------------------------
# Test 3: empty hands → []
# ---------------------------------------------------------------------------

def test_empty_hands_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_weights, factory, _ = _make_state_dict_fixture(tmp_path, 0, monkeypatch)

    clf = VisionTechniqueClassifier(weights_path=fake_weights, model_factory=factory)
    result = clf.classify([])
    assert result == [], "Expected empty list for empty hands input"


# ---------------------------------------------------------------------------
# Test 4: no model_factory with weights_path → ValueError
# ---------------------------------------------------------------------------

def test_missing_factory_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_weights = tmp_path / "model.pt"
    fake_weights.write_bytes(b"")

    monkeypatch.setattr(
        "guitarvideo2tab.vision.technique.torch.load",
        lambda path, map_location=None, weights_only=False: {},
    )

    clf = VisionTechniqueClassifier(weights_path=fake_weights, model_factory=None)
    with pytest.raises(ValueError, match="model_factory"):
        clf.classify(_make_hands(5))


# ---------------------------------------------------------------------------
# Test 5: unsorted hands → still works (timestamp sort guard)
# ---------------------------------------------------------------------------

def test_unsorted_hands_are_handled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    label_idx = 1
    fake_weights, factory, _ = _make_state_dict_fixture(tmp_path, label_idx, monkeypatch)

    clf = VisionTechniqueClassifier(
        weights_path=fake_weights,
        model_factory=factory,
        window_ms=300,
    )
    # Deliberately shuffle timestamps
    hands = _make_hands(n=10, dt=0.020)
    hands_shuffled = list(reversed(hands))
    result = clf.classify(hands_shuffled)

    assert isinstance(result, list)
    # Should produce same result as sorted input (no crash, valid labels)
    for ann in result:
        assert ann.technique in _LABELS
