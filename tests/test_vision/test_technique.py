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
    """Return a mock nn.Module whose __call__ returns fixed logits."""
    logits = torch.zeros(1, _N_CLASSES)
    logits[0, label_idx] = 10.0  # argmax → label_idx

    model = MagicMock(spec=torch.nn.Module)
    model.eval.return_value = model
    model.return_value = logits
    model.__call__ = MagicMock(return_value=logits)
    return model


# ---------------------------------------------------------------------------
# Test 1: weights_path=None → []
# ---------------------------------------------------------------------------

def test_no_weights_returns_empty() -> None:
    clf = VisionTechniqueClassifier(weights_path=None)
    hands = _make_hands(5)
    result = clf.classify(hands)
    assert result == [], "Expected empty list when weights_path is None"


# ---------------------------------------------------------------------------
# Test 2: monkeypatched torch.load → list of TechniqueAnnotation
# ---------------------------------------------------------------------------

def test_with_fake_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    label_idx = 3  # e.g. "pull-off"
    expected_label = _LABELS[label_idx]

    fake_weights = tmp_path / "model.pt"
    fake_weights.write_bytes(b"")  # empty sentinel file

    fake_model = _fake_model(label_idx)

    # Patch torch.load so no real file is read
    monkeypatch.setattr(
        "guitarvideo2tab.vision.technique.torch.load",
        lambda path, map_location=None: fake_model,
    )

    clf = VisionTechniqueClassifier(weights_path=fake_weights, window_ms=300)
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
    fake_weights = tmp_path / "model.pt"
    fake_weights.write_bytes(b"")

    monkeypatch.setattr(
        "guitarvideo2tab.vision.technique.torch.load",
        lambda path, map_location=None: _fake_model(0),
    )

    clf = VisionTechniqueClassifier(weights_path=fake_weights)
    result = clf.classify([])
    assert result == [], "Expected empty list for empty hands input"
