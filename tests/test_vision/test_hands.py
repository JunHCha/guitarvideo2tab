"""Unit tests for HandTracker (mock-based, no real model/video loading)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import guitarvideo2tab.vision.hands as hands_module
from guitarvideo2tab.models import HandKeypoints
from guitarvideo2tab.vision.hands import HandTracker

# ---------------------------------------------------------------------------
# Helpers — fake mediapipe result objects
# ---------------------------------------------------------------------------

def _make_landmark(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def _make_hand_landmarks(n: int = 21) -> SimpleNamespace:
    landmarks = [_make_landmark(float(i) / 100, float(i) / 200) for i in range(n)]
    return SimpleNamespace(landmark=landmarks)


def _make_handedness(label: str) -> SimpleNamespace:
    classification = [SimpleNamespace(label=label)]
    return SimpleNamespace(classification=classification)


def _make_mp_result(hands_info: list[tuple[str, int]] | None) -> SimpleNamespace:
    """Build a fake mediapipe Hands.process() result.

    hands_info: list of (label, n_landmarks) or None for no detection.
    """
    if not hands_info:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)

    return SimpleNamespace(
        multi_hand_landmarks=[_make_hand_landmarks(n) for _, n in hands_info],
        multi_handedness=[_make_handedness(label) for label, _ in hands_info],
    )


# ---------------------------------------------------------------------------
# Fake cv2 capture
# ---------------------------------------------------------------------------

_DUMMY_FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


class FakeCapture:
    """Fake cv2.VideoCapture that yields a fixed list of frames."""

    def __init__(self, frames: list, fps: float = 30.0) -> None:
        self._frames = list(frames)
        self._fps = fps
        self._idx = 0
        self.released = False

    def get(self, prop_id: int) -> float:
        import cv2
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def read(self):
        if self._idx < len(self._frames):
            frame = self._frames[self._idx]
            self._idx += 1
            return True, frame
        return False, None

    def release(self) -> None:
        self.released = True


# ---------------------------------------------------------------------------
# Fake mediapipe Hands object (returned by _make_hands_solution)
# ---------------------------------------------------------------------------

class FakeHands:
    """Fake hands tracker with pre-baked per-frame results."""

    def __init__(self, per_frame_results: list, init_kwargs: dict | None = None) -> None:
        self._results = per_frame_results
        self._call_count = 0
        self.closed = False
        self.init_kwargs: dict = init_kwargs or {}

    def process(self, image):
        if self._call_count < len(self._results):
            result = self._results[self._call_count]
        else:
            result = _make_mp_result(None)
        self._call_count += 1
        return result

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _patch_cv2(monkeypatch, frames, fps: float = 30.0) -> FakeCapture:
    """Replace cv2.VideoCapture and cv2.cvtColor in the hands module."""
    fake_cap = FakeCapture(frames, fps=fps)
    monkeypatch.setattr(hands_module.cv2, "VideoCapture", lambda path: fake_cap)
    monkeypatch.setattr(hands_module.cv2, "cvtColor", lambda frame, code: frame)
    return fake_cap


def _patch_hands_factory(
    monkeypatch,
    per_frame_results: list,
    captured_kwargs: dict | None = None,
) -> FakeHands:
    """Replace _make_hands_solution so no real mediapipe model is created."""
    fake = FakeHands(per_frame_results)

    def fake_factory(min_detection_confidence: float, min_tracking_confidence: float):
        if captured_kwargs is not None:
            captured_kwargs["min_detection_confidence"] = min_detection_confidence
            captured_kwargs["min_tracking_confidence"] = min_tracking_confidence
        return fake

    monkeypatch.setattr(hands_module, "_make_hands_solution", fake_factory)
    return fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_right_hand_only_frame(monkeypatch):
    """Frame with only right hand: right_hand has 21 tuples, left_hand is None."""
    _patch_cv2(monkeypatch, [_DUMMY_FRAME], fps=30.0)
    _patch_hands_factory(monkeypatch, [_make_mp_result([("Right", 21)])])

    results = HandTracker().track(Path("dummy.mp4"))

    assert len(results) == 1
    kp = results[0]
    assert isinstance(kp, HandKeypoints)
    assert kp.left_hand is None
    assert kp.right_hand is not None
    assert len(kp.right_hand) == 21
    assert all(isinstance(t, tuple) and len(t) == 2 for t in kp.right_hand)


def test_left_hand_only_frame(monkeypatch):
    """Frame with only left hand: left_hand has 21 tuples, right_hand is None."""
    _patch_cv2(monkeypatch, [_DUMMY_FRAME], fps=30.0)
    _patch_hands_factory(monkeypatch, [_make_mp_result([("Left", 21)])])

    results = HandTracker().track(Path("dummy.mp4"))

    assert len(results) == 1
    kp = results[0]
    assert kp.right_hand is None
    assert kp.left_hand is not None
    assert len(kp.left_hand) == 21


def test_no_hands_frame(monkeypatch):
    """Frame with no detected hands: both left_hand and right_hand are None."""
    _patch_cv2(monkeypatch, [_DUMMY_FRAME], fps=30.0)
    _patch_hands_factory(monkeypatch, [_make_mp_result(None)])

    results = HandTracker().track(Path("dummy.mp4"))

    assert len(results) == 1
    kp = results[0]
    assert kp.left_hand is None
    assert kp.right_hand is None


def test_timestamps_equal_frame_idx_over_fps(monkeypatch):
    """Timestamps must equal frame_idx / fps for every frame."""
    fps = 24.0
    n_frames = 5
    _patch_cv2(monkeypatch, [_DUMMY_FRAME] * n_frames, fps=fps)
    _patch_hands_factory(monkeypatch, [_make_mp_result(None)] * n_frames)

    results = HandTracker().track(Path("dummy.mp4"))

    assert len(results) == n_frames
    for idx, kp in enumerate(results):
        assert kp.timestamp == pytest.approx(idx / fps)


def test_both_hands_detected(monkeypatch):
    """Frame with both hands: both sides populated with 21 keypoints."""
    _patch_cv2(monkeypatch, [_DUMMY_FRAME], fps=30.0)
    _patch_hands_factory(monkeypatch, [_make_mp_result([("Left", 21), ("Right", 21)])])

    results = HandTracker().track(Path("dummy.mp4"))

    assert len(results) == 1
    kp = results[0]
    assert kp.left_hand is not None and len(kp.left_hand) == 21
    assert kp.right_hand is not None and len(kp.right_hand) == 21


def test_multiple_frames_mixed_detection(monkeypatch):
    """Multi-frame sequence: hand presence varies per frame."""
    fps = 10.0
    per_frame = [
        _make_mp_result([("Right", 21)]),  # frame 0: right only
        _make_mp_result(None),             # frame 1: no hands
        _make_mp_result([("Left", 21)]),   # frame 2: left only
    ]
    _patch_cv2(monkeypatch, [_DUMMY_FRAME] * 3, fps=fps)
    _patch_hands_factory(monkeypatch, per_frame)

    results = HandTracker().track(Path("dummy.mp4"))

    assert len(results) == 3
    assert results[0].right_hand is not None and results[0].left_hand is None
    assert results[1].right_hand is None and results[1].left_hand is None
    assert results[2].left_hand is not None and results[2].right_hand is None

    assert results[0].timestamp == pytest.approx(0.0)
    assert results[1].timestamp == pytest.approx(0.1)
    assert results[2].timestamp == pytest.approx(0.2)


def test_keypoint_values_are_floats(monkeypatch):
    """Each keypoint coordinate must be a float."""
    _patch_cv2(monkeypatch, [_DUMMY_FRAME], fps=30.0)
    _patch_hands_factory(monkeypatch, [_make_mp_result([("Right", 21)])])

    results = HandTracker().track(Path("dummy.mp4"))

    for x, y in results[0].right_hand:
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_confidence_passed_to_factory(monkeypatch):
    """HandTracker confidence fields are forwarded to _make_hands_solution."""
    captured: dict = {}
    _patch_cv2(monkeypatch, [_DUMMY_FRAME], fps=30.0)
    _patch_hands_factory(monkeypatch, [_make_mp_result(None)], captured_kwargs=captured)

    HandTracker(min_detection_confidence=0.7, min_tracking_confidence=0.8).track(
        Path("dummy.mp4")
    )

    assert captured.get("min_detection_confidence") == pytest.approx(0.7)
    assert captured.get("min_tracking_confidence") == pytest.approx(0.8)
