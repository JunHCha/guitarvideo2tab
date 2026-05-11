"""Unit tests for FretboardDetector (mock-based, no real model or video)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from guitarvideo2tab.models import FretboardFrame
from guitarvideo2tab.vision.fretboard import FretboardDetector

# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

_FPS = 30.0
_FAKE_CORNERS = np.array(
    [[10.0, 20.0], [110.0, 20.0], [110.0, 80.0], [10.0, 80.0]], dtype=np.float32
)
_FAKE_H = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def _make_fake_obb(conf_value: float) -> MagicMock:
    """Return a fake OBB object with one detection at *conf_value* confidence."""
    obb = MagicMock()

    # conf is an array-like; argmax() returns 0
    conf_tensor = MagicMock()
    conf_tensor.__len__ = lambda self: 1
    conf_tensor.argmax.return_value = 0
    conf_tensor.__getitem__ = lambda self, idx: conf_value
    # float() of the indexed item
    obb.conf = conf_tensor

    # xyxyxyxy[0] returns a tensor whose .cpu().numpy() gives _FAKE_CORNERS
    corners_tensor = MagicMock()
    corners_tensor.cpu.return_value.numpy.return_value = _FAKE_CORNERS.copy()
    obb.xyxyxyxy = [corners_tensor]

    return obb


def _make_fake_result(conf_value: float | None) -> MagicMock:
    """Return a fake ultralytics result. If conf_value is None → no detections."""
    result = MagicMock()
    if conf_value is None:
        result.obb = None
    else:
        result.obb = _make_fake_obb(conf_value)
    return result


class FakeYOLO:
    """Drop-in replacement for ultralytics.YOLO."""

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self._results: list = []  # injected per-test

    def __call__(self, frame: np.ndarray):
        return self._results.pop(0) if self._results else [_make_fake_result(None)]


def _make_fake_cap(frames: list[np.ndarray], fps: float = _FPS):
    """Return a fake cv2.VideoCapture that yields *frames* then stops."""

    class FakeCap:
        def __init__(self, path: str) -> None:
            self._frames = list(frames)
            self._fps = fps

        def get(self, prop_id: int) -> float:
            import cv2

            if prop_id == cv2.CAP_PROP_FPS:
                return self._fps
            return 0.0

        def read(self):
            if self._frames:
                return True, self._frames.pop(0)
            return False, None

        def release(self) -> None:
            pass

    return FakeCap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_visible_frame(monkeypatch):
    """A frame whose detection confidence exceeds threshold → visible=True."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fake_yolo = FakeYOLO("yolov8n-obb.pt")
    fake_yolo._results = [[_make_fake_result(0.9)]]

    monkeypatch.setattr("guitarvideo2tab.vision.fretboard.YOLO", lambda w: fake_yolo)
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.VideoCapture",
        _make_fake_cap([fake_frame]),
    )
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.findHomography",
        lambda src, dst: (_FAKE_H.copy(), None),
    )

    detector = FretboardDetector(confidence_threshold=0.5)
    results = detector.detect(Path("dummy.mp4"))

    assert len(results) == 1
    frame = results[0]
    assert isinstance(frame, FretboardFrame)
    assert frame.visible is True
    assert frame.homography is not None
    assert frame.corners is not None
    # homography must be a list-of-lists (3×3)
    assert len(frame.homography) == 3
    assert all(len(row) == 3 for row in frame.homography)
    # corners must be a list with 4 items
    assert len(frame.corners) == 4


def test_detect_invisible_frame_no_detection(monkeypatch):
    """A frame with no OBB detection → visible=False, homography/corners None."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fake_yolo = FakeYOLO("yolov8n-obb.pt")
    fake_yolo._results = [[_make_fake_result(None)]]

    monkeypatch.setattr("guitarvideo2tab.vision.fretboard.YOLO", lambda w: fake_yolo)
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.VideoCapture",
        _make_fake_cap([fake_frame]),
    )

    detector = FretboardDetector(confidence_threshold=0.5)
    results = detector.detect(Path("dummy.mp4"))

    assert len(results) == 1
    frame = results[0]
    assert frame.visible is False
    assert frame.homography is None
    assert frame.corners is None


def test_detect_invisible_frame_low_confidence(monkeypatch):
    """A detection below confidence_threshold → visible=False."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fake_yolo = FakeYOLO("yolov8n-obb.pt")
    fake_yolo._results = [[_make_fake_result(0.3)]]  # below 0.5 threshold

    monkeypatch.setattr("guitarvideo2tab.vision.fretboard.YOLO", lambda w: fake_yolo)
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.VideoCapture",
        _make_fake_cap([fake_frame]),
    )

    detector = FretboardDetector(confidence_threshold=0.5)
    results = detector.detect(Path("dummy.mp4"))

    assert len(results) == 1
    assert results[0].visible is False
    assert results[0].homography is None


def test_detect_timestamp_computed_as_frame_idx_over_fps(monkeypatch):
    """Timestamps are frame_index / fps for each frame."""
    fps = 24.0
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

    # Pre-build results: frame 0 → no detection, frame 1 → detected, frame 2 → no detection
    per_frame_results = [
        [_make_fake_result(None)],
        [_make_fake_result(0.9)],
        [_make_fake_result(None)],
    ]

    class SequentialYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights
            self._queue = list(per_frame_results)

        def __call__(self, frame: np.ndarray):
            return self._queue.pop(0)

    monkeypatch.setattr("guitarvideo2tab.vision.fretboard.YOLO", SequentialYOLO)
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.VideoCapture",
        _make_fake_cap(frames, fps=fps),
    )
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.findHomography",
        lambda src, dst: (_FAKE_H.copy(), None),
    )

    detector = FretboardDetector(confidence_threshold=0.5)
    results = detector.detect(Path("dummy.mp4"))

    assert len(results) == 3
    assert results[0].timestamp == pytest.approx(0.0 / fps)
    assert results[1].timestamp == pytest.approx(1.0 / fps)
    assert results[2].timestamp == pytest.approx(2.0 / fps)

    # frame 0 and 2 → invisible; frame 1 → visible
    assert results[0].visible is False
    assert results[1].visible is True
    assert results[2].visible is False


def test_detect_uses_custom_weights(monkeypatch):
    """When weights_path is given, YOLO is initialized with that path string."""
    captured_weights: list[str] = []

    def capturing_yolo(weights: str):
        captured_weights.append(weights)
        yolo = FakeYOLO(weights)
        yolo._results = [[_make_fake_result(None)]]
        return yolo

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    monkeypatch.setattr("guitarvideo2tab.vision.fretboard.YOLO", capturing_yolo)
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.VideoCapture",
        _make_fake_cap([fake_frame]),
    )

    detector = FretboardDetector(weights_path=Path("/custom/weights.pt"))
    detector.detect(Path("dummy.mp4"))

    assert captured_weights == ["/custom/weights.pt"]


def test_detect_default_weights_when_none(monkeypatch):
    """When weights_path is None, YOLO is initialized with yolov8n-obb.pt."""
    captured_weights: list[str] = []

    def capturing_yolo(weights: str):
        captured_weights.append(weights)
        yolo = FakeYOLO(weights)
        yolo._results = [[_make_fake_result(None)]]
        return yolo

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    monkeypatch.setattr("guitarvideo2tab.vision.fretboard.YOLO", capturing_yolo)
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.VideoCapture",
        _make_fake_cap([fake_frame]),
    )

    detector = FretboardDetector()
    detector.detect(Path("dummy.mp4"))

    assert captured_weights == ["yolov8n-obb.pt"]


def test_homography_is_list_of_lists(monkeypatch):
    """homography field must be a plain Python list-of-lists, not numpy array."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fake_yolo = FakeYOLO("yolov8n-obb.pt")
    fake_yolo._results = [[_make_fake_result(0.8)]]

    monkeypatch.setattr("guitarvideo2tab.vision.fretboard.YOLO", lambda w: fake_yolo)
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.VideoCapture",
        _make_fake_cap([fake_frame]),
    )
    monkeypatch.setattr(
        "guitarvideo2tab.vision.fretboard.cv2.findHomography",
        lambda src, dst: (_FAKE_H.copy(), None),
    )

    results = FretboardDetector().detect(Path("dummy.mp4"))
    homography = results[0].homography

    assert isinstance(homography, list)
    assert all(isinstance(row, list) for row in homography)
    assert all(isinstance(v, float) for row in homography for v in row)
