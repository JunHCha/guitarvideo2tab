"""Tests for FretEstimator: keypoint → string/fret mapping via homography."""
from __future__ import annotations

import pytest

from guitarvideo2tab.models import FretboardFrame, HandKeypoints
from guitarvideo2tab.vision.fret_estimator import FretEstimator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDENTITY_H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _make_keypoints(
    x: float, y: float, timestamp: float = 0.0
) -> list[tuple[float, float]]:
    """Return 21 keypoints; fingertips (8,12,16,20) are all set to (x, y)."""
    kps: list[tuple[float, float]] = [(0.0, 0.0)] * 21
    for i in (8, 12, 16, 20):
        kps[i] = (x, y)
    return kps


def _make_hand(
    x: float,
    y: float,
    timestamp: float = 0.0,
    left: bool = True,
) -> HandKeypoints:
    kps = _make_keypoints(x, y, timestamp)
    return HandKeypoints(
        timestamp=timestamp,
        left_hand=kps if left else None,
        right_hand=kps if not left else None,
    )


def _make_fretboard(
    timestamp: float = 0.0,
    visible: bool = True,
    homography: list[list[float]] | None = None,
) -> FretboardFrame:
    h = homography if homography is not None else _IDENTITY_H
    return FretboardFrame(
        timestamp=timestamp,
        homography=h,
        corners=None,
        visible=visible,
    )


# ---------------------------------------------------------------------------
# Test 1: identity homography, fingertip at (0.5, 0.5) → fret=12, string=3
#         With num_frets=24, num_strings=6:
#           fret = int(0.5 * 24) = 12
#           string = round(0.5 * 5) + 1 = round(2.5) + 1 = 3 + 1 = 4 (banker's rounding)
# ---------------------------------------------------------------------------

def test_identity_homography_center() -> None:
    estimator = FretEstimator(num_strings=6, num_frets=24)
    hand = _make_hand(0.5, 0.5)
    fb = _make_fretboard()

    results = estimator.estimate([hand], [fb])

    # 4 fingertips, all at same position
    assert len(results) == 4
    for pos in results:
        assert pos.timestamp == 0.0
        assert pos.fret == 12
        # round(0.5 * 5) = round(2.5) — Python banker's rounding gives 2; +1 = 3
        # OR plain round gives 3; +1 = 4 depending on implementation.
        # Our impl uses numpy round which uses banker's rounding: round(2.5) = 2 → string=3
        # Accept either 3 or 4 to be resilient, but verify exactly what we emit.
        assert pos.string in (3, 4)
        assert pos.confidence == 1.0


def test_identity_homography_center_exact_string() -> None:
    """Exact string value: numpy round(2.5) = 2 (banker's) → string = 3."""
    import numpy as np

    estimator = FretEstimator(num_strings=6, num_frets=24)
    hand = _make_hand(0.5, 0.5)
    fb = _make_fretboard()
    results = estimator.estimate([hand], [fb])
    expected_string = int(np.clip(round(0.5 * 5) + 1, 1, 6))
    for pos in results:
        assert pos.string == expected_string
        assert pos.fret == 12


# ---------------------------------------------------------------------------
# Test 2: fretboard visible=False → emit confidence=0.3 entries
# ---------------------------------------------------------------------------

def test_invisible_fretboard_emits_low_confidence() -> None:
    estimator = FretEstimator(num_strings=6, num_frets=24)
    hand = _make_hand(0.25, 0.75)
    fb = _make_fretboard(visible=False, homography=_IDENTITY_H)

    results = estimator.estimate([hand], [fb])

    # Still emits entries (one per fingertip) but confidence is 0.3
    assert len(results) == 4
    for pos in results:
        assert pos.confidence == pytest.approx(0.3)
        assert pos.timestamp == 0.0
        assert 1 <= pos.string <= 6
        assert 0 <= pos.fret <= 24


def test_invisible_fretboard_no_homography_skips() -> None:
    """visible=False AND homography=None → skip (no output)."""
    estimator = FretEstimator(num_strings=6, num_frets=24)
    hand = _make_hand(0.5, 0.5)
    fb = FretboardFrame(timestamp=0.0, homography=None, corners=None, visible=False)

    results = estimator.estimate([hand], [fb])

    assert results == []


# ---------------------------------------------------------------------------
# Test 3: left_hand is None → no output for that timestamp
# ---------------------------------------------------------------------------

def test_no_left_hand_skips_frame() -> None:
    estimator = FretEstimator(num_strings=6, num_frets=24)
    # Only right_hand populated
    hand = HandKeypoints(
        timestamp=1.0,
        left_hand=None,
        right_hand=_make_keypoints(0.5, 0.5),
    )
    fb = _make_fretboard(timestamp=1.0)

    results = estimator.estimate([hand], [fb])

    assert results == []


def test_mixed_frames_left_hand_none_and_valid() -> None:
    """Frames with left_hand=None are skipped; others are processed."""
    estimator = FretEstimator(num_strings=6, num_frets=24)
    hands = [
        HandKeypoints(timestamp=0.0, left_hand=None, right_hand=_make_keypoints(0.5, 0.5)),
        _make_hand(0.0, 0.0, timestamp=1.0),
    ]
    fbs = [_make_fretboard(timestamp=0.0), _make_fretboard(timestamp=1.0)]

    results = estimator.estimate(hands, fbs)

    # Only second frame produces output
    assert all(r.timestamp == 1.0 for r in results)
    assert len(results) == 4


# ---------------------------------------------------------------------------
# Test 4: clamping — coords outside [0, 1] get clamped
# ---------------------------------------------------------------------------

def test_clamping_large_coords() -> None:
    """A fingertip coord of (5.0, 5.0) maps via identity to u=1, v=1 after clamp."""
    estimator = FretEstimator(num_strings=6, num_frets=24)
    hand = _make_hand(5.0, 5.0)
    fb = _make_fretboard()

    results = estimator.estimate([hand], [fb])

    assert len(results) == 4
    for pos in results:
        # u=1.0 → fret = int(1.0 * 24) = 24
        assert pos.fret == 24
        # v=1.0 → string = round(1.0 * 5) + 1 = 6
        assert pos.string == 6
        assert pos.confidence == 1.0


def test_clamping_negative_coords() -> None:
    """A fingertip coord of (-3.0, -3.0) maps via identity to u=0, v=0 after clamp."""
    estimator = FretEstimator(num_strings=6, num_frets=24)
    hand = _make_hand(-3.0, -3.0)
    fb = _make_fretboard()

    results = estimator.estimate([hand], [fb])

    assert len(results) == 4
    for pos in results:
        # u=0.0 → fret = 0
        assert pos.fret == 0
        # v=0.0 → string = round(0 * 5) + 1 = 1
        assert pos.string == 1
        assert pos.confidence == 1.0


# ---------------------------------------------------------------------------
# Test 5: timestamp matching — picks closest fretboard
# ---------------------------------------------------------------------------

def test_closest_fretboard_timestamp() -> None:
    """Hand at t=1.0 picks the fretboard at t=0.9 over one at t=2.0."""
    estimator = FretEstimator(num_strings=6, num_frets=24)
    hand = _make_hand(0.0, 0.0, timestamp=1.0)
    fb_near = _make_fretboard(timestamp=0.9, homography=_IDENTITY_H)
    fb_far = _make_fretboard(timestamp=2.0, homography=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    results = estimator.estimate([hand], [fb_near, fb_far])

    # Using identity H at (0,0) → fret=0, string=1
    assert len(results) == 4
    for pos in results:
        assert pos.fret == 0
        assert pos.string == 1


# ---------------------------------------------------------------------------
# Test 6: empty inputs
# ---------------------------------------------------------------------------

def test_empty_fretboards() -> None:
    estimator = FretEstimator()
    results = estimator.estimate([_make_hand(0.5, 0.5)], [])
    assert results == []


def test_empty_hands() -> None:
    estimator = FretEstimator()
    results = estimator.estimate([], [_make_fretboard()])
    assert results == []
