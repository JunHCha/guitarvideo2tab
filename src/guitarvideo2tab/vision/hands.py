"""MediaPipe Hands로 좌·우손 21 keypoint 시계열 추출."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp

from ..models import HandKeypoints


def _make_hands_solution(
    min_detection_confidence: float,
    min_tracking_confidence: float,
    model_asset_path: Path | None = None,
):
    """Create and return a mediapipe Hands-compatible object.

    Wraps the API difference between mediapipe < 0.10 (solutions) and
    >= 0.10 (tasks). Returns an object with .process(rgb) -> result and
    .close() methods, where result has .multi_hand_landmarks and
    .multi_handedness matching the legacy solutions contract.
    """
    # Legacy solutions API (mediapipe < 0.10)
    try:
        solutions_hands = mp.solutions.hands  # type: ignore[attr-defined]
        return solutions_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass

    # Modern Tasks API (mediapipe >= 0.10)
    return _TasksHandsAdapter(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_asset_path=model_asset_path,
    )


class _TasksHandsAdapter:
    """Thin adapter that wraps the Tasks API to look like the legacy API."""

    def __init__(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_asset_path: Path | None = None,
    ) -> None:
        if model_asset_path is None:
            raise FileNotFoundError(
                "MediaPipe Tasks API requires a .task model file. "
                "Set HandTracker(model_asset_path=...)"
            )

        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions  # type: ignore[attr-defined]

        options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(model_asset_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        mp_image = mp.Image(  # type: ignore[attr-defined]
            image_format=mp.ImageFormat.SRGB,  # type: ignore[attr-defined]
            data=rgb_frame,
        )
        tasks_result = self._landmarker.detect(mp_image)
        return _adapt_tasks_result(tasks_result)

    def close(self) -> None:
        self._landmarker.close()


def _adapt_tasks_result(tasks_result) -> SimpleNamespace:
    """Convert Tasks API result to legacy solutions-style namespace."""
    if not tasks_result.hand_landmarks:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)

    multi_hand_landmarks = []
    for lm_list in tasks_result.hand_landmarks:
        landmark_ns = [SimpleNamespace(x=lm.x, y=lm.y) for lm in lm_list]
        multi_hand_landmarks.append(SimpleNamespace(landmark=landmark_ns))

    multi_handedness = []
    for handedness_list in tasks_result.handedness:
        label = handedness_list[0].category_name  # "Left" / "Right"
        classification = [SimpleNamespace(label=label)]
        multi_handedness.append(SimpleNamespace(classification=classification))

    return SimpleNamespace(
        multi_hand_landmarks=multi_hand_landmarks,
        multi_handedness=multi_handedness,
    )


@dataclass
class HandTracker:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_asset_path: Path | None = None

    def track(self, video_path: Path) -> list[HandKeypoints]:
        hands = _make_hands_solution(
            self.min_detection_confidence,
            self.min_tracking_confidence,
            self.model_asset_path,
        )
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(
                f"Invalid FPS value ({fps}) reported by VideoCapture for {video_path}. "
                "Ensure the file is a valid video."
            )

        results_list: list[HandKeypoints] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                left_hand: list[tuple[float, float]] | None = None
                right_hand: list[tuple[float, float]] | None = None

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        label = handedness.classification[0].label
                        keypoints = [
                            (lm.x, lm.y) for lm in hand_landmarks.landmark
                        ]
                        if label == "Left":
                            left_hand = keypoints
                        elif label == "Right":
                            right_hand = keypoints

                timestamp = frame_idx / fps
                results_list.append(
                    HandKeypoints(
                        timestamp=timestamp,
                        left_hand=left_hand,
                        right_hand=right_hand,
                    )
                )
                frame_idx += 1
        finally:
            hands.close()
            cap.release()

        return results_list
