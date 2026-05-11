"""YOLOv8-OBB 프렛보드 검출 + 호모그래피 워핑."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ..models import FretboardFrame

_DST_CANONICAL = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


@dataclass
class FretboardDetector:
    weights_path: Path | None = None
    confidence_threshold: float = 0.5

    def detect(self, video_path: Path) -> list[FretboardFrame]:
        model = YOLO(str(self.weights_path) if self.weights_path is not None else "yolov8n-obb.pt")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0

        frames: list[FretboardFrame] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps
                fretboard_frame = self._process_frame(model, frame, timestamp)
                frames.append(fretboard_frame)
                frame_idx += 1
        finally:
            cap.release()

        return frames

    def _process_frame(self, model: YOLO, frame: np.ndarray, timestamp: float) -> FretboardFrame:
        results = model(frame, verbose=False)

        obb = results[0].obb if results and results[0].obb is not None else None

        if obb is None:
            return FretboardFrame(timestamp=timestamp, homography=None, corners=None, visible=False)

        # Pick the detection with highest confidence
        confidences = obb.conf
        if len(confidences) == 0:
            return FretboardFrame(timestamp=timestamp, homography=None, corners=None, visible=False)

        best_idx = int(confidences.argmax())

        if float(confidences[best_idx]) < self.confidence_threshold:
            return FretboardFrame(timestamp=timestamp, homography=None, corners=None, visible=False)

        corners_tensor = obb.xyxyxyxy[best_idx]  # shape (4, 2)
        corners = corners_tensor.cpu().numpy().astype(np.float32)

        homography_matrix, _ = cv2.findHomography(corners, _DST_CANONICAL)

        if homography_matrix is None:
            return FretboardFrame(timestamp=timestamp, homography=None, corners=None, visible=False)

        return FretboardFrame(
            timestamp=timestamp,
            homography=homography_matrix.tolist(),
            corners=corners.tolist(),
            visible=True,
        )
