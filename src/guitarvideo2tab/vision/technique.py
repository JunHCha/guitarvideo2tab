"""TCN/Transformer on hand keypoint 시계열 — 비전 표현 기법 분류기.

Notes
-----
Mitsou 2023 pre-trained weights are not yet publicly released.

* If ``weights_path`` is ``None`` (default) the classifier returns an empty
  list — no labels are produced.  This is the expected behaviour for the
  current inference shell and will be updated once the weights are available.

* If ``weights_path`` is supplied the classifier loads the **state-dict**
  (``weights_only=True``) via ``model_factory``, slides a ``window_ms``-wide
  window over the hand-keypoint time-series, builds a ``(window_len, 21*2*2)``
  feature tensor per window, passes it through the model, and decodes the
  per-window argmax into a ``TechniqueAnnotation``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, get_args

import numpy as np
import torch

from ..models import HandKeypoints, TechniqueAnnotation, TechniqueLabel

# Ordered label list derived from the Literal type — index == logit position.
_LABELS: list[TechniqueLabel] = list(get_args(TechniqueLabel))  # type: ignore[assignment]
_N_KEYPOINTS = 21
_COORDS = 2  # (x, y)
_HANDS = 2   # left + right
_FEATURE_DIM = _N_KEYPOINTS * _COORDS * _HANDS  # 84


def _build_feature(
    frames: list[HandKeypoints],
) -> np.ndarray:
    """Return float32 array of shape ``(T, 84)`` for a list of frames.

    Missing hands are zero-filled so the tensor is always dense.
    """
    n_frames = len(frames)
    feat = np.zeros((n_frames, _FEATURE_DIM), dtype=np.float32)
    for i, kp in enumerate(frames):
        row: list[float] = []
        for hand in (kp.left_hand, kp.right_hand):
            if hand is not None and len(hand) == _N_KEYPOINTS:
                for x, y in hand:
                    row.extend([float(x), float(y)])
            else:
                # Zero-fill missing / malformed hand
                row.extend([0.0] * (_N_KEYPOINTS * _COORDS))
        feat[i] = row
    return feat


@dataclass
class VisionTechniqueClassifier:
    """Classify guitar-playing techniques from hand-keypoint time-series.

    Parameters
    ----------
    weights_path:
        Path to a ``torch.save``'d state-dict (``nn.Module.state_dict()``).
        When ``None`` (default) ``classify`` returns ``[]``; the fallback is
        intentional until Mitsou 2023 model weights become available.
    model_factory:
        Callable that returns a fresh ``nn.Module`` instance.  **Required**
        when ``weights_path`` is not ``None`` — ``_load_model`` calls this to
        instantiate the architecture before loading the state-dict.
    window_ms:
        Sliding-window width in milliseconds (default 300 ms).
    """

    weights_path: Path | None = None
    model_factory: Callable[[], torch.nn.Module] | None = None
    window_ms: int = 300

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> torch.nn.Module:
        assert self.weights_path is not None
        if self.model_factory is None:
            raise ValueError(
                "model_factory must be provided when weights_path is set. "
                "Supply a callable that returns an nn.Module instance."
            )
        state_dict = torch.load(
            self.weights_path, map_location="cpu", weights_only=True
        )
        model = self.model_factory()
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _window_indices(
        self, hands: list[HandKeypoints]
    ) -> list[tuple[int, int]]:
        """Return (start_idx, end_idx) pairs covering hands with window_ms."""
        if not hands:
            return []
        # Ensure frames are in ascending timestamp order before slicing.
        hands = sorted(hands, key=lambda h: h.timestamp)
        window_s = self.window_ms / 1000.0
        t_start = hands[0].timestamp
        t_end = hands[-1].timestamp
        windows: list[tuple[int, int]] = []
        t0 = t_start
        while t0 <= t_end:
            t1 = t0 + window_s
            idxs = [
                i
                for i, kp in enumerate(hands)
                if t0 <= kp.timestamp < t1
            ]
            if idxs:
                windows.append((idxs[0], idxs[-1] + 1))
            t0 = t1
        return windows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, hands: list[HandKeypoints]) -> list[TechniqueAnnotation]:
        """Classify guitar techniques from a sequence of hand keyframes.

        Parameters
        ----------
        hands:
            Time-ordered list of :class:`~guitarvideo2tab.models.HandKeypoints`.

        Returns
        -------
        list[TechniqueAnnotation]
            One annotation per non-empty sliding window, or ``[]`` when
            ``weights_path`` is ``None`` (no-weights fallback).
        """
        # Fallback: weights not yet available — by design.
        if self.weights_path is None:
            return []

        if not hands:
            return []

        model = self._load_model()
        annotations: list[TechniqueAnnotation] = []

        for start_idx, end_idx in self._window_indices(hands):
            window_frames = hands[start_idx:end_idx]
            if not window_frames:
                continue

            feat = _build_feature(window_frames)  # (T, 84)
            # Add batch dimension: (1, T, 84)
            tensor = torch.from_numpy(feat).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor)
                # Expected output shapes:
                #   (B, C)      — model already collapses the time axis
                #   (B, T, C)   — per-frame logits; we mean-pool to (B, C)
                # Any other rank indicates an incompatible model and is an error.
                if logits.dim() == 3:
                    logits = logits.mean(dim=1)  # (B, T, C) → (B, C)
                elif logits.dim() != 2:
                    raise ValueError(
                        f"Model output has unexpected rank {logits.dim()}. "
                        "Expected 2 (B, C) or 3 (B, T, C)."
                    )

            probs = torch.softmax(logits, dim=-1)
            best_idx = int(probs.argmax(dim=-1).item())
            confidence = float(probs[0, best_idx].item())
            label: TechniqueLabel = _LABELS[best_idx]

            t0 = window_frames[0].timestamp
            t1 = window_frames[-1].timestamp

            annotations.append(
                TechniqueAnnotation(
                    technique=label,
                    confidence=confidence,
                    source="vision",
                    params={"window_start": t0, "window_end": t1},
                )
            )

        return annotations
