"""운지 손 keypoint를 프렛보드 좌표계로 매핑하여 (string, fret) 추정."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import FretboardFrame, FretPosition, HandKeypoints

# MediaPipe fingertip landmark indices: index/middle/ring/pinky tips
_FINGERTIP_INDICES = (8, 12, 16, 20)

# Assumption: fretting hand is the LEFT hand (left_hand field).
# Right-handed players fret with the left hand — the vast majority of guitarists.
# Left-handed players who use the right hand for fretting are not supported;
# this is documented here as a known limitation.


@dataclass
class FretEstimator:
    num_strings: int = 6
    num_frets: int = 24

    def estimate(
        self,
        hands: list[HandKeypoints],
        fretboards: list[FretboardFrame],
    ) -> list[FretPosition]:
        """Map fretting-hand fingertip keypoints to (string, fret) positions.

        For each HandKeypoints frame:
        1. Find the closest FretboardFrame by timestamp.
        2. If fretboard is invisible or has no homography, emit occlusion fallback
           entries (confidence=0.3) for each fingertip — one per tip with the
           canonical (string, fret) clamped from raw image coords.
           If homography is None, skip entirely (no coordinate info).
        3. Apply the 3x3 homography H to each fingertip (x, y) → canonical (u, v).
        4. Map u → fret, v → string and emit FretPosition(confidence=1.0).

        Args:
            hands: Per-frame hand keypoints from the pose estimator.
            fretboards: Per-frame fretboard detections with homography.

        Returns:
            List of FretPosition, one per fingertip per frame (skipping frames
            where left_hand is None or homography is None).
        """
        if not fretboards:
            return []

        fb_timestamps = np.array([fb.timestamp for fb in fretboards])
        results: list[FretPosition] = []

        for hand in hands:
            # Skip frames with no fretting-hand data
            if hand.left_hand is None:
                continue

            # Find closest fretboard frame
            idx = int(np.argmin(np.abs(fb_timestamps - hand.timestamp)))
            fb = fretboards[idx]

            # Occlusion: visible=False or no homography
            if not fb.visible or fb.homography is None:
                if fb.homography is None:
                    # Cannot map coordinates at all — skip
                    continue
                # visible=False but homography exists — emit low-confidence entries
                hmat = np.array(fb.homography, dtype=np.float64)
                for tip_idx in _FINGERTIP_INDICES:
                    x, y = hand.left_hand[tip_idx]
                    u, v = self._apply_homography(hmat, x, y)
                    if u is None or v is None:
                        # Degenerate homography point — emit zero-confidence entry
                        results.append(
                            FretPosition(
                                timestamp=hand.timestamp,
                                string=1,
                                fret=0,
                                confidence=0.0,
                            )
                        )
                        continue
                    string, fret = self._map_to_string_fret(u, v)
                    results.append(
                        FretPosition(
                            timestamp=hand.timestamp,
                            string=string,
                            fret=fret,
                            confidence=0.3,
                        )
                    )
                continue

            # Normal visible fretboard with homography
            hmat = np.array(fb.homography, dtype=np.float64)
            for tip_idx in _FINGERTIP_INDICES:
                x, y = hand.left_hand[tip_idx]
                u, v = self._apply_homography(hmat, x, y)
                if u is None or v is None:
                    # Degenerate homography point — emit zero-confidence entry
                    results.append(
                        FretPosition(
                            timestamp=hand.timestamp,
                            string=1,
                            fret=0,
                            confidence=0.0,
                        )
                    )
                    continue
                string, fret = self._map_to_string_fret(u, v)
                results.append(
                    FretPosition(
                        timestamp=hand.timestamp,
                        string=string,
                        fret=fret,
                        confidence=1.0,
                    )
                )

        return results

    def _apply_homography(
        self, hmat: np.ndarray, x: float, y: float
    ) -> tuple[float, float] | tuple[None, None]:
        """Apply 3x3 homography matrix to image point (x, y).

        Returns canonical (u, v) clamped to [0, 1].
        Returns (None, None) when the projective weight w is near zero
        (degenerate or behind-camera point).  Callers must check for this
        sentinel and emit a FretPosition with confidence=0.0.
        """
        p = hmat @ np.array([x, y, 1.0], dtype=np.float64)
        w = p[2]
        if abs(w) < 1e-10:
            return None, None
        u = float(np.clip(p[0] / w, 0.0, 1.0))
        v = float(np.clip(p[1] / w, 0.0, 1.0))
        return u, v

    def _map_to_string_fret(self, u: float, v: float) -> tuple[int, int]:
        """Map canonical (u, v) in [0,1] to (string, fret).

        Convention: num_frets=24 means frets 0..24 (25 valid values).
        The fretboard width [0, 1] is divided into num_frets+1 equal bins:
          fret 0  (open/nut) ← u in [0,       1/25)
          fret 1             ← u in [1/25,    2/25)
          …
          fret 24 (last)     ← u in [24/25,   1.0]

        u → fret: 0 = open/nut, num_frets = highest fret
        v → string: 1 = lowest string index, num_strings = highest
        """
        fret = min(int(u * (self.num_frets + 1)), self.num_frets)
        string = int(np.clip(round(v * (self.num_strings - 1)) + 1, 1, self.num_strings))
        return string, fret
