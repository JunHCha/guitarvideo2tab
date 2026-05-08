"""Basic Pitch (Spotify) AMT — 오디오 → MIDI + pitch-bend curve."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import MidiEvent


@dataclass
class BasicPitchTranscriber:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3

    def transcribe(self, audio_path: Path) -> list[MidiEvent]:
        raise NotImplementedError(
            "basic-pitch CLI/API 호출하되 --save-note-events 사용 필수. "
            "각 노트의 pitch contour를 PitchContour로 보존 (MIDI 단순 환원 금지, ADR-001 D5)."
        )
