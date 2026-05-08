"""Basic Pitch (Spotify) AMT — 오디오 → MIDI + pitch-bend curve."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
from basic_pitch.inference import predict

from ..models import MidiEvent, PitchContour


@dataclass
class BasicPitchTranscriber:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3

    def transcribe(self, audio_path: Path) -> list[MidiEvent]:
        _, _, note_events = predict(
            str(audio_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            multiple_pitch_bends=True,
            melodia_trick=True,
        )
        frame_rate = AUDIO_SAMPLE_RATE / FFT_HOP
        return [self._to_midi_event(idx, evt, frame_rate) for idx, evt in enumerate(note_events)]

    def _to_midi_event(
        self, idx: int, evt: tuple, frame_rate: float
    ) -> MidiEvent:
        start, end, pitch, amplitude, pitch_bends = evt
        velocity = max(0, min(127, int(round(amplitude * 127))))
        pitch_contour = self._to_pitch_contour(idx, start, pitch, pitch_bends, frame_rate)
        return MidiEvent(
            pitch=pitch,
            start_time=start,
            end_time=end,
            velocity=velocity,
            pitch_contour=pitch_contour,
        )

    def _to_pitch_contour(
        self,
        idx: int,
        start: float,
        pitch: int,
        pitch_bends: Sequence[float] | None,
        frame_rate: float,
    ) -> PitchContour | None:
        if not pitch_bends:
            return None
        curve = [(start + i / frame_rate, pitch + bend) for i, bend in enumerate(pitch_bends)]
        bend_semitones = max(abs(b) for b in pitch_bends)
        return PitchContour(
            note_id=f"{idx:06d}",
            time_pitch_curve=curve,
            bend_semitones=bend_semitones,
        )
