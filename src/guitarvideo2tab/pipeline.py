"""End-to-end pipeline orchestrator: video input → .gpx output."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio.technique import TARTTechniqueClassifier
from .audio.transcriber import BasicPitchTranscriber
from .fusion.late_fusion import LateFusion
from .output.tab_writer import TabWriter
from .preprocessing.downloader import download_video
from .preprocessing.separator import split_audio_video
from .preprocessing.stem import separate_guitar_stem
from .vision.fret_estimator import FretEstimator
from .vision.fretboard import FretboardDetector
from .vision.hands import HandTracker
from .vision.technique import VisionTechniqueClassifier


@dataclass
class Pipeline:
    workdir: Path
    save_intermediates: bool = False

    def run(self, source: str, output_path: Path) -> Path:
        raise NotImplementedError(
            "Pipeline.run: orchestrate preprocessing → audio/vision paths → "
            "late fusion → tab writer. See architecture.md for data flow."
        )

    def _run_audio_path(self, guitar_stem_wav: Path):
        transcriber = BasicPitchTranscriber()
        midi_events = transcriber.transcribe(guitar_stem_wav)
        audio_classifier = TARTTechniqueClassifier()
        audio_techniques = audio_classifier.classify(midi_events, guitar_stem_wav)
        return midi_events, audio_techniques

    def _run_vision_path(self, video_path: Path):
        fretboard_detector = FretboardDetector()
        fretboards = fretboard_detector.detect(video_path)
        hand_tracker = HandTracker()
        hands = hand_tracker.track(video_path)
        fret_estimator = FretEstimator()
        fret_positions = fret_estimator.estimate(hands, fretboards)
        vision_classifier = VisionTechniqueClassifier()
        vision_techniques = vision_classifier.classify(hands)
        return fret_positions, vision_techniques

    def _fuse(self, midi_events, audio_techniques, fret_positions, vision_techniques):
        fusion = LateFusion()
        return fusion.fuse(midi_events, audio_techniques, fret_positions, vision_techniques)

    def _write_output(self, notes, output_path: Path) -> Path:
        writer = TabWriter()
        return writer.write_gpx(notes, output_path)
