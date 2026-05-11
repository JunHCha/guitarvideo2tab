"""End-to-end pipeline orchestrator: video input → .gpx output.

Stages:
1. download_video       — yt-dlp 또는 로컬 파일
2. split_audio_video    — ffmpeg, PTS 보존
3. separate_guitar_stem — Demucs 6s, guitar stem 추출
4. _run_audio_path      — Basic Pitch AMT + TART 기법 분류
5. _run_vision_path     — YOLO 프렛보드 + MediaPipe Hands + fret/기법
6. _fuse                — LateFusion (ADR-001 D4)
7. _write_output        — PyGuitarPro .gpx 직렬화

Weights-dependent stages(audio/technique, vision/technique, vision/fretboard)는
weights_path/model_factory가 None이면 빈 결과를 반환하는 폴백 구조이므로
가중치 없이도 전체 파이프라인이 동작한다(다만 기법 어노테이션이 비어 있음).
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    audio_weights: Path | None = None
    vision_weights: Path | None = None
    fretboard_weights: Path | None = None
    hands_model_asset: Path | None = None

    _intermediate_paths: dict[str, Path] = field(default_factory=dict, init=False)

    def run(self, source: str, output_path: Path) -> Path:
        self.workdir.mkdir(parents=True, exist_ok=True)

        video_path = download_video(source, self.workdir)
        self._intermediate_paths["video"] = video_path

        audio_wav, video_only = split_audio_video(video_path, self.workdir)
        self._intermediate_paths["audio_wav"] = audio_wav
        self._intermediate_paths["video_only"] = video_only

        guitar_wav = separate_guitar_stem(audio_wav, self.workdir)
        self._intermediate_paths["guitar_wav"] = guitar_wav

        midi_events, audio_techniques = self._run_audio_path(guitar_wav)
        fret_positions, vision_techniques = self._run_vision_path(video_only)

        notes = self._fuse(
            midi_events, audio_techniques, fret_positions, vision_techniques
        )
        return self._write_output(notes, output_path)

    def _run_audio_path(self, guitar_stem_wav: Path):
        transcriber = BasicPitchTranscriber()
        midi_events = transcriber.transcribe(guitar_stem_wav)
        audio_classifier = TARTTechniqueClassifier(weights_path=self.audio_weights)
        audio_techniques = audio_classifier.classify(midi_events, guitar_stem_wav)
        return midi_events, audio_techniques

    def _run_vision_path(self, video_path: Path):
        fretboard_detector = FretboardDetector(weights_path=self.fretboard_weights)
        fretboards = fretboard_detector.detect(video_path)
        hand_tracker = HandTracker(model_asset_path=self.hands_model_asset)
        hands = hand_tracker.track(video_path)
        fret_estimator = FretEstimator()
        fret_positions = fret_estimator.estimate(hands, fretboards)
        vision_classifier = VisionTechniqueClassifier(weights_path=self.vision_weights)
        vision_techniques = vision_classifier.classify(hands)
        return fret_positions, vision_techniques

    def _fuse(self, midi_events, audio_techniques, fret_positions, vision_techniques):
        fusion = LateFusion()
        return fusion.fuse(midi_events, audio_techniques, fret_positions, vision_techniques)

    def _write_output(self, notes, output_path: Path) -> Path:
        writer = TabWriter()
        return writer.write_gpx(notes, output_path)
