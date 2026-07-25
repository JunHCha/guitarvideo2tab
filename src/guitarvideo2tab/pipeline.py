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

`save_intermediates=True` 면 각 단계의 산출물을 workdir/intermediates/*.json 로
dump 한다 — 최종 GP5 결과가 깨졌을 때 어느 단계에서 무엇이 잘못됐는지 추적용.

참고: ``intermediates/{NN_name}.json`` 의 prefix 번호는 위 stage 번호와 다른
'산출물(artifact) 번호' 이며, ``run()`` 내부 dump 호출 순서를 그대로 따른다.
"""
from __future__ import annotations

import dataclasses
import json
import time
import warnings
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

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


def _json_default(obj: Any) -> Any:
    """JSON 직렬화 fallback: numpy / dataclass / Path 등을 일반 타입으로 변환."""
    # numpy 는 optional 의존성이지만 audio/vision stage 가 numpy 를 통과시키는
    # 케이스가 흔하므로 lazy import.
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:  # pragma: no cover
        pass

    if is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _to_serializable(data: Any) -> Any:
    """dataclass / dataclass list 를 dict / list[dict] 로 정규화."""
    if isinstance(data, list):
        return [_to_serializable(item) for item in data]
    if is_dataclass(data) and not isinstance(data, type):
        return dataclasses.asdict(data)
    return data


@dataclass
class Pipeline:
    workdir: Path
    save_intermediates: bool = False
    audio_weights: Path | None = None
    vision_weights: Path | None = None
    fretboard_weights: Path | None = None
    hands_model_asset: Path | None = None

    _intermediate_paths: dict[str, Path] = field(default_factory=dict, init=False)
    _summary: list[dict] = field(default_factory=list, init=False)

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
        fret_positions, vision_techniques, fretboards, hands = self._run_vision_path(
            video_only
        )

        notes = self._fuse(
            midi_events, audio_techniques, fret_positions, vision_techniques
        )

        if self.save_intermediates:
            try:
                self._write_summary()
            except (TypeError, OSError, ValueError) as exc:
                warnings.warn(
                    f"Failed to write intermediate summary: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return self._write_output(notes, output_path)

    def _run_audio_path(self, guitar_stem_wav: Path):
        transcriber = BasicPitchTranscriber()

        t0 = time.perf_counter()
        midi_events = transcriber.transcribe(guitar_stem_wav)
        self._dump_stage("01_midi_events", midi_events, elapsed_sec=time.perf_counter() - t0)

        audio_classifier = TARTTechniqueClassifier(weights_path=self.audio_weights)
        t0 = time.perf_counter()
        audio_techniques = audio_classifier.classify(midi_events, guitar_stem_wav)
        self._dump_stage(
            "02_audio_techniques", audio_techniques, elapsed_sec=time.perf_counter() - t0
        )
        return midi_events, audio_techniques

    def _run_vision_path(self, video_path: Path):
        fretboard_detector = FretboardDetector(weights_path=self.fretboard_weights)
        t0 = time.perf_counter()
        fretboards = fretboard_detector.detect(video_path)
        self._dump_stage("03_fretboards", fretboards, elapsed_sec=time.perf_counter() - t0)

        hand_tracker = HandTracker(model_asset_path=self.hands_model_asset)
        t0 = time.perf_counter()
        hands = hand_tracker.track(video_path)
        self._dump_stage("04_hands", hands, elapsed_sec=time.perf_counter() - t0)

        fret_estimator = FretEstimator()
        t0 = time.perf_counter()
        fret_positions = fret_estimator.estimate(hands, fretboards)
        self._dump_stage(
            "05_fret_positions", fret_positions, elapsed_sec=time.perf_counter() - t0
        )

        vision_classifier = VisionTechniqueClassifier(weights_path=self.vision_weights)
        t0 = time.perf_counter()
        vision_techniques = vision_classifier.classify(hands)
        self._dump_stage(
            "06_vision_techniques", vision_techniques, elapsed_sec=time.perf_counter() - t0
        )
        return fret_positions, vision_techniques, fretboards, hands

    def _fuse(self, midi_events, audio_techniques, fret_positions, vision_techniques):
        fusion = LateFusion()
        t0 = time.perf_counter()
        notes = fusion.fuse(midi_events, audio_techniques, fret_positions, vision_techniques)
        self._dump_stage("07_notes_fused", notes, elapsed_sec=time.perf_counter() - t0)
        return notes

    def _write_output(self, notes, output_path: Path) -> Path:
        writer = TabWriter()
        return writer.write_gpx(notes, output_path)

    # ------------------------------------------------------------------
    # Intermediate dump helpers
    # ------------------------------------------------------------------
    def _dump_stage(
        self, name: str, data: Any, elapsed_sec: float = 0.0
    ) -> Path | None:
        if not self.save_intermediates:
            return None

        try:
            out_dir = self.workdir / "intermediates"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{name}.json"

            serializable = _to_serializable(data)
            out_path.write_text(
                json.dumps(serializable, default=_json_default, indent=2, ensure_ascii=False)
            )

            count = len(data) if hasattr(data, "__len__") else 1
            sample_src = serializable[:3] if isinstance(serializable, list) else [serializable]
            self._summary.append(
                {
                    "stage": name,
                    "count": count,
                    "elapsed_sec": float(elapsed_sec),
                    "sample": sample_src,
                    "output_path": str(out_path),
                }
            )
            return out_path
        except (TypeError, OSError, ValueError) as exc:
            warnings.warn(
                f"Failed to dump intermediate {name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _write_summary(self) -> Path | None:
        if not self.save_intermediates:
            return None
        try:
            out_dir = self.workdir / "intermediates"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "summary.json"
            out_path.write_text(
                json.dumps(self._summary, default=_json_default, indent=2, ensure_ascii=False)
            )
            return out_path
        except (TypeError, OSError, ValueError) as exc:
            warnings.warn(
                f"Failed to write intermediate summary: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
