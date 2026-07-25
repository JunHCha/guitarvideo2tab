"""End-to-end pipeline orchestrator: video input → MIDI + MusicXML.

Stages:
1. download_video       — yt-dlp 또는 로컬 파일
2. split_audio_video    — ffmpeg, PTS 보존
3. separate_guitar_stem — Demucs 6s, guitar stem 추출
4. _run_audio_path      — Basic Pitch AMT + TART 기법 분류
5. _run_vision_path     — YOLO 프렛보드 + MediaPipe Hands + fret/기법
6. _fuse                — LateFusion (ADR-001 D4)
7. _write_output        — 리듬 양자화 → .mid / .musicxml

실행 하나가 ``runs/<run_id>/`` 디렉토리 하나를 통째로 쓴다(:mod:`workspace`).
입력·중간산출물·결과가 실행별로 격리되고 ``manifest.json`` 에 무엇을 어떤
설정으로 돌렸는지가 남으므로 trial 을 반복해도 추적이 가능하다.

Weights-dependent stages(audio/technique, vision/technique, vision/fretboard)는
weights_path/model_factory가 None이면 빈 결과를 반환하는 폴백 구조이므로
가중치 없이도 전체 파이프라인이 동작한다(다만 기법 어노테이션이 비어 있음).
"""
from __future__ import annotations

import dataclasses
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio.technique import TARTTechniqueClassifier
from .audio.transcriber import BasicPitchTranscriber
from .fusion.late_fusion import LateFusion
from .models import NoteEvent
from .output.midi_writer import write_midi
from .output.musicxml_writer import write_musicxml
from .output.rhythm import RhythmGrid, build_grid
from .preprocessing.downloader import download_video
from .preprocessing.separator import split_audio_video
from .preprocessing.stem import separate_guitar_stem
from .vision.fret_estimator import FretEstimator
from .vision.fretboard import FretboardDetector
from .vision.hands import HandTracker
from .vision.technique import VisionTechniqueClassifier
from .workspace import RunWorkspace

STANDARD_TUNING = (40, 45, 50, 55, 59, 64)  # EADGBE (MIDI)


def _json_default(obj: Any) -> Any:
    """dataclass / numpy / Path 등을 JSON 으로 직렬화한다."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    for attr in ("tolist", "item"):  # numpy ndarray / scalar
        method = getattr(obj, attr, None)
        if callable(method):
            return method()
    raise TypeError(f"직렬화할 수 없는 타입: {type(obj).__name__}")


@dataclass
class Pipeline:
    """영상 → 악보 파이프라인.

    Args:
        workspace: 이 실행이 쓸 :class:`RunWorkspace`. 산출물이 모두 여기 쌓인다.
        save_intermediates: 단계별 JSON dump 여부(미디어 파일은 항상 저장된다).
        tempo_bpm: 지정 시 tempo 추정을 건너뛴다.
    """

    workspace: RunWorkspace
    save_intermediates: bool = True
    audio_weights: Path | None = None
    vision_weights: Path | None = None
    fretboard_weights: Path | None = None
    hands_model_asset: Path | None = None
    tempo_bpm: float | None = None
    numerator: int = 4
    denominator: int = 4
    subdivision: int = 16
    tuning: tuple[int, ...] = STANDARD_TUNING

    _started_at: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source: str) -> dict[str, Path]:
        """파이프라인을 끝까지 실행하고 ``{"midi": …, "musicxml": …}`` 을 반환한다."""
        self._started_at = time.perf_counter()
        self.workspace.set_config(
            tempo_bpm=self.tempo_bpm,
            time_signature=f"{self.numerator}/{self.denominator}",
            subdivision=self.subdivision,
            tuning=list(self.tuning),
            save_intermediates=self.save_intermediates,
            weights={
                "audio": str(self.audio_weights) if self.audio_weights else None,
                "vision": str(self.vision_weights) if self.vision_weights else None,
                "fretboard": str(self.fretboard_weights) if self.fretboard_weights else None,
            },
        )

        video_path = self._timed(
            "01_download", download_video, source, self.workspace.stages_dir
        )
        audio_wav, video_only = self._timed(
            "02_split", split_audio_video, video_path, self.workspace.stages_dir
        )
        guitar_wav = self._timed(
            "03_stem", separate_guitar_stem, audio_wav, self.workspace.stages_dir
        )

        midi_events, audio_techniques = self._run_audio_path(guitar_wav)
        fret_positions, vision_techniques = self._run_vision_path(video_only)

        notes = self._fuse(midi_events, audio_techniques, fret_positions, vision_techniques)
        outputs = self._write_output(notes, guitar_wav)

        self.workspace.record_totals(
            midi_events=len(midi_events),
            notes_total=len(notes),
            notes_resolved=len([n for n in notes if n.string >= 1 and n.fret >= 0]),
            elapsed_sec=round(time.perf_counter() - self._started_at, 2),
        )
        self.workspace.save_manifest()
        return outputs

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _run_audio_path(self, guitar_stem_wav: Path):
        transcriber = BasicPitchTranscriber()
        midi_events = self._timed("04_transcribe", transcriber.transcribe, guitar_stem_wav)
        self._dump("04_midi_events", midi_events)

        classifier = TARTTechniqueClassifier(weights_path=self.audio_weights)
        audio_techniques = self._timed(
            "05_audio_technique", classifier.classify, midi_events, guitar_stem_wav
        )
        self._dump("05_audio_techniques", audio_techniques)
        return midi_events, audio_techniques

    def _run_vision_path(self, video_path: Path):
        detector = FretboardDetector(weights_path=self.fretboard_weights)
        fretboards = self._timed("06_fretboard", detector.detect, video_path)
        self._dump("06_fretboards", fretboards)

        tracker = HandTracker(model_asset_path=self.hands_model_asset)
        hands = self._timed("07_hands", tracker.track, video_path)
        self._dump("07_hands", hands)

        estimator = FretEstimator()
        fret_positions = self._timed("08_fret_estimate", estimator.estimate, hands, fretboards)
        self._dump("08_fret_positions", fret_positions)

        classifier = VisionTechniqueClassifier(weights_path=self.vision_weights)
        vision_techniques = self._timed("09_vision_technique", classifier.classify, hands)
        self._dump("09_vision_techniques", vision_techniques)
        return fret_positions, vision_techniques

    def _fuse(self, midi_events, audio_techniques, fret_positions, vision_techniques):
        fusion = LateFusion()
        notes = self._timed(
            "10_fusion",
            fusion.fuse,
            midi_events,
            audio_techniques,
            fret_positions,
            vision_techniques,
        )
        self._dump("10_notes_fused", notes)
        return notes

    def _write_output(self, notes: list[NoteEvent], audio_path: Path) -> dict[str, Path]:
        grid = self._build_grid(notes, audio_path)
        self.workspace.set_config(tempo_bpm_used=round(grid.tempo_bpm, 3))

        midi_path = self._timed(
            "11_midi", write_midi, notes, self.workspace.output("tab.mid"), grid
        )
        self.workspace.record_output("midi", midi_path)

        result = self._timed(
            "12_musicxml",
            write_musicxml,
            notes,
            self.workspace.output("tab.musicxml"),
            grid,
            self.tuning,
        )
        self.workspace.record_output("musicxml", result.path)
        self.workspace.record_totals(
            tempo_bpm=round(grid.tempo_bpm, 2),
            measures=result.measures,
            notes_written=result.notes,
            chords=result.chords,
            rests=result.rests,
            # (string, fret) 이 pitch 와 일치해 운지를 적을 수 있었던 비율.
            # 낮으면 비전 경로의 string/fret 배정이 신뢰할 수 없다는 뜻이다.
            fingering_match_ratio=round(result.technical_match_ratio, 3),
            fingering_skipped=result.technical_skipped,
        )
        return {"midi": midi_path, "musicxml": result.path}

    def _build_grid(self, notes: list[NoteEvent], audio_path: Path) -> RhythmGrid:
        return build_grid(
            [n.midi_event for n in notes],
            audio_path=audio_path,
            tempo_bpm=self.tempo_bpm,
            numerator=self.numerator,
            denominator=self.denominator,
            subdivision=self.subdivision,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _timed(self, name: str, func, *args, **kwargs):
        """단계를 실행하고 소요 시간과 결과 크기를 manifest 에 기록한다."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        count = len(result) if isinstance(result, (list, tuple, set, dict)) else None
        path = result if isinstance(result, Path) else None
        self.workspace.record_stage(name, count=count, elapsed_sec=elapsed, path=path)
        return result

    def _dump(self, name: str, data: Any) -> Path | None:
        """중간 산출물을 JSON 으로 남긴다. 실패해도 파이프라인을 멈추지 않는다."""
        if not self.save_intermediates:
            return None
        out_path = self.workspace.stage(f"{name}.json")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(data, default=_json_default, indent=2, ensure_ascii=False)
            )
        except (TypeError, OSError, ValueError) as exc:
            warnings.warn(
                f"중간 산출물 저장 실패 {name!r}: {exc}", RuntimeWarning, stacklevel=2
            )
            return None
        return out_path
