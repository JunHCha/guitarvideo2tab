# Source Index

> 최종 동기화: `e5ff632` (2026-05-11). PR #4/#5/#6 까지 반영.
> 구현 상태: ✅ 구현+테스트, 🟡 시그니처/스켈레톤만, ❌ 미착수.

## 파이프라인 진입점

| 파일 | 핵심 객체 | 상태 | 비고 |
|------|----------|------|------|
| [src/guitarvideo2tab/__main__.py](src/guitarvideo2tab/__main__.py) | `main(argv)` / `argparse` CLI | 🟡 | `Pipeline.run()`이 `NotImplementedError`라 실행 불가 |
| [src/guitarvideo2tab/pipeline.py](src/guitarvideo2tab/pipeline.py) | `Pipeline.run`, `_run_audio_path`, `_run_vision_path`, `_fuse`, `_write_output` | 🟡 | 모듈 와이어링만, `run()` 본체 미구현 |

## 데이터 모델

| 파일 | 데이터클래스 | 상태 |
|------|------------|------|
| [src/guitarvideo2tab/models.py](src/guitarvideo2tab/models.py) | `MidiEvent`, `PitchContour`, `TechniqueAnnotation`, `HandKeypoints`, `FretboardFrame`, `FretPosition`, `NoteEvent`, `TechniqueLabel`, `ModalitySource` | ✅ |

`tests/test_models/test_models.py`에서 dataclass 동작 검증.

## 전처리 모듈

| 파일 | 핵심 함수/클래스 | 상태 |
|------|-----------------|------|
| [preprocessing/downloader.py](src/guitarvideo2tab/preprocessing/downloader.py) | `download_video(source, output_dir)` | ✅ yt-dlp 래퍼, 로컬/URL 분기, 테스트 포함 |
| [preprocessing/separator.py](src/guitarvideo2tab/preprocessing/separator.py) | `split_audio_video(video_path, output_dir)` | ❌ ffmpeg-python, PTS 보존 (`-copyts`) |
| [preprocessing/stem.py](src/guitarvideo2tab/preprocessing/stem.py) | `separate_guitar_stem(audio_path, output_dir)` | ❌ Demucs `htdemucs_6s` |

## 오디오 경로

| 파일 | 클래스 | 상태 |
|------|-------|------|
| [audio/transcriber.py](src/guitarvideo2tab/audio/transcriber.py) | `BasicPitchTranscriber` | ✅ Basic Pitch ICASSP 2022, `multiple_pitch_bends=True`, `PitchContour` 생성, 테스트 5개 |
| [audio/technique.py](src/guitarvideo2tab/audio/technique.py) | `TARTTechniqueClassifier` | ❌ TART 2단계 MLP, `weights_path` 옵션 |

## 비전 경로

| 파일 | 클래스 | 상태 |
|------|-------|------|
| [vision/fretboard.py](src/guitarvideo2tab/vision/fretboard.py) | `FretboardDetector` | ❌ YOLOv8-OBB, 호모그래피 |
| [vision/hands.py](src/guitarvideo2tab/vision/hands.py) | `HandTracker` | ❌ MediaPipe Hands |
| [vision/fret_estimator.py](src/guitarvideo2tab/vision/fret_estimator.py) | `FretEstimator(num_strings=6, num_frets=24)` | ❌ keypoint → (string, fret) |
| [vision/technique.py](src/guitarvideo2tab/vision/technique.py) | `VisionTechniqueClassifier(window_ms=300, model_arch="tcn")` | ❌ TCN/Transformer |

## Fusion

| 파일 | 클래스 | 상태 |
|------|-------|------|
| [fusion/late_fusion.py](src/guitarvideo2tab/fusion/late_fusion.py) | `LateFusion(confidence_high=0.8, confidence_low=0.5)` | ❌ ADR-001 D4 규칙 미구현 |

## 출력

| 파일 | 클래스 | 상태 |
|------|-------|------|
| [output/tab_writer.py](src/guitarvideo2tab/output/tab_writer.py) | `TabWriter.write_gpx / write_gp5` | ❌ PyGuitarPro `bendPoints/slideType/...` 매핑 |

기본 튜닝: 표준 EADGBE — `tuning = (40, 45, 50, 55, 59, 64)` (MIDI).

---

## 테스트 인덱스

| 파일 | 대상 | 상태 |
|------|------|------|
| [tests/test_models/test_models.py](tests/test_models/test_models.py) | dataclass 모델 | ✅ PR #4 |
| [tests/test_preprocessing/test_downloader.py](tests/test_preprocessing/test_downloader.py) | `download_video` (`_FakeYDL`로 yt-dlp 모킹) | ✅ PR #5 |
| [tests/test_audio/test_transcriber.py](tests/test_audio/test_transcriber.py) | `BasicPitchTranscriber` (`predict` 모킹) | ✅ PR #6 |

미존재 — 향후 필요:
- `tests/test_preprocessing/test_separator.py`
- `tests/test_preprocessing/test_stem.py`
- `tests/test_audio/test_technique.py`
- `tests/test_vision/test_fretboard.py`
- `tests/test_vision/test_hands.py`
- `tests/test_vision/test_fret_estimator.py`
- `tests/test_vision/test_technique.py`
- `tests/test_fusion/test_late_fusion.py`
- `tests/test_output/test_tab_writer.py`
- `tests/test_pipeline.py` (통합)
- `tests/integration/` (E2E 짧은 클립)

---

## 외부 모델/가중치

| 모델 | 위치 (예정) | 상태 |
|------|-----------|------|
| Basic Pitch ICASSP-2022 | `basic_pitch.ICASSP_2022_MODEL_PATH` | ✅ 패키지 번들 |
| Demucs htdemucs_6s | HF 자동 다운로드 | ❌ stem 모듈 구현 시 |
| YOLOv8-OBB fretboard | `models/yolo/fretboard.pt` | ❌ 자체 학습 필요 |
| 비전 기법 분류기 | `models/vision_technique/` | ❌ Mitsou 2023 데이터 학습 |
| TART MLP | `models/tart/` | ❌ 공개 가중치 확인 필요 |

---

## 구현 진행률

전체 11개 핵심 모듈 중 **3개 구현 완료** (downloader, transcriber, models), 나머지 8개는 시그니처만 정의되고 `NotImplementedError` 상태.

Pipeline 실행 가능 시점 = separator / stem / late_fusion / tab_writer 4개가 모두 구현된 시점 (오디오 전용 MVP). 비전 경로(fretboard/hands/fret_estimator/vision-technique)가 추가되면 전체 멀티모달 파이프라인 완성.
