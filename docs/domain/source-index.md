# Source Index

> 최종 동기화: `6695437` (2026-05-11). PR #4~#18 까지 반영 (#9는 #17로 재생성됨).
> 구현 상태: ✅ 구현+테스트, 🟡 시그니처/스켈레톤만, ❌ 미착수.

## 파이프라인 진입점

| 파일 | 핵심 객체 | 상태 | 비고 |
|------|----------|------|------|
| [src/guitarvideo2tab/__main__.py](src/guitarvideo2tab/__main__.py) | `main(argv)` / `argparse` CLI | ✅ | `python -m guitarvideo2tab <input> [-o OUT]` |
| [src/guitarvideo2tab/pipeline.py](src/guitarvideo2tab/pipeline.py) | `Pipeline.run`, 10개 모듈 오케스트레이션 | ✅ PR #18 | 가중치 경로 4종 (`audio_weights` 등) 노출, `_intermediate_paths` 기록 |

## 데이터 모델

| 파일 | 데이터클래스 | 상태 |
|------|------------|------|
| [src/guitarvideo2tab/models.py](src/guitarvideo2tab/models.py) | `MidiEvent`, `PitchContour`, `TechniqueAnnotation`, `HandKeypoints`, `FretboardFrame`, `FretPosition`, `NoteEvent`, `TechniqueLabel`, `ModalitySource` | ✅ PR #4 |

## 전처리 모듈

| 파일 | 핵심 함수/클래스 | 상태 |
|------|-----------------|------|
| [preprocessing/downloader.py](src/guitarvideo2tab/preprocessing/downloader.py) | `download_video(source, output_dir)` | ✅ PR #5 — yt-dlp 래퍼, URL/로컬 분기 |
| [preprocessing/separator.py](src/guitarvideo2tab/preprocessing/separator.py) | `split_audio_video(video_path, output_dir)` | ✅ PR #8 — ffmpeg-python, `-copyts` PTS 보존 |
| [preprocessing/stem.py](src/guitarvideo2tab/preprocessing/stem.py) | `separate_guitar_stem(audio_path, output_dir)` | ✅ PR #17 — Demucs `htdemucs_6s`, demucs>=4.0.0 핀 |

## 오디오 경로

| 파일 | 클래스 | 상태 |
|------|-------|------|
| [audio/transcriber.py](src/guitarvideo2tab/audio/transcriber.py) | `BasicPitchTranscriber` | ✅ PR #6 — ICASSP-2022, `multiple_pitch_bends=True`, `PitchContour` |
| [audio/technique.py](src/guitarvideo2tab/audio/technique.py) | `TARTTechniqueClassifier` | ✅ PR #10 — state_dict 패턴, `weights_only=True`, `model_factory` 필요 |

## 비전 경로

| 파일 | 클래스 | 상태 |
|------|-------|------|
| [vision/fretboard.py](src/guitarvideo2tab/vision/fretboard.py) | `FretboardDetector` | ✅ PR #11 — YOLOv8-OBB, 호모그래피, VideoCapture/findHomography 가드 |
| [vision/hands.py](src/guitarvideo2tab/vision/hands.py) | `HandTracker` | ✅ PR #12 — MediaPipe legacy + Tasks API 어댑터, `model_asset_path` 필드 |
| [vision/fret_estimator.py](src/guitarvideo2tab/vision/fret_estimator.py) | `FretEstimator(num_strings=6, num_frets=24)` | ✅ PR #13 — uniform binning(`num_frets+1` bins), w-degeneracy sentinel |
| [vision/technique.py](src/guitarvideo2tab/vision/technique.py) | `VisionTechniqueClassifier(window_ms=300)` | ✅ PR #14 — state_dict 패턴, `weights_only=True`, `model_factory` 필요 |

## Fusion

| 파일 | 클래스 | 상태 |
|------|-------|------|
| [fusion/late_fusion.py](src/guitarvideo2tab/fusion/late_fusion.py) | `LateFusion(confidence_high=0.8, confidence_low=0.5)` | ✅ PR #15 — ADR-001 D4 규칙 (confirm은 AND), 점유 폴백 TODO 명시 |

## 출력

| 파일 | 클래스 | 상태 |
|------|-------|------|
| [output/tab_writer.py](src/guitarvideo2tab/output/tab_writer.py) | `TabWriter.write_gpx / write_gp5` | ✅ PR #16 — PyGuitarPro 매핑(bend/slide/hammer/vibrato/palmMute/tapping), unresolved skip |

기본 튜닝: 표준 EADGBE — `tuning = (40, 45, 50, 55, 59, 64)` (MIDI).

---

## 테스트 인덱스

| 파일 | 대상 | 테스트 수 |
|------|------|----------|
| [tests/test_models/test_models.py](tests/test_models/test_models.py) | dataclass 모델 | (PR #4) |
| [tests/test_preprocessing/test_downloader.py](tests/test_preprocessing/test_downloader.py) | `download_video` | (PR #5) |
| [tests/test_preprocessing/test_separator.py](tests/test_preprocessing/test_separator.py) | `split_audio_video` | 4 |
| [tests/test_preprocessing/test_stem.py](tests/test_preprocessing/test_stem.py) | `separate_guitar_stem` | 5 |
| [tests/test_audio/test_transcriber.py](tests/test_audio/test_transcriber.py) | `BasicPitchTranscriber` | (PR #6) |
| [tests/test_audio/test_technique.py](tests/test_audio/test_technique.py) | `TARTTechniqueClassifier` | 8 |
| [tests/test_vision/test_fretboard.py](tests/test_vision/test_fretboard.py) | `FretboardDetector` | 9 |
| [tests/test_vision/test_hands.py](tests/test_vision/test_hands.py) | `HandTracker` | 10 |
| [tests/test_vision/test_fret_estimator.py](tests/test_vision/test_fret_estimator.py) | `FretEstimator` | 11 |
| [tests/test_vision/test_technique.py](tests/test_vision/test_technique.py) | `VisionTechniqueClassifier` | 5 |
| [tests/test_fusion/test_late_fusion.py](tests/test_fusion/test_late_fusion.py) | `LateFusion` | 8 |
| [tests/test_output/test_tab_writer.py](tests/test_output/test_tab_writer.py) | `TabWriter` | 5 |
| [tests/test_pipeline.py](tests/test_pipeline.py) | `Pipeline.run()` 통합 | 4 |

총 **104 tests**, 모두 PASS · ruff clean.

미존재 — 향후 필요:
- `tests/integration/` (E2E 실제 짧은 클립으로 검증)

---

## 외부 모델/가중치

| 모델 | 위치 (예정) | 상태 |
|------|-----------|------|
| Basic Pitch ICASSP-2022 | `basic_pitch.ICASSP_2022_MODEL_PATH` | ✅ 패키지 번들 |
| Demucs htdemucs_6s | HF 자동 다운로드 (demucs.api) | 🟡 실 사용 시 가중치 자동 다운로드 |
| YOLOv8-OBB fretboard | `models/yolo/fretboard.pt` | ❌ 자체 학습 필요 (현재 기본 yolov8n-obb.pt) |
| 비전 기법 분류기 | `models/vision_technique/` (state_dict) | ❌ Mitsou 2023 데이터 학습 필요. `model_factory` 함께 제공 |
| TART MLP | `models/tart/` (state_dict) | ❌ 가중치 공개 여부 확인 필요. `model_factory` 함께 제공 |

---

## 구현 진행률

전체 12개 핵심 모듈 (`pipeline` 포함) **모두 구현 완료**. NotImplementedError 0건.

가중치 미공개 분류기 2종 (TART, Mitsou 비전 기법)은 **state_dict + model_factory 패턴**으로 `weights_only=True` 보안 로딩 — 가중치만 확보되면 즉시 활성. 미설정 시 빈 라벨 리스트 폴백.

`python -m guitarvideo2tab <local.mp4>` 실행 시:
1. ✅ 다운로드/분리/stem 추출 동작
2. ✅ Basic Pitch AMT 동작
3. 🟡 기법 분류기는 빈 라벨 (가중치 없음)
4. ✅ 비전 경로 동작 (YOLO 기본 가중치 사용 — 정확도는 낮음)
5. ✅ Late Fusion + .gpx 출력

따라서 현재 상태로도 **음표 위치(string/fret)는 추정**, 기법은 미부착의 TAB이 생성된다.
