# Source Index

> 최종 동기화: `e650ede` (2026-07-26). PR #4~#23 까지 반영 (#9는 #17로 재생성됨).
> 구현 상태: ✅ 구현+테스트, 🟡 시그니처/스켈레톤만, ❌ 미착수.

## 파이프라인 진입점

| 파일 | 핵심 객체 | 상태 | 비고 |
|------|----------|------|------|
| [src/guitarvideo2tab/__main__.py](src/guitarvideo2tab/__main__.py) | `main(argv)` — `run` / `list` 하위 명령 | ✅ PR #23 | `python -m guitarvideo2tab <input>` (하위 명령 생략 시 `run`) |
| [src/guitarvideo2tab/pipeline.py](src/guitarvideo2tab/pipeline.py) | `Pipeline.run(source) -> {midi, musicxml}` | ✅ PR #23 | `RunWorkspace` 주입, 단계별 `_timed`/`_dump`, 가중치 경로 4종 노출 |
| [src/guitarvideo2tab/workspace.py](src/guitarvideo2tab/workspace.py) | `RunWorkspace`, `list_runs`, `slugify` | ✅ PR #23 | 실행별 `runs/<run_id>/` 격리 + `manifest.json` |

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

## 리듬 해석 · 출력

| 파일 | 핵심 API | 상태 |
|------|---------|------|
| [output/rhythm.py](src/guitarvideo2tab/output/rhythm.py) | `RhythmGrid`, `QuantizedBeat`, `build_grid`, `quantize_notes`, `ticks_to_duration` | ✅ PR #22 — tempo 추정(librosa/IOI), 격자 스냅, 화음 그룹핑, 마디 분할, 쉼표 채움 |
| [output/midi_writer.py](src/guitarvideo2tab/output/midi_writer.py) | `write_midi(notes, path, grid)` | ✅ PR #23 — mido, `ticks_per_beat=960` |
| [output/musicxml_writer.py](src/guitarvideo2tab/output/musicxml_writer.py) | `write_musicxml(...) -> MusicXMLResult`, `pitch_to_xml`, `to_musicxml_string` | ✅ PR #23 — score-partwise 4.0, `divisions=960`, pitch 일치 시에만 `<technical>` |
| ~~output/tab_writer.py~~ | ~~`TabWriter.write_gpx / write_gp5`~~ | ❌ **삭제됨 (PR #23)** — pyguitarpro 의존 제거, MIDI+MusicXML 로 대체 |

기본 튜닝: 표준 EADGBE — `tuning = (40, 45, 50, 55, 59, 64)` (MIDI).

> ⚠️ **현 번호 규약**: 이 코드베이스는 `string=1` 이 최저음현(`tuning[0]`)이다.
> MusicXML 은 반대(1 = 최고음현)이므로 `to_musicxml_string()` 이 반전한다.

> 틱 규약: 4분음표 = 960틱. MIDI `ticks_per_beat` 이자 MusicXML `divisions` 로 공유되어
> 두 출력의 리듬이 정확히 일치한다.

---

## 테스트 인덱스

| 파일 | 대상 | 테스트 수 |
|------|------|----------|
| [tests/test_models/test_models.py](tests/test_models/test_models.py) | dataclass 모델 | 26 |
| [tests/test_workspace.py](tests/test_workspace.py) | `RunWorkspace`, `slugify`, `list_runs` | 12 |
| [tests/test_preprocessing/test_downloader.py](tests/test_preprocessing/test_downloader.py) | `download_video` | 4 |
| [tests/test_preprocessing/test_separator.py](tests/test_preprocessing/test_separator.py) | `split_audio_video` | 4 |
| [tests/test_preprocessing/test_stem.py](tests/test_preprocessing/test_stem.py) | `separate_guitar_stem` | 5 |
| [tests/test_audio/test_transcriber.py](tests/test_audio/test_transcriber.py) | `BasicPitchTranscriber` | 5 |
| [tests/test_audio/test_technique.py](tests/test_audio/test_technique.py) | `TARTTechniqueClassifier` | 8 |
| [tests/test_vision/test_fretboard.py](tests/test_vision/test_fretboard.py) | `FretboardDetector` | 9 |
| [tests/test_vision/test_hands.py](tests/test_vision/test_hands.py) | `HandTracker` | 10 |
| [tests/test_vision/test_fret_estimator.py](tests/test_vision/test_fret_estimator.py) | `FretEstimator` | 11 |
| [tests/test_vision/test_technique.py](tests/test_vision/test_technique.py) | `VisionTechniqueClassifier` | 5 |
| [tests/test_fusion/test_late_fusion.py](tests/test_fusion/test_late_fusion.py) | `LateFusion` | 8 |
| [tests/test_output/test_rhythm.py](tests/test_output/test_rhythm.py) | 양자화·마디 분할·화음 | 28 |
| [tests/test_output/test_midi_writer.py](tests/test_output/test_midi_writer.py) | `write_midi` (실파일 라운드트립) | 7 |
| [tests/test_output/test_musicxml_writer.py](tests/test_output/test_musicxml_writer.py) | `write_musicxml` (실파일 파싱) | 15 |
| [tests/test_pipeline.py](tests/test_pipeline.py) | `Pipeline.run()` 통합 + manifest | 8 |

총 **165 tests**, 모두 PASS · ruff clean.

> 📌 **직렬화는 모킹하지 말 것.** PR #22 에서 `guitarpro.write` 를 모킹한 탓에
> "한 beat 안의 현 중복" 으로 파일이 깨지는 버그를 유닛 테스트가 놓쳤다.
> 출력 라이터는 실제로 파일을 쓰고 되읽는 **라운드트립 테스트**로 검증한다.

`tests/` 이하에는 `__init__.py` 가 없다(PEP 420 namespace packages).
같은 basename 테스트 파일 충돌을 피하려고 pytest `--import-mode=importlib` 를 쓴다.

미존재 — 향후 필요:
- `tests/integration/` (E2E 실제 짧은 클립으로 검증)

---

## 외부 모델/가중치

| 모델 | 위치 (예정) | 상태 |
|------|-----------|------|
| Basic Pitch ICASSP-2022 | `basic_pitch.ICASSP_2022_MODEL_PATH` | ✅ 패키지 번들 |
| Demucs htdemucs_6s | HF 자동 다운로드 (`demucs.pretrained.get_model`) | ✅ 첫 실행 시 자동 다운로드 |
| MediaPipe hand_landmarker | `~/.cache/guitarvideo2tab/models/hand_landmarker.task` | ✅ 없으면 1회 자동 다운로드 (PR #21) |
| YOLOv8-OBB fretboard | `models/yolo/fretboard.pt` | ❌ 자체 학습 필요 (현재 기본 yolov8n-obb.pt) |
| 비전 기법 분류기 | `models/vision_technique/` (state_dict) | ❌ Mitsou 2023 데이터 학습 필요. `model_factory` 함께 제공 |
| TART MLP | `models/tart/` (state_dict) | ❌ 가중치 공개 여부 확인 필요. `model_factory` 함께 제공 |

---

## 구현 진행률

전체 14개 핵심 모듈 (`pipeline`, `workspace` 포함) **모두 구현 완료**. NotImplementedError 0건.

가중치 미공개 분류기 2종 (TART, Mitsou 비전 기법)은 **state_dict + model_factory 패턴**으로 `weights_only=True` 보안 로딩 — 가중치만 확보되면 즉시 활성. 미설정 시 빈 라벨 리스트 폴백.

`python -m guitarvideo2tab <local.mp4>` 실행 시:
1. ✅ 다운로드/분리/stem 추출 동작
2. ✅ Basic Pitch AMT 동작
3. 🟡 기법 분류기는 빈 라벨 (가중치 없음)
4. ✅ 비전 경로 동작 (YOLO 기본 가중치 사용 — 정확도는 낮음)
5. ✅ Late Fusion → 리듬 양자화 → MIDI + MusicXML 출력
6. ✅ 산출물은 `runs/<run_id>/` 에 격리, `manifest.json` 에 통계 기록

---

## 알려진 결함

| # | 위치 | 증상 | 상태 |
|---|------|------|------|
| 1 | `fusion/late_fusion.py` `_resolve_string_fret` | **`event.pitch` 를 참조하지 않음.** 동시 발음이 전부 같은 `(string, fret)` 을 받고, 배정된 자리가 그 음을 못 냄 (`pitches=[58,46] → (6,14),(6,14)`). 실측 `fingering_match_ratio = 0.0` | 🔴 미해결 · 최우선 |
| 2 | `fusion/late_fusion.py` `_resolve_string_fret` | 후보 없을 때 `(-1,-1)` 반환. ADR-001 D4 의 폐색 폴백(오디오 prior + 손 위치) 미구현 | ⚠️ 미해결 |
| 3 | `output/rhythm.py` | 마디 경계를 넘는 긴 음이 타이 없이 잘림. 박자표 4/4 고정, 잇단음표 미지원 | ⚠️ 알려진 한계 |
| 4 | `output/rhythm.py` `estimate_tempo_from_audio` | librosa 가 2배 tempo 를 잡을 수 있음(실측 143.6 vs IOI 추정 93.75) | ⚠️ 검증 필요 |

1번 수정 방향: pitch 로 후보를 먼저 제약(`fret = pitch - tuning[string-1]`, 6개 이하)한 뒤
비전이 관측한 손 위치에 가장 가까운 것을 택하고, 동시 발음은 서로 다른 현에 이분 매칭.
