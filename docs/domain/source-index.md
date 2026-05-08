# Source Index

> 프로젝트 초기 단계 — 아직 소스 파일이 없음. 구현 후 업데이트 예정.

## 계획된 핵심 모듈

### 파이프라인 진입점

| 파일 (예정) | 역할 |
|------------|------|
| `src/guitarvideo2tab/pipeline.py` | 전체 파이프라인 오케스트레이터. URL 입력 → .gpx 출력 |
| `src/guitarvideo2tab/__main__.py` | CLI 진입점 |

### 전처리 모듈

| 파일 (예정) | 핵심 함수/클래스 | 역할 |
|------------|----------------|------|
| `preprocessing/downloader.py` | `download_video(url)` | yt-dlp로 YouTube 영상 다운로드 |
| `preprocessing/separator.py` | `split_audio_video(path)` | ffmpeg으로 오디오(WAV) + 비디오 분리, PTS 보존 |
| `preprocessing/stem.py` | `separate_guitar_stem(audio_path)` | Demucs 6s로 기타 stem 분리 |

### 오디오 경로 모듈

| 파일 (예정) | 핵심 함수/클래스 | 역할 |
|------------|----------------|------|
| `audio/transcriber.py` | `BasicPitchTranscriber.transcribe()` | Basic Pitch AMT — MIDI + pitch-bend curve |
| `audio/technique.py` | `TARTTechniqueClassifier.classify()` | TART 2단계 MLP — 기법 라벨 + 신뢰도 |

### 비전 경로 모듈

| 파일 (예정) | 핵심 함수/클래스 | 역할 |
|------------|----------------|------|
| `vision/fretboard.py` | `FretboardDetector.detect()` | YOLOv8-OBB 프렛보드 검출 + 호모그래피 |
| `vision/hands.py` | `HandTracker.track()` | MediaPipe Hands 21 keypoint 시계열 |
| `vision/fret_estimator.py` | `FretEstimator.estimate()` | keypoint → (string, fret) 후보 |
| `vision/technique.py` | `VisionTechniqueClassifier.classify()` | TCN/Transformer on keypoint 시계열 |

### Fusion 모듈

| 파일 (예정) | 핵심 함수/클래스 | 역할 |
|------------|----------------|------|
| `fusion/late_fusion.py` | `LateFusion.fuse()` | 오디오+비전 신뢰도 가중 투표 |

### 출력 모듈

| 파일 (예정) | 핵심 함수/클래스 | 역할 |
|------------|----------------|------|
| `output/tab_writer.py` | `TabWriter.write_gpx()` | AlphaTab/PyGuitarPro → .gpx/.gp5 |

---

## 핵심 데이터 타입 (예정)

```python
@dataclass
class MidiEvent:
    pitch: int
    start_time: float
    end_time: float
    velocity: int
    pitch_contour: list[tuple[float, float]]  # (time, pitch) 곡선

@dataclass
class TechniqueAnnotation:
    technique: str          # "bend", "slide", "hammer-on", ...
    confidence: float       # 0.0 ~ 1.0
    source: str             # "audio", "vision", "fusion"
    params: dict            # 기법별 파라미터 (bend: semitones, slide: direction)

@dataclass
class NoteEvent:
    midi_event: MidiEvent
    string: int             # 1-6
    fret: int               # 0-24
    technique: TechniqueAnnotation | None
```

---

## 외부 모델/가중치

| 모델 | 위치 (예정) | 비고 |
|------|-----------|------|
| Demucs htdemucs_6s | `models/demucs/` | Hugging Face 자동 다운로드 |
| YOLOv8-OBB fretboard | `models/yolo/fretboard.pt` | 직접 학습 필요 |
| 비전 기법 분류기 | `models/vision_technique/` | Mitsou 데이터셋으로 학습 |
| TART MLP | `models/tart/` | TART 논문 공개 여부 확인 필요 |

---

_구현 시작 후 실제 파일 경로와 함수 시그니처로 업데이트할 것._
