# Coding Style

## 기술 스택

### 언어
- **Python 3.10+** — 전체 파이프라인, ML/오디오/비전 처리

### 핵심 라이브러리

| 분류 | 라이브러리 | 용도 |
|------|-----------|------|
| 미디어 | yt-dlp | YouTube 영상 다운로드 |
| 미디어 | ffmpeg-python | 오디오/비디오 분리, 트랜스코딩 |
| 오디오 분리 | demucs | 6-stem 소스 분리 |
| AMT | basic-pitch (Spotify) | 오디오 → MIDI + pitch-bend |
| 컴퓨터 비전 | opencv-python | 프레임 처리, 호모그래피 |
| 컴퓨터 비전 | ultralytics (YOLOv8) | 프렛보드 OBB 검출 |
| 손 추적 | mediapipe | 손 21 keypoint 추출 |
| ML | torch, torchvision | TCN/Transformer 기법 분류기 |
| 음악 출력 | pyguitarpro | .gp5 파일 생성 |
| 음악 출력 | alphatab (Python 바인딩) | .gpx 파일 생성 |
| 수치 | numpy, scipy | 신호 처리, 행렬 연산 |
| 데이터 | pandas | 어노테이션, 타임라인 관리 |

### 개발 도구
- **uv** — 패키지 관리 (pip 대신)
- **ruff** — 린터 + 포매터
- **pytest** — 테스트
- **black** — 코드 포매팅 (ruff format으로 대체 가능)

---

## 프로젝트 구조 (목표)

```
guitarvideo2tab/
├── src/
│   └── guitarvideo2tab/
│       ├── __init__.py
│       ├── pipeline.py          # 전체 파이프라인 오케스트레이터
│       ├── preprocessing/
│       │   ├── downloader.py    # yt-dlp 래퍼
│       │   ├── separator.py     # ffmpeg 오디오/비디오 분리
│       │   └── stem.py          # Demucs stem 분리
│       ├── audio/
│       │   ├── transcriber.py   # Basic Pitch AMT
│       │   └── technique.py     # TART 기법 분류기
│       ├── vision/
│       │   ├── fretboard.py     # YOLOv8-OBB + 호모그래피
│       │   ├── hands.py         # MediaPipe 손 추적
│       │   ├── fret_estimator.py# String/Fret 위치 추정
│       │   └── technique.py     # TCN/Transformer 기법 분류기
│       ├── fusion/
│       │   └── late_fusion.py   # Late Fusion 통합 모듈
│       └── output/
│           └── tab_writer.py    # AlphaTab/PyGuitarPro 출력
├── models/                      # 학습된 모델 가중치
├── data/                        # 데이터셋 (gitignore)
├── tests/
├── docs/
│   └── domain/
├── pyproject.toml
└── README.md
```

---

## 코딩 규칙

### 일반
- 타입 힌트 필수 (Python 3.10+ union syntax: `X | Y`)
- 함수/클래스 docstring: 한 줄 요약만 (WHY가 자명하면 생략)
- 모듈별 단일 책임 원칙

### 모듈 인터페이스
각 파이프라인 모듈은 다음 패턴을 따름:

```python
class AudioTechniqueClassifier:
    def classify(self, midi_events: list[MidiEvent]) -> list[TechniqueAnnotation]:
        ...
```

- 입력/출력 타입을 명확히 정의한 dataclass 사용
- 신뢰도(confidence: float) 항상 포함

### Pitch Bend 처리
- Basic Pitch `note_events` 출력의 피치 벤드 정보를 `PitchContour` 객체로 보존
- MIDI note-on/off로 단순화 금지 — 연속 곡선 유지 필수

### Late Fusion 규칙
신뢰도 임계값은 `fusion/late_fusion.py` 상단 상수로 관리:

```python
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5
```

### 테스트
- 단위 테스트: 각 모듈별 독립 테스트
- 통합 테스트: 짧은 기타 클립(~10초)으로 전체 파이프라인 검증
- TART/SpectroFusionNet 재현 수치와 비교 벤치마크

---

## 학습 데이터

| 데이터셋 | 용도 | 위치 |
|---------|------|------|
| Mitsou et al. (2023) | 비전 기법 분류기 학습 (9기법, 549 MP4) | `data/mitsou2023/` |
| GuitarSet | 오디오 AMT 검증 | `data/guitarset/` |
| GAPS | 오디오 기법 분류기 학습 (TART 참조) | `data/gaps/` |

---

## 린터 설정 (ruff)

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
```
