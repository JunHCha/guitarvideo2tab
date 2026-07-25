# Build & Test

## 환경 설정

### 요구사항
- Python 3.10+
- CUDA 11.8+ (GPU 학습/추론) 또는 MPS (Apple Silicon)
- ffmpeg 시스템 설치 필수

### 패키지 관리 (uv)

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 + 의존성 설치
uv sync

# 개발 의존성 포함
uv sync --extra dev
```

### 시스템 의존성

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt-get install ffmpeg

# ffmpeg 설치 확인
ffmpeg -version
```

---

## 의존성 설치 (pyproject.toml 기준)

```bash
# 핵심 의존성
uv add yt-dlp ffmpeg-python demucs basic-pitch
uv add opencv-python ultralytics mediapipe
uv add torch torchvision
uv add mido pretty_midi numpy scipy pandas

# 개발 의존성 (optional-dependencies 의 dev extra)
uv add --optional dev pytest pytest-cov ruff
```

---

## 실행

산출물은 실행마다 `runs/<타임스탬프>_<슬러그>/` 아래에 격리된다.

### 기본 실행

```bash
# YouTube URL → MIDI + MusicXML
uv run python -m guitarvideo2tab "https://youtube.com/watch?v=..."

# 로컬 파일 입력
uv run python -m guitarvideo2tab path/to/video.mp4

# tempo 고정 (추정 건너뜀 — librosa 가 2배로 잡을 때 유용)
uv run python -m guitarvideo2tab video.mp4 --tempo 72

# 양자화 해상도 변경 (기본 16 = 16분음표)
uv run python -m guitarvideo2tab video.mp4 --subdivision 8

# 단계별 JSON dump 생략 (미디어 파일은 그대로 저장됨)
uv run python -m guitarvideo2tab video.mp4 --no-intermediates

# 워크스페이스 위치 변경 (기본 runs/)
uv run python -m guitarvideo2tab video.mp4 --runs-dir /tmp/experiments
```

### 지난 실행 조회

```bash
uv run python -m guitarvideo2tab list
# 20260726-011530_주이름-찬양-Verse-1  22마디 음표 272 129.6s

open runs/latest/output/tab.musicxml     # 최신 실행 결과
cat runs/latest/manifest.json            # 설정·통계·git 커밋
```

### 산출물 구조

```
runs/<run_id>/
├── manifest.json   입력·설정·git 커밋·단계별 소요시간·결과 통계
├── input/
├── stages/         video.mp4 · audio.wav · guitar.wav
│                   04_midi_events.json ~ 10_notes_fused.json
└── output/         tab.mid · tab.musicxml
```

상세는 [`runs/README.md`](runs/README.md) 참조.

### 실측 소요 시간

37.6초 / 1129프레임 영상, Apple M1 Pro (CPU 추론) 기준 — 총 **173초**.

| 단계 | 시간 | 산출 |
|------|-----:|------|
| 02 오디오/비디오 분리 (ffmpeg) | 10.9s | 2 |
| 03 Demucs 기타 stem | 30.6s | — |
| 04 Basic Pitch 채보 | 1.4s | 348 events |
| **06 YOLOv8-OBB 프렛보드** | **82.1s** | 1129 frames |
| **07 MediaPipe Hands** | **46.9s** | 1129 frames |
| 08 fret 추정 | 0.02s | 1440 |
| 10 Late Fusion | 0.03s | 348 |
| 11·12 MIDI/MusicXML 출력 | 0.03s | — |

**비전 경로가 전체의 75%** 를 차지한다. 프레임 단위 추론이라 영상 길이에
선형 비례하므로, 긴 영상은 프레임 샘플링이나 GPU 가속이 필요하다.

### 디버깅

특정 단계가 의심스러우면 `stages/` 의 JSON 을 직접 확인한다.

```bash
# 단계별 소요 시간과 산출 개수
jq '.stages[] | {stage, count, elapsed_sec}' runs/latest/manifest.json

# 운지 일치율 (0.0 이면 비전 경로의 string/fret 배정이 신뢰 불가)
jq '.totals.fingering_match_ratio' runs/latest/manifest.json
```

---

---

## 테스트

```bash
# 전체 테스트 (165 tests)
uv run pytest

# 특정 모듈 테스트
uv run pytest tests/test_audio/
uv run pytest tests/test_vision/
uv run pytest tests/test_fusion/
uv run pytest tests/test_output/     # 리듬 양자화 + MIDI/MusicXML

# 통합 테스트 (짧은 클립)
uv run pytest tests/integration/ -v

# 커버리지
uv run pytest --cov=guitarvideo2tab --cov-report=html
```

### 테스트 데이터
- 단위 테스트: `tests/fixtures/` — 5~10초 샘플 오디오/비디오 클립
- 통합 테스트: `tests/integration/fixtures/` — 알려진 기법이 포함된 30초 클립
- 벤치마크: TART/SpectroFusionNet 논문 재현 수치와 비교

---

## 린트 & 포매팅

```bash
# 린트 검사
uv run ruff check src/

# 자동 수정
uv run ruff check --fix src/

# 포매팅
uv run ruff format src/
```

---

## 모델 다운로드

```bash
# Demucs 모델 (첫 실행 시 자동 다운로드)
python -c "import demucs; print('OK')"

# YOLOv8 기본 모델
python -c "from ultralytics import YOLO; YOLO('yolov8n-obb.pt')"

# MediaPipe hand_landmarker — 첫 실행 시 ~/.cache/guitarvideo2tab/models 로 자동 다운로드
```

---

## 데이터셋 준비

### Mitsou et al. (2023) — 비전 기법 분류기 학습용
```
data/mitsou2023/
├── videos/          # 549개 MP4
├── annotations/     # 기법 라벨
└── splits/          # train/val/test 분할
```

### GuitarSet — AMT 검증용
```
data/guitarset/
├── audio/
└── annotations/
```

---

## 환경 변수

```bash
# GPU 설정
CUDA_VISIBLE_DEVICES=0

# 모델 캐시 경로
HF_HOME=~/.cache/huggingface
TORCH_HOME=~/.cache/torch

# 데이터 경로
GUITARVIDEO2TAB_DATA_DIR=./data
GUITARVIDEO2TAB_MODEL_DIR=./models
```

---

## CI/CD (계획)

```yaml
# .github/workflows/test.yml
- Python 3.10, 3.11 매트릭스
- ffmpeg 설치
- CPU 전용 torch로 단위 테스트
- ruff lint 검사
```
