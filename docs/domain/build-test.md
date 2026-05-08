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
uv sync --dev
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
uv add pyguitarpro numpy scipy pandas

# 개발 의존성
uv add --dev pytest ruff
```

---

## 실행

### 기본 실행 (구현 후)

```bash
# YouTube URL → .gpx 변환
uv run python -m guitarvideo2tab "https://youtube.com/watch?v=..."

# 로컬 파일 입력
uv run python -m guitarvideo2tab path/to/video.mp4 --output output.gpx

# 중간 결과 저장 (디버깅용)
uv run python -m guitarvideo2tab "URL" --save-intermediates
```

### 파이프라인 단계별 실행

```bash
# 오디오 분리만
uv run python -m guitarvideo2tab.preprocessing.stem audio.wav

# 오디오 기법 분류만
uv run python -m guitarvideo2tab.audio.technique midi.mid

# 비전 기법 분류기 학습
uv run python -m guitarvideo2tab.vision.technique train --data data/mitsou2023/
```

---

## 테스트

```bash
# 전체 테스트
uv run pytest

# 특정 모듈 테스트
uv run pytest tests/test_audio/
uv run pytest tests/test_vision/
uv run pytest tests/test_fusion/

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
