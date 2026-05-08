# guitarvideo2tab

기타 연주 영상(YouTube 등) → Guitar Pro TAB(.gpx/.gp5) 자동 변환 멀티모달 파이프라인.

> 표현 기법(bend, slide, hammer-on, vibrato 등)까지 포함한 종단간 변환을 목표로 합니다.

## 아키텍처

**"오디오는 무엇을(what), 비디오는 어디서·어떻게(where & how)"**

- **오디오 경로**: Demucs 6s → Basic Pitch (MIDI + pitch-bend) → TART 2단계 MLP 기법 분류
- **비디오 경로**: YOLOv8-OBB 프렛보드 → MediaPipe Hands → TCN/Transformer 기법 분류
- **Late Fusion**: string/fret은 비전 우선, 기법은 신뢰도 가중 투표
- **출력**: AlphaTab / PyGuitarPro로 .gpx/.gp5 생성

자세한 내용은 [docs/domain/](docs/domain/)를 참조하세요.

## 문서

- [docs/domain/architecture.md](docs/domain/architecture.md) — 전체 파이프라인 구조
- [docs/domain/features.md](docs/domain/features.md) — F1~F9 기능 목록
- [docs/domain/coding-style.md](docs/domain/coding-style.md) — 기술 스택과 코딩 규칙
- [docs/domain/build-test.md](docs/domain/build-test.md) — 빌드/테스트 환경
- [docs/domain/source-index.md](docs/domain/source-index.md) — 모듈/함수 인덱스
- [docs/domain/decisions/](docs/domain/decisions/) — Architecture Decision Records

## 빠른 시작

```bash
# 의존성 설치 (uv 권장)
uv sync

# CLI 실행 (구현 후)
uv run python -m guitarvideo2tab "https://youtube.com/watch?v=..." -o output.gpx
```

시스템 의존성: `ffmpeg`

## 상태

초기 skeleton 단계. 모듈 구현 진행 중.

## 라이선스

TBD
