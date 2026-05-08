# Architecture

## 프로젝트 개요

기타 연주 영상(YouTube 등)을 입력으로 받아 Guitar Pro 형식(.gpx/.gp5)의 기타 악보(TAB)를 자동 생성하는 멀티모달 파이프라인 시스템.

**핵심 원칙**: "오디오는 무엇을(what), 비디오는 어디서·어떻게(where & how)"

---

## 전체 파이프라인

```
YouTube 영상
   │
   ├── ffmpeg ──→ [오디오 트랙]          ├── ffmpeg ──→ [비디오 트랙]
   │                 │                                      │
   │            Demucs 6s                            YOLOv8-OBB
   │            (기타 stem 분리)                     (프렛보드 검출 → 호모그래피)
   │                 │                                      │
   │         ┌───────┴───────┐                    MediaPipe Hands
   │         │               │                    (21 keypoint × T 프레임)
   │    Basic Pitch      오디오 기법                         │
   │    (MIDI + pitch-   분류기                    ┌─────────┴─────────┐
   │     bend curve)   (TART 2단계)          String/Fret        비전 기법
   │         │               │                추정기             분류기
   │    MIDI events    기법 라벨               │              (TCN/Transformer
   │    + bend curve   + 신뢰도         (string, fret)      on keypoint seq)
   │         │               │                후보                  │
   └─────────┴───────────────┴────────────────┴──────────────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │   Late Fusion 모듈   │
                              │                     │
                              │  string/fret: 비전 ≫ │
                              │  기법: 가중 투표      │
                              └─────────┬──────────┘
                                        │
                              MIDI + string/fret + technique 어노테이션
                                        │
                              AlphaTab / PyGuitarPro
                                        │
                                   .gpx / .gp5
```

---

## 모듈별 역할

### 1. 전처리 계층

| 모듈 | 라이브러리 | 출력 |
|------|-----------|------|
| 영상 다운로드 | yt-dlp | MP4 파일 |
| 오디오/비디오 분리 | ffmpeg | WAV + 프레임 시퀀스 |
| 기타 stem 분리 | Demucs 6-stem | guitar WAV |
| 프렛보드 검출 | YOLOv8-OBB | 4점 코너 → 호모그래피 행렬 |

### 2. 오디오 경로

| 모듈 | 라이브러리 | 출력 |
|------|-----------|------|
| AMT (음 인식) | Basic Pitch (Spotify) | MIDI + pitch-bend curve |
| 오디오 기법 분류기 | TART 2단계 MLP | 기법 라벨 + 신뢰도 |

- Pitch bend 정보는 `--save-note-events`로 별도 보존 (단순 MIDI로 환원 금지)
- 연속 pitch contour를 저장 후 .gpx 변환 시 bend 기호로 매핑

### 3. 비디오 경로

| 모듈 | 라이브러리 | 출력 |
|------|-----------|------|
| 손 keypoint 추출 | MediaPipe Hands | 21 keypoint × 2D × T 프레임 |
| String/Fret 추정 | 커스텀 모델 | (string, fret) 후보 per 프레임 |
| 비전 기법 분류기 | 1D-CNN / TCN / Transformer | 기법 라벨 + 신뢰도 |

- 슬라이딩 윈도우 ±300ms 단위로 시계열 분석
- 학습 데이터: Mitsou et al.(2023) 멀티모달 데이터셋 (549개 MP4, 9개 기법)

### 4. Late Fusion 모듈

의사결정 규칙:

```
오디오 신뢰도 > 0.8 AND 비전 신뢰도 > 0.8 AND 일치 → 확정
오디오 ≠ 비전               → 비전 우선 (직접 관찰)
비전 실패 (가림)             → 오디오만 사용
빠른 패시지 (비전 흐릿)      → 오디오 + 비전 trajectory 평균
```

String/Fret: 비전 직접 관찰 우선
기법: 가중 투표 (Bayesian fusion)

### 5. 출력 계층

| 모듈 | 라이브러리 | 출력 |
|------|-----------|------|
| TAB 생성 | AlphaTab / PyGuitarPro | .gpx / .gp5 |

Guitar Pro 형식 지원 기법:
- `Note.bendPoints` — bend (반음 단위 시간-피치 곡선)
- `Note.slideInType / slideOutType` — slide (legato/shift)
- `Note.isHammerPullOrigin` — hammer-on / pull-off
- `Note.vibrato` — vibrato
- palm mute, tapping, harmonics, dead note

---

## 데이터 흐름 타임라인

```
t=0        t=1        t=2        t=3
  영상/오디오 분리
             └── 오디오 처리 (MIDI + 기법)
                  └── 비디오 처리 (keypoint + 기법)
                           └── 동기화 (PTS 기반 타임스탬프)
                                    └── Fusion → TAB 생성
```

PTS(Presentation Timestamp) 보존으로 오디오-비디오 동기화 유지.

---

## 모듈 교체 가능성

각 모듈이 독립적으로 검증·교체 가능한 구조:

- Basic Pitch → 후속 AMT 모델로 교체 시 오디오 경로만 수정
- MediaPipe → 더 정확한 손 추적기 교체 시 비디오 경로만 수정
- TART → 새 기법 분류기로 교체 시 오디오 기법 모듈만 수정

---

## 관련 선행 연구

| 연구 | 기여 | 한계 |
|------|------|------|
| TART (UC Berkeley, 2025) | MLP 기반 오디오 기법 분류기 + Basic Pitch MIDI | 비디오 미활용 |
| SpectroFusionNet (2025) | 99.12% 기법 분류 (격리 환경), late fusion max voting 검증 | 실세계 70.9%로 저하 |
| Mitsou et al. (2023) | 9개 기법 549개 멀티모달 데이터셋 공개 | 종단간 시스템 없음 |
| Paleari & Huet (2008) | 시청각 융합으로 89% 음 모호성 해소 | 기법 미포함 |
| UIST 베이스 (2025) | 비디오 기반 핑거링 Bi-LSTM 추정 | 베이스 전용, 기법 미포함 |
