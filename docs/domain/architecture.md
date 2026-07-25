# Architecture

## 프로젝트 개요

기타 연주 영상(YouTube 등)을 입력으로 받아 **MIDI(.mid)** 와 **MusicXML(.musicxml)** 기타 악보를 자동 생성하는 멀티모달 파이프라인 시스템.

MusicXML 은 MuseScore·Guitar Pro·TuxGuitar 등이 모두 읽는 표준 교환 포맷이며, `<technical><string>/<fret>` 으로 TAB 운지를 표기한다.

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
                              ┌─────────▼──────────┐
                              │   리듬 양자화 모듈    │
                              │  (output/rhythm)    │
                              │                     │
                              │  tempo 추정 → 격자   │
                              │  스냅 → 화음 그룹핑  │
                              │  → 마디 분할/쉼표    │
                              └─────────┬──────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │                           │
                     mido (MIDI)            ElementTree (MusicXML)
                          │                           │
                     tab.mid                    tab.musicxml
```

출력은 **하나의 틱 격자**(4분음표 = 960틱)에서 파생되므로 MIDI 와 악보의 리듬이
정확히 일치한다. 이 값은 MIDI 의 `ticks_per_beat` 이자 MusicXML 의 `divisions` 다.

---

## 모듈별 역할

### 1. 전처리 계층

| 모듈 | 라이브러리 | 출력 |
|------|-----------|------|
| 영상 다운로드 | yt-dlp | MP4 파일 |
| 오디오/비디오 분리 | ffmpeg | WAV + 프레임 시퀀스 |
| 기타 stem 분리 | Demucs 6-stem | guitar WAV |
| 프렛보드 검출 | YOLOv8-OBB | 4점 코너 → 호모그래피 행렬 |
| 실행 워크스페이스 | 자체 (`workspace.py`) | `runs/<run_id>/` + manifest.json |

### 2. 오디오 경로

| 모듈 | 라이브러리 | 출력 |
|------|-----------|------|
| AMT (음 인식) | Basic Pitch (Spotify) | MIDI + pitch-bend curve |
| 오디오 기법 분류기 | TART 2단계 MLP | 기법 라벨 + 신뢰도 |

- Pitch bend 정보는 `PitchContour` 로 별도 보존 (단순 MIDI note-on/off 로 환원 금지)
- 연속 pitch contour 는 악보 변환 시 bend 기호로 매핑 (현재 미구현 — 기법 가중치 미공개)

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

### 5. 리듬 해석 계층

| 모듈 | 라이브러리 | 출력 |
|------|-----------|------|
| tempo 추정 | librosa `beat_track` | BPM + 첫 다운비트 |
| 양자화·마디 분할 | 자체 (`output/rhythm.py`) | `QuantizedBeat` 목록 |

Basic Pitch 의 절대 시각(초)을 틱 격자로 옮기는 단계다. 이 계층이 없으면 모든
음이 한 마디에 4분음표로 쌓여 악보를 읽을 수 없다.

처리 순서: tempo 추정 → 격자 스냅 → 화음 그룹핑 → 길이 결정 → 마디 분할 → 쉼표 채움.

- 표현 불가능한 길이는 **내림**하고 잔여분을 쉼표로 메워 마디 총합을 항상 맞춘다
- 한 현에 두 음이 겹치면 velocity 가 큰 쪽만 남긴다(물리적 제약)
- 현이 미해결(`string=-1`)인 음은 현이 아니라 **pitch 기준**으로 추린다

### 6. 출력 계층

| 모듈 | 라이브러리 | 출력 |
|------|-----------|------|
| MIDI 생성 | mido | `.mid` |
| 악보 생성 | 표준 `xml.etree.ElementTree` | `.musicxml` (score-partwise 4.0) |

MusicXML 은 MuseScore / Guitar Pro / TuxGuitar 가 모두 읽는 표준 포맷이라
바이너리 포맷 라이브러리 없이 표준 라이브러리만으로 정확히 생성할 수 있다.

**pitch 와 운지의 관계** — `(string, fret)` 은 음높이를 정확히 결정한다
(`tuning[string-1] + fret`). 비전 경로가 이 제약을 검증하지 않아 둘이 어긋날 수
있으므로:

- `<pitch>` 는 **항상** Basic Pitch 값을 쓴다 (들리는 음의 정본)
- `<technical><string>/<fret>` 은 **둘이 일치할 때만** 기록한다

어긋난 운지를 적으면 연주자가 다른 음을 내게 되므로, 적지 않는 편이 낫다.
생략 비율은 `manifest.json` 의 `fingering_match_ratio` 로 노출되어 비전 경로의
품질 지표가 된다.

> ⚠️ 현 번호 규약이 서로 다르다. 이 코드베이스는 `string=1` 이 **최저음현**이지만
> MusicXML 은 기타 관습대로 **1 이 최고음현**이다. `to_musicxml_string()` 으로
> 반전하지 않으면 TAB 이 위아래로 뒤집혀 표시된다.

### 7. 실행 워크스페이스 계층

| 모듈 | 역할 |
|------|------|
| `workspace.py` | 실행 하나당 디렉토리 하나를 보장하고 `manifest.json` 기록 |

trial 을 반복해도 산출물이 섞이지 않도록 `runs/<run_id>/` 아래에 입력·중간
산출물·결과를 격리한다. 상세는 [`runs/README.md`](runs/README.md) 참조.

```
runs/<타임스탬프>_<슬러그>/
├── manifest.json   입력·설정·git 커밋·단계별 소요시간·결과 통계
├── input/
├── stages/         미디어 + 단계별 JSON dump
└── output/         tab.mid · tab.musicxml
```

---

## 데이터 흐름 타임라인

```
t=0        t=1        t=2        t=3
  영상/오디오 분리
             └── 오디오 처리 (MIDI + 기법)
                  └── 비디오 처리 (keypoint + 기법)
                           └── 동기화 (PTS 기반 타임스탬프)
                                    └── Fusion → 리듬 양자화 → MIDI/MusicXML
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
