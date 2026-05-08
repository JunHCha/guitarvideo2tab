---
id: "001"
title: "멀티모달 Late Fusion 파이프라인 채택"
status: Accepted
date: "2026-05-08"
deciders: ["프로젝트 설계 세션 (선행 연구 기반 워크플로우 개선 논의)"]
supersedes: null
---

# ADR-001: 멀티모달 Late Fusion 파이프라인 채택

기타 연주 영상 → Guitar Pro TAB 변환의 전체 아키텍처를 결정한다.
초기 "snapshot 기반" 워크플로우의 구조적 한계를 선행 연구 분석 기반으로 보강한 단일 의사결정.

## Context

### 초기 워크플로우의 구조적 한계

처음 제안된 워크플로우는 다음 골격을 가졌다: YouTube 다운로드 → 오디오/비디오 분리 → Audio→MIDI → MIDI 음 등장 시점의 단일 프레임에서 손 위치 추정 → string/fret 결정 → .gpx 출력.

방향성은 정확하지만 네 가지 결정적 결함이 발견됨:

1. **"Snapshot" 가정의 오류** — 단일 프레임 스냅샷에서 정적 (string, fret)만 추출하는 접근. 그러나 표현 기법(bend 0.2~0.5초, slide 연속 이동, vibrato 주기적 진동)은 본질적으로 시간에 걸친 동작이므로 한 프레임으로는 영원히 잡을 수 없음.
2. **Audio→MIDI 단계에서 표현 정보 소실** — 일반 MIDI는 note-on/off의 이산 이벤트. 벤딩이 "여러 개의 짧은 음"으로 채보되어 사라짐.
3. **String/Fret 결정만 시각에 의존** — 시각이 줄 수 있는 두 가지 정보(공간적 string/fret, 시간적 동작 패턴) 중 후자 미활용.
4. **.gpx 출력에 기법 표기 누락** — 표현 기법 없는 TAB은 멜로디만 따라치는 수준. Guitar Pro 형식이 지원하는 bend/slide/hammer-on/vibrato 메타데이터 손실.

### 선행 연구 검토 결과

| 연구 | 기여 | 한계 |
|------|------|------|
| TART (UC Berkeley, 2025) | MLP 기반 오디오 기법 분류기 + Basic Pitch MIDI | 비디오 미활용 |
| SpectroFusionNet (2025, Scientific Reports) | 9기법 99.12% 분류, max voting late fusion 검증 | 실세계 70.9%로 격차 큼 |
| Mitsou et al. (2023) | 9기법 549개 멀티모달 데이터셋 (3 기타 × 3 앰프) | 종단간 시스템 없음 |
| Paleari & Huet (2008) | 시청각 융합으로 89% 음 모호성 해소 | 표현 기법 미포함 |
| UIST 베이스 (2025) | 비디오 기반 핑거링 Bi-LSTM 추정 | 베이스 전용, 기법 미포함 |

**핵심 학계 공백**: 멀티모달(오디오+비디오) 입력 + 표현 기법까지 처리하는 종단간 TAB 생성 시스템은 사실상 부재. 이것이 본 프로젝트의 차별점이자 ROI가 가장 높은 contribution 지점.

### 시각적으로 표현 기법이 식별 가능함 (오디오 단독 대비 우위 근거)

| 기법 | 시각적 단서 |
|------|-----------|
| Bending | 운지 손가락의 줄에 수직 방향 움직임 |
| Slide | 운지 손가락의 줄과 평행한 프렛 사이 이동 |
| Hammer-on/Pull-off | 피킹 손이 정지한 상태에서 운지 손만 변화 |
| Vibrato | 운지 손가락의 빠른 좌우 진동 |
| Palm muting | 피킹 손바닥이 브릿지 근처 줄에 접촉 |
| Tapping | 보통 운지 외 손가락(또는 양손)이 프렛보드 위에 닿음 |
| Sweep picking | 피킹 손의 연속적인 단일 방향 스트로크 |

오디오만으로는 음색이 유사한 hammer-on/pull-off 분리가 어렵지만, 시각적으로는 피킹 손 동작 유무가 명확하다.

## Decision

**개선된 멀티모달 Late Fusion 파이프라인을 채택한다.**

핵심 원칙: **"오디오는 무엇을(what), 비디오는 어디서·어떻게(where & how)"**

### 전체 구조

```
YouTube 영상
   │
   ├── ffmpeg ──→ [오디오] ──→ Demucs 6s ──→ guitar stem
   │                                              │
   │                            ┌─────────────────┴──────────────┐
   │                            │                                 │
   │                       Basic Pitch                       오디오 기법
   │                       (MIDI + pitch-bend curve)         분류기 (TART 2단계 MLP)
   │
   └── ffmpeg ──→ [비디오] ──→ YOLOv8-OBB (프렛보드 → 호모그래피)
                                  │
                          MediaPipe Hands (21 keypoint × T)
                                  │
                       ┌──────────┴──────────┐
                       │                      │
                  String/Fret 추정     비전 기법 분류기
                  (프레임별 위치)      (TCN/Transformer on
                                       keypoint 시계열, ±300ms)
   
                                     ↓
                        ┌────────────────────────────┐
                        │   Late Fusion 통합 모듈      │
                        │  string/fret: 비전 우선      │
                        │  기법: 신뢰도 가중 투표       │
                        └────────────────────────────┘
                                     ↓
                       MIDI + string/fret + technique 어노테이션
                                     ↓
                       AlphaTab / PyGuitarPro → .gpx
```

### 다섯 가지 하위 결정

본 의사결정은 다음 다섯 가지 구체적 선택을 포함한다.

#### D1. 멀티모달 (오디오 + 비디오) 채택

오디오와 비디오를 **동등한 모달리티**로 격상. 비디오를 단순 보조가 아니라 (i) 공간적 string/fret 직접 관찰, (ii) 시간적 동작 패턴 학습의 두 채널로 활용한다.

#### D2. TART 오디오 기법 분류기 모듈 재사용

오디오 경로에서 표현 기법 분류기를 처음부터 만들지 않는다. TART(UC Berkeley, 2025)가 이미 검증한 "Basic Pitch 계열로 MIDI 추출 → MLP 분류기로 기법 라벨링" 2단계 구조를 모듈로 차용한다. 자체 학습은 본질적 R&D가 되므로 프로젝트 novelty(멀티모달 fusion)에서 벗어남.

#### D3. 비디오 경로에 시계열 모션 분류기 추가

비전 기법 분류기는 **TCN (또는 Transformer) on hand keypoint sequence**.
- 입력: MediaPipe 21 keypoint × 2D × T 프레임
- 슬라이딩 윈도우: ±300ms (~9프레임 @ 30fps)
- 학습 데이터: Mitsou et al.(2023) 멀티모달 데이터셋 (549 MP4, 9기법)
- 차원: `(T, 21, 2)` → flatten → `(T, 42)` (낮은 차원으로 가벼운 학습 가능)

초기 모델은 TCN, 데이터 확장 시 Transformer 업그레이드 검토.

#### D4. Late Fusion (Early/Joint Fusion 기각)

Early/joint fusion은 동기화 어노테이션 포함 대규모 멀티모달 데이터셋이 필요하지만 현재 549개 규모로는 불가. SpectroFusionNet이 max voting을 가장 효과적인 late fusion 기법으로 식별한 결과를 참고.

**결합 규칙:**
- String/fret: 비전 신뢰도 > 0.8 → 비전 우선 (직접 관찰); 비전 실패(가림) → 오디오 피치 + hand position prior
- 기법: 오디오·비전 신뢰도 가중 투표
  - 오디오 ≥ 0.8 AND 비전 ≥ 0.8 AND 일치 → 확정
  - 오디오·비전 불일치 → 비전 우선 (운지 위치 직접 관찰)
  - 비전 가림 → 오디오 단독
  - 빠른 패시지(비전 흐릿) → 오디오 + trajectory 평균
- 임계값: `CONFIDENCE_HIGH = 0.8`, `CONFIDENCE_LOW = 0.5`

#### D5. Pitch Bend 정보 보존 (단순 MIDI 환원 금지)

Basic Pitch `--save-note-events` 출력의 피치 벤드 정보를 별도 `PitchContour` 객체로 보존. 연속 pitch contour를 .gpx 변환 시 `Note.bendPoints`에 시간-피치 곡선으로 매핑. AlphaTab 데이터 모델이 bend point들을 직접 표현 가능.

```python
@dataclass
class PitchContour:
    note_id: str
    time_pitch_curve: list[tuple[float, float]]  # (초, MIDI pitch)
    bend_semitones: float
```

### 기법 정보의 .gpx 인코딩

융합 모듈 출력의 기법 라벨을 다음 필드에 직접 매핑:

| 기법 | Guitar Pro 인코딩 |
|------|------------------|
| Bend | `Note.bendPoints` (반음 단위 시간-피치 곡선) |
| Slide | `Note.slideInType` / `Note.slideOutType` (legato/shift) |
| Hammer-on / Pull-off | `Note.isHammerPullOrigin` |
| Vibrato | `Note.vibrato` |
| Palm mute | `Note.palmMute` |
| Tapping | `Note.isTapping` |
| Harmonic | `Note.harmonicType` |
| Dead note | `Note.isDeadNote` |

## Consequences

### 긍정적

- **String/fret 모호성 해소**: 비전 직접 관찰로 오디오 단독(TART) 대비 정확도 향상. Paleari & Huet(2008)의 89% 모호성 해소 결과 참고.
- **Hammer-on/Pull-off 분리 가능**: 피킹 손 동작 유무가 시각적으로 명확하므로 음색 유사로 인한 오디오 분류 한계를 극복.
- **오클루전 강건성**: 한 모달리티가 실패해도 다른 쪽이 백업.
- **모듈 교체 가능성**: 각 모듈(Basic Pitch, MediaPipe, TART, 비전 분류기)이 독립적으로 검증·교체 가능. 더 좋은 후속 모델로 교체 시 해당 모듈만 수정.
- **학계 공백 공략**: 멀티모달 + 표현 기법 종단간 시스템 부재 영역 → 연구 contribution 가능.
- **학습 데이터 효율**: TART 2단계 분류기 재사용 + Mitsou 데이터셋(이미 공개)로 비전 분류기만 학습.
- **표현 기법 완전 인코딩**: bend, slide, hammer-on, vibrato, palm mute 등 .gpx에 정확히 표기.

### 부정적

- **구현 복잡도 증가**: 오디오 단독 대비 파이프라인 구성 요소 약 2배.
- **오디오-비디오 동기화 위험**: PTS(Presentation Timestamp) 보존 실패 시 기법 매핑 오류. `ffmpeg` 분리 단계에서 PTS 보존 필수.
- **임계값 튜닝 필요**: `CONFIDENCE_HIGH/LOW` 값이 성능에 민감하므로 검증 데이터로 캘리브레이션 필요.
- **549개 샘플로 9클래스 학습**: 클래스당 ~60개로 과적합 위험 → TCN 우선, Transformer는 나중에.
- **MediaPipe 추적 실패 처리**: keypoint 누락 프레임 보간 로직 필요.
- **TART 공개 여부 불확실**: 공개 가중치가 없으면 논문 구조 재현 비용 발생.
- **빠른 패시지 비전 블러**: 폴백으로 오디오 단독 사용 → 해당 구간 기법 분류 정확도 일부 저하.

## Alternatives

### A. 초기 워크플로우 (Snapshot 기반) 그대로 유지
- **장점**: 구현 가장 단순, 빠른 프로토타입 가능.
- **단점**: 표현 기법 완전 누락, .gpx 실용성 크게 저하, 단일 프레임으로는 시간 동작 인식 원천 불가.
- **기각 이유**: 프로젝트 핵심 가치인 "기법 포함 TAB"을 달성 불가.

### B. 오디오 단독 (TART 그대로 활용)
- **장점**: 즉시 사용 가능한 SOTA, 구현 단순, 영상 처리 인프라 불필요.
- **단점**: String/fret 모호성 미해결, 시각 기법 정보 미활용, hammer-on/pull-off 분리 어려움, 일렉기타 이펙트에 약함.
- **기각 이유**: 프로젝트 차별점 부재 → 이미 존재하는 시스템 재구현에 그침.

### C. 비디오 단독
- **장점**: 시각만으로 string/fret 직접 관찰.
- **단점**: AMT(음 인식)를 비전만으로 하기 매우 어려움, 빠른 패시지에서 프레임 블러로 정확도 급락.
- **기각 이유**: AMT 품질이 오디오 단독 대비 현저히 낮을 것으로 예상.

### D. Early Fusion (Spectrogram + Optical Flow 결합)
- **장점**: 두 모달리티 상호작용을 end-to-end 학습.
- **단점**: 동기화 어노테이션 포함 대규모 멀티모달 데이터셋 필요. 현재 데이터(549개)로 학습 불가.
- **기각 이유**: 데이터 병목으로 비현실적.

### E. Cross-Attention (Joint Fusion)
- **장점**: 두 모달리티 간 주의집중 학습 가능.
- **단점**: Early fusion과 동일한 데이터 문제 + 구현 복잡도 매우 높음.
- **기각 이유**: ROI 낮음.

### F. SpectroFusionNet (오디오만의 MFCC-Gammatone late fusion)
- **장점**: 격리 환경 99.12% 정확도, late fusion 방식 검증.
- **단점**: 실세계 70.9%로 급락, 자체 학습 필요, 비전 정보 미활용.
- **기각 이유**: 실세계 성능 불확실 + 멀티모달 차별점 부재.

### G. 비전 단일 프레임 CNN (Snapshot)
- **장점**: 구현 단순.
- **단점**: 시간적 동작 패턴(bend, slide, vibrato) 인식 원천 불가.
- **기각 이유**: 표현 기법 인식 목표 달성 불가.

### H. 3D CNN (Video-level)
- **장점**: 시공간 특징 동시 학습.
- **단점**: 549개 데이터로 3D CNN 학습 불가, 계산 비용 매우 높음.
- **기각 이유**: 데이터 부족.

### I. 단순 MIDI note-on/off만 사용 (pitch bend 폐기)
- **장점**: 기존 MIDI 라이브러리 그대로 활용.
- **단점**: 벤딩 정보 완전 소실 → TAB 실용성 대폭 저하.
- **기각 이유**: 표현 기법 인코딩 목표 위반.

## References

### 선행 연구
- TART (UC Berkeley, 2025) — "Tablature-Aware Automatic Transcription and Recognition of Guitar Techniques"
- SpectroFusionNet (2025, Scientific Reports) — MFCC-Gammatone late fusion, 99.12% (격리) / 70.9% (실세계)
- Mitsou et al. (2023) — 멀티모달 기타 데이터셋 (549 MP4, 9기법)
- Paleari & Huet (2008) — "Audio-visual guitar transcription", 시청각 융합으로 89% 음 모호성 해소
- Chen et al. (2015, ISMIR) — 멜로디 컨투어 추출 + 기법 후보 식별 2단계 구조
- D'Hooge et al. (2023) — 벤딩 특화 25개 고수준 특징 기반 예측
- UIST 베이스 기타 (2025) — 비디오 기반 핑거링 Bi-LSTM 추정

### 도구 / 라이브러리
- Basic Pitch (Spotify): https://github.com/spotify/basic-pitch
- Demucs: https://github.com/facebookresearch/demucs
- YOLOv8-OBB (Ultralytics): https://docs.ultralytics.com/tasks/obb/
- MediaPipe Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- AlphaTab: https://www.alphatab.net/
- PyGuitarPro: https://github.com/Perlence/PyGuitarPro
- yt-dlp: https://github.com/yt-dlp/yt-dlp

### 데이터셋
- Mitsou et al. (2023): 549 MP4, 9 기법 (alternate picking, bend, hammer-on, legato, pull-off, slide, sweep picking, tapping, vibrato)
- GuitarSet: AMT 검증
- GAPS: 오디오 기법 분류 학습 (TART 참조)
