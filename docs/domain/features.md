# Features

## 핵심 기능

### F1. 영상 입력 및 미디어 분리
- YouTube URL 또는 로컬 MP4 파일 입력 지원 (yt-dlp)
- ffmpeg으로 오디오 트랙(WAV)과 비디오 프레임 분리
- PTS(Presentation Timestamp) 보존으로 오디오-비디오 동기화

### F2. 기타 음원 분리 (Source Separation)
- Demucs 6-stem 모델로 믹스에서 기타 stem 분리
- 드럼, 베이스, 보컬 등 배경 잡음 제거
- 분리된 기타 오디오만 AMT 파이프라인에 전달

### F3. 자동 음악 채보 (AMT)
- Basic Pitch (Spotify)로 기타 stem → MIDI 변환
- pitch-bend curve 정보 별도 보존 (`--save-note-events`)
- note-on/off 타임스탬프, 강도(velocity), 피치 정보 추출

### F4. 프렛보드 검출 및 좌표 변환
- YOLOv8-OBB(Oriented Bounding Box)로 비디오에서 프렛보드 사각형 검출
- 호모그래피 워핑으로 프렛보드를 정규화된 평면으로 변환
- 가려짐(occlusion) 프레임 자동 감지 및 마킹

### F5. 손 Keypoint 추출
- MediaPipe Hands로 좌손(운지)·우손(피킹) 21 keypoint 추출
- 프레임별 2D 좌표 시계열 저장
- 호모그래피 변환 후 프렛보드 좌표계로 정규화

### F6. String/Fret 위치 추정
- 운지 손 keypoint → 프렛보드 좌표계 매핑
- 어느 줄(string 1-6)의 어느 프렛(fret 0-24)에 위치하는지 추정
- 오디오 MIDI 피치와 교차 검증으로 모호성 해소

### F7. 표현 기법 인식 (Expression Techniques)

#### 오디오 기반
TART 2단계 MLP 분류기 활용:
- Bending (피치 벤드 커브 분석)
- Slide (연속 피치 변화)
- Hammer-on / Pull-off (어택 특성)
- Vibrato (주기적 피치 변동)
- Palm muting (음색 변화)

#### 비전 기반 (시계열 모션 분석)
슬라이딩 윈도우 ±300ms, TCN/Transformer:

| 기법 | 시각적 단서 |
|------|-----------|
| Bending | 운지 손가락의 줄에 수직 방향 움직임 |
| Slide | 프렛 사이 평행 이동 trajectory |
| Hammer-on/Pull-off | 피킹 손 정지 + 운지 손만 변화 |
| Vibrato | 운지 손가락의 빠른 좌우 진동 |
| Palm muting | 피킹 손바닥이 브릿지 근처 접촉 |
| Tapping | 양손 손가락이 프렛보드에 닿음 |
| Sweep picking | 피킹 손의 단일 방향 연속 스트로크 |

### F8. Late Fusion (멀티모달 통합)
- String/Fret: 비전 직접 관찰 우선
- 기법: 오디오 × 비전 신뢰도 가중 투표
- 오클루전 시 오디오 단독 폴백
- 빠른 패시지 시 trajectory 평균화

### F9. Guitar Pro 파일 생성 (.gpx / .gp5)
AlphaTab / PyGuitarPro를 통한 완전한 TAB 인코딩:
- 음표 위치 (string, fret)
- 리듬 정보 (note duration, tempo)
- 표현 기법 어노테이션:
  - bend (반음 단위 시간-피치 곡선)
  - slide (legato/shift 구분)
  - hammer-on / pull-off
  - vibrato, tapping, palm mute, harmonic, dead note

---

## 지원 표현 기법 목록

| 기법 | 오디오 검출 | 비전 검출 | .gpx 인코딩 |
|------|:---------:|:--------:|:----------:|
| Bend | ✓ (TART) | ✓ | ✓ bendPoints |
| Slide | ✓ (TART) | ✓ | ✓ slideType |
| Hammer-on | ✓ (TART) | ✓ | ✓ isHammerPull |
| Pull-off | ✓ (TART) | ✓ | ✓ isHammerPull |
| Vibrato | ✓ (TART) | ✓ | ✓ vibrato |
| Palm muting | ✓ (TART) | ✓ | ✓ |
| Tapping | ✓ (TART) | ✓ | ✓ |
| Sweep picking | ✓ (TART) | ✓ | - |
| Alternate picking | - | ✓ | - |

---

## 비기능 요구사항

- 모듈 교체 가능성: 각 모듈 독립적 교체 가능 (인터페이스 기반 설계)
- 가림(occlusion) 강건성: 비전 실패 시 오디오 폴백
- 배치 처리: 여러 영상 순차/병렬 처리 지원
