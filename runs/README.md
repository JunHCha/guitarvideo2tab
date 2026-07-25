# runs/ — 실행별 산출물 워크스페이스

파이프라인 실행 **한 번이 디렉토리 하나**를 쓴다. trial 을 반복해도 어느 파일이
어느 실행에서 나왔는지 섞이지 않는다.

> 이 디렉토리의 내용물은 `.gitignore` 로 전부 제외된다 (`runs/*`).
> 추적되는 건 이 `README.md` 와 `.gitkeep` 뿐이다.

## 레이아웃

```
runs/
├── latest -> 20260725-234512_주이름-찬양-Verse-1     심볼릭 링크 (최신 실행)
└── 20260725-234512_주이름-찬양-Verse-1/
    ├── manifest.json      실행 메타데이터 — 무엇을 어떤 설정으로 돌렸는가
    ├── input/             원본 입력
    ├── stages/            단계별 중간 산출물
    │   ├── video.mp4              01 다운로드
    │   ├── audio.wav              02 오디오 분리
    │   ├── video-only.mp4         02 비디오 분리
    │   ├── guitar.wav             03 Demucs 기타 stem
    │   ├── 04_midi_events.json    Basic Pitch 채보
    │   ├── 05_audio_techniques.json
    │   ├── 06_fretboards.json     YOLOv8-OBB
    │   ├── 07_hands.json          MediaPipe Hands
    │   ├── 08_fret_positions.json
    │   ├── 09_vision_techniques.json
    │   └── 10_notes_fused.json    Late Fusion 결과
    └── output/
        ├── tab.mid                MIDI (양자화된 리듬)
        └── tab.musicxml           MusicXML (MuseScore/Guitar Pro/TuxGuitar)
```

run_id 는 `<타임스탬프>_<입력이름 슬러그>` 형식이라 정렬하면 시간순이 된다.
같은 초에 두 번 실행하면 `-2`, `-3` 접미사가 붙어 덮어쓰지 않는다.

## manifest.json

```jsonc
{
  "run_id": "20260725-234512_주이름-찬양-Verse-1",
  "created_at": "2026-07-25T23:45:12",
  "source": "/path/to/input.mp4",
  "git_commit": "b4b6578",              // 어느 코드로 돌렸는지
  "config": { "tempo_bpm": null, "subdivision": 16, "tuning": [40, 45, ...] },
  "stages": [                            // 단계별 소요 시간과 산출 개수
    { "stage": "01_download", "elapsed_sec": 0.4, "path": "stages/video.mp4" },
    { "stage": "04_transcribe", "count": 352, "elapsed_sec": 9.1 }
  ],
  "outputs": { "midi": "output/tab.mid", "musicxml": "output/tab.musicxml" },
  "totals": {
    "midi_events": 352,
    "notes_resolved": 227,
    "measures": 22,
    "notes_written": 272,
    "chords": 43,
    "fingering_match_ratio": 0.0,        // ← 품질 지표, 아래 참고
    "elapsed_sec": 129.6
  }
}
```

### `fingering_match_ratio` 읽는 법

`(string, fret)` 은 음높이를 정확히 결정한다(`tuning[string-1] + fret`).
이 값이 Basic Pitch 가 채보한 pitch 와 일치한 음표의 비율이다.

- **1.0** — 비전이 배정한 운지가 들리는 음과 모두 맞다
- **0.0** — 운지가 전부 모순된다. MusicXML 에는 pitch 만 기록되고 TAB 운지는 생략된다

현재 0.0 이 나오는 것은 알려진 버그다 — `LateFusion._resolve_string_fret` 이
후보를 고를 때 `event.pitch` 를 전혀 참조하지 않는다.

## 사용법

```bash
# 실행 (기본으로 runs/ 아래에 새 디렉토리 생성)
uv run python -m guitarvideo2tab "https://youtube.com/watch?v=..."
uv run python -m guitarvideo2tab path/to/video.mp4 --tempo 72

# 지난 실행 목록
uv run python -m guitarvideo2tab list

# 최신 실행 결과 열기
open runs/latest/output/tab.musicxml
```

### 정리

```bash
# 30일 지난 실행 삭제
find runs -maxdepth 1 -type d -name "20*" -mtime +30 -exec rm -rf {} +
```
