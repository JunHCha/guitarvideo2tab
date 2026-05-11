"""TART 2단계 MLP 기반 오디오 표현 기법 분류기.

TART(Technique-Aware Real-Time) 는 per-note 스펙트럴 특징 + pitch contour 를
입력으로 받아 기타 표현 기법 라벨과 신뢰도를 추출하는 2단계 MLP 모델이다.

현재 TART 사전 학습 가중치는 공개되지 않았으므로:
  - weights_path=None  → 빈 리스트 반환 (라벨 없음). 의도된 동작.
  - weights_path 지정 시 → torch.load 로 모델 로드 후 per-note 추론 수행.

오디오 특징 추출:
  scipy.signal.spectrogram 을 기본으로 사용한다 (librosa 는 선택적 의존성).
  scipy 는 pyproject.toml 에 명시된 의존성이므로 항상 사용 가능하다.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..models import MidiEvent, TechniqueAnnotation

try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    import torch  # noqa: F811

# TechniqueLabel 순서 — models.py 의 Literal 순서와 일치
_LABELS: list[str] = [
    "bend",
    "slide",
    "hammer-on",
    "pull-off",
    "vibrato",
    "palm-mute",
    "tapping",
    "sweep-picking",
    "alternate-picking",
    "legato",
]


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """오디오 파일을 float32 mono 배열로 읽는다.

    scipy.io.wavfile → soundfile → wave 순으로 시도하며,
    어느 백엔드도 없으면 RuntimeError 를 발생시킨다.
    """
    sf_mod = importlib.util.find_spec("soundfile")
    try:
        import scipy.io.wavfile as wf  # type: ignore[import]

        sr, data = wf.read(str(audio_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = data.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.iinfo(data.dtype).max  # type: ignore[attr-defined]
        return audio, int(sr)
    except Exception:
        pass

    if sf_mod is not None:
        import soundfile as sf  # type: ignore[import]

        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    raise RuntimeError(
        f"오디오 파일 읽기 실패: {audio_path}. "
        "scipy 또는 soundfile 이 필요합니다."
    )


def _extract_feature(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    n_mels: int = 64,
    n_fft: int = 1024,
) -> "torch.Tensor":
    """주어진 시간 구간의 per-note 스펙트럴 특징 텐서를 반환한다.

    scipy.signal.spectrogram 으로 log-power 스펙트럼을 계산한 뒤 mel filterbank
    로 압축한다. librosa 가 없어도 동작하도록 scipy 만 사용한다.

    Args:
        audio: mono float32 waveform (shape: [T]).
        sr: 샘플링 레이트 (Hz).
        start_time: 노트 시작 시각 (초).
        end_time: 노트 종료 시각 (초).
        n_mels: mel bin 수.
        n_fft: FFT 윈도우 크기.

    Returns:
        shape (1, n_mels, frames) 의 float32 Tensor.
    """
    from scipy.signal import spectrogram  # type: ignore[import]

    # --- 샘플 슬라이싱 -------------------------------------------------------
    i_start = max(0, int(start_time * sr))
    i_end = min(len(audio), int(end_time * sr))
    if i_end <= i_start:
        # 길이가 0인 구간 → 영 텐서 반환
        return torch.zeros(1, n_mels, 1, dtype=torch.float32)

    segment = audio[i_start:i_end]

    # --- scipy 스펙트로그램 ---------------------------------------------------
    hop = n_fft // 4
    freqs, _times, sxx = spectrogram(
        segment,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        window="hann",
        scaling="spectrum",
    )

    # --- 간단한 mel filterbank (삼각형) ------------------------------------
    f_min, f_max = 27.5, min(float(sr) / 2, 8000.0)
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, len(freqs) - 1)

    filterbank = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        if f_center > f_left:
            filterbank[m - 1, f_left:f_center] = (
                np.arange(f_left, f_center) - f_left
            ) / (f_center - f_left)
        if f_right > f_center:
            filterbank[m - 1, f_center:f_right] = (
                f_right - np.arange(f_center, f_right)
            ) / (f_right - f_center)

    mel_spec = filterbank @ sxx.astype(np.float32)  # (n_mels, frames)
    log_mel = np.log1p(mel_spec)

    tensor = torch.from_numpy(log_mel).unsqueeze(0)  # (1, n_mels, frames)
    return tensor


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@dataclass
class TARTTechniqueClassifier:
    """TART per-note 기타 표현 기법 분류기.

    TART(Technique-Aware Real-Time) 가중치는 현재 공개되지 않았다.
    weights_path=None 일 때는 빈 리스트를 반환한다(의도된 기본 동작).
    가중치가 공개되면 weights_path 를 지정하여 실제 추론이 가능하다.

    Args:
        weights_path: 사전 학습 가중치 파일 경로.
                      None 이면 추론 없이 빈 리스트 반환.
    """

    weights_path: Path | None = None

    def classify(
        self,
        midi_events: list[MidiEvent],
        audio_path: Path,
    ) -> list[TechniqueAnnotation]:
        """노트 이벤트 목록에 대한 기법 라벨을 반환한다.

        Args:
            midi_events: BasicPitchTranscriber 가 생성한 노트 이벤트 목록.
            audio_path: 분석할 오디오 파일 경로.

        Returns:
            MidiEvent 와 1:1 대응하는 TechniqueAnnotation 목록.
            weights_path=None 이면 빈 리스트 반환(TART 가중치 미공개).
        """
        # --- 가중치 없음: 의도된 폴백 ----------------------------------------
        if self.weights_path is None:
            # TART 사전 학습 가중치가 공개되지 않아 추론 불가.
            # 파이프라인이 오류 없이 동작하도록 빈 리스트를 반환한다.
            return []

        # --- 모델 로드 --------------------------------------------------------
        model = torch.load(self.weights_path, map_location="cpu")
        model.eval()

        # --- 오디오 로드 ------------------------------------------------------
        audio, sr = _load_audio(Path(audio_path))

        # --- per-note 추론 ----------------------------------------------------
        annotations: list[TechniqueAnnotation] = []
        with torch.no_grad():
            for event in midi_events:
                feature = _extract_feature(
                    audio, sr, event.start_time, event.end_time
                )
                logits = model(feature)  # (1, num_classes) or (num_classes,)
                logits = logits.squeeze(0) if logits.dim() > 1 else logits

                probs = torch.softmax(logits.float(), dim=-1)
                label_idx = int(probs.argmax().item())
                confidence = float(probs[label_idx].item())

                label = _LABELS[label_idx % len(_LABELS)]
                annotations.append(
                    TechniqueAnnotation(
                        technique=label,  # type: ignore[arg-type]
                        confidence=confidence,
                        source="audio",
                        params={},
                    )
                )

        return annotations
