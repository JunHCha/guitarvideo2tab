"""Demucs 6-stem 모델로 믹스 → 기타 stem 분리."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _run_demucs(audio_path: Path) -> tuple[dict[str, Any], int]:
    """Run htdemucs_6s on *audio_path* and return ``(stems, samplerate)``.

    Returns a mapping from stem name (``"guitar"``, ``"drums"``, …) to a
    ``[channels, time]`` tensor, plus the model's native sample rate.
    """
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    model = get_model("htdemucs_6s")
    model.eval()

    wav = AudioFile(audio_path).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(
        model,
        wav[None],
        device="cpu",
        shifts=1,
        split=True,
        overlap=0.25,
    )[0]
    sources = sources * ref.std() + ref.mean()

    return dict(zip(model.sources, sources)), model.samplerate


def _save_audio(tensor: Any, path: str, samplerate: int) -> None:
    """Write a ``[channels, time]`` tensor to *path* as WAV."""
    import soundfile as sf

    array = tensor.detach().cpu().numpy().T
    sf.write(path, array, samplerate)


def separate_guitar_stem(audio_path: Path, output_dir: Path) -> Path:
    """Separate the guitar stem from *audio_path* using htdemucs_6s.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory where ``guitar.wav`` will be written.

    Returns:
        Path to the saved ``guitar.wav`` file.

    Raises:
        KeyError: If the model did not produce a ``"guitar"`` stem.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        stems, samplerate = _run_demucs(audio_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "demucs>=4.0.0 and torchaudio are required. Run: uv sync"
        ) from e

    if "guitar" not in stems:
        available = list(stems.keys())
        raise KeyError(
            f"'guitar' stem not found in model output. "
            f"Available stems: {available}. "
            "Make sure you are using a 6-stem model (htdemucs_6s)."
        )

    out = output_dir / "guitar.wav"
    _save_audio(stems["guitar"], str(out), samplerate)
    return out
