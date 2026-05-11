"""Demucs 6-stem 모델로 믹스 → 기타 stem 분리."""
from __future__ import annotations

from pathlib import Path


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
    from demucs.api import Separator, save_audio  # lazy import — keeps tests fast

    output_dir.mkdir(parents=True, exist_ok=True)

    sep = Separator(model="htdemucs_6s")
    _origin, stems = sep.separate_audio_file(audio_path)

    if "guitar" not in stems:
        available = list(stems.keys())
        raise KeyError(
            f"'guitar' stem not found in model output. "
            f"Available stems: {available}. "
            "Make sure you are using a 6-stem model (htdemucs_6s)."
        )

    guitar_tensor = stems["guitar"]
    out = output_dir / "guitar.wav"
    save_audio(guitar_tensor, str(out), sep.samplerate)
    return out
