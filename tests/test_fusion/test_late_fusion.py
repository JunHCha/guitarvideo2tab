"""Tests for LateFusion (ADR-001 D4 weighted-voting merge)."""
from __future__ import annotations

from guitarvideo2tab.fusion.late_fusion import LateFusion
from guitarvideo2tab.models import FretPosition, MidiEvent, TechniqueAnnotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _midi(start: float = 0.0, end: float = 1.0) -> MidiEvent:
    return MidiEvent(pitch=60, start_time=start, end_time=end, velocity=80)


def _fret_pos(
    timestamp: float = 0.5,
    string: int = 2,
    fret: int = 5,
    confidence: float = 0.9,
) -> FretPosition:
    return FretPosition(timestamp=timestamp, string=string, fret=fret, confidence=confidence)


def _audio_tech(label: str = "bend", confidence: float = 0.9) -> TechniqueAnnotation:
    return TechniqueAnnotation(technique=label, confidence=confidence, source="audio", params={})


def _vision_tech(
    label: str = "bend",
    confidence: float = 0.9,
    window_start: float = 0.0,
    window_end: float = 1.0,
) -> TechniqueAnnotation:
    return TechniqueAnnotation(
        technique=label,
        confidence=confidence,
        source="vision",
        params={"window_start": window_start, "window_end": window_end},
    )


# ---------------------------------------------------------------------------
# Test 1: Single MidiEvent + matching FretPosition (high confidence)
# ---------------------------------------------------------------------------

def test_fret_position_high_confidence_used():
    """FretPosition within event window at high confidence → (string, fret) from vision."""
    fusion = LateFusion()
    event = _midi(0.0, 1.0)
    fp = _fret_pos(timestamp=0.5, string=3, fret=7, confidence=0.95)

    results = fusion.fuse(
        midi_events=[event],
        audio_techniques=[],
        fret_positions=[fp],
        vision_techniques=[],
    )

    assert len(results) == 1
    note = results[0]
    assert note.string == 3
    assert note.fret == 7
    assert note.midi_event is event


# ---------------------------------------------------------------------------
# Test 2: No matching FretPosition → string=-1, fret=-1
# ---------------------------------------------------------------------------

def test_no_fret_position_returns_unresolved():
    """When no FretPosition overlaps the event window → string=-1, fret=-1."""
    fusion = LateFusion()
    event = _midi(2.0, 3.0)
    fp = _fret_pos(timestamp=0.5)  # outside [2, 3]

    results = fusion.fuse(
        midi_events=[event],
        audio_techniques=[],
        fret_positions=[fp],
        vision_techniques=[],
    )

    note = results[0]
    assert note.string == -1
    assert note.fret == -1


# ---------------------------------------------------------------------------
# Test 3: Audio + vision agree, high confidence → fused label, source="fusion"
# ---------------------------------------------------------------------------

def test_audio_vision_agree_high_confidence_fused():
    """Both sources agree with >= 0.8 confidence → technique source is 'fusion'."""
    fusion = LateFusion()
    event = _midi(0.0, 1.0)
    audio = _audio_tech("slide", confidence=0.9)
    vision = _vision_tech("slide", confidence=0.85, window_start=0.0, window_end=1.0)

    results = fusion.fuse(
        midi_events=[event],
        audio_techniques=[audio],
        fret_positions=[],
        vision_techniques=[vision],
    )

    note = results[0]
    assert note.technique is not None
    assert note.technique.technique == "slide"
    assert note.technique.source == "fusion"
    # Confidence should be average
    assert abs(note.technique.confidence - (0.9 + 0.85) / 2) < 1e-9


# ---------------------------------------------------------------------------
# Test 4: Audio + vision disagree → vision wins, source="fusion"
# ---------------------------------------------------------------------------

def test_audio_vision_disagree_vision_wins():
    """Disagreement → vision label is chosen, source='fusion'."""
    fusion = LateFusion()
    event = _midi(0.0, 1.0)
    audio = _audio_tech("bend", confidence=0.9)
    vision = _vision_tech("hammer-on", confidence=0.85, window_start=0.0, window_end=1.0)

    results = fusion.fuse(
        midi_events=[event],
        audio_techniques=[audio],
        fret_positions=[],
        vision_techniques=[vision],
    )

    note = results[0]
    assert note.technique is not None
    assert note.technique.technique == "hammer-on"
    assert note.technique.source == "fusion"


# ---------------------------------------------------------------------------
# Test 5: Only audio (vision list empty) → audio label, source="audio"
# ---------------------------------------------------------------------------

def test_only_audio_technique_used():
    """No vision techniques → audio annotation passes through unchanged."""
    fusion = LateFusion()
    event = _midi(0.0, 1.0)
    audio = _audio_tech("vibrato", confidence=0.75)

    results = fusion.fuse(
        midi_events=[event],
        audio_techniques=[audio],
        fret_positions=[],
        vision_techniques=[],
    )

    note = results[0]
    assert note.technique is not None
    assert note.technique.technique == "vibrato"
    assert note.technique.source == "audio"


# ---------------------------------------------------------------------------
# Test 6: Only vision → vision label, source="vision"
# ---------------------------------------------------------------------------

def test_only_vision_technique_used():
    """No audio techniques (mismatched lengths) → vision annotation passes through."""
    fusion = LateFusion()
    event = _midi(0.0, 1.0)
    vision = _vision_tech("palm-mute", confidence=0.88, window_start=0.0, window_end=1.0)

    # No audio_techniques → audio_aligned=False → audio skipped
    results = fusion.fuse(
        midi_events=[event],
        audio_techniques=[],
        fret_positions=[],
        vision_techniques=[vision],
    )

    note = results[0]
    assert note.technique is not None
    assert note.technique.technique == "palm-mute"
    assert note.technique.source == "vision"


# ---------------------------------------------------------------------------
# Test 7: Neither audio nor vision → technique=None
# ---------------------------------------------------------------------------

def test_no_technique_returns_none():
    """No technique from either source → NoteEvent.technique is None."""
    fusion = LateFusion()
    event = _midi(0.0, 1.0)

    results = fusion.fuse(
        midi_events=[event],
        audio_techniques=[],
        fret_positions=[],
        vision_techniques=[],
    )

    note = results[0]
    assert note.technique is None
