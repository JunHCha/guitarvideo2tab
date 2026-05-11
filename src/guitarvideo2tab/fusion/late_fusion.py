"""Late Fusion: 오디오·비전 결과를 신뢰도 가중 투표로 결합 (ADR-001 D4).

Assumptions about audio_techniques timing
------------------------------------------
TARTTechniqueClassifier produces TechniqueAnnotation objects whose ``params``
dict may be empty (no explicit time window stored).  Because TART classifies
one technique per MIDI event, positional alignment is used:

    audio_techniques[i]  ↔  midi_events[i]   (when lengths are equal)

If the lists differ in length every audio technique is skipped (no reliable
mapping exists).  This is the single source of ambiguity in the current design
and should be revisited once TART stores ``params["start"]``/``params["end"]``
timestamps.

Vision technique timing
-----------------------
VisionTechniqueClassifier stores the analysis window in
``params["window_start"]`` and ``params["window_end"]`` (floats, seconds).
Overlap with [event.start_time, event.end_time] is used for matching.

Occlusion fallback (TODO)
-------------------------
When ``_resolve_string_fret`` finds no FretPosition candidate it currently
returns ``(-1, -1)`` to signal an unresolved note.  ADR-001 D4 mandates a
fallback to audio prior + hand position estimate in that case.  This fallback
is **not yet implemented**.  Downstream consumers MUST filter out ``string=-1``
notes before rendering tabs.  Tracked as a gap against ADR-001 D4.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    FretPosition,
    MidiEvent,
    NoteEvent,
    TechniqueAnnotation,
)

CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5


@dataclass
class LateFusion:
    confidence_high: float = CONFIDENCE_HIGH
    confidence_low: float = CONFIDENCE_LOW

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        midi_events: list[MidiEvent],
        audio_techniques: list[TechniqueAnnotation],
        fret_positions: list[FretPosition],
        vision_techniques: list[TechniqueAnnotation],
    ) -> list[NoteEvent]:
        """Merge multimodal signals into a list of NoteEvent objects.

        Parameters
        ----------
        midi_events:
            Ordered MIDI events from the audio transcription stage.
        audio_techniques:
            Technique annotations from the audio classifier (TART).
            Positionally aligned to *midi_events* when ``len`` matches;
            otherwise skipped entirely.
        fret_positions:
            Per-frame fret/string positions from the vision pipeline.
        vision_techniques:
            Technique annotations from the vision classifier.  Each entry
            must carry ``params["window_start"]`` and ``params["window_end"]``
            (seconds) so that temporal overlap with a MIDI event can be tested.
        """
        audio_aligned = len(audio_techniques) == len(midi_events)

        results: list[NoteEvent] = []
        for idx, event in enumerate(midi_events):
            string, fret = self._resolve_string_fret(event, fret_positions)

            audio_tech = audio_techniques[idx] if audio_aligned else None
            technique = self._resolve_technique(event, audio_tech, vision_techniques)

            results.append(
                NoteEvent(
                    midi_event=event,
                    string=string,
                    fret=fret,
                    technique=technique,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_string_fret(
        self,
        event: MidiEvent,
        fret_positions: list[FretPosition],
    ) -> tuple[int, int]:
        """Return (string, fret) for this MIDI event.

        Selection logic (ADR-001 D4):
        1. Collect all FretPosition whose timestamp falls within
           [event.start_time, event.end_time].
        2. If multiple candidates, pick the one closest to start_time;
           break ties by highest confidence.
        3. If no candidate is found return (-1, -1) — unresolved.
        4. Regardless of confidence level, use the vision observation
           (it is the only direct spatial measurement available).
        """
        candidates = [
            fp for fp in fret_positions
            if event.start_time <= fp.timestamp <= event.end_time
        ]

        if not candidates:
            # TODO: ADR-001 D4 occlusion fallback (audio prior + hand position)
            # not yet implemented — downstream must filter string==-1 notes.
            return -1, -1

        # Closest by time, then highest confidence on tie
        best = min(
            candidates,
            key=lambda fp: (
                abs(fp.timestamp - event.start_time),
                -fp.confidence,
            ),
        )
        return best.string, best.fret

    def _resolve_technique(
        self,
        event: MidiEvent,
        audio_tech: TechniqueAnnotation | None,
        vision_techniques: list[TechniqueAnnotation],
    ) -> TechniqueAnnotation | None:
        """Return the fused technique for this MIDI event.

        Fusion rules (ADR-001 D4):
        - Both agree AND max confidence >= confidence_high
            → fused label, source="fusion", confidence=avg
        - They disagree
            → vision wins (direct observation), source="fusion"
        - Only one source available
            → use that source, source unchanged
        - Neither available
            → None
        """
        vision_candidates = self._overlapping_vision_techniques(event, vision_techniques)
        vision_tech = self._best_vision_technique(vision_candidates)

        # Determine what we have
        has_audio = audio_tech is not None
        has_vision = vision_tech is not None

        if has_audio and has_vision:
            return self._fuse_both(audio_tech, vision_tech)  # type: ignore[arg-type]

        if has_vision:
            return vision_tech

        if has_audio:
            return audio_tech

        return None

    def _overlapping_vision_techniques(
        self,
        event: MidiEvent,
        vision_techniques: list[TechniqueAnnotation],
    ) -> list[TechniqueAnnotation]:
        """Return vision techniques whose window overlaps [start, end]."""
        overlapping: list[TechniqueAnnotation] = []
        for vt in vision_techniques:
            w_start = vt.params.get("window_start")
            w_end = vt.params.get("window_end")
            if w_start is None or w_end is None:
                continue
            # Overlap: not (w_end < event.start or w_start > event.end)
            if w_end >= event.start_time and w_start <= event.end_time:
                overlapping.append(vt)
        return overlapping

    def _best_vision_technique(
        self,
        candidates: list[TechniqueAnnotation],
    ) -> TechniqueAnnotation | None:
        """Pick the highest-confidence candidate from a vision window."""
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.confidence)

    def _fuse_both(
        self,
        audio: TechniqueAnnotation,
        vision: TechniqueAnnotation,
    ) -> TechniqueAnnotation:
        """Fuse audio and vision annotations into one."""
        agree = audio.technique == vision.technique

        if (
            agree
            and audio.confidence >= self.confidence_high
            and vision.confidence >= self.confidence_high
        ):
            # D4: both sources HIGH + agreement → confirmed, average confidence
            return TechniqueAnnotation(
                technique=audio.technique,
                confidence=(audio.confidence + vision.confidence) / 2,
                source="fusion",
                params={"audio_conf": audio.confidence, "vision_conf": vision.confidence},
            )

        # Disagree OR low confidence → vision wins (direct observation)
        return TechniqueAnnotation(
            technique=vision.technique,
            confidence=vision.confidence,
            source="fusion",
            params={},
        )
