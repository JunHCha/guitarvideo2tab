"""양자화된 리듬을 MusicXML(score-partwise)로 출력한다.

MuseScore / Guitar Pro / TuxGuitar 등이 모두 읽는 표준 포맷이라 별도 바이너리
포맷 라이브러리 없이 표준 ElementTree 만으로 정확히 생성할 수 있다.

pitch 와 (string, fret) 의 관계
--------------------------------
``(string, fret)`` 은 pitch 를 정확히 결정한다(``tuning[string-1] + fret``).
현재 비전 경로는 이 제약을 검증하지 않아 둘이 어긋나는 경우가 있으므로:

* ``<pitch>`` 는 **항상** Basic Pitch 의 값을 쓴다(들리는 음의 정본).
* ``<technical><string>/<fret>`` 은 **둘이 일치할 때만** 기록한다.

일치하지 않아 생략된 개수는 ``write_musicxml`` 이 돌려주므로 품질 지표로 쓸 수
있다. 어긋난 운지를 악보에 적으면 연주자가 다른 음을 내게 되므로, 적지 않는
편이 낫다.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom

from ..models import NoteEvent
from .rhythm import QUARTER_TICKS, QuantizedBeat, RhythmGrid, quantize_notes, ticks_to_duration

# MusicXML <type> 이름. rhythm.ticks_to_duration 이 돌려주는 value 와 대응한다.
_TYPE_NAMES = {
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "16th",
    32: "32nd",
    64: "64th",
}

# 샤프 기준 음이름 (0 = C)
_STEPS = [
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
]

_DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


@dataclass
class MusicXMLResult:
    path: Path
    measures: int
    notes: int
    chords: int
    rests: int
    technical_written: int
    technical_skipped: int

    @property
    def technical_match_ratio(self) -> float:
        total = self.technical_written + self.technical_skipped
        return self.technical_written / total if total else 0.0


def pitch_to_xml(pitch: int) -> tuple[str, int, int]:
    """MIDI 번호를 ``(step, alter, octave)`` 로 바꾼다. MIDI 60 = C4."""
    step, alter = _STEPS[pitch % 12]
    octave = pitch // 12 - 1
    return step, alter, octave


def to_musicxml_string(string: int, string_count: int) -> int:
    """내부 현 번호를 MusicXML 현 번호로 뒤집는다.

    이 코드베이스는 ``string=1`` 이 **최저음현**(6번 줄 E2)이고 번호가 커질수록
    음이 높아진다(``tuning`` 이 오름차순). 반면 MusicXML 의 ``<string>`` 은
    기타 관습대로 **1 이 최고음현**이다. 변환하지 않으면 악보 소프트웨어에서
    TAB 의 현이 위아래로 뒤집혀 표시된다.
    """
    return string_count - string + 1


def _sub(parent: ET.Element, tag: str, text: str | int | None = None) -> ET.Element:
    element = ET.SubElement(parent, tag)
    if text is not None:
        element.text = str(text)
    return element


def _append_note(
    measure: ET.Element,
    note_event: NoteEvent,
    duration_ticks: int,
    type_value: int,
    is_dotted: bool,
    is_chord: bool,
    tuning: tuple[int, ...],
) -> bool:
    """음표 하나를 추가하고, technical(string/fret)을 기록했는지 반환한다."""
    note = _sub(measure, "note")
    if is_chord:
        _sub(note, "chord")

    step, alter, octave = pitch_to_xml(note_event.midi_event.pitch)
    pitch_el = _sub(note, "pitch")
    _sub(pitch_el, "step", step)
    if alter:
        _sub(pitch_el, "alter", alter)
    _sub(pitch_el, "octave", octave)

    _sub(note, "duration", duration_ticks)
    _sub(note, "voice", 1)
    _sub(note, "type", _TYPE_NAMES.get(type_value, "quarter"))
    if is_dotted:
        _sub(note, "dot")

    # (string, fret) 이 pitch 와 모순되지 않을 때만 운지를 적는다.
    string, fret = note_event.string, note_event.fret
    consistent = (
        1 <= string <= len(tuning)
        and fret >= 0
        and tuning[string - 1] + fret == note_event.midi_event.pitch
    )
    if consistent:
        notations = _sub(note, "notations")
        technical = _sub(notations, "technical")
        _sub(technical, "string", to_musicxml_string(string, len(tuning)))
        _sub(technical, "fret", fret)
    return consistent


def _append_rest(
    measure: ET.Element, duration_ticks: int, type_value: int, is_dotted: bool
) -> None:
    note = _sub(measure, "note")
    _sub(note, "rest")
    _sub(note, "duration", duration_ticks)
    _sub(note, "voice", 1)
    _sub(note, "type", _TYPE_NAMES.get(type_value, "quarter"))
    if is_dotted:
        _sub(note, "dot")


def _build_part_list(root: ET.Element, part_name: str) -> None:
    part_list = _sub(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", {"id": "P1"})
    _sub(score_part, "part-name", part_name)
    _sub(score_part, "part-abbreviation", "Gtr.")
    instrument = ET.SubElement(score_part, "score-instrument", {"id": "P1-I1"})
    _sub(instrument, "instrument-name", "Acoustic Guitar (steel)")
    midi_instrument = ET.SubElement(score_part, "midi-instrument", {"id": "P1-I1"})
    _sub(midi_instrument, "midi-channel", 1)
    _sub(midi_instrument, "midi-program", 26)  # GM 1-based: Acoustic Guitar (steel)


def _append_attributes(measure: ET.Element, grid: RhythmGrid) -> None:
    attributes = _sub(measure, "attributes")
    _sub(attributes, "divisions", QUARTER_TICKS)  # 4분음표당 division 수
    key = _sub(attributes, "key")
    _sub(key, "fifths", 0)
    time = _sub(attributes, "time")
    _sub(time, "beats", grid.numerator)
    _sub(time, "beat-type", grid.denominator)
    clef = _sub(attributes, "clef")
    _sub(clef, "sign", "G")
    _sub(clef, "line", 2)
    # 기타는 실음보다 한 옥타브 높게 기보하는 관습을 따른다.
    _sub(clef, "clef-octave-change", -1)


def _append_tempo(measure: ET.Element, grid: RhythmGrid) -> None:
    direction = ET.SubElement(measure, "direction", {"placement": "above"})
    direction_type = _sub(direction, "direction-type")
    metronome = _sub(direction_type, "metronome")
    _sub(metronome, "beat-unit", "quarter")
    _sub(metronome, "per-minute", int(round(grid.tempo_bpm)))
    ET.SubElement(direction, "sound", {"tempo": str(int(round(grid.tempo_bpm)))})


def build_score(
    beats: list[QuantizedBeat],
    grid: RhythmGrid,
    tuning: tuple[int, ...],
    title: str,
    part_name: str = "Guitar",
) -> tuple[ET.Element, dict[str, int]]:
    """score-partwise 루트 엘리먼트와 집계 통계를 만든다."""
    root = ET.Element("score-partwise", {"version": "4.0"})
    work = _sub(root, "work")
    _sub(work, "work-title", title)
    identification = _sub(root, "identification")
    encoding = _sub(identification, "encoding")
    _sub(encoding, "software", "guitarvideo2tab")

    _build_part_list(root, part_name)
    part = ET.SubElement(root, "part", {"id": "P1"})

    by_measure: dict[int, list[QuantizedBeat]] = {}
    for beat in beats:
        by_measure.setdefault(beat.measure_index, []).append(beat)
    measure_count = max(by_measure, default=-1) + 1
    measure_count = max(measure_count, 1)

    stats = {"notes": 0, "chords": 0, "rests": 0, "technical": 0, "skipped": 0}

    for index in range(measure_count):
        measure = ET.SubElement(part, "measure", {"number": str(index + 1)})
        if index == 0:
            _append_attributes(measure, grid)
            _append_tempo(measure, grid)

        measure_beats = sorted(by_measure.get(index, []), key=lambda b: b.offset_ticks)
        if not measure_beats:
            # 빈 마디는 온쉼표 하나로 채운다.
            value, dotted = ticks_to_duration(grid.measure_ticks)
            _append_rest(measure, grid.measure_ticks, value, dotted)
            stats["rests"] += 1
            continue

        for beat in measure_beats:
            value, dotted = ticks_to_duration(beat.duration_ticks)
            if beat.is_rest:
                _append_rest(measure, beat.duration_ticks, value, dotted)
                stats["rests"] += 1
                continue
            if len(beat.notes) > 1:
                stats["chords"] += 1
            for position, note_event in enumerate(beat.notes):
                wrote = _append_note(
                    measure,
                    note_event,
                    beat.duration_ticks,
                    value,
                    dotted,
                    is_chord=position > 0,
                    tuning=tuning,
                )
                stats["notes"] += 1
                stats["technical" if wrote else "skipped"] += 1

    return root, stats


def write_musicxml(
    notes: list[NoteEvent],
    output_path: Path,
    grid: RhythmGrid,
    tuning: tuple[int, ...],
    title: str = "guitarvideo2tab",
) -> MusicXMLResult:
    """NoteEvent 목록을 양자화하여 ``.musicxml`` 로 쓴다."""
    beats = quantize_notes(notes, grid)
    root, stats = build_score(beats, grid, tuning, title)

    rough = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    # minidom 이 붙인 선언을 걷어내고 DOCTYPE 을 넣은 표준 헤더로 교체한다.
    body = "\n".join(line for line in pretty.split("\n")[1:] if line.strip())
    document = f'<?xml version="1.0" encoding="UTF-8"?>\n{_DOCTYPE}\n{body}\n'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")

    measures = len({b.measure_index for b in beats}) or 1
    return MusicXMLResult(
        path=output_path,
        measures=measures,
        notes=stats["notes"],
        chords=stats["chords"],
        rests=stats["rests"],
        technical_written=stats["technical"],
        technical_skipped=stats["skipped"],
    )
