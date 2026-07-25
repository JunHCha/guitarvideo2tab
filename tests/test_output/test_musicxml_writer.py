"""MusicXML 출력: 파싱 가능성, 마디 길이, pitch/운지 일관성."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from guitarvideo2tab.models import MidiEvent, NoteEvent
from guitarvideo2tab.output.musicxml_writer import (
    pitch_to_xml,
    to_musicxml_string,
    write_musicxml,
)
from guitarvideo2tab.output.rhythm import QUARTER_TICKS, RhythmGrid

TUNING = (40, 45, 50, 55, 59, 64)  # EADGBE
_MEASURE_TICKS = 3840


def _note(start: float, end: float, pitch: int, string: int, fret: int) -> NoteEvent:
    return NoteEvent(
        midi_event=MidiEvent(pitch=pitch, start_time=start, end_time=end, velocity=80),
        string=string,
        fret=fret,
    )


def _consistent(start: float, end: float, string: int, fret: int) -> NoteEvent:
    """pitch 와 (string, fret) 이 일치하는 음."""
    return _note(start, end, TUNING[string - 1] + fret, string, fret)


def _grid(**kwargs) -> RhythmGrid:
    return RhythmGrid(tempo_bpm=120.0, **kwargs)


def _parse(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


# ---------------------------------------------------------------------------
# pitch 변환
# ---------------------------------------------------------------------------


def test_pitch_to_xml_middle_c() -> None:
    assert pitch_to_xml(60) == ("C", 0, 4)


def test_pitch_to_xml_sharp_and_octaves() -> None:
    assert pitch_to_xml(61) == ("C", 1, 4)
    assert pitch_to_xml(40) == ("E", 0, 2)   # 6번 개방현
    assert pitch_to_xml(64) == ("E", 0, 4)   # 1번 개방현


# ---------------------------------------------------------------------------
# 문서 구조
# ---------------------------------------------------------------------------


def test_written_musicxml_is_wellformed(tmp_path: Path) -> None:
    out = tmp_path / "tab.musicxml"
    write_musicxml([_consistent(0.0, 0.5, 1, 0)], out, _grid(), TUNING)

    root = _parse(out)
    assert root.tag == "score-partwise"
    assert root.get("version") == "4.0"
    assert root.find("part-list/score-part") is not None
    assert root.find("part") is not None


def test_declaration_and_doctype_present(tmp_path: Path) -> None:
    out = tmp_path / "tab.musicxml"
    write_musicxml([_consistent(0.0, 0.5, 1, 0)], out, _grid(), TUNING)

    head = out.read_text(encoding="utf-8").splitlines()[:2]
    assert head[0].startswith("<?xml version=")
    assert "score-partwise" in head[1] and "DOCTYPE" in head[1]


def test_attributes_carry_divisions_and_time_signature(tmp_path: Path) -> None:
    out = tmp_path / "tab.musicxml"
    write_musicxml([_consistent(0.0, 0.5, 1, 0)], out, _grid(numerator=3), TUNING)

    attributes = _parse(out).find("part/measure/attributes")
    assert attributes.findtext("divisions") == str(QUARTER_TICKS)
    assert attributes.findtext("time/beats") == "3"
    assert attributes.findtext("time/beat-type") == "4"
    # 기타는 실음보다 한 옥타브 높게 기보한다
    assert attributes.findtext("clef/clef-octave-change") == "-1"


def test_tempo_direction_is_written(tmp_path: Path) -> None:
    out = tmp_path / "tab.musicxml"
    write_musicxml([_consistent(0.0, 0.5, 1, 0)], out, RhythmGrid(tempo_bpm=96.0), TUNING)

    root = _parse(out)
    assert root.findtext("part/measure/direction/direction-type/metronome/per-minute") == "96"


# ---------------------------------------------------------------------------
# 리듬 정확성
# ---------------------------------------------------------------------------


def test_every_measure_sums_to_full_measure(tmp_path: Path) -> None:
    notes = [_consistent(t, t + 0.3, 1, 0) for t in (0.0, 0.75, 1.5, 2.25, 3.1, 5.0)]
    out = tmp_path / "tab.musicxml"
    write_musicxml(notes, out, _grid(), TUNING)

    for measure in _parse(out).findall("part/measure"):
        total = 0
        for note in measure.findall("note"):
            if note.find("chord") is not None:  # 화음 구성음은 시간을 소비하지 않는다
                continue
            total += int(note.findtext("duration"))
        assert total == _MEASURE_TICKS, f"마디 {measure.get('number')} 길이 {total}"


def test_notes_are_split_across_measures(tmp_path: Path) -> None:
    notes = [_consistent(0.0, 0.5, 1, 0), _consistent(2.0, 2.5, 2, 3)]
    out = tmp_path / "tab.musicxml"
    write_musicxml(notes, out, _grid(), TUNING)

    measures = _parse(out).findall("part/measure")
    assert [m.get("number") for m in measures] == ["1", "2"]


def test_rests_are_emitted(tmp_path: Path) -> None:
    out = tmp_path / "tab.musicxml"
    write_musicxml([_consistent(0.0, 0.5, 1, 0)], out, _grid(), TUNING)

    rests = _parse(out).findall("part/measure/note/rest")
    assert len(rests) >= 1


def test_chord_marks_subsequent_notes(tmp_path: Path) -> None:
    notes = [
        _consistent(0.0, 0.5, 1, 0),
        _consistent(0.0, 0.5, 2, 0),
        _consistent(0.0, 0.5, 3, 0),
    ]
    out = tmp_path / "tab.musicxml"
    result = write_musicxml(notes, out, _grid(), TUNING)

    first_measure = _parse(out).find("part/measure")
    pitched = [n for n in first_measure.findall("note") if n.find("rest") is None]
    assert len(pitched) == 3
    assert pitched[0].find("chord") is None
    assert all(n.find("chord") is not None for n in pitched[1:])
    assert result.chords == 1


# ---------------------------------------------------------------------------
# pitch ↔ 운지 일관성 (버그 ③-a 방어)
# ---------------------------------------------------------------------------


def test_to_musicxml_string_inverts_numbering() -> None:
    """내부 1(최저음) ↔ MusicXML 6(최저음). 변환하지 않으면 TAB 이 뒤집힌다."""
    assert to_musicxml_string(1, 6) == 6
    assert to_musicxml_string(6, 6) == 1
    assert to_musicxml_string(3, 6) == 4


def test_technical_written_when_fingering_matches_pitch(tmp_path: Path) -> None:
    out = tmp_path / "tab.musicxml"
    # 내부 string=1 은 최저음현 → MusicXML 에서는 6번 줄로 적혀야 한다
    result = write_musicxml([_consistent(0.0, 0.5, 1, 3)], out, _grid(), TUNING)

    technical = _parse(out).find("part/measure/note/notations/technical")
    assert technical is not None
    assert technical.findtext("string") == "6"
    assert technical.findtext("fret") == "3"
    assert result.technical_written == 1
    assert result.technical_skipped == 0


def test_technical_omitted_when_fingering_contradicts_pitch(tmp_path: Path) -> None:
    """string=1(=E2, MIDI 40) 14프렛은 MIDI 54 다. pitch 58 과 모순이므로 운지를 생략한다."""
    out = tmp_path / "tab.musicxml"
    result = write_musicxml([_note(0.0, 0.5, pitch=58, string=1, fret=14)], out, _grid(), TUNING)

    root = _parse(out)
    assert root.find("part/measure/note/notations/technical") is None
    # pitch 는 그대로 기록된다 (들리는 음의 정본)
    assert root.findtext("part/measure/note/pitch/step") == "A"
    assert result.technical_written == 0
    assert result.technical_skipped == 1
    assert result.technical_match_ratio == 0.0


def test_out_of_range_string_does_not_crash(tmp_path: Path) -> None:
    out = tmp_path / "tab.musicxml"
    result = write_musicxml([_note(0.0, 0.5, pitch=60, string=9, fret=2)], out, _grid(), TUNING)

    assert out.exists()
    assert result.technical_skipped == 1


def test_empty_notes_still_produce_valid_score(tmp_path: Path) -> None:
    out = tmp_path / "tab.musicxml"
    result = write_musicxml([], out, _grid(), TUNING)

    root = _parse(out)
    assert len(root.findall("part/measure")) == 1
    assert root.find("part/measure/note/rest") is not None
    assert result.notes == 0
