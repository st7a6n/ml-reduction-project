"""
data_pipeline.py
================
Orchestral-to-Piano Reduction System — Data Preprocessing & Feature Extraction

NOTES ON THIS DATASET:
  - Instrument names appear in TWO forms in MusicXML:
      <part-name>       Italian original  e.g. "Corno 1 in D", "Fagotto 2"
      <instrument-name> music21-normalised English  e.g. "Horn in D", "Bassoon"
  - All transposing instruments are in concert pitch
  - Time/key signatures can change mid-excerpt

Requirements:
    pip install music21 pandas numpy scikit-learn
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from music21 import chord, converter, instrument, note, stream

"""HEre is a library of all the possible instrument names"""
INSTRUMENT_FAMILY_PATTERNS: List[Tuple[str, int]] = [
("corno di bassetto",   1),
("basset horn",         1),
("basset-horn",         1),
("cor de basset",       1),
("corno inglese",       1),
("cor anglais",         1),
("english horn",        1),
("oboe da caccia",      1),
("bass clarinet",       1),
("clarinetto basso",    1),
("klarinette bass",     1),
("ophicleide",          2),
("serpent",             2),
("glockenspiel",        4),
("xylophone",           4),
("marimba",             4),
("vibraphone",          4),
("campanella",          4),
("campane",             4),
("tubular bell",        4),
("crotales",            4),
("celesta",             6),
("timpani",             4),
("tympani",             4),
("pauken",              4),
("timbales",            4),
("triangolo",           4),
("triangle",            4),
("triangel",            4),
("cinelli",             4),
("piatti",              4),
("cymbal",              4),
("becken",              4),
("gran tamburo",        4),
("grosse caisse",       4),
("bass drum",           4),
("große trommel",       4),
("cassa",               4),
("tamburo",             4),
("snare",               4),
("rullante",            4),
("caisse claire",       4),
("kleine trommel",      4),
("tamburin",            4),
("tambourine",          4),
("tamburino",           4),
("tam-tam",             4),
("gong",                4),
("castagnette",         4),
("castanets",           4),
("wood block",          4),
("temple block",        4),
("claves",              4),
("maracas",             4),
("bongo",               4),
("conga",               4),
("wind machine",        4),
("ratchet",             4),
("piano",               6),
("pianoforte",          6),
("fortepiano",          6),
("harpsichord",         6),
("cembalo",             6),
("clavicembalo",        6),
("clavecin",            6),
("virginal",            6),
("spinet",              6),
("organ",               6),
("organo",              6),
("orgue",               6),
("orgel",               6),
("harmonium",           6),
("accordion",           6),
("harp",                6),
("arpa",                6),
("harfe",               6),
("harpe",               6),
("soprano",             5),
("mezzo",               5),
("alto",                5),
("contralto",           5),
("tenore",              5),
("tenor",               5),
("baritone",            5),
("bariton",             5),
("bass solo",           5),
("basso solo",          5),
("vocal",               5),
("voice",               5),
("vox",                 5),
("gesang",              5),
("singstimme",          5),
("chant",               5),
("coro",                5),
("choir",               5),
("chorus",              5),
("choeur",              5),
("trombone",            2),
("posaune",             2),
("trombón",             2),
("tromba",              2),
("trumpet",             2),
("trompete",            2),
("trompette",           2),
("clarino",             2),
("corno",               2),
("horn",                2),
("waldhorn",            2),
("cor ",                2),
("cornet",              2),
("kornett",             2),
("cornetto",            2),
("tuba",                2),
("euphonium",           2),
("helicon",             2),
("sousaphone",          2),
("basstuba",            2),
("cimbasso",            2),
("flauto piccolo",      1),
("piccolo",             1),
("flûte piccolo",       1),
("flauto",              1),
("flute",               1),
("flöte",               1),
("flûte",               1),
("traverso",            1),
("fife",                1),
("oboe",                1),
("hautbois",            1),
("oboe d'amore",        1),
("clarinetto",          1),
("clarinet",            1),
("klarinette",          1),
("clarinette",          1),
("chalumeau",           1),
("fagotto",             1),
("controfagotto",       1),
("bassoon",             1),
("contrabassoon",       1),
("kontrafagott",        1),
("basson",              1),
("contrebasson",        1),
("saxophone",           1),
("saxophon",            1),
("saxofon",             1),
("sax",                 1),
("violini",             3),
("violino",             3),
("violin",              3),
("violine",             3),
("violon",              3),
("viole",               3),
("viola",               3),
("bratsche",            3),
("alto strings",        3),
("violoncelli",         3),
("violoncello",         3),
("cello",               3),
("violoncelle",         3),
("contrabbassi",        3),
("contrabasso",         3),
("contrabass",          3),
("double bass",         3),
("kontrabass",          3),
("contrebasse",         3),
("basso",               3),
("viola d'amore",       3),
("viola da gamba",      3),
("viola da braccio",    3),
("violetta",            3),
("lute",                3),
("liuto",               3),
("laute",               3),
("theorbo",             3),
("mandolin",            3),
("banjo",               3),
("guitar",              3),
("chitarra",            3),
("gitarre",             3),
]

"""Weights of all the instruments by priority"""
INSTRUMENT_PRIORITY: Dict[int, float] = {
    1: 0.75,  # Woodwind — often melodic; moderate density
    2: 0.60,  # Brass — harmonic filler; kept selectively
    3: 0.90,  # Strings — structural backbone; highest orchestral priority
    4: 0.10,  # Percussion — rarely transcribed to piano directly
    5: 1.00,  # Voice — text-bearing melody; always the top priority
    6: 0.80,  # Keyboard/Harp — already piano-idiomatic; high priority
}

"""Readable names"""
FAMILY_NAMES: Dict[int, str] = {
    1: "Woodwind",
    2: "Brass",
    3: "Strings",
    4: "Percussion",
    5: "Voice",
    6: "Keyboard/Harp",
}


def get_instrument_family(part_name: str, instrument_name: str = "") -> int:
    """
    Combines the MusicXML <part-name> (original language, e.g. Italian)
    and <instrument-name> (music21-normalised English) into a single
    string and checks it against INSTRUMENT_FAMILY_PATTERNS in order.

    Args:
        part_name:       Raw <part-name> from MusicXML.
        instrument_name: music21-normalised <instrument-name>.

    Returns:
        Integer family code. Defaults to 3 (strings) — the most common
        and structurally important family — if no pattern matches.
    """
    combined = f"{part_name} | {instrument_name}".lower()
    for pattern, family in INSTRUMENT_FAMILY_PATTERNS:
        if pattern in combined:
            return family
    return 3  # safe default: strings


def is_unpitched_percussion(part_name: str, instrument_name: str = "") -> bool:
    """
    Return True for percussion parts that should contribute rhythm/accent
    information without becoming piano pitches.
    """
    combined = f"{part_name} | {instrument_name}".lower()
    pitched_keywords = (
        "timpani", "tympani", "pauken", "glockenspiel", "xylophone",
        "marimba", "vibraphone", "tubular bell", "campane", "crotales",
    )
    if any(keyword in combined for keyword in pitched_keywords):
        return False
    return get_instrument_family(part_name, instrument_name) == 4


def is_pitched_instrument(part_name: str, instrument_name: str = "") -> bool:
    """Return True when a part should be extracted as pitched musical notes."""
    return not is_unpitched_percussion(part_name, instrument_name)


def extract_unpitched_percussion_rhythm(score: stream.Score) -> List[Dict]:
    """
    Extract onset/accent events from unpitched percussion parts.

    The returned events are intended as context for pitched notes. They are not
    added to the pitch stream, which keeps harmonic features clean.
    """
    events: List[Dict] = []
    for part in score.parts:
        part_name, instr_name = _get_part_names(part)
        if not is_unpitched_percussion(part_name, instr_name):
            continue

        for element in part.flatten().notes:
            try:
                beat_strength = float(element.beatStrength)
            except Exception:
                beat_strength = 0.0

            duration_ql = float(element.duration.quarterLength)
            accent = min(1.0, max(beat_strength, 0.25) + min(duration_ql, 2.0) * 0.1)
            events.append({
                "offset": round(float(element.offset), 6),
                "measure_number": element.measureNumber or 0,
                "duration_ql": duration_ql,
                "beat_strength": beat_strength,
                "accent_level": accent,
                "part_name": part_name,
            })

    return events


def enhance_beat_strength_with_percussion(
    notes: List[Dict],
    percussion_events: List[Dict],
    tolerance: float = 0.125,
) -> List[Dict]:
    """
    Add percussion support features to pitched notes and boost metric weight
    near unpitched percussion hits.
    """
    if not notes:
        return notes

    events_by_measure: Dict[int, List[Dict]] = {}
    for ev in percussion_events:
        events_by_measure.setdefault(ev["measure_number"], []).append(ev)

    for n in notes:
        supported = [
            ev for ev in events_by_measure.get(n["measure_number"], [])
            if abs(float(n["offset"]) - float(ev["offset"])) <= tolerance
        ]
        accent = max((ev["accent_level"] for ev in supported), default=0.0)
        n["has_percussion_support"] = int(accent > 0.0)
        n["percussion_accent_level"] = float(accent)
        if accent:
            n["beat_strength"] = min(1.0, float(n.get("beat_strength", 0.0)) + 0.35 * accent)

    return notes


"""FUnctions for loading and processing the scores"""
def load_score(filepath: str) -> stream.Score:
    """
    Parse a MusicXML file and return a music21 Score in concert pitch.

    Transposing instruments are converted via toSoundingPitch() so that
    all pitch analysis is consistent: a Horn in D writing C4 becomes D4,
    a B♭ Clarinet writing C4 becomes B♭3, etc.

    Args:
        filepath: Path to a .musicxml or .xml file.

    Returns:
        music21 Score in concert (sounding) pitch.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Score not found: {filepath}")

    print(f"  Loading: {os.path.basename(filepath)}")
    try:
        score = converter.parse(filepath)
    except Exception as e:
        raise RuntimeError(f"music21 parse error on {filepath}: {e}") from e

    try:
        score = score.toSoundingPitch()
    except Exception:
        pass 

    return score


def _get_part_names(part: stream.Part) -> Tuple[str, str]:
    """
    Return (part_name, instrument_name) from a music21 Part.

    part_name comes from  <part-name>  (original language).
    instrument_name comes from the first Instrument object music21
    attaches to the part (normalised English).
    """
    part_name = part.partName or part.id or ""
    instr_name = ""
    for el in part.recurse().getElementsByClass(instrument.Instrument):
        instr_name = el.instrumentName or ""
        break
    return part_name, instr_name


def _is_piano_placeholder(part: stream.Part) -> bool:
    """
    In the Beethoven 9th dataset the orchestral file includes a Piano part
    (P41) containing only whole-measure rests; it was inserted as a layout aid.
    This part must be skipped; including it would add zero-note rows.

    Returns True if the part is named "piano" but contains no pitched notes.
    """
    part_name, instr_name = _get_part_names(part)
    if "piano" not in f"{part_name} {instr_name}".lower():
        return False
    # Look for any sounding note
    for _ in part.recurse().getElementsByClass(note.Note):
        return False
    for _ in part.recurse().getElementsByClass(chord.Chord):
        return False
    return True


def extract_notes_from_score(
    score: stream.Score,
    exclude_families: Optional[List[int]] = None,
    include_percussion_context: bool = True,
) -> List[Dict]:
    """
    Extract all sounding notes from an orchestral score as a flat list
    of dicts, one dict per note event (chords are expanded per pitch).

    Percussion parts (family 4) are excluded by default since they are
    rarely transcribed directly into a piano reduction.

    Args:
        score:            Parsed music21 Score in concert pitch.
        exclude_families: Family codes to skip entirely (default: [4]).

    Returns:
        List of note dicts with keys:
            midi_pitch, pitch_class, octave, duration_ql, offset,
            measure_number, beat, beat_strength, part_name,
            instrument_family, instrument_priority, voice
    """
    if exclude_families is None:
        exclude_families = [4]

    notes = []
    percussion_events = (
        extract_unpitched_percussion_rhythm(score)
        if include_percussion_context
        else []
    )

    for part in score.parts:
        if _is_piano_placeholder(part):
            print(f"    Skipping silent piano placeholder.")
            continue

        part_name, instr_name = _get_part_names(part)
        family   = get_instrument_family(part_name, instr_name)
        priority = INSTRUMENT_PRIORITY.get(family, 0.5)

        if family in exclude_families:
            continue

        for element in part.flatten().notes:
            offset      = float(element.offset)
            measure_num = element.measureNumber or 0
            duration_ql = float(element.duration.quarterLength)

            try:
                beat_strength = float(element.beatStrength)
            except Exception:
                beat_strength = 0.0

            try:
                beat = float(element.beat)
            except Exception:
                beat = 1.0

            try:
                voice_num = int(element.voice) if element.voice else 1
            except (ValueError, TypeError, AttributeError):
                voice_num = 1

            if isinstance(element, note.Note):
                p = element.pitch
                notes.append({
                    "midi_pitch":          p.midi,
                    "pitch_class":         p.pitchClass,
                    "octave":              p.octave or 4,
                    "duration_ql":         duration_ql,
                    "offset":              offset,
                    "measure_number":      measure_num,
                    "beat":                beat,
                    "beat_strength":       beat_strength,
                    "part_name":           part_name,
                    "instrument_family":   family,
                    "instrument_priority": priority,
                    "voice":               voice_num,
                    "has_percussion_support": 0,
                    "percussion_accent_level": 0.0,
                })
            elif isinstance(element, chord.Chord):
                for p in element.pitches:
                    notes.append({
                        "midi_pitch":          p.midi,
                        "pitch_class":         p.pitchClass,
                        "octave":              p.octave or 4,
                        "duration_ql":         duration_ql,
                        "offset":              offset,
                        "measure_number":      measure_num,
                        "beat":                beat,
                        "beat_strength":       beat_strength,
                        "part_name":           part_name,
                        "instrument_family":   family,
                        "instrument_priority": priority,
                        "voice":               voice_num,
                        "has_percussion_support": 0,
                        "percussion_accent_level": 0.0,
                    })

    if percussion_events:
        enhance_beat_strength_with_percussion(notes, percussion_events)

    print(f"    Extracted {len(notes):,} note events "
          f"({len(score.parts)} parts in score).")
    return notes


def extract_piano_notes(score: stream.Score) -> List[Dict]:
    """
    Extract all notes from a piano reduction score.

    Finds the Piano part by name (or falls back to the sole part) and extracts all notes with staff assignment (1=RH treble, 2=LH bass).

    This returns a list of dicts: midi_pitch, pitch_class, octave, duration_ql, offset, measure_number, beat, staff.

    Raises ValueError if no Piano part can be identified.
    """
    #parts could be in english or japanese
    _PIANO_NAMES = {"piano", "pianoforte", "ピアノ"}
    piano_parts = [
        p for p in score.parts
        if any(kw in (p.partName or p.id or "").lower() for kw in _PIANO_NAMES)
        or any(kw in (p.partName or p.id or "") for kw in {"ピアノ"})
    ]
    if not piano_parts:
        piano_parts = list(score.parts)  # piano-reduction file: treat all staves

    notes = []
    for staff_idx, piano_part in enumerate(piano_parts, start=1):
        for element in piano_part.flatten().notes:
            offset      = float(element.offset)
            measure_num = element.measureNumber or 0
            duration_ql = float(element.duration.quarterLength)
            try:
                beat = float(element.beat) if element.beat else 1.0
            except Exception:
                beat = 1.0
            try:
                staff_num = int(element.editorial.staffNumber)
            except (AttributeError, TypeError, ValueError):
                staff_num = staff_idx

            if isinstance(element, note.Note):
                notes.append({
                    "midi_pitch":     element.pitch.midi,
                    "pitch_class":    element.pitch.pitchClass,
                    "octave":         element.pitch.octave or 4,
                    "duration_ql":    duration_ql,
                    "offset":         offset,
                    "measure_number": measure_num,
                    "beat":           beat,
                    "staff":          staff_num,
                })
            elif isinstance(element, chord.Chord):
                for p in element.pitches:
                    notes.append({
                        "midi_pitch":     p.midi,
                        "pitch_class":    p.pitchClass,
                        "octave":         p.octave or 4,
                        "duration_ql":    duration_ql,
                        "offset":         offset,
                        "measure_number": measure_num,
                        "beat":           beat,
                        "staff":          staff_num,
                    })

    print(f"    Extracted {len(notes):,} notes from piano reduction.")
    return notes


def load_pair(
    orchestral_path: str,
    piano_path: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load a matched orchestral/piano pair from two MusicXML files.

    Returns:
        (orchestral_notes, piano_notes) — flat lists of note dicts.
    """
    print(f"\nLoading: {os.path.basename(orchestral_path)}")
    orch_score  = load_score(orchestral_path)
    piano_score = load_score(piano_path)
    orch_notes  = extract_notes_from_score(orch_score)
    piano_notes = extract_piano_notes(piano_score)
    return orch_notes, piano_notes

"""Function to match orch notes to piano notes """
def match_notes(
    orch_notes: List[Dict],
    piano_notes: List[Dict],
    offset_tolerance: float = 0.25,
) -> List[int]:
    """
    Label each orchestral note KEEP (1) or REMOVE (0).

    Only a match if 
      1. Same MIDI pitch (after toSoundingPitch, enharmonics collapse correctly)
      2. Same measure number (prevents cross-bar false positives)
      3. Offset within ±offset_tolerance quarter-lengths

    +- 0.25 ql because piano reductions sometimes re-articulate notes slightly
    early/late (ornaments, arpeggiation, rhythmic simplification). 0.25 ql
    (one 16th note) covers this without over-matching across different beats.

    Args:
        orch_notes:       Output of extract_notes_from_score().
        piano_notes:      Output of extract_piano_notes().
        offset_tolerance: Max offset delta to count as a match (default 0.25).

    Returns:
        List of int labels (1=KEEP, 0=REMOVE), same length as orch_notes.
    """
    piano_lookup: Dict[Tuple[int, int], List[float]] = {}
    for pn in piano_notes:
        key = (pn["measure_number"], pn["midi_pitch"])
        piano_lookup.setdefault(key, []).append(pn["offset"])

    labels = []
    for on in orch_notes:
        key     = (on["measure_number"], on["midi_pitch"])
        offsets = piano_lookup.get(key, [])
        match   = any(abs(on["offset"] - po) <= offset_tolerance for po in offsets)
        labels.append(1 if match else 0)

    kept  = sum(labels)
    total = len(labels)
    print(f"    Labels → {kept:,} KEEP ({kept/total:.1%}), "
          f"{total-kept:,} REMOVE ({(total-kept)/total:.1%})")
    return labels

def _build_context_arrays(
    notes: List[Dict],
) -> Tuple[
    Dict[Tuple[int, float], int],
    Dict[Tuple[int, float], int],
    Dict[Tuple[int, float], int],
    Dict[Tuple[int, float], List[int]],
]:
    """
    Pre-compute per-timepoint context keyed by (measure_number, offset).

    Returns four dicts:
      note_count:  simultaneous note count at each timepoint
      highest:     highest MIDI pitch at each timepoint
      lowest:      lowest MIDI pitch at each timepoint
      all_pitches: list of all MIDI pitches at each timepoint
    """
    tp_pitches: Dict[Tuple[int, float], List[int]] = {}
    for n in notes:
        key = (n["measure_number"], round(n["offset"], 6))
        tp_pitches.setdefault(key, []).append(n["midi_pitch"])

    note_count = {k: len(v) for k, v in tp_pitches.items()}
    highest    = {k: max(v) for k, v in tp_pitches.items()}
    lowest     = {k: min(v) for k, v in tp_pitches.items()}
    return note_count, highest, lowest, tp_pitches


def _is_chord_tone(pitch_class: int, bass_pitch_class: int) -> int:
    """
    Heuristic: does this pitch fit a triad or 7th chord on the bass note?

    Checks standard intervals above the bass (mod 12):
        0=unison, 3=m3, 4=M3, 7=P5, 10=m7, 11=M7

    Intentionally excludes the tritone (6) and augmented intervals, which
    are dissonant passing tones rarely kept in sparse piano reductions.

    Args:
        pitch_class:      Pitch class (0-11) of the note being tested.
        bass_pitch_class: Pitch class of the lowest simultaneous note.

    Returns:
        1 if the note is a likely chord tone, 0 otherwise.
    """
    interval = (pitch_class - bass_pitch_class) % 12
    return 1 if interval in {0, 3, 4, 7, 10, 11} else 0


def extract_features(
    notes: List[Dict],
    all_notes: Optional[List[Dict]] = None,
    phrase_start_measure: Optional[int] = None,
    phrase_end_measure: Optional[int] = None,
    phrase_position: int = 0,
    piece_min_measure: Optional[int] = None,
    piece_max_measure: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract the music-theoretic, percussion-context, and phrase features for each note.

    Feature groups (for report clarity):

    PITCH (5):
        midi_pitch         MIDI note number (21-108)
        pitch_class        Pitch class 0=C … 11=B
        octave             Octave number (typically 1-8)
        is_melody          1 if highest pitch at this timepoint
        is_bass            1 if lowest pitch at this timepoint

    RHYTHM (4):
        duration_ql        Duration in quarter-lengths (0.25=16th, 4.0=whole)
        offset_in_measure  Absolute offset from score start (ql)
        beat_strength      music21 metric weight (1.0=downbeat, 0.0625=weakest)
        is_strong_beat     1 if beat_strength ≥ 0.5 (beats 1 & 3 in 4/4)

    HARMONIC (3):
        interval_from_bass Semitones above the lowest simultaneous note (≥0)
        is_chord_tone      1 if the interval fits a triad/7th on the bass
        simultaneous_notes Total notes sounding at this timepoint

    SOURCE / CONTEXT (4):
        instrument_family   Family code 1-6
        instrument_priority Reduction priority weight 0.1-1.0
        measure_norm        Measure index within excerpt, normalised 0-1
        dist_to_melody      Semitones below the highest note (0 if this IS melody)

    TEXTURE (2):
        total_notes_at_tp  Alias of simultaneous_notes (for explainability)
        voice_number       Voice index within the part

    Args:
        notes:     List of note dicts to featurise.
        all_notes: Full texture note list for context computation.
                   If None, context is computed within `notes` alone.

    Returns:
        pd.DataFrame with FEATURE_COLUMNS, one row per note.
    """
    context = all_notes if all_notes is not None else notes
    note_count, highest, lowest, _ = _build_context_arrays(context)

    all_measures = [n["measure_number"] for n in notes]
    min_m  = min(all_measures) if all_measures else 0
    max_m  = max(all_measures) if all_measures else 1
    phrase_start_measure = phrase_start_measure if phrase_start_measure is not None else min_m
    phrase_end_measure = phrase_end_measure if phrase_end_measure is not None else max_m
    piece_min_measure = piece_min_measure if piece_min_measure is not None else min_m
    piece_max_measure = piece_max_measure if piece_max_measure is not None else max_m
    span   = max(max_m - min_m, 1)

    rows = []
    for n in notes:
        tp  = (n["measure_number"], round(n["offset"], 6))
        cnt = note_count.get(tp, 1)
        hi  = highest.get(tp, n["midi_pitch"])
        lo  = lowest.get(tp, n["midi_pitch"])

        rows.append({
            # Pitch
            "midi_pitch":          n["midi_pitch"],
            "pitch_class":         n["pitch_class"],
            "octave":              n["octave"],
            "is_melody":           int(n["midi_pitch"] == hi),
            "is_bass":             int(n["midi_pitch"] == lo),
            # Rhythm
            "duration_ql":         n["duration_ql"],
            "offset_in_measure":   n["offset"],
            "beat_strength":       n["beat_strength"],
            "is_strong_beat":      int(n["beat_strength"] >= 0.5),
            "has_percussion_support": int(n.get("has_percussion_support", 0)),
            "percussion_accent_level": float(n.get("percussion_accent_level", 0.0)),
            # Harmonic
            "interval_from_bass":  n["midi_pitch"] - lo,
            "is_chord_tone":       _is_chord_tone(n["pitch_class"], lo % 12),
            "simultaneous_notes":  cnt,
            # Source / context
            "instrument_family":   n["instrument_family"],
            "instrument_priority": n["instrument_priority"],
            "measure_norm":        round((n["measure_number"] - min_m) / span, 4),
            "dist_to_melody":      hi - n["midi_pitch"],
            # Texture
            "total_notes_at_tp":   cnt,
            "voice_number":        n["voice"],
            # Phrase context
            "is_phrase_start":     int(n["measure_number"] == phrase_start_measure),
            "is_phrase_end":       int(n["measure_number"] == phrase_end_measure),
            "phrase_position":     phrase_position,
            "is_opening":          int(phrase_start_measure <= piece_min_measure),
            "is_closing":          int(phrase_end_measure >= piece_max_measure),
        })

    return pd.DataFrame(rows)

"""Build the training dataset"""
def build_training_dataset(
    orch_notes: List[Dict],
    piano_notes: List[Dict],
    offset_tolerance: float = 0.25,
) -> pd.DataFrame:
    """
    Combine feature extraction and label matching into one DataFrame.

    Each row = one orchestral note.
    Feature columns are defined by FEATURE_COLUMNS.
    Column 'label' = 1 (KEEP) or 0 (REMOVE).
    """
    labels = match_notes(orch_notes, piano_notes, offset_tolerance)
    df     = extract_features(orch_notes, all_notes=orch_notes)
    df["label"] = labels
    print(f"    Dataset shape: {df.shape}")
    return df

def build_dataset_from_folder(
    orchestral_dir: str,
    piano_dir: str,
    offset_tolerance: float = 0.25,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Process all matched pairs across orchestral_dir / piano_dir.

    Pairs are matched by filename. Errors on individual pairs are caught
    and logged so that the remaining pairs continue processing.

    Returns:
        Combined DataFrame with a 'pair_id' column tracing each row to
        its source file (e.g. "pair3_m0201-0208").
    """
    orch_path  = Path(orchestral_dir)
    piano_path = Path(piano_dir)

    if not orch_path.exists():
        raise FileNotFoundError(f"Orchestral dir not found: {orchestral_dir}")
    if not piano_path.exists():
        raise FileNotFoundError(f"Piano dir not found: {piano_dir}")

    orch_files = sorted(
        list(orch_path.glob("*.musicxml")) + list(orch_path.glob("*.xml"))
    )
    if not orch_files:
        raise ValueError(f"No .musicxml/.xml files found in {orchestral_dir}")

    all_dfs     = []
    n_processed = 0
    n_errors    = 0

    for orch_file in orch_files:
        if max_pairs and n_processed >= max_pairs:
            break

        piano_file = piano_path / orch_file.name
        if not piano_file.exists():
            print(f"  WARNING: no matching piano file for {orch_file.name} — skipping.")
            continue

        try:
            orch_notes, piano_notes = load_pair(str(orch_file), str(piano_file))
            if not orch_notes or not piano_notes:
                print(f"  WARNING: empty notes in {orch_file.name} — skipping.")
                continue

            df = build_training_dataset(orch_notes, piano_notes, offset_tolerance)
            df["pair_id"] = orch_file.stem
            all_dfs.append(df)
            n_processed += 1

        except Exception as e:
            print(f"  ERROR on {orch_file.name}: {e}")
            n_errors += 1

    if not all_dfs:
        raise ValueError("No pairs successfully processed.")

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"  Pairs processed  : {n_processed}")
    print(f"  Pairs with errors: {n_errors}")
    print(f"  Total note rows  : {len(combined):,}")
    print(f"  KEEP  (label=1)  : {combined['label'].sum():,} "
          f"({combined['label'].mean():.1%})")
    print(f"  REMOVE (label=0) : {(combined['label']==0).sum():,} "
          f"({(combined['label']==0).mean():.1%})")
    print(f"{'='*60}\n")

    return combined


FEATURE_COLUMNS = [
    # Pitch
    "midi_pitch", "pitch_class", "octave", "is_melody", "is_bass",
    # Rhythm
    "duration_ql", "offset_in_measure", "beat_strength", "is_strong_beat",
    "has_percussion_support", "percussion_accent_level",
    # Harmonic
    "interval_from_bass", "is_chord_tone", "simultaneous_notes",
    # Source
    "instrument_family", "instrument_priority", "measure_norm", "dist_to_melody",
    # Texture
    "total_notes_at_tp", "voice_number",
    # Phrase context
    "is_phrase_start", "is_phrase_end", "phrase_position", "is_opening", "is_closing",
]

LABEL_COLUMN = "label"

"""NOW we want to save the dataset"""
def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save training DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"Dataset saved → {path}  ({len(df):,} rows)")


def load_dataset(path: str) -> pd.DataFrame:
    """Load training DataFrame from CSV."""
    df = pd.read_csv(path)
    print(f"Dataset loaded ← {path}  ({len(df):,} rows)")
    return df

"""TESTING LOCALLY HERE """
if __name__ == "__main__":
    """ Quick test: process one pair and print feature/label summary."""
    import sys

    ORCH_FILE  = "data/orchestral/pair3_m0201-0208.musicxml"
    PIANO_FILE = "data/piano/pair3_m0201-0208.musicxml"

    if not os.path.exists(ORCH_FILE) or not os.path.exists(PIANO_FILE):
        print(f"Test files not found. Edit ORCH_FILE / PIANO_FILE in the script.")
        sys.exit(1)

    print("=" * 60)
    print("DATA PIPELINE — SINGLE PAIR TEST")
    print("=" * 60)

    orch_notes, piano_notes = load_pair(ORCH_FILE, PIANO_FILE)
    df = build_training_dataset(orch_notes, piano_notes)

    print("\nFeature statistics:")
    print(df[FEATURE_COLUMNS].describe().round(3).to_string())

    print("\nLabel distribution:")
    print(df[LABEL_COLUMN].value_counts())

    print("\nKEEP rate by instrument family:")
    keep_rate = df.groupby("instrument_family")[LABEL_COLUMN].mean()
    for fam, rate in keep_rate.sort_values(ascending=False).items():
        print(f"  {FAMILY_NAMES.get(int(fam), str(fam)):15s}: {rate:.1%}")

    save_dataset(df, "training_data_pair3.csv")

    """Test the family mapper with possible outliers"""
    print("\nMapper spot-checks:")
    test_cases = [
        ("Corno di bassetto", ""),   # expect 1 (Woodwind)
        ("Corno 1 in D", "Horn in D"),  # expect 2 (Brass)
        ("Fagotto 2", "Bassoon"),    # expect 1 (Woodwind)
        ("Violini I", "Violins"),    # expect 3 (Strings)
        ("Soprano solo", "Soprano"), # expect 5 (Voice)
        ("Gran Tamburo", "Acoustic Bass Drum"),  # expect 4 (Perc)
        ("Piano", "Piano"),          # expect 6 (Keyboard)
        ("Cor anglais", ""),         # expect 1 (Woodwind)
        ("Posaune 1", "Trombone"),   # expect 2 (Brass)
    ]
    for pname, iname in test_cases:
        fam = get_instrument_family(pname, iname)
        print(f"  {pname!r:30s} → family {fam} ({FAMILY_NAMES[fam]})")

    print("\nDone.")
