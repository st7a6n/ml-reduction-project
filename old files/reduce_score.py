"""
This file takes any MusicXML file (any instruments, any length), runs the trained
Random Forest model on consecutive 8-bar phrases, and writes a two-staff
piano reduction as a new MusicXML file.

Options:
    --model      PATH   Trained model file (default: models/rf_reducer.joblib)
    --phrase-bars N     Process in N-bar chunks (default: 8)
    --threshold   F     Extra keep threshold above top-K (default: 0.30)
    --top3-conf   F     Mean P(KEEP) cutoff for 3 vs 2 notes/tp (default: 0.20)
    --split-pitch N     MIDI pitch for RH/LH split (default: 60 = C4)
    -v, --verbose       Print per-measure detail
"""

import argparse
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from music21 import (
    chord as m21chord,
    clef as m21clef,
    converter,
    dynamics as m21dynamics,
    expressions as m21expressions,
    instrument as m21instrument,
    key as m21key,
    meter,
    note as m21note,
    spanner as m21spanner,
    stream,
    tempo as m21tempo,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import (
    FEATURE_COLUMNS,
    extract_features,
    extract_notes_from_score,
    load_score,
)
from ml_model import NoteClassifier
from mdp_reducer import ReductionMDP, group_by_timepoint, hybrid_reduction


"""THese are valid MusixXMl note lengths converted to digits"""
_VALID_QL = sorted([
    0.0625,          # 64th note
    0.125,           # 32nd note
    0.1875,          # dotted 32nd
    0.25,            # 16th note
    0.375,           # dotted 16th
    0.5,             # 8th note
    0.75,            # dotted 8th
    1.0,             # quarter note
    1.5,             # dotted quarter
    2.0,             # half note
    3.0,             # dotted half
    4.0,             # whole note
    6.0,             # dotted whole
    8.0,             # double whole
])

"""Quantization grid for note offsets: 32nd note"""
_GRID = 0.125


def _snap_offset(val: float) -> float:
    """Snap an offset to the nearest 32nd-note grid point."""
    return round(round(val / _GRID) * _GRID, 6)


def _snap_dur(ql: float) -> float:
    """Snap a duration to the nearest valid MusicXML quarter-length."""
    ql = max(ql, _GRID)
    return min(_VALID_QL, key=lambda v: abs(v - ql))


def _snap_dur_floor(ql: float) -> float:
    """Snap a duration DOWN to the largest valid QL that fits within ql.
    Used for gap-filling rests so they never overflow their allotted space."""
    ql = max(ql, _GRID)
    candidates = [v for v in _VALID_QL if v <= ql + 1e-9]
    return max(candidates) if candidates else _GRID


def build_measure_map(score: stream.Score) -> dict:
    """
    Return {measure_number: (abs_offset, bar_duration_ql)} from the first part.

    Walks every Measure object so time-signature changes are correctly
    reflected in each bar's duration.  Falls back to 4/4 if no measures found.
    """
    measure_map = {}
    ref_part = None

    # Pick the part with the most measures as reference
    for part in score.parts:
        ms = list(part.getElementsByClass(stream.Measure))
        if ms and (ref_part is None or
                   len(ms) > len(list(ref_part.getElementsByClass(stream.Measure)))):
            ref_part = part

    if ref_part is None:
        return measure_map

    for idx, m in enumerate(ref_part.getElementsByClass(stream.Measure), start=1):
        abs_off = float(m.offset)
        bar_dur = float(m.barDuration.quarterLength)
        measure_map[idx] = (abs_off, bar_dur)

    return measure_map


def normalize_measure_map(measure_map: dict) -> dict:
    """
    Return a sequential {measure_index: (abs_offset, bar_duration_ql)} map.

    Some imported CCARH/MuseData MusicXML files preserve corrupted printed
    measure numbers from individual parts. The offsets are still reliable, so
    use those to create stable piano-output measure numbers.
    """
    if not measure_map:
        return measure_map
    ordered = sorted(measure_map.values(), key=lambda item: item[0])
    return {idx + 1: value for idx, value in enumerate(ordered)}


def assign_measure_numbers_by_offset(notes: list, measure_map: dict) -> None:
    """Mutate note dicts so phrase processing uses normalized measure numbers."""
    import bisect

    measure_starts = sorted(
        (abs_off, mnum, bar_dur)
        for mnum, (abs_off, bar_dur) in measure_map.items()
    )
    if not measure_starts:
        return

    ms_offsets = [x[0] for x in measure_starts]
    for n in notes:
        idx = bisect.bisect_right(ms_offsets, n["offset"] + 1e-6) - 1
        idx = max(0, min(idx, len(measure_starts) - 1))
        _, mnum, _ = measure_starts[idx]
        n["measure_number"] = mnum


def collect_tempo_markings(score: stream.Score) -> dict:
    """Collect tempo markings from all source parts, deduped by offset."""
    tempo_at_offset = {}
    for tm in score.recurse().getElementsByClass(m21tempo.MetronomeMark):
        off = round(float(tm.getOffsetInHierarchy(score)), 6)
        if off not in tempo_at_offset:
            tempo_at_offset[off] = deepcopy(tm)
    return tempo_at_offset


def collect_expressive_markings(score: stream.Score) -> tuple[dict, dict]:
    """
    Collect dynamics and dynamic-like text from the source score.

    Multiple orchestral parts often repeat the same dynamic at an offset. We
    keep one representative marking per offset for the piano reduction.
    """
    dynamic_by_offset = {}
    text_by_offset = {}
    dynamic_rank = {
        "pppp": 0, "ppp": 1, "pp": 2, "p": 3, "mp": 4, "mf": 5,
        "f": 6, "ff": 7, "fff": 8, "ffff": 9, "sf": 10, "sfz": 11,
        "fp": 12,
    }

    grouped = defaultdict(list)
    for dyn in score.recurse().getElementsByClass(m21dynamics.Dynamic):
        off = round(float(dyn.getOffsetInHierarchy(score)), 6)
        grouped[off].append(deepcopy(dyn))

    for off, dyns in grouped.items():
        dyns.sort(key=lambda d: dynamic_rank.get((d.value or "").lower(), -1), reverse=True)
        dynamic_by_offset[off] = dyns[0]

    dynamic_words = ("cresc", "dim", "decresc", "rit", "rall", "accel", "sempre")
    for text in score.recurse().getElementsByClass(m21expressions.TextExpression):
        content = (getattr(text, "content", "") or "").strip()
        if not content:
            continue
        lower = content.lower()
        if not any(word in lower for word in dynamic_words):
            continue
        off = round(float(text.getOffsetInHierarchy(score)), 6)
        if off not in text_by_offset:
            text_by_offset[off] = deepcopy(text)

    return dynamic_by_offset, text_by_offset


def insert_expressive_markings(
    rh_part: stream.Part,
    lh_part: stream.Part,
    tempo_at_offset: dict,
    measure_map: dict,
    dynamic_by_offset: dict,
    text_by_offset: dict,
) -> None:
    """Insert collected tempo/dynamics/text inside output measures."""
    import bisect

    measure_starts = sorted(
        (abs_off, mnum, bar_dur)
        for mnum, (abs_off, bar_dur) in measure_map.items()
    )
    ms_offsets = [x[0] for x in measure_starts]

    def insert_at_abs(part: stream.Part, abs_off: float, obj) -> bool:
        if not measure_starts:
            return False
        idx = bisect.bisect_right(ms_offsets, abs_off + 1e-6) - 1
        idx = max(0, min(idx, len(measure_starts) - 1))
        measure_start, mnum, bar_dur = measure_starts[idx]
        rel_off = max(0.0, min(float(abs_off) - measure_start, max(bar_dur - _GRID, 0.0)))
        measure = part.measure(mnum)
        if measure is None:
            return False
        measure.insert(_snap_offset(rel_off), deepcopy(obj))
        return True

    # Tempo belongs once in the piano system; put it on the upper staff.
    for off, tm in sorted(tempo_at_offset.items()):
        insert_at_abs(rh_part, off, tm)

    # Dynamics are copied to both staves so LH-only passages still show them.
    for off, dyn in sorted(dynamic_by_offset.items()):
        insert_at_abs(rh_part, off, dyn)
        insert_at_abs(lh_part, off, dyn)

    for off, text in sorted(text_by_offset.items()):
        insert_at_abs(rh_part, off, text)


def _element_pitches(el) -> list[int]:
    if isinstance(el, m21note.Note):
        return [el.pitch.midi]
    if isinstance(el, m21chord.Chord):
        return [p.midi for p in el.pitches]
    return []


def collect_slur_specs(score: stream.Score) -> list[tuple[float, int, float, int]]:
    """
    Collect source slur endpoint specs as (start_offset, start_pitch,
    end_offset, end_pitch). This is best-effort because output notes are rebuilt.
    """
    specs = []
    for slur in score.recurse().getElementsByClass(m21spanner.Slur):
        elements = [el for el in slur.getSpannedElements() if _element_pitches(el)]
        if len(elements) < 2:
            continue
        start = elements[0]
        end = elements[-1]
        start_pitches = _element_pitches(start)
        end_pitches = _element_pitches(end)
        specs.append((
            round(float(start.getOffsetInHierarchy(score)), 6),
            max(start_pitches),
            round(float(end.getOffsetInHierarchy(score)), 6),
            max(end_pitches),
        ))
    return specs


def collect_fermata_specs(score: stream.Score) -> list[tuple[float, tuple[int, ...]]]:
    """
    Collect fermatas attached to source notes/chords.

    Returns (absolute_offset, pitches) specs. The reduction rebuilds notes from
    scratch, so these specs are later matched onto surviving output events.
    """
    specs = []
    seen = set()
    for el in score.recurse().notes:
        fermatas = [
            expr for expr in getattr(el, "expressions", [])
            if isinstance(expr, m21expressions.Fermata)
        ]
        if not fermatas:
            continue
        off = _snap_offset(float(el.getOffsetInHierarchy(score)))
        pitches = tuple(sorted(_element_pitches(el)))
        key = (off, pitches)
        if key in seen:
            continue
        seen.add(key)
        specs.append(key)
    return specs


def apply_fermatas_to_output(out_score: stream.Score, fermata_specs: list[tuple[float, tuple[int, ...]]]) -> int:
    """Attach fermatas to matching output notes/chords when source events survive."""
    if not fermata_specs:
        return 0

    events_by_offset = defaultdict(list)
    for el in out_score.recurse().notes:
        off = _snap_offset(float(el.getOffsetInHierarchy(out_score)))
        events_by_offset[off].append(el)

    added = 0
    applied_ids = set()
    for off, source_pitches in fermata_specs:
        candidates = events_by_offset.get(off, [])
        if not candidates:
            continue

        source_pitch_set = set(source_pitches)
        target = None
        for el in candidates:
            if source_pitch_set.intersection(_element_pitches(el)):
                target = el
                break

        # If the exact fermata pitch was reduced away, keep the expressive stop
        # on another note/chord at that same onset.
        if target is None and len(candidates) == 1:
            target = candidates[0]

        if target is None or id(target) in applied_ids:
            continue
        target.expressions.append(m21expressions.Fermata())
        applied_ids.add(id(target))
        added += 1
    return added


def apply_slurs_to_output(out_score: stream.Score, slur_specs: list[tuple[float, int, float, int]]) -> int:
    """Best-effort slur recreation by matching source endpoints to output events."""
    if not slur_specs:
        return 0

    event_lookup = defaultdict(list)
    for el in out_score.recurse().notes:
        off = _snap_offset(float(el.getOffsetInHierarchy(out_score)))
        for pitch in _element_pitches(el):
            event_lookup[(off, pitch)].append(el)

    added = 0
    for start_off, start_pitch, end_off, end_pitch in slur_specs:
        start_el = event_lookup.get((_snap_offset(start_off), start_pitch), [None])[0]
        end_el = event_lookup.get((_snap_offset(end_off), end_pitch), [None])[0]
        if start_el is None or end_el is None or start_el is end_el:
            continue
        out_score.insert(0, m21spanner.Slur(start_el, end_el))
        added += 1
    return added


def select_notes(
    phrase_notes: list,
    clf: NoteClassifier,
    threshold: float,
    top3_conf: float,
) -> list:
    """Run the classifier on a phrase and return the list of kept note dicts."""
    if not phrase_notes:
        return []

    df    = extract_features(phrase_notes, all_notes=phrase_notes)
    X     = clf._feature_matrix(df)
    proba = clf.model.predict_proba(X)[:, 1]   # P(KEEP)

    # Group notes by timepoint.
    tp_col = (df["measure_norm"].round(6).astype(str)
              + "_"
              + df["offset_in_measure"].round(6).astype(str))
    keep_set = set()

    for _, grp_idx in df.groupby(tp_col, sort=False).groups.items():
        grp_idx = list(grp_idx)
        ranked  = sorted(grp_idx, key=lambda i: proba[i], reverse=True)

        # Adaptive top-K meausre 
        top3 = ranked[:min(3, len(ranked))]
        min_keep = 3 if np.mean([proba[i] for i in top3]) >= top3_conf else 2

        for i in ranked[:min_keep]:
            keep_set.add(i)
        for i in ranked[min_keep:]:
            if proba[i] >= threshold:
                keep_set.add(i)

    return [phrase_notes[i] for i in sorted(keep_set)]


def select_mdp_notes(phrase_notes: list, mdp_model: ReductionMDP) -> list:
    """Select notes with the MDP only, using musical reward heuristics."""
    if not phrase_notes:
        return []
    return mdp_model.process_phrase(group_by_timepoint(phrase_notes), train=False)


def build_measure(
    mnum: int,
    kept_notes: list,
    abs_offset: float,
    bar_dur: float,
    ts: meter.TimeSignature,
    add_ts: bool,
) -> stream.Measure:
    """
    Build a music21 Measure from a list of kept note dicts.

    This function takes in:
    mnum: Measure number (for display).
    kept_notes: Note dicts that fall in this measure.
    abs_offset: Absolute score offset where this measure starts.
    bar_dur: Duration of this bar in quarter-lengths.
    ts: TimeSignature for this bar.
    add_ts: Whether to insert the time signature into this measure.
    """
    m = stream.Measure(number=mnum)
    if add_ts:
        m.insert(0.0, deepcopy(ts))

    if not kept_notes:
        r = m21note.Rest(quarterLength=bar_dur)
        m.insert(0.0, r)
        return m

    # Group by snapped relative offset to form chords
    tp_map = defaultdict(list)
    for n in kept_notes:
        rel = _snap_offset(n["offset"] - abs_offset)
        rel = max(0.0, min(rel, bar_dur - _GRID))
        tp_map[rel].append(n)

    # Build a sorted list of (offset, element) events.
    # Cap each note's duration to the NEXT event's onset (or bar end),
    # whichever is shorter.  This prevents whole-notes from overlapping
    # subsequent 8th/16th events in the same voice.
    sorted_offsets = sorted(tp_map.keys())
    events = []
    for i, off in enumerate(sorted_offsets):
        ns      = tp_map[off]
        pitches = sorted(set(n_["midi_pitch"] for n_ in ns))
        dur     = max(n_["duration_ql"] for n_ in ns)

        # Ceiling 1: don't exceed the bar end
        max_dur = bar_dur - off
        # Ceiling 2: don't overlap the next event's onset
        if i + 1 < len(sorted_offsets):
            max_dur = min(max_dur, sorted_offsets[i + 1] - off)

        dur = _snap_dur_floor(min(dur, max_dur))
        dur = max(dur, _GRID)

        if len(pitches) == 1:
            el = m21note.Note(pitches[0])
        else:
            el = m21chord.Chord(pitches)
        el.quarterLength = dur
        events.append((off, el))

    # Fill gaps with rests so the bar is exactly bar_dur long.
    # Use _snap_dur_floor for rests so they never exceed the gap.
    cursor = 0.0
    for off, el in events:
        # Fill any gap before this event with rest(s)
        while off > cursor + 1e-6:
            gap  = off - cursor
            rdur = _snap_dur_floor(gap)
            r    = m21note.Rest(quarterLength=rdur)
            m.insert(cursor, r)
            cursor = round(cursor + rdur, 9)

        m.insert(off, el)
        cursor = round(off + el.quarterLength, 9)

    # Trailing rest(s) to fill remainder of bar
    while bar_dur - cursor > 1e-6:
        remaining = bar_dur - cursor
        rdur      = _snap_dur_floor(remaining)
        r         = m21note.Rest(quarterLength=rdur)
        m.insert(cursor, r)
        cursor    = round(cursor + rdur, 9)

    return m


def build_part(
    note_list: list,
    part_name: str,
    clef_obj,
    measure_map: dict,
    m_min: int,
    m_max: int,
    ts_at_offset: dict,        # abs_offset -> TimeSignature
    tempo_at_offset: dict,     # abs_offset -> MetronomeMark
) -> stream.Part:
    """
    Assemble a complete Part from selected notes.

    Every measure from m_min to m_max is included (empty bars get whole rests).
    Measures are inserted at their correct absolute offsets.
    """
    import bisect

    part = stream.Part()
    part.partName = part_name
    part.insert(0, m21instrument.Piano())
    part.insert(0, deepcopy(clef_obj))

    #Group notes by measure using ABSOLUTE OFFSET RANGE (not measure_number, which can be inconsistent across parts in music21).
    # Build sorted list of measure boundaries for faster binary search.
    measure_starts = sorted(
        (abs_off, mnum, bar_dur)
        for mnum, (abs_off, bar_dur) in measure_map.items()
    )
    ms_offsets = [x[0] for x in measure_starts]  # sorted abs offsets

    notes_by_measure = defaultdict(list)
    for n in note_list:
        idx = bisect.bisect_right(ms_offsets, n["offset"] + 1e-6) - 1
        idx = max(0, min(idx, len(measure_starts) - 1))
        _, mnum, _ = measure_starts[idx]
        notes_by_measure[mnum].append(n)

    # Track which time sig is currently active
    prev_ts  = meter.TimeSignature("4/4")
    seen_ts  = set()

    for mnum in range(m_min, m_max + 1):
        if mnum not in measure_map:
            continue
        abs_off, bar_dur = measure_map[mnum]

        # Find the active time signature for this measure
        active_ts = prev_ts
        for ts_off in sorted(ts_at_offset):
            if ts_off <= abs_off + 1e-6:
                active_ts = ts_at_offset[ts_off]
        prev_ts = active_ts

        # Add TS to measure only on first occurrence or when it changes
        ts_sig = active_ts.ratioString
        add_ts = ts_sig not in seen_ts or (
            mnum > m_min
            and abs_off in ts_at_offset
            and ts_at_offset[abs_off].ratioString != ts_sig
        )
        if add_ts:
            seen_ts.add(ts_sig)

        m = build_measure(
            mnum        = mnum,
            kept_notes  = notes_by_measure.get(mnum, []),
            abs_offset  = abs_off,
            bar_dur     = bar_dur,
            ts          = active_ts,
            add_ts      = (mnum == m_min) or (abs_off in ts_at_offset),
        )
        part.insert(abs_off, m)

    return part


def reduce(
    input_path: str,
    output_path: str,
    model_path: str  = "models/rf_reducer.joblib",
    phrase_bars: int = 8,
    threshold: float = 0.30,
    top3_conf: float = 0.20,
    split_pitch: int = 60,
    verbose: bool    = False,
    mode: str = "hybrid",
    mdp_model_path: str = "models/mdp_reducer.joblib",
) -> None:
    """
    Full pipeline.

    Takes in an orchestral MusicXML file and parses into:
    input_path: Path to orchestral MusicXML file.
    output_path: Path to write the piano reduction.
    model_path: Path to trained RF model (.joblib).
    phrase_bars: Number of bars per processing chunk.
    threshold: Additional keep threshold for notes above top-K.
    top3_conf: Mean P(KEEP) threshold to use 3 vs 2 notes per timepoint.
    split_pitch: MIDI pitch dividing right hand (≥) from left hand (<).
    verbose: Print per-phrase detail.
    """
    """Load score"""
    print(f"Loading RF model: {model_path}")
    clf = NoteClassifier.load(model_path)
    mdp_model = None
    if mode in {"hybrid", "mdp_only"}:
        if mdp_model_path and os.path.exists(mdp_model_path):
            mdp_model = ReductionMDP.load(mdp_model_path)
        else:
            print("  MDP model not found; using untrained heuristic MDP.")
            mdp_model = ReductionMDP(epsilon=0.0)

    """Parse score"""
    print(f"Parsing: {input_path}")
    score     = load_score(input_path)
    all_notes = extract_notes_from_score(score)

    if not all_notes:
        raise ValueError("No notes extracted from score.")

    """Build measure map from the score's structure. This gives us each measure's absolute offset and bar duration, which are crucial for correctly grouping notes into measures and preserving rhythms."""
    measure_map = build_measure_map(score)
    measure_map = normalize_measure_map(measure_map)

    """Fall back: if some measures weren't found in measure_map, infer from 4/4"""
    if not measure_map:
        print("WARNING: No measures found in score structure — assuming 4/4")
        measures = sorted(set(n["measure_number"] for n in all_notes))
        for mnum in measures:
            measure_map[mnum] = ((mnum - 1) * 4.0, 4.0)

    assign_measure_numbers_by_offset(all_notes, measure_map)
    measures = sorted(measure_map)
    m_min, m_max = measures[0], measures[-1]
    note_measure_count = len(set(n["measure_number"] for n in all_notes))
    print(f"  {len(all_notes):,} notes across {note_measure_count} active measures "
          f"(output m{m_min}–m{m_max}), {len(score.parts)} parts")

    """Collect time signatures from the reference part"""
    ts_at_offset = {}
    ref_part     = list(score.parts)[0]
    for ts in ref_part.flatten().getElementsByClass(meter.TimeSignature):
        ts_at_offset[float(ts.offset)] = ts
    if not ts_at_offset:
        ts_at_offset[0.0] = meter.TimeSignature("4/4")

    """Collect tempo/dynamic/slur markings from the source score."""
    tempo_at_offset = collect_tempo_markings(score)
    dynamic_by_offset, text_by_offset = collect_expressive_markings(score)
    slur_specs = collect_slur_specs(score)
    fermata_specs = collect_fermata_specs(score)

    print(f"  Time signatures: "
          + ", ".join(f"{ts.ratioString}@{off:.1f}ql"
                      for off, ts in sorted(ts_at_offset.items())))
    print(f"  Markings: {len(tempo_at_offset)} tempos, "
          f"{len(dynamic_by_offset)} dynamics, "
          f"{len(text_by_offset)} text directions, "
          f"{len(slur_specs)} slurs, "
          f"{len(fermata_specs)} fermatas")

    """Process in phrases and select notes"""
    print(f"\nReducing in {phrase_bars}-bar phrases ({mode}) …")
    all_kept = []
    phrase_start = m_min

    while phrase_start <= m_max:
        phrase_end   = min(phrase_start + phrase_bars - 1, m_max)
        phrase_notes = [n for n in all_notes
                        if phrase_start <= n["measure_number"] <= phrase_end]

        if mode == "rf_only":
            kept = select_notes(phrase_notes, clf, threshold, top3_conf)
        elif mode == "mdp_only":
            kept = select_mdp_notes(phrase_notes, mdp_model)
        elif mode == "hybrid":
            kept = hybrid_reduction(phrase_notes, clf, mdp_model)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        all_kept.extend(kept)

        if verbose or True:   # always show phrase summary
            rate = len(kept) / len(phrase_notes) if phrase_notes else 0
            print(f"  m{phrase_start:4d}–m{phrase_end:4d}: "
                  f"{len(phrase_notes):5d} notes → {len(kept):4d} kept ({rate:.0%})")

        phrase_start += phrase_bars

    total_rate = len(all_kept) / len(all_notes) if all_notes else 0
    print(f"\nTotal: {len(all_notes):,} → {len(all_kept):,} kept ({total_rate:.1%})")

    """Split into two hands."""
    rh = [n for n in all_kept if n["midi_pitch"] >= split_pitch]
    lh = [n for n in all_kept if n["midi_pitch"] <  split_pitch]

    """If one hand ended up empty, split by median pitch instead of split_pitch cutoff."""
    if all_kept and (not rh or not lh):
        mid = int(np.median([n["midi_pitch"] for n in all_kept]))
        rh  = [n for n in all_kept if n["midi_pitch"] >= mid]
        lh  = [n for n in all_kept if n["midi_pitch"] <  mid]

    print(f"  RH ({split_pitch}+): {len(rh):,} notes")
    print(f"  LH (<{split_pitch}): {len(lh):,} notes")

    """VBUILD THE OUTPUT PARTS"""
    print("\nBuilding piano reduction …")

    rh_part = build_part(
        note_list       = rh,
        part_name       = "Piano",
        clef_obj        = m21clef.TrebleClef(),
        measure_map     = measure_map,
        m_min           = m_min,
        m_max           = m_max,
        ts_at_offset    = ts_at_offset,
        tempo_at_offset = tempo_at_offset,
    )
    lh_part = build_part(
        note_list       = lh,
        part_name       = "Piano",
        clef_obj        = m21clef.BassClef(),
        measure_map     = measure_map,
        m_min           = m_min,
        m_max           = m_max,
        ts_at_offset    = ts_at_offset,
        tempo_at_offset = tempo_at_offset,
    )

    insert_expressive_markings(
        rh_part,
        lh_part,
        tempo_at_offset,
        measure_map,
        dynamic_by_offset,
        text_by_offset,
    )

    out_score = stream.Score()
    out_score.insert(0, rh_part)
    out_score.insert(0, lh_part)
    fermatas_added = apply_fermatas_to_output(out_score, fermata_specs)
    slurs_added = apply_slurs_to_output(out_score, slur_specs)

    """Verify and notate it so it works"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Writing: {output_path}")
    out_score.write("musicxml", fp=output_path)

    verification = converter.parse(output_path)
    parts_info   = []
    for p in verification.parts:
        ms = list(p.getElementsByClass(stream.Measure))
        ns = list(p.flatten().notes)
        parts_info.append(f"{len(ms)} measures, {len(ns)} events")

    expected_dur = max(
        abs_off + bar_dur
        for abs_off, bar_dur in measure_map.values()
        if abs_off is not None
    ) if measure_map else 0.0

    print(f"\n{'='*50}")
    print(f"  Output: {output_path}")
    print(f"  RH: {parts_info[0] if parts_info else 'N/A'}")
    print(f"  LH: {parts_info[1] if len(parts_info) > 1 else 'N/A'}")
    print(f"  Duration: {verification.highestTime:.3f} ql "
          f"(expected ≈ {expected_dur:.3f} ql)")
    print(f"  Keep rate: {total_rate:.1%}")
    print(f"  Fermatas transferred: {fermatas_added}/{len(fermata_specs)}")
    print(f"  Slurs transferred: {slurs_added}/{len(slur_specs)}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Reduce an orchestral MusicXML score to a piano reduction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  help="Input orchestral MusicXML file")
    parser.add_argument("output", help="Output piano reduction MusicXML file")
    parser.add_argument(
        "--model", default="models/rf_reducer.joblib",
        help="Path to trained model (default: models/rf_reducer.joblib)"
    )
    parser.add_argument(
        "--phrase-bars", type=int, default=8,
        help="Bars per processing phrase (default: 8)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.30,
        help="Additional keep threshold above top-K (default: 0.30)"
    )
    parser.add_argument(
        "--top3-conf", type=float, default=0.20,
        help="Mean P(KEEP) cutoff for keeping 3 vs 2 notes/timepoint (default: 0.20)"
    )
    parser.add_argument(
        "--split-pitch", type=int, default=60,
        help="MIDI pitch dividing RH (>=) from LH (<) (default: 60 = C4)"
    )
    parser.add_argument(
        "--mode", choices=["rf_only", "mdp_only", "hybrid"], default="hybrid",
        help="Reduction mode: rf_only, mdp_only, or hybrid (default: hybrid)"
    )
    parser.add_argument(
        "--mdp-model", default="models/mdp_reducer.joblib",
        help="Path to trained MDP model (default: models/mdp_reducer.joblib)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-measure detail"
    )
    args = parser.parse_args()

    # Ensure model_path is absolute
    model_path = os.path.abspath(args.model)
    model_path = args.model
    if not os.path.isabs(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        print("       Train a model first with ml_model.py", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    reduce(
        input_path  = args.input,
        output_path = args.output,
        model_path  = model_path,
        phrase_bars = args.phrase_bars,
        threshold   = args.threshold,
        top3_conf   = args.top3_conf,
        split_pitch = args.split_pitch,
        verbose     = args.verbose,
        mode        = args.mode,
        mdp_model_path = args.mdp_model,
    )


if __name__ == "__main__":
    main()
