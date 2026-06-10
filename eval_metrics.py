"""
eval_metrics.py

Code by Sean Tan, Spencer Cha, Theresa Jiao

Runs all four evaluation metrics for a piano reduction vs orchestral score:
  1. Vertical Harmonies: Weighted Jaccard, chord quality, root and bass note matching
  2. Melodic Shape: Contour, interval profile, range preservation
  3. Rhythmic Quality: Onset preservation, syncopation, rhythm pattern similarity
  4. Voice Leading: Melody, bass, and inner voice preservation, playability

To use, run:
    python evaluate_reduction.py score.xml piano.xml 
"""

import sys
import argparse
import os
import tempfile
import numpy as np
import music21
from music21 import converter, note, chord, meter
from collections import Counter
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
try:
    from dtw import dtw
except ImportError:
    def dtw(a, b, dist=euclidean):
        """Small fallback compatible with the external dtw package's first result."""
        n, m = len(a), len(b)
        costs = np.full((n + 1, m + 1), np.inf)
        costs[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                step = dist(a[i - 1], b[j - 1])
                costs[i, j] = step + min(costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1])
        return (costs[n, m],)


def evaluate_on_beethoven(model, symphony_num, movement_num):
    """Load Beethoven, apply model, return quality metrics.

    Beethoven support is validation-only here: the function combines local Kern
    parts, generates a reduction, and evaluates the generated piano texture
    against the orchestral source using the existing metric suite.
    """
    if int(symphony_num) != 5:
        raise ValueError("Only Beethoven Symphony No. 5 is available in symph5data.")

    from kern_loader import load_beethoven_movement
    from reduce_score import reduce

    model_path = model
    cleanup_model = None
    if not isinstance(model, (str, os.PathLike)):
        cleanup_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        cleanup_model.close()
        model.save(cleanup_model.name)
        model_path = cleanup_model.name

    with tempfile.TemporaryDirectory() as tmpdir:
        orch_path = os.path.join(tmpdir, f"beethoven5_mvt{movement_num}.musicxml")
        reduction_path = os.path.join(tmpdir, f"beethoven5_mvt{movement_num}_reduction.musicxml")
        score = load_beethoven_movement(int(movement_num))
        score.write("musicxml", fp=orch_path)
        reduce(orch_path, reduction_path, model_path=str(model_path), mode="hybrid")
        results = evaluate_all(orch_path, reduction_path)
        section_scores = compute_section_scores(results)
        results.update(section_scores)
        results["final_score"] = compute_final_score(section_scores)

    if cleanup_model is not None:
        os.unlink(cleanup_model.name)
    return results


def load_stream(path):
    '''
    Returns stream object from inputted path.
    '''
    score = converter.parse(path)
    if not isinstance(score, music21.stream.Score):
        score = score.toScore()
    return score


def get_clean_pitched_stream(input_stream):
    '''
    Strips unpitched percussion while preserving offsets to prevent errors from unpitched notes.
        
    :param input_stream: score or piano stream 
    :type input_stream: music21.stream.Score
    :return: cleaned score
    :rtype: Score
    '''

    clean_score = music21.stream.Score()
    if input_stream.metadata:
        clean_score.metadata = input_stream.metadata

    for part in input_stream.parts:
        clean_part = music21.stream.Part()
        clean_part.id = part.id
        for el in part.recurse().notes:
            if el.classes[0] == 'Unpitched':
                continue
            clean_part.insert(el.getOffsetInHierarchy(part), el)
        if len(clean_part) > 0:
            clean_score.insert(0, clean_part)

    return clean_score


### Vertical harmonies ###

CHORD_POSITION_WEIGHTS = {"root": 3.0, "fifth": 1.5, "other": 1.0}
def build_chord_map(chordified_stream):
    '''
    Builds chord map dictionary of all of the chords in inputted sream
    
    :param chordified_stream: score or piano stream 
    :type chordified_stream: music21.stream.Score
    :return: chord map
    :rtype: dict
    '''
    chord_map = {}
    for el in chordified_stream.recurse().getElementsByClass(music21.chord.Chord):
        offset = round(float(el.offset), 6)
        if offset not in chord_map:
            chord_map[offset] = el
    return chord_map


def get_chord_position_weights(chord_obj):
    '''
    Get weights for pitchs in chords 
    
    :param chord_obj: music21 chord 
    :return: weights 
    :rtype: dict
    '''
    weights = {}
    try:
        root_pc  = chord_obj.root().pitchClass
        fifth_cp = (root_pc + 7) % 12
        for cp in chord_obj.pitchClasses:
            if cp == root_pc:
                weights[cp] = CHORD_POSITION_WEIGHTS["root"]
            elif cp == fifth_cp:
                weights[cp] = CHORD_POSITION_WEIGHTS["fifth"]
            else:
                weights[cp] = CHORD_POSITION_WEIGHTS["other"]
    except Exception:
        for cp in chord_obj.pitchClasses:
            weights[cp] = CHORD_POSITION_WEIGHTS["other"]
    return weights


def weighted_jaccard(c1, c2):
    '''
    Get weighted jaccard for positions
    
    :param c1: music21 chord object
    :param c2: music21 chord object
    '''
    w1 = get_chord_position_weights(c1)
    w2 = get_chord_position_weights(c2)
    all_pcs = set(w1.keys()) | set(w2.keys())
    if not all_pcs:
        return 1.0
    num = sum(min(w1.get(pc, 0.0), w2.get(pc, 0.0)) for pc in all_pcs)
    den = sum(max(w1.get(pc, 0.0), w2.get(pc, 0.0)) for pc in all_pcs)
    if den > 0:
        return num / den
    return 1.0


def chord_quality_similarity(c1, c2):
    '''
    Compare quality of two chords
    
    :param c1: music21 chord object
    :param c2: music21 chord object
    '''
    try:
        q1, q2 = c1.quality, c2.quality
        if q1 == q2:
            return 1.0 if c1.commonName == c2.commonName else 0.5
        return 0.0
    except Exception:
        return 0.0


def evaluate_vertical_harmonies(stream1, stream2):
    '''
    Full evaluation function for vertical harmonies. This looks at the 
    weighted Jaccard chord position similarity, chord quality match,
    root note match, and bass note match across all beats.
    
    :param stream1: score stream
    :type stream1: music21.stream.Score
    :param stream2: piano stream
    :type stream2: music21.stream.Score
    :return: Dictionary of different vertical harmoniy comparasions
    :rtype: dict
    '''

    map1 = build_chord_map(stream1.chordify())
    map2 = build_chord_map(stream2.chordify())

    shared    = sorted(set(map1) & set(map2))
    only_in_1 = len(set(map1) - set(map2))
    only_in_2 = len(set(map2) - set(map1))

    if not shared:
        return {
            "harmonic_weighted_jaccard_mean":     0.0,
            "harmonic_weighted_jaccard_per_beat": {},
            "harmonic_quality_match_mean":        0.0,
            "harmonic_quality_exact_match_pct":   0.0,
            "harmonic_root_match_pct":            0.0,
            "harmonic_bass_match_pct":            0.0,
            "harmonic_compared_beats":            0,
            "harmonic_beats_only_in_score":       only_in_1,
            "harmonic_beats_only_in_piano":       only_in_2,
        }

    wjaccard, quality_scores, quality_exact, root_matches, bass_matches = [], [], [], [], []

    for offset in shared:
        c1, c2 = map1[offset], map2[offset]

        wjaccard.append(weighted_jaccard(c1, c2))

        qs = chord_quality_similarity(c1, c2)
        quality_scores.append(qs)
        quality_exact.append(1 if qs == 1.0 else 0)

        try:
            root_matches.append(int(c1.root().pitchClass == c2.root().pitchClass))
        except Exception:
            pass
        try:
            bass_matches.append(int(c1.bass().pitchClass == c2.bass().pitchClass))
        except Exception:
            pass

    return {
        "harmonic_weighted_jaccard_mean":     sum(wjaccard) / len(wjaccard),
        "harmonic_weighted_jaccard_per_beat": dict(zip(shared, wjaccard)),
        "harmonic_quality_match_mean":        sum(quality_scores) / len(quality_scores),
        "harmonic_quality_exact_match_pct":   sum(quality_exact) / len(quality_exact),
        "harmonic_root_match_pct":            sum(root_matches) / len(root_matches) if root_matches else 0.0,
        "harmonic_bass_match_pct":            sum(bass_matches) / len(bass_matches) if bass_matches else 0.0,
        "harmonic_compared_beats":            len(shared),
        "harmonic_beats_only_in_score":       only_in_1,
        "harmonic_beats_only_in_piano":       only_in_2,
    }


### Melodic shape ###

def extract_melody_notes(stream_obj):
    '''
    Get the notes of the melody by taking the soprano line, highest pitch
    
    :param stream_obj: score or piano stream
    :return: list of melofy notes 
    '''
    chordified = stream_obj.chordify()
    melody = []
    for el in chordified.recurse().getElementsByClass(music21.chord.Chord):
        pitches = [p.ps for p in el.pitches]
        if pitches:
            melody.append((float(el.offset), max(pitches)))
    melody.sort(key=lambda x: x[0])
    return melody


def extract_bass_notes(stream_obj):
    '''
    Get the bass notes of the melody by taing the lowest note
    
    :param stream_obj: score or piano stream
    :return: list of bass notes
    '''

    chordified = stream_obj.chordify()
    bass = []
    for el in chordified.recurse().getElementsByClass(music21.chord.Chord):
        pitches = [p.ps for p in el.pitches]
        if pitches:
            bass.append((float(el.offset), min(pitches)))
    bass.sort(key=lambda x: x[0])
    return bass


def to_interval_sequence(note_sequence):
    '''
    Get interval shape sequence, looking at intervals between each note

    :param note_sequence: list of notes pitches
     :return: list of pitch differences
    '''

    pitches = [p for _, p in note_sequence]
    return [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]


def to_contour(interval_sequence):
    '''
    Get direction of interval travel to see if the pitch went up, down, or stayed same 

    :param interval_sequence: list of pitch/interval differences 
    :return: list of directions in the form of -1, 0, 1
    '''
    return [int(np.sign(i)) for i in interval_sequence]


def dtw_similarity(seq1, seq2):
    '''
    Apply dynamic time warping to account for slightly shifted phrases 
    
    :param seq1: score contour 
    :param seq2: piano contour
    :return: dtw similarity score 
    :rtype: float
    '''
    if not seq1 or not seq2:
        return 0.0
    a = np.array(seq1, dtype=float).reshape(-1, 1)
    b = np.array(seq2, dtype=float).reshape(-1, 1)
    result = dtw(a, b, dist=euclidean)
    normalised = result[0] / (len(seq1) + len(seq2))
    return float(np.exp(-normalised / 12.0))


def contour_exact_match(c1, c2):
    '''
    Evaluate how much of the actually contour matches 
    
    :param c1: score contour
    :param c2: piano contour
    :return: total number of exact matches
    :rtype: float
    '''
    if not c1 or not c2:
        return 0.0
    n = min(len(c1), len(c2))
    return sum(1 for i in range(n) if c1[i] == c2[i]) / n


def interval_cosine(i1, i2):
    '''
    Look at distribution of intervals
    
    :param i1: score intervals
    :param i2: piano intercals
    :return: cosine similarity
    :rtype: float
    '''
    if not i1 or not i2:
        return 0.0
    bins = np.arange(-24, 25)
    h1, _ = np.histogram(i1, bins=bins)
    h2, _ = np.histogram(i2, bins=bins)
    if np.all(h1 == 0) or np.all(h2 == 0):
        return 0.0
    return float(1.0 - cosine(h1.astype(float), h2.astype(float)))


def pitch_range_series(stream_obj, beat_step=1.0):
    '''
    Get pitch range across each beat, how wide the spead of the pitches are
    
    :param stream_obj: score or piano stream
    :param beat_step: list of pitch ranges for each beat
    '''
    chordified = stream_obj.chordify()
    ranges = []
    for offset in np.arange(0, chordified.highestTime, beat_step):
        els = chordified.getElementsByOffset(
            offset, offset + beat_step,
            includeEndBoundary=False, mustBeginInSpan=False).stream()
        pitches = [p.ps for p in els.recurse().pitches]
        ranges.append(max(pitches) - min(pitches) if len(pitches) > 1 else 0.0)
    return np.array(ranges)


def evaluate_melodic_shape(stream1, stream2):
    '''
    Full function to evaluate melodic shape. Look at the pitch contour, interval profile,
    and the range preservation between beat and comapres the score and piano reduction. 
    
    :param stream1: score stream
    :type stream1: music21.stream.Score
    :param stream2: piano stream
    :type stream2: music21.stream.Score
    :return: Dictionary of different melodic shape data and comparasions
    :rtype: dict
    '''

    orch_melody  = extract_melody_notes(stream1)
    piano_melody = extract_melody_notes(stream2)
    orch_bass    = extract_bass_notes(stream1)
    piano_bass   = extract_bass_notes(stream2)

    orch_mel_int  = to_interval_sequence(orch_melody)
    piano_mel_int = to_interval_sequence(piano_melody)
    orch_bas_int  = to_interval_sequence(orch_bass)
    piano_bas_int = to_interval_sequence(piano_bass)

    orch_mel_con  = to_contour(orch_mel_int)
    piano_mel_con = to_contour(piano_mel_int)
    orch_bas_con  = to_contour(orch_bas_int)
    piano_bas_con = to_contour(piano_bas_int)

    o_range = pitch_range_series(stream1)
    p_range = pitch_range_series(stream2)
    n = min(len(o_range), len(p_range))
    if n > 1 and np.std(o_range[:n]) > 0 and np.std(p_range[:n]) > 0:
        range_pres, _ = pearsonr(o_range[:n], p_range[:n])
    else:
        range_pres = 0.0

    return {
        "melodic_contour_exact_match_soprano": contour_exact_match(orch_mel_con, piano_mel_con),
        "melodic_contour_dtw_soprano":         dtw_similarity(orch_mel_con, piano_mel_con),
        "melodic_contour_exact_match_bass":    contour_exact_match(orch_bas_con, piano_bas_con),
        "melodic_contour_dtw_bass":            dtw_similarity(orch_bas_con, piano_bas_con),
        "melodic_interval_cosine_soprano":     interval_cosine(orch_mel_int, piano_mel_int),
        "melodic_interval_dtw_soprano":        dtw_similarity(orch_mel_int, piano_mel_int),
        "melodic_interval_cosine_bass":        interval_cosine(orch_bas_int, piano_bas_int),
        "melodic_interval_dtw_bass":           dtw_similarity(orch_bas_int, piano_bas_int),
        "melodic_range_preservation":          float(range_pres),
    }


### Rhythmic quality ###

def extract_events(stream_obj):
    '''
    Extract and separate different features from the score, separating notes, chords, and rests 
    
    :param stream_obj: score or piano stream
    :return: list of all of the the features categorized from the steam 
    :rtype: list
    '''
    events = []
    for el in stream_obj.flatten().notesAndRests:
        if isinstance(el, note.Note) and el.pitch:
            events.append({"offset": el.offset, "duration": el.duration.quarterLength, "type": "note"})
        elif isinstance(el, chord.Chord):
            events.append({"offset": el.offset, "duration": el.duration.quarterLength, "type": "chord"})
        elif isinstance(el, note.Rest):
            events.append({"offset": el.offset, "duration": el.duration.quarterLength, "type": "rest"})
    return events


def count_syncopations(stream_obj):
    '''
    Count syncopation instances 
    
    :param stream_obj: score or piano stream
    :return: list of syncopation instances 
    :rtype: list
    '''
    syncs = []
       # Analyze original for syncopation
    for part in stream_obj.parts:
        for measure in part.getElementsByClass(music21.stream.Measure):
            time_sig = measure.timeSignature
            if not time_sig:
                time_sig = measure.getContextByClass(meter.TimeSignature)
            
            if time_sig:
                beat_duration = time_sig.beatDuration.quarterLength
                
                for element in measure.flatten().notesAndRests:
                    if isinstance(element, (note.Note, chord.Chord)):
                        # Check if onset is off the beat
                        onset_in_measure = element.offset - measure.offset
                        beat_position = onset_in_measure % beat_duration
                        
                        # Syncopation: onset not on beat (with small tolerance)
                        if beat_position > 0.05 and beat_position < beat_duration - 0.05:
                            syncs.append(element.offset)

    return syncs


def evaluate_rhythmic_quality(stream1, stream2, stream1_raw, stream2_raw, onset_tolerance: float = 0.1):
    '''
    Full eval function of rhythmic quality, looking at onset preservation of notes, rhytmic 
    patern similarity, syncopation preservation, rest preservation, and rhythmic complexity 
    
    :param stream1: score stream
    :type stream1: music21.stream.Score
    :param stream2: piano stream
    :type stream2: music21.stream.Score
    :param stream1_raw: score raw
    :type stream1_raw: music21.stream.Score
    :param stream2_raw: piano raw
    :type stream2_raw: music21.stream.Score
    :param onset_tolerance: tolerance for onset of notes to account for shifting
    :type onset_tolerance: float
    :return: Dictionary of rhythmic quality data and comparasions
    :rtype: dict
    '''

    ev1 = extract_events(stream1)
    ev2 = extract_events(stream2)

    onsets1 = set(e["offset"] for e in ev1)
    onsets2 = set(e["offset"] for e in ev2)

    matched = sum(1 for o in onsets1 if any(abs(o - r) <= onset_tolerance for r in onsets2))
    onset_preservation = matched / len(onsets1) if onsets1 else 0.0

    dur1 = [e["duration"] for e in ev1 if e["type"] != "rest"]
    dur2 = [e["duration"] for e in ev2 if e["type"] != "rest"]
    all_durs = set(Counter(dur1)) | set(Counter(dur2))
    dur_diffs = [
        abs(Counter(dur1).get(d, 0) / max(len(dur1), 1) -
            Counter(dur2).get(d, 0) / max(len(dur2), 1))
        for d in all_durs
    ]
    rhythm_pattern_sim = 1.0 - np.mean(dur_diffs) if dur_diffs else 0.0

    sync1 = count_syncopations(stream1_raw)
    sync2 = count_syncopations(stream2_raw)
    matched_sync = sum(1 for s in sync1 if any(abs(s - r) <= onset_tolerance for r in sync2))
    sync_preservation = matched_sync / len(sync1) if sync1 else 1.0

    rests1 = set(e["offset"] for e in ev1 if e["type"] == "rest")
    rests2 = set(e["offset"] for e in ev2 if e["type"] == "rest")
    matched_rests = sum(1 for r in rests1 if any(abs(r - x) <= onset_tolerance for x in rests2))
    rest_preservation = matched_rests / len(rests1) if rests1 else 1.0

    unique1 = len(set(dur1))
    unique2 = len(set(dur2))

    return {
        "rhythmic_onset_preservation":        onset_preservation,
        "rhythmic_pattern_similarity":         float(rhythm_pattern_sim),
        "rhythmic_syncopation_preservation":   sync_preservation,
        "rhythmic_rest_preservation":          rest_preservation,
        "rhythmic_complexity_ratio":           unique2 / unique1 if unique1 else 0.0,
        "rhythmic_original_syncopations":      len(sync1),
        "rhythmic_reduction_syncopations":     len(sync2),
        "rhythmic_original_unique_durations":  unique1,
        "rhythmic_reduction_unique_durations": unique2,
    }


### Voice leading ###

def evaluate_voice_leading(stream1, stream2, melody_parts: list = [0], bass_parts: list = [-1], treble_threshold: float = 60.0):
    '''
    Eval function for voice leading. This looks at the preservation of the melody (soprano), the inner voices,
    and the bass voices. It also check for playability.
    
    :param stream1: score stream
    :type stream1: music21.stream.Score
    :param stream2: piano stream 
    :type stream2: music21.stream.Score
    :param melody_parts: melodic part of score (taking the highest part)
    :type melody_parts: list
    :param bass_parts: bass part of score (taking the lowest part)
    :type bass_parts: list
    :param treble_threshold: cutoff from left hand to right hand 
    :type treble_threshold: float
    :return: Dictionary of voice leading and playability data 
    :rtype: dict
    '''

    results  = {}
    n_parts  = len(list(stream1.parts))

    melody_notes = set()
    bass_notes   = set()
    inner_notes  = set()

    # Melody notes
    for idx in melody_parts:
        if idx < len(stream1.parts):
            for el in stream1.parts[idx].flatten().notesAndRests:
                if isinstance(el, note.Note):
                    melody_notes.add(el.pitch.ps)
                elif isinstance(el, chord.Chord):
                    melody_notes.add(max(p.ps for p in el.pitches))

    # Bass notes
    for idx in bass_parts:
        actual = idx if idx >= 0 else n_parts + idx
        if 0 <= actual < n_parts:
            for el in stream1.parts[actual].flatten().notesAndRests:
                if isinstance(el, note.Note):
                    bass_notes.add(el.pitch.ps)
                elif isinstance(el, chord.Chord):
                    bass_notes.add(min(p.ps for p in el.pitches))

    # Inner voice notes
    used = set((i if i >= 0 else n_parts + i) for i in melody_parts + bass_parts)
    for idx in set(range(n_parts)) - used:
        for el in stream1.parts[idx].flatten().notesAndRests:
            if isinstance(el, note.Note):
                inner_notes.add(el.pitch.ps)
            elif isinstance(el, chord.Chord):
                inner_notes.update(p.ps for p in el.pitches)

    # Piano reduction note sets
    rh, lh, all_red = set(), set(), set()
    for el in stream2.flatten().notesAndRests:
        pitches = ([el.pitch] if isinstance(el, note.Note) and el.pitch
                   else el.pitches if isinstance(el, chord.Chord) else [])
        for p in pitches:
            all_red.add(p.ps)
            (rh if p.midi >= treble_threshold else lh).add(p.ps)

    def preservation(src, tgt):
        return len(src & tgt) / len(src) if src else 0.0

    results["voiceleading_melody_in_right_hand"]     = preservation(melody_notes, rh)
    results["voiceleading_melody_total"]              = preservation(melody_notes, all_red)
    results["voiceleading_bass_in_left_hand"]         = preservation(bass_notes, lh)
    results["voiceleading_bass_total"]                = preservation(bass_notes, all_red)
    results["voiceleading_inner_voice_preservation"]  = preservation(inner_notes, all_red)
    results["voiceleading_inner_voice_omission"]      = 1.0 - results["voiceleading_inner_voice_preservation"]
    results["voiceleading_primary_coverage"]          = preservation(melody_notes | bass_notes, all_red)

    # Playability checks
    overlap, bad_rh, bad_lh, total = 0, 0, 0, 0
    for el in stream2.flatten().notesAndRests:
        if not isinstance(el, chord.Chord):
            continue
        total += 1
        midi    = sorted(p.midi for p in el.pitches)
        rh_midi = [m for m in midi if m >= treble_threshold]
        lh_midi = [m for m in midi if m < treble_threshold]
        if rh_midi and lh_midi and min(rh_midi) < max(lh_midi):
            overlap += 1
        if len(rh_midi) > 1 and (max(rh_midi) - min(rh_midi)) > 12:
            bad_rh += 1
        if len(lh_midi) > 1 and (max(lh_midi) - min(lh_midi)) > 12:
            bad_lh += 1

    results["voiceleading_voice_overlap_rate"]         = overlap / total if total else 0.0
    results["voiceleading_unfeasible_right_hand_rate"] = bad_rh  / total if total else 0.0
    results["voiceleading_unfeasible_left_hand_rate"]  = bad_lh  / total if total else 0.0

    # Melodic contour preservation
    orig_intervals, prev = [], None
    for el in stream1.parts[melody_parts[0]].flatten().notesAndRests:
        if isinstance(el, note.Note):
            if prev is not None:
                orig_intervals.append(el.pitch.ps - prev)
            prev = el.pitch.ps

    red_intervals, prev_top = [], None
    for el in sorted(stream2.flatten().notesAndRests, key=lambda x: x.offset):
        if isinstance(el, (note.Note, chord.Chord)):
            top = (el.pitch.ps if isinstance(el, note.Note)
                   else max(p.ps for p in el.pitches))
            if prev_top is not None:
                red_intervals.append(top - prev_top)
            prev_top = top

    orig_dir = [np.sign(i) for i in orig_intervals]
    red_dir  = [np.sign(i) for i in red_intervals]
    n = min(len(orig_dir), len(red_dir))
    results["voiceleading_melodic_contour_preservation"] = (
        sum(1 for i in range(n) if orig_dir[i] == red_dir[i]) / n if n else 0.0
    )

    return results



def evaluate_all(score_path, piano_path, melody_parts: list = [0], bass_parts: list = [-1], treble_threshold: float = 60.0, onset_tolerance: float = 0.1,):
    '''
    Run all evaluations. 
    
    :param score_path: path to score file
    :type score_path: str
    :param piano_path: path to piano reduction file 
    :type piano_path: str
    :param melody_parts: Melodic part (highest)
    :type melody_parts: list
    :param bass_parts: Bass part (lowest)
    :type bass_parts: list
    :param treble_threshold: cutoff for right-left hand assignment 
    :type treble_threshold: float
    :param onset_tolerance: toleracne for onset of notes when evaluating rhythm
    :type onset_tolerance: float
    :return: Dictionary of all eval data
    :rtype: dict
    '''
  
    print(f"Loading score : {score_path}")
    score_raw = load_stream(score_path)
    score = get_clean_pitched_stream(score_raw)

    print(f"Loading piano : {piano_path}")
    piano_raw = load_stream(piano_path)
    piano = get_clean_pitched_stream(piano_raw)

    print("Running evaluations...")
    results = {}

    print("  [1/4] Vertical harmonies...")
    results.update(evaluate_vertical_harmonies(score, piano))

    print("  [2/4] Melodic shape...")
    results.update(evaluate_melodic_shape(score, piano))

    print("  [3/4] Rhythmic quality...")
    results.update(evaluate_rhythmic_quality(score, piano, score_raw, piano_raw, onset_tolerance=onset_tolerance))

    print("  [4/4] Voice leading...")
    results.update(evaluate_voice_leading(
        score_raw, piano_raw,
        melody_parts=melody_parts,
        bass_parts=bass_parts,
        treble_threshold=treble_threshold,
    ))

    return results




def compute_section_scores(metrics):
    '''
    Combine metrics into one score per section, applying weights to each applicable similary
    score. Note: many parts of the data are left out because they are absolute numbers (reported counts, etc.)
    Predetermined weights based on content.
    
    :param metrics: metrics from eval functions
    :type metrics: dict
    :return: Dictionary of weighted data 
    :rtype: dict
    '''


    # VERTICAL HARMONIES
    harmonic = (
        0.25 * metrics["harmonic_root_match_pct"] +
        0.40 * metrics["harmonic_weighted_jaccard_mean"] +
        0.20 * metrics["harmonic_quality_match_mean"] +
        0.15 * metrics["harmonic_bass_match_pct"]
    )

    # MELODIC SHAPE
    melodic = (
        0.20 * metrics["melodic_contour_exact_match_soprano"] +
        0.15 * metrics["melodic_contour_exact_match_bass"] +
        0.30 * metrics["melodic_interval_cosine_soprano"] +
        0.25 * metrics["melodic_interval_cosine_bass"] +
        0.10 * metrics["melodic_range_preservation"]
    )

    # RHYTHMIC QUALITY
    rhythmic = (
        0.35 * metrics["rhythmic_onset_preservation"] +
        0.35 * metrics["rhythmic_pattern_similarity"] +
        0.20 * metrics["rhythmic_syncopation_preservation"] +
        0.10 * metrics["rhythmic_rest_preservation"]
    )

    # VOICE LEADING
    voiceleading = (
        0.10 * metrics["voiceleading_melody_in_right_hand"] +
        0.10 * metrics["voiceleading_bass_in_left_hand"] +
        0.19 * metrics["voiceleading_primary_coverage"] +
        0.15 * metrics["voiceleading_melodic_contour_preservation"] +
        0.10 * (1.0 - metrics["voiceleading_voice_overlap_rate"]) +
        0.18 * (1.0 - metrics["voiceleading_unfeasible_right_hand_rate"]) +
        0.18 * (1.0 - metrics["voiceleading_unfeasible_left_hand_rate"])
    )

    return {
        "harmonic_score":    harmonic,
        "melodic_score":     melodic,
        "rhythmic_score":    rhythmic,
        "voiceleading_score": voiceleading,
    }


# Section weights — predetermined based on musicality
SECTION_WEIGHTS = {
    "harmonic_score":     0.15,  
    "voiceleading_score": 0.25,  
    "melodic_score":      0.25,  
    "rhythmic_score":     0.35,  
}

def compute_final_score(section_scores):
    '''
    Compute final score with section weights 
    
    :param section_scores: scores per section
    :type section_scores: dict
    :return: combined score with applied section weights
    :rtype: float
    '''
    return sum(
        SECTION_WEIGHTS[k] * section_scores[k]
        for k in SECTION_WEIGHTS
    )



def print_report(results, score_path, piano_path):
    '''
    Print report of the eval data and weighted scores 
    
    :param results: Dictionary of resukts from eval functions
    :type results: dict
    :param score_path: path to score
    :type score_path: str
    :param piano_path: path to piano reduction
    :type piano_path: str
    '''
    print()
    print("=" * 62)
    print("  FULL REDUCTION EVALUATION REPORT")
    print(f"  Score : {score_path}")
    print(f"  Piano : {piano_path}")
    print("=" * 62)

    sections = {
        "VERTICAL HARMONIES": [k for k in results if k.startswith("harmonic_")],
        "MELODIC SHAPE":      [k for k in results if k.startswith("melodic_")],
        "RHYTHMIC QUALITY":   [k for k in results if k.startswith("rhythmic_")],
        "VOICE LEADING":      [k for k in results if k.startswith("voiceleading_")],
    }

    for section, keys in sections.items():
        print(f"\n  ── {section} ──")
        for k in keys:
            v = results[k]
            if isinstance(v, float):
                print(f"    {k:<54} {v:.4f}")
            else:
                print(f"    {k:<54} {v}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_reduction.py <score.xml> <piano.xml>")
        sys.exit(1)

    score_path = sys.argv[1]
    piano_path = sys.argv[2]

    results = evaluate_all(score_path=score_path, piano_path=piano_path)
    print_report(results, score_path, piano_path)
    
    section_scores = compute_section_scores(results)
    final_score = compute_final_score(section_scores)

    print(section_scores)
    print("\nFinal score: " + str(final_score))


if __name__ == "__main__":
    main()
