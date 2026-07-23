"""
mdp_reducer.py
==============
Markov Decision Process approach to piano reduction that considers
sequential dependencies and voice leading context.
"""

import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
import pandas as pd
import joblib
import os


def _q_action_table():
    """Pickle-safe factory for per-state Q action values."""
    return defaultdict(float)

class ReductionMDP:
    """
    MDP-based note selector that makes sequential decisions about
    which notes to keep, considering context and voice leading.
    
    State: Recent kept notes + current harmonic/rhythmic context
    Action: KEEP or REMOVE for each candidate note
    Reward: Based on musical quality metrics
    """
    
    def __init__(
        self,
        context_window: int = 4,      # how many timepoints to remember
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,          # exploration rate
    ):
        self.context_window = context_window
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: state_hash -> {action: q_value}
        # action is tuple: (note_index, KEEP=1/REMOVE=0)
        self.q_table = defaultdict(_q_action_table)
        
        # Track recent history for state construction
        self.recent_kept_notes = deque(maxlen=context_window)
        
    def _extract_state_features(
        self,
        candidate_notes: List[Dict],
        kept_history: List[List[Dict]],  # recent timepoints of kept notes
        current_offset: float,
    ) -> Tuple:
        """
        Extract state representation from current context.
        
        State components:
        - Last kept bass pitch (or -1 if none)
        - Last kept soprano pitch (or -1 if none)  
        - Number of notes currently kept at this timepoint
        - Beat strength bucket (0=weak, 1=medium, 2=strong)
        - Pitch range of candidates (discretized)
        """
        # Get last kept notes from history
        last_bass = -1
        last_soprano = -1
        if kept_history:
            last_timepoint = kept_history[-1]
            if last_timepoint:
                pitches = [n['midi_pitch'] for n in last_timepoint]
                last_bass = min(pitches)
                last_soprano = max(pitches)
        
        # Current timepoint info
        current_kept = sum(1 for n in candidate_notes if n.get('_kept', False))
        
        # Beat strength discretization
        beat_strengths = [n['beat_strength'] for n in candidate_notes]
        avg_beat = np.mean(beat_strengths) if beat_strengths else 0
        beat_bucket = 2 if avg_beat >= 0.5 else (1 if avg_beat >= 0.25 else 0)
        
        # Pitch range bucket
        pitches = [n['midi_pitch'] for n in candidate_notes]
        pitch_range = max(pitches) - min(pitches) if pitches else 0
        range_bucket = min(pitch_range // 6, 4)  # 0-4 buckets
        
        return (last_bass, last_soprano, current_kept, beat_bucket, range_bucket)
    
    def _compute_reward(
        self,
        kept_notes: List[Dict],
        candidate_notes: List[Dict],
        history: List[List[Dict]],
    ) -> float:
        """
        Compute immediate reward for the current decision.
        
        Rewards:
        + Harmonic completeness (bass + melody preserved)
        + Smooth voice leading (small intervals from previous)
        + Playability (notes within hand span)
        - Penalties for too sparse/dense texture
        """
        reward = 0.0
        
        if not kept_notes:
            return -10.0  # penalty for removing everything
        
        pitches = [n['midi_pitch'] for n in kept_notes]
        
        # 1. Voice leading reward
        if history and history[-1]:
            prev_pitches = [n['midi_pitch'] for n in history[-1]]
            
            # Soprano voice leading (smooth = good)
            if max(pitches) and max(prev_pitches):
                soprano_leap = abs(max(pitches) - max(prev_pitches))
                reward += 2.0 if soprano_leap <= 2 else (1.0 if soprano_leap <= 5 else -1.0)
            
            # Bass voice leading
            if min(pitches) and min(prev_pitches):
                bass_leap = abs(min(pitches) - min(prev_pitches))
                reward += 1.5 if bass_leap <= 3 else (0.5 if bass_leap <= 7 else -1.0)
        
        # 2. Harmonic coverage reward
        # Prefer keeping bass and soprano from candidates
        candidate_pitches = [n['midi_pitch'] for n in candidate_notes]
        has_bass = min(pitches) == min(candidate_pitches)
        has_soprano = max(pitches) == max(candidate_pitches)
        
        reward += 3.0 if has_bass else -2.0
        reward += 3.0 if has_soprano else -2.0
        
        # 3. Texture density reward (prefer 2-4 notes per timepoint)
        n_notes = len(kept_notes)
        if 2 <= n_notes <= 4:
            reward += 2.0
        elif n_notes == 1:
            reward -= 1.0
        elif n_notes > 5:
            reward -= 2.0
        
        # 4. Playability reward (check hand span)
        rh_pitches = [p for p in pitches if p >= 60]
        lh_pitches = [p for p in pitches if p < 60]
        
        if rh_pitches and (max(rh_pitches) - min(rh_pitches)) > 12:
            reward -= 3.0  # RH span > octave = bad
        if lh_pitches and (max(lh_pitches) - min(lh_pitches)) > 12:
            reward -= 3.0  # LH span > octave = bad
        
        # 5. Priority-based reward (prefer high-priority instruments)
        priority_sum = sum(n.get('instrument_priority', 0.5) for n in kept_notes)
        reward += priority_sum * 0.5

        # 6. RF guidance for the hybrid reducer. RF remains a suggestion;
        # the sequential MDP can override it when voice leading/playability wins.
        reward += sum(float(n.get('rf_keep_probability', 0.5)) * 2.0 for n in kept_notes)
        for note in candidate_notes:
            if note not in kept_notes:
                rf_prob = float(note.get('rf_keep_probability', 0.5))
                if rf_prob > 0.7:
                    reward -= (rf_prob - 0.7) * 3.0
        
        return reward
    
    def select_action(
        self,
        state: Tuple,
        candidate_notes: List[Dict],
        explore: bool = True,
    ) -> List[int]:
        """
        Select which notes to KEEP using epsilon-greedy policy.
        
        Returns: List of indices into candidate_notes that should be kept.
        """
        if explore and np.random.random() < self.epsilon:
            # Exploration: random selection (but always keep 1-4 notes)
            n_keep = np.random.randint(1, min(5, len(candidate_notes) + 1))
            return np.random.choice(len(candidate_notes), n_keep, replace=False).tolist()
        
        # Exploitation: greedy selection based on Q-values
        # Try different combinations and pick best Q-value
        state_hash = hash(state)
        
        best_combo = None
        best_q = float('-inf')
        
        # Heuristic: try combinations of 1-4 notes
        from itertools import combinations
        
        for n_keep in range(1, min(5, len(candidate_notes) + 1)):
            for combo in combinations(range(len(candidate_notes)), n_keep):
                action_hash = hash(tuple(sorted(combo)))
                q_val = self.q_table[state_hash][action_hash]
                
                if q_val > best_q:
                    best_q = q_val
                    best_combo = list(combo)
        
        # If no Q-values yet, use heuristic (keep bass + soprano + high priority)
        if best_combo is None:
            priorities = [
                (i, n.get('instrument_priority', 0.5)) 
                for i, n in enumerate(candidate_notes)
            ]
            priorities.sort(key=lambda x: x[1], reverse=True)
            best_combo = [i for i, _ in priorities[:min(3, len(priorities))]]
        
        return best_combo
    
    def update_q_value(
        self,
        state: Tuple,
        action: List[int],
        reward: float,
        next_state: Tuple,
    ):
        """Q-learning update rule."""
        state_hash = hash(state)
        action_hash = hash(tuple(sorted(action)))
        next_state_hash = hash(next_state)
        
        # Get max Q-value for next state
        next_q_values = self.q_table[next_state_hash].values()
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        old_q = self.q_table[state_hash][action_hash]
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state_hash][action_hash] = new_q
    
    def process_phrase(
        self,
        notes_by_timepoint: Dict[Tuple, List[Dict]],
        train: bool = True,
    ) -> List[Dict]:
        """
        Process a phrase (sequence of timepoints) and select notes.
        
        Args:
            notes_by_timepoint: Dict mapping (measure, offset) -> list of notes
            train: If True, update Q-values during processing
        
        Returns:
            List of all kept notes
        """
        kept_history = []
        all_kept = []
        
        # Sort timepoints chronologically
        timepoints = sorted(notes_by_timepoint.keys())
        
        for i, tp in enumerate(timepoints):
            candidates = notes_by_timepoint[tp]
            
            # Extract current state
            state = self._extract_state_features(
                candidates, kept_history, tp[1]
            )
            
            # Select action (which notes to keep)
            kept_indices = self.select_action(state, candidates, explore=train)
            kept_notes = [candidates[i] for i in kept_indices]
            
            # Compute reward
            reward = self._compute_reward(kept_notes, candidates, kept_history)
            
            # Get next state (if not last timepoint)
            next_state = None
            if i < len(timepoints) - 1:
                next_tp = timepoints[i + 1]
                next_candidates = notes_by_timepoint[next_tp]
                next_state = self._extract_state_features(
                    next_candidates, kept_history + [kept_notes], next_tp[1]
                )
            else:
                # Terminal state
                next_state = (-1, -1, -1, -1, -1)
            
            # Update Q-values if training
            if train:
                self.update_q_value(state, kept_indices, reward, next_state)
            
            # Update history
            kept_history.append(kept_notes)
            if len(kept_history) > self.context_window:
                kept_history.pop(0)
            
            all_kept.extend(kept_notes)
        
        return all_kept

    def save(self, path: str) -> None:
        """Serialize the MDP reducer to a joblib file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"  MDP model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ReductionMDP":
        """Load a serialized MDP reducer."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected ReductionMDP in {path}, got {type(model)!r}")
        print(f"  MDP model loaded ← {path}")
        return model

    def __getstate__(self):
        """Make the nested Q-table pickle-safe."""
        state = self.__dict__.copy()
        state["q_table"] = {
            state_hash: dict(action_values)
            for state_hash, action_values in self.q_table.items()
        }
        return state

    def __setstate__(self, state):
        """Restore the nested Q-table with its default factories."""
        q_table = defaultdict(_q_action_table)
        for state_hash, action_values in state.get("q_table", {}).items():
            q_table[state_hash].update(action_values)
        state["q_table"] = q_table
        self.__dict__.update(state)


def train_mdp_reducer(
    orchestral_dir: str,
    piano_dir: str,
    n_episodes: int = 10,
    phrase_bars: int = 8,
) -> ReductionMDP:
    """
    Train MDP reducer on paired orchestral-piano data.
    
    Each episode processes all pairs, updating Q-values based on
    how well the MDP's choices match the ground truth reductions.
    """
    from data_pipeline import load_pair
    from pathlib import Path
    
    mdp = ReductionMDP()
    
    orch_path = Path(orchestral_dir)
    piano_path = Path(piano_dir)
    
    pairs = []
    for orch_file in sorted(list(orch_path.glob("*.musicxml")) + list(orch_path.glob("*.xml"))):
        piano_file = piano_path / orch_file.name
        if piano_file.exists():
            pairs.append((str(orch_file), str(piano_file)))
    
    print(f"Training on {len(pairs)} pairs for {n_episodes} episodes...")

    cached_pairs = []
    for orch_file, piano_file in pairs:
        orch_notes, piano_notes = load_pair(orch_file, piano_file)
        cached_pairs.append((orch_notes, piano_notes))
    
    for episode in range(n_episodes):
        total_reward = 0
        
        for orch_notes, piano_notes in cached_pairs:
            # Group by timepoint
            notes_by_tp = defaultdict(list)
            for n in orch_notes:
                tp = (n['measure_number'], round(n['offset'], 6))
                notes_by_tp[tp].append(n)
            
            # Process phrase by phrase
            phrase_start = min(n['measure_number'] for n in orch_notes)
            phrase_end = max(n['measure_number'] for n in orch_notes)
            
            current = phrase_start
            while current <= phrase_end:
                phrase_tps = {
                    tp: notes for tp, notes in notes_by_tp.items()
                    if current <= tp[0] < current + phrase_bars
                }
                
                if phrase_tps:
                    kept = mdp.process_phrase(phrase_tps, train=True)
                    # Accumulate reward (you could also compare to ground truth here)
                
                current += phrase_bars
        
        # Decay exploration over time
        mdp.epsilon = max(0.01, mdp.epsilon * 0.95)
        
        print(f"Episode {episode + 1}/{n_episodes} complete, ε={mdp.epsilon:.3f}")
    
    return mdp

def group_by_timepoint(notes: List[Dict]) -> Dict[Tuple[int, float], List[Dict]]:
    """Group note dictionaries by measure and absolute offset."""
    grouped = defaultdict(list)
    for n in notes:
        grouped[(int(n.get("measure_number", 0)), round(float(n.get("offset", 0.0)), 6))].append(n)
    return grouped


def hybrid_reduction(phrase_notes, rf_model, mdp_model):
    """
    Use RF to get keep probabilities, MDP to make final decisions.
    Add RF probability as feature in MDP reward function.
    """
    if not phrase_notes:
        return []

    from data_pipeline import extract_features

    df = extract_features(phrase_notes, all_notes=phrase_notes)
    if hasattr(rf_model, "predict_proba") and not hasattr(rf_model, "_feature_matrix"):
        keep_probs = rf_model.predict_proba(df.to_numpy())[:, 1]
    else:
        X = rf_model._feature_matrix(df) if hasattr(rf_model, "_feature_matrix") else df.to_numpy()
        keep_probs = rf_model.model.predict_proba(X)[:, 1]

    annotated_notes = []
    for note, prob in zip(phrase_notes, keep_probs):
        enriched = dict(note)
        enriched["rf_keep_probability"] = float(prob)
        annotated_notes.append(enriched)

    notes_by_tp = group_by_timepoint(annotated_notes)
    mdp_kept = mdp_model.process_phrase(notes_by_tp, train=False)
    bass_restored = _restore_structural_bass(
        mdp_kept,
        notes_by_tp,
        min_rf_probability=0.15,
        restore_any_strong_bass=True,
        restore_very_low_bass=True,
    )
    return _restore_right_hand_details(bass_restored, notes_by_tp)


def _note_identity(n: Dict) -> Tuple:
    """Stable identity for note dictionaries during post-selection passes."""
    return (
        n.get("measure_number"),
        round(float(n.get("offset", 0.0)), 6),
        n.get("midi_pitch"),
        n.get("part_name"),
    )


def _restore_structural_bass(
    kept_notes: List[Dict],
    notes_by_tp: Dict[Tuple[int, float], List[Dict]],
    bass_ceiling: int = 60,
    min_rf_probability: float = 0.12,
    high_confidence_probability: float = 0.25,
    min_beat_strength: float = 0.25,
    restore_any_strong_bass: bool = False,
    restore_very_low_bass: bool = False,
) -> List[Dict]:
    """
    Preserve local bass anchors after MDP compression.

    The MDP is useful for sequential playability, but an aggressive policy can
    over-favor melody/middle texture. This post-pass keeps the lowest candidate
    at a timepoint when it is a plausible left-hand/bass note and no selected
    note already covers that bass region.
    """
    kept_by_identity = {_note_identity(n) for n in kept_notes}
    restored = list(kept_notes)

    for tp, candidates in notes_by_tp.items():
        bass = min(candidates, key=lambda n: (n.get("midi_pitch", 127), -float(n.get("rf_keep_probability", 0.0))))
        bass_pitch = int(bass.get("midi_pitch", 127))
        if bass_pitch >= bass_ceiling:
            continue

        selected_here = [
            n for n in kept_notes
            if (int(n.get("measure_number", 0)), round(float(n.get("offset", 0.0)), 6)) == tp
        ]
        has_bass_cover = any(int(n.get("midi_pitch", 127)) <= bass_pitch + 7 for n in selected_here)
        if has_bass_cover:
            continue

        rf_prob = float(bass.get("rf_keep_probability", 0.5))
        beat_strength = float(bass.get("beat_strength", 0.0))
        is_supported_anchor = beat_strength >= min_beat_strength and rf_prob >= min_rf_probability
        is_strong_bass = restore_any_strong_bass and beat_strength >= min_beat_strength
        is_very_low_bass = restore_very_low_bass and bass_pitch < 48
        if is_supported_anchor or rf_prob >= high_confidence_probability or is_strong_bass or is_very_low_bass:
            key = _note_identity(bass)
            if key not in kept_by_identity:
                restored.append(bass)
                kept_by_identity.add(key)

    return sorted(restored, key=lambda n: (n.get("offset", 0.0), n.get("midi_pitch", 0)))


def _restore_right_hand_details(
    kept_notes: List[Dict],
    notes_by_tp: Dict[Tuple[int, float], List[Dict]],
    min_pitch: int = 60,
    max_rh_notes_per_timepoint: int = 5,
    max_extra_per_timepoint: int = 2,
) -> List[Dict]:
    """
    Add back a small number of salient RH details after MDP compression.

    This is deliberately targeted: it favors local melody/top-neighbor notes,
    short rhythmic details, and notes the RF gave at least some support, while
    capping additions per timepoint for playability.
    """
    kept_by_identity = {_note_identity(n) for n in kept_notes}
    restored = list(kept_notes)
    selected_by_tp: Dict[Tuple[int, float], List[Dict]] = defaultdict(list)
    for n in kept_notes:
        selected_by_tp[(int(n.get("measure_number", 0)), round(float(n.get("offset", 0.0)), 6))].append(n)

    for tp, candidates in notes_by_tp.items():
        selected_here = selected_by_tp.get(tp, [])
        rh_selected = [n for n in selected_here if int(n.get("midi_pitch", 0)) >= min_pitch]
        selected_pitches = {int(n.get("midi_pitch", 0)) for n in selected_here}
        if len(rh_selected) >= max_rh_notes_per_timepoint:
            continue

        highest = max(int(n.get("midi_pitch", 0)) for n in candidates)
        selected_highest = max((int(n.get("midi_pitch", 0)) for n in rh_selected), default=-1)
        additions = 0

        def detail_rank(n: Dict) -> Tuple:
            pitch = int(n.get("midi_pitch", 0))
            rf_prob = float(n.get("rf_keep_probability", 0.0))
            duration = float(n.get("duration_ql", 1.0))
            beat = float(n.get("beat_strength", 0.0))
            is_local_melody = int(pitch == highest)
            is_upper_neighbor = int(pitch >= highest - 5)
            is_fast_detail = int(duration <= 0.5)
            return (
                is_local_melody,
                is_upper_neighbor,
                is_fast_detail,
                rf_prob,
                beat,
                pitch,
            )

        ranked = sorted(
            [
                n for n in candidates
                if int(n.get("midi_pitch", 0)) >= min_pitch
                and int(n.get("midi_pitch", 0)) not in selected_pitches
                and _note_identity(n) not in kept_by_identity
            ],
            key=detail_rank,
            reverse=True,
        )

        for cand in ranked:
            if additions >= max_extra_per_timepoint or len(rh_selected) >= max_rh_notes_per_timepoint:
                break

            pitch = int(cand.get("midi_pitch", 0))
            rf_prob = float(cand.get("rf_keep_probability", 0.0))
            duration = float(cand.get("duration_ql", 1.0))
            beat = float(cand.get("beat_strength", 0.0))
            is_local_melody = pitch == highest
            is_upper_neighbor = pitch >= highest - 5
            is_fast_detail = duration <= 0.5
            melody_missing = is_local_melody and selected_highest < highest - 2

            restore = (
                (melody_missing and (rf_prob >= 0.04 or beat >= 0.25 or is_fast_detail))
                or (is_upper_neighbor and is_fast_detail and rf_prob >= 0.05)
                or (is_upper_neighbor and beat >= 0.25 and rf_prob >= 0.08)
                or (rf_prob >= 0.18 and (is_fast_detail or beat >= 0.25))
            )
            if not restore:
                continue

            projected = [int(n.get("midi_pitch", 0)) for n in rh_selected] + [pitch]
            if len(projected) > 1 and max(projected) - min(projected) > 16:
                continue

            restored.append(cand)
            kept_by_identity.add(_note_identity(cand))
            rh_selected.append(cand)
            selected_pitches.add(pitch)
            selected_highest = max(selected_highest, pitch)
            additions += 1

    return sorted(restored, key=lambda n: (n.get("offset", 0.0), n.get("midi_pitch", 0)))


# Backwards-compatible name for earlier experiments.
hybrid_mdp_rf_reduction = hybrid_reduction
