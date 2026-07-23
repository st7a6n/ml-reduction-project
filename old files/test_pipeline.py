"""
test_pipeline.py
================
Smoke tests for the enhanced reduction pipeline.

Run:
    python test_pipeline.py
"""

from __future__ import annotations

import numpy as np
from music21 import instrument, note, stream

from data_pipeline import (
    FEATURE_COLUMNS,
    extract_features,
    extract_notes_from_score,
)
from kern_loader import DEFAULT_DATA_DIR, inspect_kern_file
from mdp_reducer import ReductionMDP, hybrid_reduction


class _DummyRFInner:
    n_features_in_ = len(FEATURE_COLUMNS)

    def predict_proba(self, X):
        pitch = X[:, FEATURE_COLUMNS.index("midi_pitch")]
        probs = np.clip((pitch - 48) / 36, 0.1, 0.9)
        return np.column_stack([1.0 - probs, probs])


class _DummyRF:
    def __init__(self):
        self.model = _DummyRFInner()

    def _feature_matrix(self, df):
        return df[FEATURE_COLUMNS].to_numpy()


def _synthetic_score():
    score = stream.Score()
    violin = stream.Part()
    violin.partName = "Violin I"
    violin.insert(0, instrument.Violin())
    for offset, pitch_name in [(0.0, "C5"), (1.0, "E5"), (2.0, "G4")]:
        n = note.Note(pitch_name, quarterLength=1.0)
        violin.insert(offset, n)

    drums = stream.Part()
    drums.partName = "Bass Drum"
    drums.insert(0, instrument.Woodblock())
    for offset in [0.0, 2.0]:
        u = note.Unpitched(quarterLength=0.25)
        drums.insert(offset, u)

    score.insert(0, violin)
    score.insert(0, drums)
    return score


def main():
    score = _synthetic_score()
    notes = extract_notes_from_score(score)
    assert notes, "synthetic score should produce pitched notes"

    df = extract_features(notes, all_notes=notes)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    assert not missing, f"missing feature columns: {missing}"
    assert "has_percussion_support" in df.columns

    kept = hybrid_reduction(notes, _DummyRF(), ReductionMDP(epsilon=0.0))
    assert kept, "hybrid reducer should keep at least one note"

    if DEFAULT_DATA_DIR.exists():
        stats = inspect_kern_file(str(DEFAULT_DATA_DIR / "01"))
        assert stats["part_files"] > 0, "Beethoven movement 1 should have part files"

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
