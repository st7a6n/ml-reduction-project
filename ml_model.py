"""
ml_model.py
===========
Trains and evaluates a Random Forest on the music-theoretic features
extracted by data_pipeline.py to predict KEEP (1) / REMOVE (0) for each
orchestral note.

Both are trained with class_weight='balanced' to handle KEEP/REMOVE imbalance.

Requirements:
    pip install music21 scikit-learn pandas numpy
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore", category=UserWarning, module="music21")

from data_pipeline import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    extract_features,
    load_pair,
    match_notes,
)


ModelType = Literal["random_forest", "gradient_boosting"]

_RANDOM_FOREST_PARAMS: Dict = dict(
    n_estimators=300,
    max_depth=None,          # grow full trees; regularised by min_samples_leaf
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

_GRADIENT_BOOSTING_PARAMS: Dict = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)


def _make_model(model_type: ModelType):
    if model_type == "random_forest":
        return RandomForestClassifier(**_RANDOM_FOREST_PARAMS)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**_GRADIENT_BOOSTING_PARAMS)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. "
                         "Choose 'random_forest' or 'gradient_boosting'.")


def _pair_group(pair_id: str) -> str:
    """
    Extract the piece-group prefix from a pair_id.

    e.g. "pair3_m0201-0208" → "pair3"
    Used to define LOGO cross-validation groups.
    """
    return pair_id.split("_")[0]


def _composer_group(pair_id: str) -> str:
    """Best-effort composer grouping from filenames like mozart_pair1_..."""
    first = pair_id.split("_")[0].lower()
    known = {
        "bach", "beethoven", "brahms", "debussy", "dvorak", "haydn",
        "mahler", "mozart", "rachmaninoff", "ravel", "schubert",
        "schumann", "stravinsky", "tchaikovsky", "wagner",
    }
    return first if first in known else _pair_group(pair_id)


def load_dataset_from_folder(
    orchestral_dir: str,
    piano_dir: str,
    offset_tolerance: float = 0.25,
    max_pairs: Optional[int] = None,
    verbose: bool = True,
    phrase_bars: int = 8,
    overlap: int = 0,
) -> pd.DataFrame:
    """
    Build a training DataFrame from all matched pairs in a directory.

    Returns a DataFrame with FEATURE_COLUMNS + 'label' + 'pair_id' columns.
    Skips pairs with errors (logged as warnings).
    """
    orch_path  = Path(orchestral_dir)
    piano_path = Path(piano_dir)

    orch_files = sorted(
        list(orch_path.glob("*.musicxml")) + list(orch_path.glob("*.xml"))
    )
    if not orch_files:
        raise ValueError(f"No MusicXML files in {orchestral_dir}")

    dfs      = []
    n_ok     = 0
    n_errors = 0

    for orch_file in orch_files:
        if max_pairs is not None and n_ok >= max_pairs:
            break
        piano_file = piano_path / orch_file.name
        if not piano_file.exists():
            if verbose:
                print(f"  SKIP (no piano match): {orch_file.name}")
            continue
        try:
            orch_notes, piano_notes = load_pair(str(orch_file), str(piano_file))
            if not orch_notes or not piano_notes:
                continue
            labels = match_notes(orch_notes, piano_notes, offset_tolerance)
            df = _extract_phrase_training_rows(
                orch_notes,
                labels,
                pair_id=orch_file.stem,
                phrase_bars=phrase_bars,
                overlap=overlap,
            )
            dfs.append(df)
            n_ok += 1
        except Exception as e:
            if verbose:
                print(f"  ERROR {orch_file.name}: {e}")
            n_errors += 1

    if not dfs:
        raise ValueError("No pairs loaded successfully.")

    combined = pd.concat(dfs, ignore_index=True)
    if verbose:
        keep_n   = combined[LABEL_COLUMN].sum()
        total_n  = len(combined)
        print(f"\n  Loaded {n_ok} pairs ({n_errors} errors) — "
              f"{total_n:,} notes, "
              f"{keep_n/total_n:.1%} KEEP")
    return combined


def _extract_phrase_training_rows(
    orch_notes: List[Dict],
    labels: List[int],
    pair_id: str,
    phrase_bars: int = 8,
    overlap: int = 0,
) -> pd.DataFrame:
    """Create phrase-aware training rows, with optional bar overlap."""
    if not orch_notes:
        return pd.DataFrame()

    phrase_bars = max(1, int(phrase_bars))
    overlap = max(0, min(int(overlap), phrase_bars - 1))
    step = phrase_bars - overlap
    measures = sorted({int(n["measure_number"]) for n in orch_notes})
    piece_min, piece_max = measures[0], measures[-1]

    rows = []
    phrase_position = 0
    start = piece_min
    indexed = list(enumerate(orch_notes))
    while start <= piece_max:
        end = min(start + phrase_bars - 1, piece_max)
        selected = [(i, n) for i, n in indexed if start <= int(n["measure_number"]) <= end]
        if selected:
            idxs, phrase_notes = zip(*selected)
            df = extract_features(
                list(phrase_notes),
                all_notes=orch_notes,
                phrase_start_measure=start,
                phrase_end_measure=end,
                phrase_position=phrase_position,
                piece_min_measure=piece_min,
                piece_max_measure=piece_max,
            )
            df[LABEL_COLUMN] = [labels[i] for i in idxs]
            df["pair_id"] = pair_id
            rows.append(df)
            phrase_position += 1
        start += step

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


class NoteClassifier:
    """
    Random Forest (or Gradient Boosting) note-level classifier for
    orchestral-to-piano reduction.

    Parameters
    ----------
    model_type : "random_forest" | "gradient_boosting"
    """

    def __init__(self, model_type: ModelType = "random_forest") -> None:
        self.model_type   = model_type
        self.model        = _make_model(model_type)
        self._fitted      = False
        self.feature_importances_: Optional[pd.Series] = None
        self.feature_columns = list(FEATURE_COLUMNS)

    def fit(
        self,
        orchestral_dir: str,
        piano_dir: str,
        offset_tolerance: float = 0.25,
        max_pairs: Optional[int] = None,
        phrase_bars: int = 8,
        overlap: int = 0,
    ) -> "NoteClassifier":
        """
        Train on all pairs in orchestral_dir / piano_dir.

        Args:
            orchestral_dir:   Directory of orchestral MusicXML files.
            piano_dir:        Directory of piano reduction MusicXML files.
            offset_tolerance: Note-matching tolerance (quarter-lengths).
            max_pairs:        Cap number of pairs (for quick tests).

        Returns:
            self (for chaining)
        """
        print(f"\n{'='*60}")
        print(f"TRAINING  [{self.model_type}]")
        print(f"{'='*60}")

        df = load_dataset_from_folder(
            orchestral_dir, piano_dir, offset_tolerance, max_pairs,
            phrase_bars=phrase_bars, overlap=overlap,
        )
        X = self._feature_matrix(df)
        y = df[LABEL_COLUMN].to_numpy()

        print(f"  Fitting on {len(X):,} notes …")
        self.model.fit(X, y)
        self._fitted = True
        self._record_importances()
        print(f"  Done.")
        return self

    def fit_from_df(self, df: pd.DataFrame) -> "NoteClassifier":
        """Fit directly from a pre-built feature DataFrame."""
        X = self._feature_matrix(df)
        y = df[LABEL_COLUMN].to_numpy()
        self.model.fit(X, y)
        self._fitted = True
        self._record_importances()
        return self

    def _record_importances(self) -> None:
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_,
                index=self.feature_columns[:len(self.model.feature_importances_)],
            ).sort_values(ascending=False)

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return a model-ready feature matrix while preserving compatibility
        with older saved reducers trained on the original 18 columns.
        """
        expected = getattr(self.model, "n_features_in_", None)
        columns = list(getattr(self, "feature_columns", FEATURE_COLUMNS))
        if expected is not None:
            columns = columns[:expected]

        missing = [c for c in columns if c not in df.columns]
        if missing:
            df = df.copy()
            for col in missing:
                df[col] = 0
        return df[columns].to_numpy()

    def predict(self, orch_notes: List[Dict]) -> List[int]:
        """
        Predict KEEP (1) / REMOVE (0) for a list of orchestral note dicts.

        Args:
            orch_notes: Output of data_pipeline.extract_notes_from_score().

        Returns:
            List[int] of same length as orch_notes.
        """
        self._check_fitted()
        df = extract_features(orch_notes, all_notes=orch_notes)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> List[int]:
        """Predict from a pre-built feature DataFrame."""
        self._check_fitted()
        X = self._feature_matrix(df)
        return self.model.predict(X).tolist()

    def predict_proba(self, orch_notes: List[Dict]) -> np.ndarray:
        """
        Return class probabilities (shape N×2, column 1 = P(KEEP)).

        Only available for models that support predict_proba (both RF and GB do).
        """
        self._check_fitted()
        df = extract_features(orch_notes, all_notes=orch_notes)
        X  = self._feature_matrix(df)
        return self.model.predict_proba(X)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def cross_validate(
        self,
        orchestral_dir: str,
        piano_dir: str,
        offset_tolerance: float = 0.25,
        max_pairs: Optional[int] = None,
        verbose: bool = True,
        phrase_bars: int = 8,
        overlap: int = 0,
        group_by: Literal["piece", "composer"] = "piece",
    ) -> pd.DataFrame:
        """
        Leave-One-Group-Out cross-validation.

        Groups = piece prefixes (pair1 / pair2 / pair3).  Each fold trains
        on all pairs from N-1 pieces and tests on the held-out piece, so
        the model is never evaluated on its own training piece.

        Takes in directories of musicxml files

        Returns:
            DataFrame with one row per fold:
            fold, train_size, test_size, accuracy, precision, recall, f1.
        """
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION  [{self.model_type}]  (Leave-One-Group-Out)")
        print(f"{'='*60}")

        df = load_dataset_from_folder(
            orchestral_dir, piano_dir, offset_tolerance, max_pairs,
            phrase_bars=phrase_bars, overlap=overlap,
        )
        X      = df[FEATURE_COLUMNS].to_numpy()
        y      = df[LABEL_COLUMN].to_numpy()
        group_fn = _composer_group if group_by == "composer" else _pair_group
        groups = df["pair_id"].apply(group_fn).to_numpy()

        logo   = LeaveOneGroupOut()
        rows   = []

        for fold_i, (train_idx, test_idx) in enumerate(
            logo.split(X, y, groups), start=1
        ):
            held_out = groups[test_idx[0]]
            model    = _make_model(self.model_type)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])

            metrics = {
                "fold":        fold_i,
                "held_out":    held_out,
                "train_size":  len(train_idx),
                "test_size":   len(test_idx),
                "accuracy":    accuracy_score(y[test_idx], y_pred),
                "precision":   precision_score(y[test_idx], y_pred, zero_division=0),
                "recall":      recall_score(y[test_idx], y_pred, zero_division=0),
                "f1":          f1_score(y[test_idx], y_pred, zero_division=0),
            }
            rows.append(metrics)

            if verbose:
                print(f"  Fold {fold_i}  held-out={held_out:8s}  "
                      f"train={len(train_idx):,}  test={len(test_idx):,}  "
                      f"acc={metrics['accuracy']:.3f}  "
                      f"prec={metrics['precision']:.3f}  "
                      f"rec={metrics['recall']:.3f}  "
                      f"f1={metrics['f1']:.3f}")

        results = pd.DataFrame(rows)

        print(f"\n  {'':30s}  {'acc':>6}  {'prec':>6}  {'rec':>6}  {'f1':>6}")
        print(f"  {'MEAN':30s}  "
              f"{results['accuracy'].mean():6.3f}  "
              f"{results['precision'].mean():6.3f}  "
              f"{results['recall'].mean():6.3f}  "
              f"{results['f1'].mean():6.3f}")
        print(f"  {'STD':30s}  "
              f"{results['accuracy'].std():6.3f}  "
              f"{results['precision'].std():6.3f}  "
              f"{results['recall'].std():6.3f}  "
              f"{results['f1'].std():6.3f}")
        print(f"{'='*60}\n")

        return results

    """Evaluate a fitted model on every pair in a directory. THis requires the model to have been fitted beforehand (via .fit()). 
    Returns a dataFrame with per-pair metrics (pair_id, accuracy, precision, recall, f1, n_notes, keep_rate_true, keep_rate_pred)."""
    def evaluate_on_folder(
        self,
        orchestral_dir: str,
        piano_dir: str,
        offset_tolerance: float = 0.25,
        max_pairs: Optional[int] = None,
    ) -> pd.DataFrame:

        self._check_fitted()

        orch_path  = Path(orchestral_dir)
        piano_path = Path(piano_dir)

        rows     = []
        n_errors = 0

        for orch_file in sorted(
            list(orch_path.glob("*.musicxml")) + list(orch_path.glob("*.xml"))
        ):
            if max_pairs is not None and len(rows) >= max_pairs:
                break
            piano_file = piano_path / orch_file.name
            if not piano_file.exists():
                continue
            try:
                orch_notes, piano_notes = load_pair(
                    str(orch_file), str(piano_file)
                )
                y_true = np.array(
                    match_notes(orch_notes, piano_notes, offset_tolerance)
                )
                y_pred = np.array(self.predict(orch_notes))

                rows.append({
                    "pair_id":        orch_file.stem,
                    "accuracy":       accuracy_score(y_true, y_pred),
                    "precision":      precision_score(y_true, y_pred, zero_division=0),
                    "recall":         recall_score(y_true, y_pred, zero_division=0),
                    "f1":             f1_score(y_true, y_pred, zero_division=0),
                    "n_notes":        len(y_true),
                    "keep_rate_true": float(y_true.mean()),
                    "keep_rate_pred": float(y_pred.mean()),
                })
            except Exception as e:
                print(f"  ERROR {orch_file.name}: {e}")
                n_errors += 1

        results = pd.DataFrame(rows)
        if len(results):
            print(f"\n{'='*60}")
            print(f"ML MODEL [{self.model_type}] — EVALUATION")
            print(f"  Pairs: {len(results)}  Errors: {n_errors}")
            print(f"  Mean Accuracy  : {results['accuracy'].mean():.4f}")
            print(f"  Mean Precision : {results['precision'].mean():.4f}")
            print(f"  Mean Recall    : {results['recall'].mean():.4f}")
            print(f"  Mean F1        : {results['f1'].mean():.4f}")
            print(f"{'='*60}\n")
        return results

    def print_feature_importance(self, top_n: int = 18) -> None:
        """Print a ranked feature importance table."""
        self._check_fitted()
        if self.feature_importances_ is None:
            print("Feature importances not available for this model type.")
            return
        print(f"\nFeature Importance  [{self.model_type}]")
        print(f"{'Feature':25s}  {'Importance':>10}")
        print("-" * 38)
        """USE UNICODE BLOCKS AS BAR PLACEHOLDERS IT LOOKS WACK BUT IT WORKS"""
        for feat, imp in self.feature_importances_.head(top_n).items():
            bar = "█" * int(imp * 60)
            print(f"  {feat:23s}  {imp:8.4f}  {bar}")

    def save(self, path: str) -> None:
        """Serialise the fitted model to a .joblib file."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
        }, path)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NoteClassifier":
        """Load a previously saved NoteClassifier."""
        payload     = joblib.load(path)
        instance    = cls(model_type=payload["model_type"])
        instance.model   = payload["model"]
        instance._fitted = True
        expected = getattr(instance.model, "n_features_in_", len(FEATURE_COLUMNS))
        instance.feature_columns = payload.get(
            "feature_columns",
            list(FEATURE_COLUMNS)[:expected],
        )
        instance._record_importances()
        print(f"  Model loaded ← {path}")
        return instance

"""TEST IT"""
if __name__ == "__main__":
    import sys

    ORCH_DIR  = "../final_pipeline/train_data_split/orchestral"
    PIANO_DIR = "../final_pipeline/train_data_split/piano"

    if not os.path.isdir(ORCH_DIR):
        print(f"Data directory not found: {ORCH_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("ML MODEL — LEAVE-ONE-GROUP-OUT CROSS-VALIDATION")
    print("=" * 60)

    """Random Forest"""
    rf = NoteClassifier(model_type="random_forest")
    rf_cv = rf.cross_validate(ORCH_DIR, PIANO_DIR)

    """Gradient Boosting"""
    gb = NoteClassifier(model_type="gradient_boosting")
    gb_cv = gb.cross_validate(ORCH_DIR, PIANO_DIR)

    """Summary comparison"""
    print("\nModel comparison (mean CV metrics):")
    print(f"  {'Model':22s}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*52}")
    for name, cv in [("random_forest", rf_cv), ("gradient_boosting", gb_cv)]:
        print(f"  {name:22s}  "
              f"{cv['accuracy'].mean():6.3f}  "
              f"{cv['precision'].mean():6.3f}  "
              f"{cv['recall'].mean():6.3f}  "
              f"{cv['f1'].mean():6.3f}")

    """Train on all data and show feature importance"""
    print("\nTraining final Random Forest on all data …")
    rf.fit(ORCH_DIR, PIANO_DIR)
    rf.print_feature_importance()
    rf.save("models/rf_reducer.joblib")
