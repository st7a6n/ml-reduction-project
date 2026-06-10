"""
train_model.py
==============
Command-line training entry point for the orchestral-to-piano reducer.

Examples:
    python train_model.py --orch-dir data/orchestral --piano-dir data/piano
    python train_model.py --orch-dir data/orchestral --piano-dir data/piano --overlap 4 --enable-mdp
    python train_model.py --cv-group composer
"""

from __future__ import annotations

import argparse
import os

from mdp_reducer import train_mdp_reducer
from ml_model import NoteClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF/GB and optional MDP reducers.")
    parser.add_argument("--orch-dir", default="final_pipeline/train_data_split/orchestral")
    parser.add_argument("--piano-dir", default="final_pipeline/train_data_split/piano")
    parser.add_argument("--model-out", default="models/rf_reducer.joblib")
    parser.add_argument("--mdp-out", default="models/mdp_reducer.joblib")
    parser.add_argument("--model-type", choices=["random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--phrase-bars", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=0, help="Bar overlap between training phrases, e.g. 4.")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--offset-tolerance", type=float, default=0.25)
    parser.add_argument("--cv", action="store_true", help="Run cross-validation before final fit.")
    parser.add_argument("--cv-group", choices=["piece", "composer"], default="piece")
    parser.add_argument("--enable-mdp", action="store_true", help="Train and save the MDP layer too.")
    parser.add_argument("--mdp-episodes", type=int, default=10)
    args = parser.parse_args()

    if not os.path.isdir(args.orch_dir) or not os.path.isdir(args.piano_dir):
        raise FileNotFoundError(
            f"Training dirs not found: orch={args.orch_dir!r}, piano={args.piano_dir!r}"
        )

    clf = NoteClassifier(model_type=args.model_type)
    if args.cv:
        clf.cross_validate(
            args.orch_dir,
            args.piano_dir,
            offset_tolerance=args.offset_tolerance,
            max_pairs=args.max_pairs,
            phrase_bars=args.phrase_bars,
            overlap=args.overlap,
            group_by=args.cv_group,
        )

    clf.fit(
        args.orch_dir,
        args.piano_dir,
        offset_tolerance=args.offset_tolerance,
        max_pairs=args.max_pairs,
        phrase_bars=args.phrase_bars,
        overlap=args.overlap,
    )
    clf.print_feature_importance()
    clf.save(args.model_out)

    if args.enable_mdp:
        mdp = train_mdp_reducer(
            args.orch_dir,
            args.piano_dir,
            n_episodes=args.mdp_episodes,
            phrase_bars=args.phrase_bars,
        )
        os.makedirs(os.path.dirname(args.mdp_out) or ".", exist_ok=True)
        mdp.save(args.mdp_out)


if __name__ == "__main__":
    main()
