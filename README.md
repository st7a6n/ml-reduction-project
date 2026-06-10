# Orchestral-to-Piano Reduction Pipeline

This project reduces orchestral MusicXML scores into two-staff piano reductions.
The pipeline now supports three reduction modes:

- `rf_only`: original Random Forest note selector.
- `mdp_only`: sequential MDP selector using voice-leading/playability rewards.
- `hybrid`: Random Forest keep probabilities plus MDP sequential selection. This is the default.

## New Features

- Hybrid RF + MDP reduction in `reduce_score.py`.
- Percussion-aware feature extraction in `data_pipeline.py`.
  - Unpitched percussion is used as rhythmic/accent support.
  - Percussion is not added as piano pitches, preserving harmonic cleanliness.
  - New ML features: `has_percussion_support`, `percussion_accent_level`.
- Phrase-context features:
  - `is_phrase_start`, `is_phrase_end`, `phrase_position`, `is_opening`, `is_closing`.
- Overlapping phrase training via `train_model.py --overlap 4`.
- Beethoven Symphony No. 5 Kern/Humdrum loader in `kern_loader.py`.
- Validation helper: `eval_metrics.evaluate_on_beethoven(model, 5, movement_num)`.

## Training

Train the RF model:

```bash
python train_model.py
```

Train with 8-bar phrases and 4-bar overlap:

```bash
python train_model.py --overlap 4
```

Train RF plus the optional MDP layer:

```bash
python train_model.py --overlap 4 --enable-mdp
```

Run leave-one-composer-out validation when filenames include composer prefixes:

```bash
python train_model.py --cv --cv-group composer
```

## Reduction

Hybrid mode is the default:

```bash
python reduce_score.py input.musicxml output.musicxml
```

Run the original RF-only path:

```bash
python reduce_score.py input.musicxml output.musicxml --mode rf_only
```

Compare all modes:

```bash
python reduce_score.py input.musicxml out_rf.musicxml --mode rf_only
python reduce_score.py input.musicxml out_mdp.musicxml --mode mdp_only
python reduce_score.py input.musicxml out_hybrid.musicxml --mode hybrid
```

## Beethoven Kern Data

The included `symph5data/` folders contain movement-level Kern/Humdrum parts.
Inspect the local data:

```bash
python kern_loader.py --inspect
```

Combine a movement into MusicXML:

```bash
python kern_loader.py --movement 1 --write-xml beethoven5_mvt1.musicxml
```

Use Beethoven for validation first, not training:

```python
from eval_metrics import evaluate_on_beethoven
metrics = evaluate_on_beethoven("models/rf_reducer.joblib", 5, 1)
print(metrics["final_score"])
```

## Smoke Test

```bash
python test_pipeline.py
```

The smoke test checks feature extraction, percussion context, hybrid selection,
and Beethoven folder inspection without requiring private training data.

## Repository Note

The trained model binaries in `models/*.joblib` and training checkpoints in
`final_pipeline/checkpoints/*.pt` are intentionally excluded from Git because
they exceed GitHub's standard 100 MB file limit without Git LFS. They remain
available in the local project folder after training.
