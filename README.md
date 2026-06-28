# Sleepy Sentinel

This is a repo that trains a machine learning model to classify alertness levels.

This project requires `uv` to run.

## Train the model

Run `uv run train_alertness` to run training with the default parameters. By
default this trains only the baselines (majority, luminance-only, PERCLOS-only).
Add `--include-mlp` to also train the regularized MLP.

Each run writes a fixed, flat set of files into `outputs/` (and nothing else):
`manifest.json`, `metric_summary.csv`, `fold_metrics.csv`,
`video_predictions.csv`, `learning_curves.csv`, `confusion_matrices.csv`,
`diagnostics.csv`, `folds.json`, and `split_summaries.csv`.

W&B is optional and off unless you pass `--wandb-project`; each training command
is logged as a single W&B run.

## Assignment Docs

Write-ups for the assignments are kept in the `docs/` directory. They're labeled per assignment.
