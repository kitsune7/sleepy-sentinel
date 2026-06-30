"""Write the flat run-output contract from structured CV results.

The orchestration code (`train.run_cross_validation`) builds structured results
and hands them here; it never calls `.to_csv` itself. The layout is a single
flat directory (no `latest/` vs `runs/<id>/` split) -- see the train.py docstring
for why.

Output files:
    manifest.json          run config, versions, git SHA, timing
    metric_summary.csv      mean/std of headline metrics per run
    fold_metrics.csv        one row per (run, fold)
    video_predictions.csv   video-level predictions for every run/fold
    learning_curves.csv     per-epoch MLP metrics (empty-with-headers if no MLP)
    confusion_matrices.csv   long-form: run, fold, true_label, pred_label, count
    diagnostics.csv         long-form confidence + error-slice rows
    folds.json              subject-to-fold assignments (written by splits)
    split_summaries.csv     per-fold split sizes
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation import metrics
from training.config import RunResult

LEARNING_CURVE_COLUMNS = [
    "run",
    "fold",
    "epoch",
    "selected_checkpoint",
    "train_loss",
    "train_accuracy",
    "train_qwk",
    "train_rank_mae",
    "train_macro_f1",
    "validation_loss",
    "validation_accuracy",
    "validation_qwk",
    "validation_rank_mae",
    "validation_macro_f1",
]

HEADLINE_METRICS = ["qwk", "rank_mae", "accuracy", "macro_f1"]


def write_run_outputs(
    output_dir: str | Path,
    *,
    results: list[RunResult],
    split_summaries: pd.DataFrame,
    manifest: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Write the full flat output contract and return the assembled tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_predictions = _concat([r.video_predictions for r in results])
    fold_metrics = pd.DataFrame([r.fold_metrics for r in results])
    learning_curves = _learning_curves(results)
    metric_summary = _metric_summary(fold_metrics)
    confusion = _confusion_matrices(results)
    diagnostics = _diagnostics(video_predictions)

    split_summaries.to_csv(output_dir / "split_summaries.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    video_predictions.to_csv(output_dir / "video_predictions.csv", index=False)
    learning_curves.to_csv(output_dir / "learning_curves.csv", index=False)
    metric_summary.to_csv(output_dir / "metric_summary.csv")
    confusion.to_csv(output_dir / "confusion_matrices.csv", index=False)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "split_summaries": split_summaries,
        "fold_metrics": fold_metrics,
        "video_predictions": video_predictions,
        "learning_curves": learning_curves,
        "metric_summary": metric_summary,
        "confusion_matrices": confusion,
        "diagnostics": diagnostics,
    }


def _concat(tables: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _learning_curves(results: list[RunResult]) -> pd.DataFrame:
    rows = [row for result in results for row in result.learning_curve]
    # Empty-with-headers keeps the contract stable when no MLP ran.
    return pd.DataFrame(rows, columns=None if rows else LEARNING_CURVE_COLUMNS)


def _metric_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    return fold_metrics.groupby("run")[HEADLINE_METRICS].agg(["mean", "std"])


def _confusion_matrices(results: list[RunResult]) -> pd.DataFrame:
    tables = []
    for result in results:
        preds = result.video_predictions
        table = metrics.confusion_long_form(preds["label"], preds["pred_label"])
        table.insert(0, "run", result.run_name)
        table.insert(1, "fold", result.fold)
        tables.append(table)
    return _concat(tables)


def _diagnostics(video_predictions: pd.DataFrame) -> pd.DataFrame:
    if video_predictions.empty:
        return pd.DataFrame()

    tables = []
    for (run_name, fold), group in video_predictions.groupby(["run", "fold"]):
        confidence = metrics.confidence_summary(group)
        confidence.insert(0, "diagnostic", "confidence_by_correctness")
        tables.append(_label(confidence, run_name, fold))

        error = metrics.error_slice_summary(group, "label")
        error.insert(0, "diagnostic", "error_by_true_label")
        tables.append(_label(error, run_name, fold))

    return _concat(tables)


def _label(table: pd.DataFrame, run_name: str, fold: int) -> pd.DataFrame:
    table.insert(1, "run", run_name)
    table.insert(2, "fold", fold)
    return table
