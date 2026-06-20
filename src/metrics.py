"""Compute reliability-focused metrics for the alertness assignment.

This file owns Stage 7 evaluation helpers:
- compute ordinal-aware aggregate metrics like QWK and rank MAE
- build confusion matrices at the video level
- summarize metrics as mean ± std across CV folds
- compare baseline and regularized runs
- support error analysis and confidence/calibration checks

Metrics should be computed primarily on video-level predictions because the
video is the real unit of interest. Window-level metrics can still be useful for
debugging training behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def quadratic_weighted_kappa(y_true: Any, y_pred: Any) -> float:
    """Compute quadratic weighted kappa for ordinal predictions."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def rank_mae(y_true: Any, y_pred: Any) -> float:
    """Compute mean absolute error over ordinal class ranks."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def confusion_matrix_table(y_true: Any, y_pred: Any) -> pd.DataFrame:
    """Build a labeled 3x3 confusion matrix."""
    labels = [0, 1, 2]
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)


def classification_metric_summary(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the main metric bundle for one fold or split."""
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "rank_mae": rank_mae(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro")),
    }


def confidence_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize confidence behavior for correct and incorrect predictions."""
    predictions_df = predictions_df.copy()
    predictions_df["correct"] = predictions_df["label"] == predictions_df["pred_label"]
    return (
        predictions_df.groupby("correct", as_index=False)
        .agg(n=("confidence", "size"), mean_confidence=("confidence", "mean"))
        .sort_values("correct", ascending=False)
        .reset_index(drop=True)
    )


def error_slice_summary(predictions_df: pd.DataFrame, slice_column: str) -> pd.DataFrame:
    """Summarize errors by class, subject, quality bin, or another diagnostic slice."""
    rows = []

    for slice_value, slice_df in predictions_df.groupby(slice_column):
        rows.append(
            {
                slice_column: slice_value,
                "n": len(slice_df),
                "accuracy": float((slice_df["label"] == slice_df["pred_label"]).mean()),
                "rank_mae": rank_mae(slice_df["label"], slice_df["pred_label"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_cv_metrics(fold_metrics: list[dict[str, float]]) -> pd.DataFrame:
    """Aggregate fold metrics into mean and standard deviation rows."""
    metrics_df = pd.DataFrame(fold_metrics)
    return pd.DataFrame(
        {
            "metric": metrics_df.columns,
            "mean": metrics_df.mean().to_numpy(),
            "std": metrics_df.std().to_numpy(),
        }
    )
