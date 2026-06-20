from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

import metrics


def test_quadratic_weighted_kappa_matches_sklearn_for_ordinal_labels() -> None:
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 0]

    assert metrics.quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(
        cohen_kappa_score(y_true, y_pred, weights="quadratic")
    )


def test_rank_mae_penalizes_two_step_errors_twice_as_much_as_one_step_errors() -> None:
    assert metrics.rank_mae([0, 1, 2], [0, 2, 0]) == pytest.approx(1.0)


def test_confusion_matrix_table_returns_labeled_three_by_three_table() -> None:
    table = metrics.confusion_matrix_table([0, 1, 2, 2], [0, 2, 2, 1])

    assert list(table.index) == [0, 1, 2]
    assert list(table.columns) == [0, 1, 2]
    assert table.loc[0, 0] == 1
    assert table.loc[1, 2] == 1
    assert table.loc[2, 1] == 1
    assert table.loc[2, 2] == 1


def test_classification_metric_summary_contains_assignment_headline_metrics() -> None:
    summary = metrics.classification_metric_summary([0, 1, 2, 2], [0, 2, 2, 1])

    assert {"qwk", "rank_mae", "accuracy", "macro_f1"}.issubset(summary)
    assert summary["rank_mae"] == pytest.approx(0.5)
    assert summary["accuracy"] == pytest.approx(0.5)


def test_confidence_summary_compares_correct_and_incorrect_predictions() -> None:
    predictions_df = pd.DataFrame(
        {
            "label": [0, 1, 2, 2],
            "pred_label": [0, 2, 2, 1],
            "confidence": [0.9, 0.8, 0.7, 0.4],
        }
    )

    summary = metrics.confidence_summary(predictions_df)

    assert set(summary["correct"]) == {True, False}
    correct_row = summary.loc[summary["correct"]].iloc[0]
    incorrect_row = summary.loc[~summary["correct"]].iloc[0]
    assert correct_row["n"] == 2
    assert correct_row["mean_confidence"] == pytest.approx(0.8)
    assert incorrect_row["mean_confidence"] == pytest.approx(0.6)


def test_error_slice_summary_reports_counts_accuracy_and_mae_by_slice() -> None:
    predictions_df = pd.DataFrame(
        {
            "label": [0, 1, 2, 2],
            "pred_label": [0, 2, 2, 1],
            "missing_face_bin": ["low", "low", "high", "high"],
        }
    )

    summary = metrics.error_slice_summary(predictions_df, slice_column="missing_face_bin")

    assert set(summary["missing_face_bin"]) == {"low", "high"}
    assert {"n", "accuracy", "rank_mae"}.issubset(summary.columns)
    assert summary.loc[summary["missing_face_bin"] == "low", "accuracy"].item() == pytest.approx(0.5)
    assert summary.loc[summary["missing_face_bin"] == "high", "rank_mae"].item() == pytest.approx(0.5)


def test_summarize_cv_metrics_returns_mean_and_std_for_each_metric() -> None:
    fold_metrics = [
        {"qwk": 0.2, "rank_mae": 0.8},
        {"qwk": 0.4, "rank_mae": 0.6},
        {"qwk": 0.6, "rank_mae": 0.4},
    ]

    summary = metrics.summarize_cv_metrics(fold_metrics)

    assert list(summary["metric"]) == ["qwk", "rank_mae"]
    assert summary.loc[summary["metric"] == "qwk", "mean"].item() == pytest.approx(0.4)
    assert summary.loc[summary["metric"] == "qwk", "std"].item() == pytest.approx(0.2)
