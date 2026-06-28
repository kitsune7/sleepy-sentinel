from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

from training import train
from training.config import ModelConfig, TrainingConfig


def test_set_random_seeds_makes_python_numpy_and_torch_reproducible() -> None:
    train.set_random_seeds(123)
    first = (random.random(), np.random.random(), torch.rand(1).item())

    train.set_random_seeds(123)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert second == first


def test_aggregate_window_predictions_averages_probabilities_to_video_level() -> None:
    predictions_df = pd.DataFrame(
        {
            "subject_id": ["01", "01", "01", "02"],
            "video_id": ["01/0", "01/0", "01/5", "02/10"],
            "label": [0, 0, 1, 2],
            "window_idx": [0, 1, 0, 0],
            "prob_0": [0.8, 0.6, 0.2, 0.1],
            "prob_1": [0.1, 0.3, 0.7, 0.2],
            "prob_2": [0.1, 0.1, 0.1, 0.7],
        }
    )

    video_predictions = train.aggregate_window_predictions(predictions_df)

    assert list(video_predictions["video_id"]) == ["01/0", "01/5", "02/10"]
    assert video_predictions["label"].tolist() == [0, 1, 2]
    assert video_predictions["pred_label"].tolist() == [0, 1, 2]
    assert video_predictions.loc[video_predictions["video_id"] == "01/0", "prob_0"].item() == pytest.approx(0.7)


def test_run_cross_validation_writes_assignment_artifacts(tmp_path) -> None:
    rows = []
    for subject_idx in range(4):
        subject_id = f"{subject_idx + 1:02d}"
        for label in [0, 1, 2]:
            rows.append(
                {
                    "subject_id": subject_id,
                    "video_id": f"{subject_id}/{label}",
                    "label": label,
                    "window_idx": 0,
                    "frac_face_missing": 0.0,
                    "perclos": label / 2,
                    "blink_rate": 10 + subject_idx,
                    "ear_mean": 0.3 - label / 20,
                    "bright_mean": 100 + label,
                    "warmth_mean": 1.0 + label / 10,
                }
            )
    windows_path = tmp_path / "windows.parquet"
    output_dir = tmp_path / "eval"
    pd.DataFrame(rows).to_parquet(windows_path, index=False)

    results = train.run_cross_validation(
        windows_path,
        output_dir,
        n_splits=2,
        validation_subject_count=1,
        epochs=1,
        batch_size=4,
        early_stopping_patience=-1,
        random_seed=1,
        include_mlp=True,
    )

    expected_outputs = {
        "manifest.json",
        "folds.json",
        "split_summaries.csv",
        "learning_curves.csv",
        "video_predictions.csv",
        "fold_metrics.csv",
        "metric_summary.csv",
        "confusion_matrices.csv",
        "diagnostics.csv",
    }
    output_names = {path.name for path in output_dir.iterdir()}
    assert expected_outputs.issubset(output_names)
    assert not any(name.endswith("_confusion_matrix.csv") for name in output_names)
    assert set(results["fold_metrics"]["run"]) == {"majority", "perclos", "luminance", "mlp_regularized"}
    assert {
        "run",
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
        "selected_checkpoint",
    }.issubset(results["learning_curves"].columns)


def test_train_fold_stops_after_validation_qwk_patience(monkeypatch) -> None:
    validation_scores = iter([0.1, 0.2, 0.15])

    def fake_train_one_epoch(*_args):
        return {"loss": 1.0, "accuracy": 0.0, "qwk": 0.0, "rank_mae": 1.0, "macro_f1": 0.0}

    def fake_evaluate_one_epoch(*_args):
        score = next(validation_scores)
        return {"loss": 1.0 - score, "accuracy": score, "qwk": score, "rank_mae": 1.0, "macro_f1": score}

    monkeypatch.setattr(train, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train, "evaluate_one_epoch", fake_evaluate_one_epoch)
    fold_data = {
        "feature_columns": ["perclos", "blink_rate"],
        "train": {"x": np.array([[0.0, 1.0], [1.0, 0.0]]), "y": np.array([0, 1])},
        "validation": {"x": np.array([[0.5, 0.5]]), "y": np.array([1])},
        "test": {"x": np.array([[0.2, 0.8]]), "y": np.array([0])},
    }

    result = train.train_fold(
        fold_data,
        ModelConfig(hidden_dims=(4,), dropout=0.0, weight_decay=0.0, num_classes=3),
        TrainingConfig(epochs=5, batch_size=2, early_stopping_patience=1),
    )

    assert result["best_epoch"] == 2
    assert len(result["history"]) == 3
    assert result["stopped_early"] is True
