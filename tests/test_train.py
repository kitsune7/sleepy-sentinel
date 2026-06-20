from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

import train


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
        random_seed=1,
    )

    expected_outputs = {
        "folds.json",
        "split_summaries.csv",
        "learning_curves.csv",
        "video_predictions.csv",
        "fold_metrics.csv",
        "metric_summary.csv",
        "confidence_by_correctness.csv",
        "error_by_true_label.csv",
    }
    assert expected_outputs.issubset({path.name for path in output_dir.iterdir()})
    assert set(results["fold_metrics"]["run"]) == {"baseline", "regularized"}
