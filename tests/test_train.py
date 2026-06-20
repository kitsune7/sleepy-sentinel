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
