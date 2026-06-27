from __future__ import annotations

import pandas as pd

from training import baselines


def make_windows_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["01", "01", "02", "02", "03", "03"],
            "video_id": ["01/0", "01/5", "02/0", "02/10", "03/5", "03/10"],
            "window_idx": [0, 0, 0, 0, 0, 0],
            "label": [0, 1, 0, 2, 1, 2],
            "frac_face_missing": [0.0] * 6,
            "perclos": [0.1, 0.4, 0.2, 0.8, 0.5, 0.9],
            "bright_mean": [100.0, 110.0, 105.0, 150.0, 120.0, 160.0],
            "warmth_mean": [1.0, 1.1, 1.0, 1.4, 1.2, 1.5],
        }
    )


def test_available_baselines_reflect_window_schema() -> None:
    specs = baselines.available_baselines(make_windows_df())

    assert [spec.name for spec in specs] == ["majority", "perclos", "luminance"]
    assert specs[1].feature_columns == ("perclos",)
    assert specs[2].feature_columns == ("bright_mean", "warmth_mean")


def test_predict_baseline_returns_metadata_and_three_class_probabilities() -> None:
    windows_df = make_windows_df()
    spec = baselines.BaselineSpec("perclos", ("perclos",))

    predictions = baselines.predict_baseline(windows_df.iloc[:4], windows_df.iloc[4:], spec, random_seed=1)

    assert {"subject_id", "video_id", "window_idx", "label", "prob_0", "prob_1", "prob_2"}.issubset(
        predictions.columns
    )
    assert predictions[["prob_0", "prob_1", "prob_2"]].sum(axis=1).tolist() == [1.0, 1.0]


def test_majority_baseline_falls_back_to_training_majority_class() -> None:
    windows_df = make_windows_df()

    predictions = baselines.predict_baseline(
        windows_df.iloc[:4],
        windows_df.iloc[4:],
        baselines.BaselineSpec("majority"),
        random_seed=1,
    )

    assert predictions[["prob_0", "prob_1", "prob_2"]].to_numpy().tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
