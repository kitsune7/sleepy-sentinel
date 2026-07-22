from __future__ import annotations

import numpy as np
import pandas as pd

from training import temporal


def make_two_video_windows(n_windows: int = 12) -> pd.DataFrame:
    """Two videos with distinct, deterministic per-window trajectories."""
    rows = []
    for video_idx, (subject_id, video_stem, label) in enumerate([("01", "0", 0), ("02", "10", 2)]):
        for window_idx in range(n_windows):
            base = 0.1 + 0.05 * window_idx + video_idx  # rising trajectory, offset per video
            row = {
                "subject_id": subject_id,
                "video_id": f"{subject_id}/{video_stem}",
                "window_idx": window_idx,
                "label": label,
                "frac_face_missing": 0.0,
                "bright_mean": 100.0,
                "bright_std": 1.0,
                "warmth_mean": 1.0,
                "warmth_std": 0.1,
            }
            for col in temporal.TREND_COLUMNS:
                row[col] = base
            row["jaw_open_mean"] = 0.2  # one non-trend model feature
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequence_feature_columns_exclude_luminance_and_identifiers() -> None:
    columns = temporal.get_sequence_feature_columns(make_two_video_windows())

    assert "subject_id" not in columns and "label" not in columns
    assert not {"bright_mean", "bright_std", "warmth_mean", "warmth_std"}.intersection(columns)
    assert "perclos" in columns and "jaw_open_mean" in columns


def test_trend_features_are_causal() -> None:
    """Editing a later window must never change trend features at earlier windows."""
    windows_df = make_two_video_windows()
    original = temporal.add_trend_features(windows_df)

    corrupted_input = windows_df.copy()
    late_mask = (corrupted_input["video_id"] == "01/0") & (corrupted_input["window_idx"] >= 8)
    corrupted_input.loc[late_mask, list(temporal.TREND_COLUMNS)] = 99.0
    corrupted = temporal.add_trend_features(corrupted_input)

    trend_columns = [col for col in original.columns if col not in windows_df.columns]
    early = original["window_idx"] < 8
    pd.testing.assert_frame_equal(
        original.loc[early, trend_columns].reset_index(drop=True),
        corrupted.loc[early, trend_columns].reset_index(drop=True),
    )


def test_trend_features_first_window_has_zero_delta_and_drift() -> None:
    result = temporal.add_trend_features(make_two_video_windows())
    first_windows = result[result["window_idx"] == 0]

    for col in temporal.TREND_COLUMNS:
        assert (first_windows[f"{col}_delta"] == 0.0).all()
        assert (first_windows[f"{col}_drift"] == 0.0).all()
        assert (first_windows[f"{col}_slope"] == 0.0).all()


def test_trend_slope_matches_known_linear_trajectory() -> None:
    """Every trajectory rises 0.05 per window, so trailing slopes converge to 0.05."""
    result = temporal.add_trend_features(make_two_video_windows())
    settled = result[result["window_idx"] >= temporal.TREND_PAST_WINDOWS]

    for col in temporal.TREND_COLUMNS:
        assert np.allclose(settled[f"{col}_slope"], 0.05)


def test_trend_features_stay_within_video_boundaries() -> None:
    """The second video's first window must not inherit drift from the first video."""
    windows_df = make_two_video_windows()
    result = temporal.add_trend_features(windows_df)

    second_video_first = result[(result["video_id"] == "02/10") & (result["window_idx"] == 0)]
    # If state leaked across videos, drift would reflect the offset between videos (≈1.0).
    assert (second_video_first["perclos_drift"] == 0.0).all()


def test_build_video_sequences_preserves_window_order_and_shape() -> None:
    windows_df = make_two_video_windows(n_windows=5)
    shuffled = windows_df.sample(frac=1.0, random_state=7)
    feature_columns = temporal.get_sequence_feature_columns(windows_df)

    sequences = temporal.build_video_sequences(shuffled, feature_columns)

    assert len(sequences) == 2
    assert sequences["length"].tolist() == [5, 5]
    first = sequences.iloc[0]["features"]
    assert first.shape == (5, len(feature_columns))
    perclos_idx = feature_columns.index("perclos")
    assert np.all(np.diff(first[:, perclos_idx]) > 0)  # rising trajectory back in order


# pad_sequences / GruVideoClassifier tests moved to tests/test_sequence_models.py
# alongside the torch-optional split (see training.sequence_models docstring).
