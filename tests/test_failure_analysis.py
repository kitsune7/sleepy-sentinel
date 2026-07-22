from __future__ import annotations

import numpy as np
import pandas as pd
from test_temporal import make_two_video_windows

from training import failure_analysis, temporal


def make_predictions(run_name: str, rows: list[tuple[str, str, int, int, float]]) -> pd.DataFrame:
    """Build a video-prediction table: (subject, video, label, pred, confidence)."""
    records = []
    for subject_id, video_id, label, pred, confidence in rows:
        probs = {f"prob_{i}": (1 - confidence) / 2 for i in range(3)}
        probs[f"prob_{pred}"] = confidence
        records.append(
            {
                "run": run_name,
                "fold": 1,
                "subject_id": subject_id,
                "video_id": video_id,
                "label": label,
                "window_count": 10,
                **probs,
                "pred_label": pred,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(records)


def _tiny_trend_table(video_ids: list[str], n_windows: int = 10) -> pd.DataFrame:
    """A minimal trend-feature table covering the given videos."""
    rows = []
    for video_id in video_ids:
        subject_id = video_id.split("/")[0]
        for window_idx in range(n_windows):
            row = {
                "subject_id": subject_id,
                "video_id": video_id,
                "window_idx": window_idx,
                "label": 1,
                "frac_face_missing": 0.0,
            }
            for col in temporal.TREND_COLUMNS:
                row[col] = 0.1 + 0.02 * window_idx
            rows.append(row)
    return temporal.add_trend_features(pd.DataFrame(rows))


def test_truncation_drops_early_windows_and_restarts_clock() -> None:
    windows_df = make_two_video_windows(n_windows=12)
    truncated = failure_analysis.truncate_video_windows(windows_df, 0.5)

    for _, video in truncated.groupby("video_id"):
        assert len(video) == 6  # half of 12 dropped
        assert list(video["window_idx"]) == [0, 1, 2, 3, 4, 5]  # renumbered from 0
    # The kept rows are the *later* windows: values continue the rising trajectory.
    first_kept = truncated[truncated["window_idx"] == 0]
    original_seventh = windows_df[windows_df["window_idx"] == 6].sort_values("video_id")
    assert np.allclose(
        first_kept.sort_values("video_id")["perclos"].to_numpy(),
        original_seventh["perclos"].to_numpy(),
    )


def test_truncation_zero_is_identity_for_features() -> None:
    windows_df = make_two_video_windows()
    truncated = failure_analysis.truncate_video_windows(windows_df, 0.0)
    assert len(truncated) == len(windows_df)
    expected = windows_df.sort_values(["video_id", "window_idx"])["perclos"].to_numpy()
    assert np.allclose(truncated["perclos"].to_numpy(), expected)


def test_truncated_trend_features_treat_cut_point_as_session_start() -> None:
    """Drift at the first kept window must be 0 — the model forgets the removed history."""
    windows_df = make_two_video_windows(n_windows=12)
    trend = temporal.add_trend_features(failure_analysis.truncate_video_windows(windows_df, 0.5))
    first_kept = trend[trend["window_idx"] == 0]

    for col in temporal.TREND_COLUMNS:
        assert (first_kept[f"{col}_drift"] == 0.0).all()
        assert (first_kept[f"{col}_delta"] == 0.0).all()
    assert (first_kept["elapsed_min"] == 0.0).all()


def test_corner_migration_categories() -> None:
    # Video v1: corner under both. v2: corner only under anchor (fixed).
    # v3: corner only under trend (introduced). v4: never a corner (excluded).
    anchor = make_predictions(
        failure_analysis.ANCHOR_RUN_NAME,
        [("01", "v1", 0, 2, 0.6), ("02", "v2", 2, 0, 0.6), ("03", "v3", 0, 0, 0.6), ("04", "v4", 1, 1, 0.6)],
    )
    trend = make_predictions(
        failure_analysis.TREND_RUN_NAME,
        [("01", "v1", 0, 2, 0.6), ("02", "v2", 2, 2, 0.6), ("03", "v3", 0, 2, 0.6), ("04", "v4", 1, 0, 0.6)],
    )
    corners = failure_analysis.trace_corner_migration(pd.concat([anchor, trend], ignore_index=True))

    categories = dict(zip(corners["video_id"], corners["category"]))
    assert categories == {"v1": "persistent", "v2": "fixed_by_trends", "v3": "introduced_by_trends"}


def test_low_vigilant_slicing_labels_miss_directions() -> None:
    trend = make_predictions(
        failure_analysis.TREND_RUN_NAME,
        [("01", "01/5", 1, 1, 0.5), ("02", "02/5", 1, 0, 0.6), ("03", "03/5", 1, 2, 0.7), ("04", "04/0", 0, 0, 0.9)],
    )
    trend_df = _tiny_trend_table(["01/5", "02/5", "03/5", "04/0"])
    table = failure_analysis.slice_low_vigilant_misses(trend, trend_df)

    assert len(table) == 3  # the alert video is excluded
    outcomes = dict(zip(table["video_id"], table["outcome"]))
    assert outcomes == {"01/5": "hit", "02/5": "missed_as_alert", "03/5": "missed_as_drowsy"}
    hit_margin = table.loc[table["outcome"] == "hit", "true_class_margin"].iloc[0]
    assert hit_margin == 0.0  # winners lose to themselves by nothing


def test_calibration_by_class_counts_and_confidence() -> None:
    trend = make_predictions(
        failure_analysis.TREND_RUN_NAME,
        [("01", "a", 2, 2, 0.8), ("02", "b", 0, 2, 0.4), ("03", "c", 2, 2, 0.6)],
    )
    calibration = failure_analysis.calibration_by_class(trend)
    drowsy = calibration[calibration["predicted_class"] == "drowsy"].set_index("correct")

    assert drowsy.loc[True, "n_videos"] == 2 and drowsy.loc[False, "n_videos"] == 1
    assert np.isclose(drowsy.loc[True, "mean_confidence"], 0.7)
    assert np.isclose(drowsy.loc[False, "mean_confidence"], 0.4)


def test_check_reproduction_grades_the_match() -> None:
    saved = make_predictions(
        failure_analysis.TREND_RUN_NAME, [("01", "a", 0, 0, 0.6), ("02", "b", 2, 2, 0.7)]
    )
    probe = saved.copy()
    probe.insert(0, "truncation_fraction", 0.0)
    probe.loc[0, "prob_0"] += 1e-3  # version-level numerical noise, same labels

    result = failure_analysis.check_reproduction(probe, saved)
    assert result["predictions_identical"] is True
    assert result["probabilities_allclose_1e6"] is False
    assert np.isclose(result["max_prob_diff"], 1e-3)


def test_cold_start_summary_recall_per_fraction() -> None:
    cold_start = pd.DataFrame(
        {
            "truncation_fraction": [0.0, 0.0, 0.5, 0.5],
            "label": [2, 2, 2, 2],
            "pred_label": [2, 2, 2, 0],
        }
    )
    summary = failure_analysis.summarize_cold_start(cold_start).set_index("truncation_fraction")
    assert summary.loc[0.0, "drowsy_recall"] == 1.0
    assert summary.loc[0.5, "drowsy_recall"] == 0.5
