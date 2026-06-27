from __future__ import annotations

import pandas as pd
import pytest

from data_prep import windows


def make_frame_df() -> pd.DataFrame:
    """Ten frames at 10 fps with two blinks and one half-second yawn."""
    return pd.DataFrame(
        {
            "frame_idx": range(10),
            "t_ms": [i * 100 for i in range(10)],
            "face": [1] * 10,
            "eye_blink_l": [0.1, 0.1, 0.8, 0.9, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1],
            "eye_blink_r": [0.1, 0.1, 0.7, 0.8, 0.1, 0.1, 0.1, 0.9, 0.7, 0.1],
            "ear": [0.31, 0.32, 0.12, 0.11, 0.30, 0.31, 0.32, 0.13, 0.12, 0.30],
            "jaw_open": [0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.8, 0.8, 0.7, 0.7],
            "pitch": [0, 1, 2, 3, 4, 20, 22, 21, 19, 18],
            "yaw": [0, 1, -1, 0, 1, -1, 0, 1, -1, 0],
            "roll": [0, 0, 1, 1, 0, 0, -1, -1, 0, 0],
            "bright_mean": [100.0] * 10,
            "warmth": [1.2] * 10,
        }
    )


def test_load_frame_csv_reads_the_per_video_extraction_output(tmp_path) -> None:
    csv_path = tmp_path / "01" / "0.csv"
    csv_path.parent.mkdir()
    expected = make_frame_df()
    expected.to_csv(csv_path, index=False)

    actual = windows.load_frame_csv(csv_path)

    pd.testing.assert_frame_equal(actual, expected)


def test_clean_frame_features_interpolates_short_gaps_without_filling_missing_faces() -> None:
    frame_df = make_frame_df()
    frame_df.loc[3, "ear"] = None
    frame_df.loc[6:9, "eye_blink_l"] = None
    frame_df.loc[8, "face"] = 0

    cleaned = windows.clean_frame_features(frame_df, fps=10.0)

    assert not pd.isna(cleaned.loc[3, "ear"]), "single-frame numeric gaps should be interpolated"
    assert pd.isna(cleaned.loc[8, "eye_blink_l"]), "longer gaps should stay missing for the validity gate"
    assert cleaned.loc[8, "face"] == 0, "missing-face frames should remain explicitly flagged"


def test_iter_window_bounds_uses_time_order_and_exclusive_end_indexes() -> None:
    frame_df = make_frame_df()

    bounds = windows.iter_window_bounds(frame_df, window_sec=0.4, stride_sec=0.2)

    assert bounds == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_is_valid_window_keeps_windows_at_the_missing_face_threshold() -> None:
    frame_df = make_frame_df()
    frame_df.loc[[0, 1, 2], "face"] = 0

    assert windows.is_valid_window(frame_df, max_missing_face_fraction=0.30) is True

    frame_df.loc[3, "face"] = 0

    assert windows.is_valid_window(frame_df, max_missing_face_fraction=0.30) is False


def test_summarize_window_computes_assignment_features() -> None:
    summary = windows.summarize_window(make_frame_df(), fps=10.0)

    expected_keys = {
        "perclos",
        "blink_rate",
        "blink_dur_mean",
        "blink_dur_max",
        "eye_blink_mean",
        "eye_blink_std",
        "ear_mean",
        "ear_std",
        "ear_min",
        "jaw_open_mean",
        "jaw_open_max",
        "yawn_count",
        "pitch_mean",
        "pitch_std",
        "pitch_range",
        "frac_head_down",
        "yaw_std",
        "roll_std",
        "bright_mean",
        "bright_std",
        "warmth_mean",
        "warmth_std",
        "frac_face_missing",
    }
    assert set(summary) == expected_keys
    assert summary["perclos"] == pytest.approx(0.4)
    assert summary["blink_rate"] == pytest.approx(120.0)
    assert summary["blink_dur_mean"] == pytest.approx(0.2)
    assert summary["blink_dur_max"] == pytest.approx(0.2)
    assert summary["yawn_count"] == 1
    assert summary["frac_face_missing"] == 0.0


def test_build_windows_table_preserves_subject_video_and_label_identifiers(tmp_path) -> None:
    features_root = tmp_path / "features"
    for subject in ["01", "02"]:
        subject_dir = features_root / subject
        subject_dir.mkdir(parents=True)
        make_frame_df().to_csv(subject_dir / "0.csv", index=False)

    windows_df = windows.build_windows_table(features_root)

    assert {"subject_id", "video_id", "label", "window_idx"}.issubset(windows_df.columns)
    assert set(windows_df["subject_id"]) == {"01", "02"}
    assert set(windows_df["label"]) == {0}
    assert windows_df["window_idx"].min() == 0


def test_main_supports_named_cli_arguments(tmp_path) -> None:
    features_root = tmp_path / "features"
    subject_dir = features_root / "01"
    subject_dir.mkdir(parents=True)
    make_frame_df().to_csv(subject_dir / "0.csv", index=False)
    output_path = tmp_path / "windows.csv"

    windows.main(["--features_root", str(features_root), "--output_path", str(output_path)])

    actual = pd.read_csv(output_path, dtype={"subject_id": str})
    assert set(actual["subject_id"]) == {"01"}
    assert set(actual["label"]) == {0}
