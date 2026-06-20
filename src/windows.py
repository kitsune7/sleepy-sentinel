"""Build model-ready window summaries from per-frame feature CSVs.

This file owns Stage 2 of the alertness pipeline:
- read the per-video CSVs produced by `extract_features.py`
- clean short landmark gaps and smooth noisy frame-level signals
- slide fixed-duration windows through each video
- compute one row of summary features per valid window
- write the combined windows table used by training and evaluation

Do not split subjects or scale features here. This module should only convert
frame-level evidence into window-level examples while preserving identifiers
like `subject_id`, `video_id`, `label`, and `window_idx`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

WINDOW_SEC = 15.0
STRIDE_SEC = 7.5
MAX_MISSING_FACE_FRACTION = 0.30
LABEL_BY_VIDEO_STEM = {"0": 0, "5": 1, "10": 2}
HEAD_DOWN_PITCH_DEG = 15.0


def load_frame_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read one per-video feature CSV."""
    return pd.read_csv(csv_path, header=0)


def clean_frame_features(frame_df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Smooth jittery frame-level signals and interpolate short missing gaps."""
    cleaned = frame_df.copy()
    max_gap_frames = max(1, round(0.3 * fps))

    feature_cols = ["eye_blink_l", "eye_blink_r", "ear", "jaw_open", "pitch", "yaw", "roll"]
    jitter_cols = ["eye_blink_l", "eye_blink_r", "ear", "jaw_open"]

    for col in feature_cols:
        # series of True/False values indicating missing values
        missing = cleaned[col].isna()

        # create a new series of integers that group consecutive missing values (i.e. FFTTTFF -> 1122233)
        gap_ids = missing.ne(missing.shift()).cumsum()

        # `missing.groupby(gap_ids).transform("sum")` counts the number of missing values in each group (i.e. 0033300)
        # We then compare this count to `max_gap_frames` to identify groups that are longer than the threshold.
        # And we do a bitwise AND with `missing` to ensure we only mark frames that were originally missing.
        long_gap = missing & (missing.groupby(gap_ids).transform("sum") > max_gap_frames)

        # Here we fill in the all the gaps with what would naturally fall in the middle
        cleaned[col] = cleaned[col].interpolate(
            method="linear",
            limit=max_gap_frames,
            limit_area="inside",
        )

        # And we reset the long gaps since we want to preserve those
        cleaned.loc[long_gap, col] = pd.NA

    for col in jitter_cols:
        # Hold onto these so we don't overwrite them with the median filter
        missing_after_interpolation = cleaned[col].isna()

        # Remove the jitter by applying a median filter
        cleaned[col] = cleaned[col].rolling(window=5, center=True, min_periods=1).median()

        # Put any long gaps back in
        cleaned.loc[missing_after_interpolation, col] = pd.NA

    return cleaned


def iter_window_bounds(frame_df: pd.DataFrame, window_sec: float, stride_sec: float) -> list[tuple[int, int]]:
    """Return start/end frame indexes for each sliding window."""
    fps = 1000 / frame_df["t_ms"].diff().dropna().median()
    window_frames = round(window_sec * fps)
    stride_frames = round(stride_sec * fps)

    return [
        (start, start + window_frames)
        for start in range(0, len(frame_df) - window_frames + 1, stride_frames)
    ]


def is_valid_window(window_df: pd.DataFrame, max_missing_face_fraction: float) -> bool:
    """Decide whether a window has enough detected-face frames to keep."""
    missing_face_fraction = len(window_df[window_df["face"] == 0]) / len(window_df)
    return missing_face_fraction <= max_missing_face_fraction


def summarize_window(window_df: pd.DataFrame, fps: float) -> dict[str, Any]:
    """Compute blink, eye-closure, yawn, head-pose, and quality features."""
    eye_blink = window_df[["eye_blink_l", "eye_blink_r"]].mean(axis=1)
    eye_closed = eye_blink > 0.5

    blink_durations = _run_durations_seconds(eye_closed, fps)
    window_minutes = len(window_df) / fps / 60

    yawn_frames = window_df["jaw_open"] > 0.5
    yawn_count = sum(duration >= 0.5 for duration in _run_durations_seconds(yawn_frames, fps))

    return {
        "perclos": float(eye_closed.mean()),
        "blink_rate": len(blink_durations) / window_minutes if window_minutes else 0.0,
        "blink_dur_mean": _mean_or_zero(blink_durations),
        "blink_dur_max": max(blink_durations, default=0.0),
        "eye_blink_mean": float(eye_blink.mean()),
        "eye_blink_std": float(eye_blink.std()),
        "ear_mean": float(window_df["ear"].mean()),
        "ear_std": float(window_df["ear"].std()),
        "ear_min": float(window_df["ear"].min()),
        "jaw_open_mean": float(window_df["jaw_open"].mean()),
        "jaw_open_max": float(window_df["jaw_open"].max()),
        "yawn_count": yawn_count,
        "pitch_mean": float(window_df["pitch"].mean()),
        "pitch_std": float(window_df["pitch"].std()),
        "pitch_range": float(window_df["pitch"].max() - window_df["pitch"].min()),
        "frac_head_down": float((window_df["pitch"] > HEAD_DOWN_PITCH_DEG).mean()),
        "yaw_std": float(window_df["yaw"].std()),
        "roll_std": float(window_df["roll"].std()),
        "frac_face_missing": float((window_df["face"] == 0).mean()),
    }


def build_windows_table(features_root: str | Path) -> pd.DataFrame:
    """Build the full window-level table from all subject/video CSVs."""
    rows: list[dict[str, Any]] = []
    features_root = Path(features_root)

    for csv_path in sorted(features_root.glob("*/*.csv")):
        subject_id = csv_path.parent.name
        video_stem = csv_path.stem
        label = LABEL_BY_VIDEO_STEM.get(video_stem, int(video_stem))
        frame_df = load_frame_csv(csv_path)
        fps = 1000 / frame_df["t_ms"].diff().dropna().median()
        cleaned = clean_frame_features(frame_df, fps=fps)
        bounds = iter_window_bounds(cleaned, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)

        if not bounds and not cleaned.empty:
            bounds = [(0, len(cleaned))]

        window_idx = 0
        for start, end in bounds:
            window_df = cleaned.iloc[start:end]
            if not is_valid_window(window_df, max_missing_face_fraction=MAX_MISSING_FACE_FRACTION):
                continue

            rows.append(
                {
                    "subject_id": subject_id,
                    "video_id": f"{subject_id}/{video_stem}",
                    "label": label,
                    "window_idx": window_idx,
                    **summarize_window(window_df, fps=fps),
                }
            )
            window_idx += 1

    return pd.DataFrame(rows)


def write_windows_table(windows_df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist the window-level table, likely as Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        windows_df.to_csv(output_path, index=False)
    else:
        windows_df.to_parquet(output_path, index=False)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point for generating the windows table."""
    import argparse

    parser = argparse.ArgumentParser(description="Build the alertness window table from per-frame CSVs.")
    parser.add_argument("features_root", type=Path, nargs="?")
    parser.add_argument("output_path", type=Path, nargs="?")
    parser.add_argument("--features-root", "--features_root", dest="features_root_flag", type=Path)
    parser.add_argument("--output-path", "--output_path", dest="output_path_flag", type=Path)
    args = parser.parse_args(argv)

    features_root = args.features_root_flag or args.features_root
    output_path = args.output_path_flag or args.output_path
    if features_root is None or output_path is None:
        parser.error("features_root and output_path are required")

    write_windows_table(build_windows_table(features_root), output_path)


def _run_durations_seconds(mask: pd.Series, fps: float) -> list[float]:
    durations: list[float] = []
    run_length = 0

    for value in mask.fillna(False).astype(bool):
        if value:
            run_length += 1
        elif run_length:
            durations.append(run_length / fps)
            run_length = 0

    if run_length:
        durations.append(run_length / fps)

    return durations


def _mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
