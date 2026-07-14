"""Assignment 8 temporal representations: cross-window trends and sequences.

Assignment 7's coefficient evidence located the discriminative signal in
eyelid/blink dynamics summarized *within* each 15-second window, and left one
bounded question for Assignment 8: does structure *across* windows — how blink
behavior evolves over a video — carry signal the per-window tabularization
discards? This file owns the two representations that test that question:

- trend features: causal cross-window summaries (delta from recent past,
  slope of the recent past, drift from the session baseline, elapsed time)
  appended to each window row so the existing logistic path can consume them.
- sequences: per-video `(T, F)` arrays of window features, consumed by a small
  many-to-one GRU that reads the whole video before predicting.

Causality invariant: every trend feature for window `t` must be computable
from windows `0..t` only. A live early-fatigue aid never sees the future, so
features that peek ahead would inflate the offline comparison. Tests assert
this by checking that editing window `t+1` never changes features at `t`.

Split logic stays in `training.splits`; scaling stays fit-on-train-only in the
callers. This module only reshapes the windows table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from training.baselines import NON_MODEL_INPUT_COLUMNS

# Where A7's coefficient inspection found the signal: eyelid temporal dynamics.
# Trends are computed for this cluster only, to keep the added feature count
# honest at n=60 subjects.
TREND_COLUMNS = (
    "perclos",
    "blink_rate",
    "blink_dur_mean",
    "blink_dur_max",
    "eye_blink_mean",
    "eye_blink_std",
    "ear_mean",
    "ear_std",
    "ear_min",
)

# 8 windows at a 7.5 s stride ≈ one minute of past context.
TREND_PAST_WINDOWS = 8
STRIDE_SEC = 7.5  # must match data_prep.windows.STRIDE_SEC


def get_sequence_feature_columns(windows_df: pd.DataFrame) -> list[str]:
    """Return the charter model-input columns (luminance confounds excluded)."""
    return [col for col in windows_df.columns if col not in NON_MODEL_INPUT_COLUMNS]


def add_trend_features(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Append causal cross-window trend features to every window row.

    For each column in TREND_COLUMNS:
    - `{col}_delta`: current value minus the mean of the previous
      TREND_PAST_WINDOWS windows (0.0 at the first window).
    - `{col}_slope`: least-squares slope over the trailing TREND_PAST_WINDOWS
      windows including the current one (0.0 until two windows exist).
    - `{col}_drift`: current value minus the expanding mean of all *earlier*
      windows in the same video — change relative to the session's own
      baseline (0.0 at the first window).

    Plus one shared column:
    - `elapsed_min`: minutes since the start of the video (window_idx * stride).
    """
    result = windows_df.sort_values(["video_id", "window_idx"]).reset_index(drop=True).copy()
    grouped = result.groupby("video_id", sort=False)

    for col in TREND_COLUMNS:
        series = result[col]
        past_mean = grouped[col].transform(
            lambda s: s.shift(1).rolling(TREND_PAST_WINDOWS, min_periods=1).mean()
        )
        session_baseline = grouped[col].transform(lambda s: s.expanding().mean().shift(1))

        result[f"{col}_delta"] = (series - past_mean).fillna(0.0)
        result[f"{col}_slope"] = grouped[col].transform(_trailing_slopes)
        result[f"{col}_drift"] = (series - session_baseline).fillna(0.0)

    result["elapsed_min"] = result["window_idx"] * STRIDE_SEC / 60.0
    return result


def build_video_sequences(
    windows_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Collapse the window table into one row per video with a `(T, F)` array.

    Returns a DataFrame with columns: subject_id, video_id, label, length,
    features (np.ndarray of shape `(T, len(feature_columns))`, in window order).
    """
    rows = []
    ordered = windows_df.sort_values(["video_id", "window_idx"])
    for (subject_id, video_id), video_df in ordered.groupby(["subject_id", "video_id"], sort=True):
        rows.append(
            {
                "subject_id": subject_id,
                "video_id": video_id,
                "label": int(video_df["label"].iloc[0]),
                "length": len(video_df),
                "features": video_df[feature_columns].to_numpy(dtype=np.float32),
            }
        )
    return pd.DataFrame(rows)


def pad_sequences(sequences: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length `(T, F)` arrays into a zero-padded batch.

    Returns `(padded, lengths)` where `padded` has shape
    `(batch, max_T, F)` and `lengths` holds each sequence's true length so the
    GRU can ignore the padding via packing.
    """
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max())
    feature_dim = sequences[0].shape[1]
    padded = torch.zeros((len(sequences), max_len, feature_dim), dtype=torch.float32)
    for idx, seq in enumerate(sequences):
        padded[idx, : len(seq)] = torch.from_numpy(np.ascontiguousarray(seq))
    return padded, lengths


class GruVideoClassifier(torch.nn.Module):
    """Small many-to-one GRU: read a video's window sequence, predict its label."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.25, num_classes: int = 3):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, final_hidden = self.gru(packed)
        return self.head(self.dropout(final_hidden[-1]))


def _trailing_slopes(series: pd.Series) -> pd.Series:
    """Per-window least-squares slope over the trailing TREND_PAST_WINDOWS values."""
    values = series.to_numpy(dtype=float)
    slopes = np.zeros(len(values))
    for idx in range(len(values)):
        start = max(0, idx - TREND_PAST_WINDOWS + 1)
        window = values[start : idx + 1]
        finite = np.isfinite(window)
        if finite.sum() < 2:
            continue
        x = np.arange(len(window), dtype=float)[finite]
        slopes[idx] = np.polyfit(x, window[finite], 1)[0]
    return pd.Series(slopes, index=series.index)
