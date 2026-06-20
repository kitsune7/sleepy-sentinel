"""Create leakage-safe subject-wise train/validation/test splits.

This file owns Stage 4 of the alertness pipeline:
- group all examples by `subject_id`
- create cross-validation folds where held-out subjects never appear in training
- carve validation subjects out of each training fold for early stopping
- assert that train/validation/test subject sets are disjoint
- record split summaries for the assignment writeup

The core invariant is that windows from one subject must never be split across
train, validation, and test in the same fold.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_subject_ids(windows_df: pd.DataFrame) -> list[str]:
    """Return the sorted unique subject IDs in the windows table."""
    raise NotImplementedError


def make_group_folds(windows_df: pd.DataFrame, n_splits: int, random_seed: int) -> list[dict[str, list[str]]]:
    """Create subject-wise CV folds with train/test subject assignments."""
    raise NotImplementedError


def add_validation_subjects(
    fold: dict[str, list[str]],
    validation_subject_count: int,
    random_seed: int,
) -> dict[str, list[str]]:
    """Move some training subjects into a validation set for one fold."""
    raise NotImplementedError


def assert_disjoint_subjects(split_subjects: dict[str, list[str]]) -> None:
    """Raise if any subject appears in more than one split."""
    raise NotImplementedError


def filter_split(windows_df: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    """Return rows for the requested subjects."""
    raise NotImplementedError


def describe_split(windows_df: pd.DataFrame, split_subjects: dict[str, list[str]]) -> pd.DataFrame:
    """Summarize subject, video, window, and label counts for each split."""
    raise NotImplementedError


def save_fold_assignments(folds: list[dict[str, list[str]]], output_path: str | Path) -> None:
    """Persist subject-to-fold assignments for reproducibility."""
    raise NotImplementedError
