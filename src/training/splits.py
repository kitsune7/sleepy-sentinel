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

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_group_folds(windows_df: pd.DataFrame, n_splits: int, random_seed: int) -> list[dict[str, list[str]]]:
    """Create subject-wise cross-validation folds with train/test subject assignments."""
    subject_ids = get_subject_ids(windows_df)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(subject_ids):
        raise ValueError("n_splits cannot exceed the number of subjects")

    rng = np.random.default_rng(random_seed)
    shuffled_subjects = list(rng.permutation(subject_ids))
    test_groups = np.array_split(shuffled_subjects, n_splits)
    folds = []

    for test_group in test_groups:
        test_subjects = sorted(subject_id for subject_id in test_group.tolist())
        train_subjects = sorted(subject_id for subject_id in subject_ids if subject_id not in test_subjects)
        folds.append({"train": train_subjects, "test": test_subjects})

    return folds


def make_loso_folds(windows_df: pd.DataFrame) -> list[dict[str, list[str]]]:
    """Create leave-one-subject-out folds with one held-out test subject per fold."""
    subject_ids = get_subject_ids(windows_df)
    if len(subject_ids) < 2:
        raise ValueError("LOSO requires at least two subjects")

    return [
        {
            "train": [train_subject_id for train_subject_id in subject_ids if train_subject_id != test_subject_id],
            "test": [test_subject_id],
        }
        for test_subject_id in subject_ids
    ]


def get_subject_ids(windows_df: pd.DataFrame) -> list[str]:
    """Return the sorted unique subject IDs in the windows table."""
    return sorted(windows_df["subject_id"].unique())


def add_validation_subjects(
    fold: dict[str, list[str]],
    validation_subject_count: int,
    random_seed: int,
) -> dict[str, list[str]]:
    """Move some training subjects into a validation set for one fold."""
    train_subjects = fold["train"]
    if validation_subject_count < 0:
        raise ValueError("validation_subject_count cannot be negative")
    if validation_subject_count >= len(train_subjects):
        raise ValueError("validation_subject_count must leave at least one training subject")

    rng = np.random.default_rng(random_seed)
    validation_subjects = sorted(rng.choice(train_subjects, size=validation_subject_count, replace=False).tolist())
    updated_train = sorted(subject_id for subject_id in train_subjects if subject_id not in validation_subjects)

    return {
        "train": updated_train,
        "validation": validation_subjects,
        "test": list(fold["test"]),
    }


def assert_disjoint_subjects(split_subjects: dict[str, list[str]]) -> None:
    """Raise if any subject appears in more than one split."""
    seen: set[str] = set()
    for split_name, subject_ids in split_subjects.items():
        overlap = seen.intersection(subject_ids)
        if overlap:
            raise AssertionError(f"Subjects appear in multiple splits at {split_name}: {sorted(overlap)}")
        seen.update(subject_ids)


def filter_split(windows_df: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    """Return rows for the requested subjects."""
    return windows_df.loc[windows_df["subject_id"].astype(str).isin(subject_ids)].copy()


def describe_split(windows_df: pd.DataFrame, split_subjects: dict[str, list[str]]) -> pd.DataFrame:
    """Summarize subject, video, window, and label counts for each split."""
    rows = []

    for split_name, subject_ids in split_subjects.items():
        split_df = filter_split(windows_df, subject_ids)
        label_counts = split_df["label"].value_counts().to_dict()
        row = {
            "split": split_name,
            "subject_count": split_df["subject_id"].nunique(),
            "video_count": split_df["video_id"].nunique(),
            "window_count": len(split_df),
        }
        row.update({f"label_{label}": int(label_counts.get(label, 0)) for label in [0, 1, 2]})
        rows.append(row)

    return pd.DataFrame(rows)


def save_fold_assignments(folds: list[dict[str, list[str]]], output_path: str | Path) -> None:
    """Persist subject-to-fold assignments for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(folds, indent=2))
