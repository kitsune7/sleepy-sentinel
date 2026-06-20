from __future__ import annotations

import json

import pandas as pd
import pytest

import splits


def make_windows_df() -> pd.DataFrame:
    rows = []
    for subject_idx in range(6):
        subject_id = f"{subject_idx + 1:02d}"
        for label in [0, 1, 2]:
            video_id = f"{subject_id}/{label}"
            for window_idx in range(2):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "video_id": video_id,
                        "label": label,
                        "window_idx": window_idx,
                        "perclos": label + window_idx / 10,
                    }
                )
    return pd.DataFrame(rows)


def test_get_subject_ids_returns_sorted_unique_subjects() -> None:
    windows_df = make_windows_df().sample(frac=1, random_state=1)

    assert splits.get_subject_ids(windows_df) == ["01", "02", "03", "04", "05", "06"]


def test_make_group_folds_holds_out_each_subject_once_without_overlap() -> None:
    folds = splits.make_group_folds(make_windows_df(), n_splits=3, random_seed=42)

    assert len(folds) == 3
    all_test_subjects = []
    for fold in folds:
        assert set(fold) == {"train", "test"}
        assert set(fold["train"]).isdisjoint(fold["test"])
        all_test_subjects.extend(fold["test"])

    assert sorted(all_test_subjects) == ["01", "02", "03", "04", "05", "06"]


def test_add_validation_subjects_moves_subjects_out_of_training_only() -> None:
    fold = {"train": ["01", "02", "03", "04"], "test": ["05", "06"]}

    updated = splits.add_validation_subjects(fold, validation_subject_count=1, random_seed=42)

    assert set(updated) == {"train", "validation", "test"}
    assert len(updated["validation"]) == 1
    assert len(updated["train"]) == 3
    assert updated["test"] == ["05", "06"]
    assert set(updated["train"]).isdisjoint(updated["validation"])


def test_assert_disjoint_subjects_rejects_leakage_between_splits() -> None:
    splits.assert_disjoint_subjects({"train": ["01", "02"], "validation": ["03"], "test": ["04"]})

    with pytest.raises((AssertionError, ValueError)):
        splits.assert_disjoint_subjects({"train": ["01", "02"], "validation": ["02"], "test": ["03"]})


def test_filter_split_returns_only_requested_subjects() -> None:
    filtered = splits.filter_split(make_windows_df(), ["02", "04"])

    assert set(filtered["subject_id"]) == {"02", "04"}
    assert len(filtered) == 12


def test_describe_split_reports_subject_video_window_and_label_counts() -> None:
    windows_df = make_windows_df()
    split_subjects = {"train": ["01", "02"], "validation": ["03"], "test": ["04"]}

    summary = splits.describe_split(windows_df, split_subjects)

    assert set(summary["split"]) == {"train", "validation", "test"}
    assert set(["subject_count", "video_count", "window_count", "label_0", "label_1", "label_2"]).issubset(
        summary.columns
    )
    train_row = summary.loc[summary["split"] == "train"].iloc[0]
    assert train_row["subject_count"] == 2
    assert train_row["video_count"] == 6
    assert train_row["window_count"] == 12
    assert train_row["label_0"] == 4


def test_save_fold_assignments_writes_reproducible_json(tmp_path) -> None:
    folds = [{"train": ["01", "02"], "validation": ["03"], "test": ["04"]}]
    output_path = tmp_path / "folds.json"

    splits.save_fold_assignments(folds, output_path)

    assert json.loads(output_path.read_text()) == folds
