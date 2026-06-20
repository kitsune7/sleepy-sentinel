from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import dataset


def make_windows_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["01", "01", "02", "02"],
            "video_id": ["01/0", "01/5", "02/0", "02/5"],
            "window_idx": [0, 0, 0, 0],
            "label": [0, 1, 0, 1],
            "frac_face_missing": [0.0, 0.1, 0.0, 0.2],
            "perclos": [0.1, 0.2, 0.3, np.nan],
            "blink_rate": [10.0, 20.0, 30.0, 40.0],
            "ear_mean": [0.3, 0.25, 0.35, 0.2],
        }
    )


def test_get_feature_columns_excludes_ids_target_and_quality_diagnostics() -> None:
    feature_columns = dataset.get_feature_columns(make_windows_df())

    assert feature_columns == ["perclos", "blink_rate", "ear_mean"]


def test_split_features_and_target_returns_copied_inputs_and_ordinal_labels() -> None:
    windows_df = make_windows_df()

    features, target = dataset.split_features_and_target(windows_df, ["perclos", "blink_rate"])

    assert list(features.columns) == ["perclos", "blink_rate"]
    assert target.tolist() == [0, 1, 0, 1]

    features.loc[0, "perclos"] = 999
    assert windows_df.loc[0, "perclos"] == 0.1


def test_fit_and_transform_preprocessor_imputes_and_scales_from_training_data_only() -> None:
    train_x = pd.DataFrame({"perclos": [0.1, 0.2, 0.3], "blink_rate": [10.0, 20.0, 30.0]})
    validation_x = pd.DataFrame({"perclos": [None], "blink_rate": [40.0]})

    preprocessor = dataset.fit_preprocessor(train_x)
    transformed_train = dataset.transform_features(preprocessor, train_x)
    transformed_validation = dataset.transform_features(preprocessor, validation_x)

    assert transformed_train.shape == (3, 2)
    assert transformed_validation.shape == (1, 2)
    assert not np.isnan(np.asarray(transformed_validation)).any()
    assert np.asarray(transformed_train).mean(axis=0) == pytest.approx([0.0, 0.0])


def test_prepare_fold_datasets_returns_train_validation_and_test_payloads() -> None:
    windows_df = make_windows_df()

    fold_datasets = dataset.prepare_fold_datasets(
        train_df=windows_df.iloc[:2],
        validation_df=windows_df.iloc[2:3],
        test_df=windows_df.iloc[3:],
    )

    assert set(fold_datasets) == {"train", "validation", "test", "feature_columns", "preprocessor"}
    for split_name in ["train", "validation", "test"]:
        assert {"x", "y", "metadata"}.issubset(fold_datasets[split_name])
        assert len(fold_datasets[split_name]["x"]) == len(fold_datasets[split_name]["y"])


def test_make_dataloaders_returns_iterable_batches_for_each_split() -> None:
    fold_datasets = {
        "train": {"x": np.array([[0.0, 1.0], [1.0, 0.0]]), "y": np.array([0, 1])},
        "validation": {"x": np.array([[0.5, 0.5]]), "y": np.array([1])},
        "test": {"x": np.array([[0.2, 0.8]]), "y": np.array([0])},
    }

    dataloaders = dataset.make_dataloaders(fold_datasets, batch_size=2)

    assert set(dataloaders) == {"train", "validation", "test"}
    first_batch_x, first_batch_y = next(iter(dataloaders["train"]))
    assert first_batch_x.shape == (2, 2)
    assert first_batch_y.tolist() == [0, 1]
