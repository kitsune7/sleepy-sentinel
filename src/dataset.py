"""Prepare window tables for model training without leaking validation data.

This file owns Stage 3 of the alertness pipeline:
- choose which window summary columns are model inputs
- keep identifiers and quality columns available for diagnostics
- split each fold into X/y arrays
- fit imputers/scalers on the training fold only
- transform validation and test folds with the training-fitted preprocessing
- optionally build PyTorch datasets or dataloaders for training

Avoid creating train/test splits here. Splitting belongs in `splits.py`; this
module should only turn already-split DataFrames into model-ready objects.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

NON_FEATURE_COLUMNS = {"subject_id", "video_id", "window_idx", "label", "frac_face_missing"}


def get_feature_columns(windows_df: pd.DataFrame) -> list[str]:
    """Return the model input feature columns, excluding IDs, labels, and diagnostics."""
    return [col for col in windows_df.columns if col not in NON_FEATURE_COLUMNS]


def split_features_and_target(windows_df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the ordinal target label."""
    return windows_df[feature_columns].copy(), windows_df["label"].copy()


def fit_preprocessor(train_x: pd.DataFrame) -> Any:
    """Fit missing-value handling and scaling on training features only."""
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return preprocessor.fit(train_x)


def transform_features(preprocessor: Any, features: pd.DataFrame) -> Any:
    """Apply a training-fitted preprocessor to one split's features."""
    return preprocessor.transform(features)


def prepare_fold_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Create model-ready train, validation, and test objects for one fold."""
    feature_columns = get_feature_columns(train_df)
    train_x, _ = split_features_and_target(train_df, feature_columns)
    preprocessor = fit_preprocessor(train_x)

    return {
        "train": _prepare_split(train_df, feature_columns, preprocessor),
        "validation": _prepare_split(validation_df, feature_columns, preprocessor),
        "test": _prepare_split(test_df, feature_columns, preprocessor),
        "feature_columns": feature_columns,
        "preprocessor": preprocessor,
    }


def make_dataloaders(fold_datasets: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Create PyTorch dataloaders from prepared fold datasets."""
    dataloaders = {}

    for split_name in ["train", "validation", "test"]:
        split = fold_datasets[split_name]
        x = torch.as_tensor(split["x"].copy(), dtype=torch.float32)
        y = torch.as_tensor(split["y"].copy(), dtype=torch.long)
        dataloaders[split_name] = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=False,
        )

    return dataloaders


def _prepare_split(split_df: pd.DataFrame, feature_columns: list[str], preprocessor: Any) -> dict[str, Any]:
    features, target = split_features_and_target(split_df, feature_columns)
    metadata_columns = [col for col in split_df.columns if col not in feature_columns]

    return {
        "x": transform_features(preprocessor, features),
        "y": target.to_numpy(),
        "metadata": split_df[metadata_columns].copy(),
    }
