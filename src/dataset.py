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


def get_feature_columns(windows_df: pd.DataFrame) -> list[str]:
    """Return the model input feature columns, excluding IDs, labels, and diagnostics."""
    raise NotImplementedError


def split_features_and_target(windows_df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the ordinal target label."""
    raise NotImplementedError


def fit_preprocessor(train_x: pd.DataFrame) -> Any:
    """Fit missing-value handling and scaling on training features only."""
    raise NotImplementedError


def transform_features(preprocessor: Any, features: pd.DataFrame) -> Any:
    """Apply a training-fitted preprocessor to one split's features."""
    raise NotImplementedError


def prepare_fold_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Create model-ready train, validation, and test objects for one fold."""
    raise NotImplementedError


def make_dataloaders(fold_datasets: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Create PyTorch dataloaders from prepared fold datasets."""
    raise NotImplementedError
