"""Stage 0 diagnostic baselines for alertness classification.

These models are intentionally small floors, not contenders:
- majority class checks the class-balance floor
- PERCLOS-only checks the strongest eyelid-domain heuristic
- luminance-only checks whether lighting can explain the signal

They emit the same window-level probability table shape as the MLP path so the
training coordinator can reuse video aggregation and metric code unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUM_CLASSES = 3
LABELS = list(range(NUM_CLASSES))
LUMINANCE_CANDIDATE_COLUMNS = ("bright_mean", "bright_std", "warmth_mean", "warmth_std", "warmth")


@dataclass(frozen=True)
class BaselineSpec:
    """Describe one non-neural baseline run."""

    name: str
    feature_columns: tuple[str, ...] = ()


def available_baselines(windows_df: pd.DataFrame) -> list[BaselineSpec]:
    """Return baselines supported by the current window table schema."""
    baselines = [BaselineSpec("majority")]

    if "perclos" in windows_df.columns:
        baselines.append(BaselineSpec("perclos", ("perclos",)))

    luminance_columns = tuple(col for col in LUMINANCE_CANDIDATE_COLUMNS if col in windows_df.columns)
    if luminance_columns:
        baselines.append(BaselineSpec("luminance", luminance_columns))

    return baselines


def predict_baseline(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    spec: BaselineSpec,
    *,
    random_seed: int,
) -> pd.DataFrame:
    """Fit one baseline on training rows and predict probabilities for target rows."""
    if not spec.feature_columns:
        probabilities = _majority_probabilities(train_df["label"], len(target_df))
    elif train_df["label"].nunique() < 2:
        probabilities = _majority_probabilities(train_df["label"], len(target_df))
    else:
        probabilities = _logistic_probabilities(train_df, target_df, spec.feature_columns, random_seed=random_seed)

    predictions_df = _metadata(target_df)
    for class_idx in LABELS:
        predictions_df[f"prob_{class_idx}"] = probabilities[:, class_idx]

    return predictions_df


def _majority_probabilities(train_labels: pd.Series, row_count: int) -> np.ndarray:
    label_counts = train_labels.value_counts()
    majority_label = int(label_counts.idxmax())
    probabilities = np.zeros((row_count, NUM_CLASSES), dtype=float)
    probabilities[:, majority_label] = 1.0
    return probabilities


def _logistic_probabilities(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    random_seed: int,
) -> np.ndarray:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed),
            ),
        ]
    )
    model.fit(train_df.loc[:, feature_columns], train_df["label"])

    raw_probabilities = model.predict_proba(target_df.loc[:, feature_columns])
    classifier = model.named_steps["classifier"]
    probabilities = np.zeros((len(target_df), NUM_CLASSES), dtype=float)
    for raw_idx, class_label in enumerate(classifier.classes_):
        probabilities[:, int(class_label)] = raw_probabilities[:, raw_idx]

    return probabilities


def _metadata(split_df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        col
        for col in split_df.columns
        if col in {"subject_id", "video_id", "window_idx", "label", "frac_face_missing"}
    ]
    return split_df.loc[:, metadata_columns].reset_index(drop=True).copy()
