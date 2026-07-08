"""Post-hoc coefficient inspection for the Assignment 7 `logistic_full` run.

Refits the exact logistic_full pipeline (median impute -> standardize ->
balanced multinomial logistic regression) on each LOSO training fold from the
saved `outputs/folds.json`, then reports the per-class coefficients on the
standardized feature scale, averaged across folds. Because inputs are
standardized inside the pipeline, coefficient magnitudes are comparable across
features.

This is a diagnostic only: it never touches test rows for fitting and does not
change any training outputs.

Usage:
    uv run python scripts/inspect_logistic_full.py \
        [data/frame_windows.parquet] [outputs/folds.json] [outputs/logistic_full_coefficients.csv]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from training.baselines import NON_MODEL_INPUT_COLUMNS  # noqa: E402

RANDOM_SEED = 42
CLASS_NAMES = {0: "alert", 1: "low_vigilant", 2: "drowsy"}


def main() -> None:
    windows_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data/frame_windows.parquet"
    folds_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "outputs/folds.json"
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO_ROOT / "outputs/logistic_full_coefficients.csv"

    windows_df = pd.read_parquet(windows_path)
    folds = json.loads(folds_path.read_text())
    feature_columns = [col for col in windows_df.columns if col not in NON_MODEL_INPUT_COLUMNS]

    per_fold_coefs = []
    for fold_idx, fold in enumerate(folds, start=1):
        train_df = windows_df[windows_df["subject_id"].isin(fold["train"])]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED + fold_idx
                    ),
                ),
            ]
        )
        model.fit(train_df.loc[:, feature_columns], train_df["label"])
        classifier = model.named_steps["classifier"]
        coefs = np.zeros((3, len(feature_columns)))
        for raw_idx, class_label in enumerate(classifier.classes_):
            coefs[int(class_label)] = classifier.coef_[raw_idx]
        per_fold_coefs.append(coefs)

    stacked = np.stack(per_fold_coefs)  # (n_folds, 3, n_features)
    rows = []
    for class_idx, class_name in CLASS_NAMES.items():
        for feat_idx, feature in enumerate(feature_columns):
            values = stacked[:, class_idx, feat_idx]
            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "coef_mean": values.mean(),
                    "coef_std": values.std(),
                    "abs_coef_mean": np.abs(values).mean(),
                }
            )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Folds: {len(folds)}  Features: {len(feature_columns)}")
    print(f"Saved: {output_path}\n")
    print("Top features by mean |standardized coef| (averaged over classes):")
    overall = (
        result.groupby("feature")["abs_coef_mean"].mean().sort_values(ascending=False).round(3)
    )
    print(overall.to_string())
    print("\nPer-class signed means (top 8 by magnitude per class):")
    for class_name in CLASS_NAMES.values():
        class_rows = result[result["class"] == class_name].copy()
        class_rows = class_rows.reindex(class_rows["coef_mean"].abs().sort_values(ascending=False).index)
        print(f"\n[{class_name}]")
        print(class_rows.head(8)[["feature", "coef_mean", "coef_std"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
