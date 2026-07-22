"""Assignment 9 failure analysis: slice the champion's errors, probe its blind spot.

Assignment 8 adopted `trend_logistic` and left a prioritized list of failure
questions for Week 9. This module answers them without training anything new —
it reads the saved A8 prediction artifacts, slices them, and runs one bounded
*scoring* probe (the cold-start truncation test) using the already-chosen model
refit under the identical protocol. Four analyses:

1. Low-vigilant miss slicing — the 39 still-missed low-vigilant videos: which
   subjects, what the model called them instead, how confident it was, and
   whether landmark quality (`frac_face_missing`) or drift-trajectory shape
   separates the misses from the hits.
2. Corner-error migration — A7 and A8 both show 17 alert<->drowsy corner
   errors, but are they the *same* 17 videos? This table tracks which corners
   the trend features fixed, which they introduced, and which persist.
3. Per-class calibration — confidence by predicted class and correctness, to
   test whether the confidence gap A8 reported survives per-class inspection
   (a warning threshold is the stakeholder-facing output, so this matters).
4. Cold-start truncation probe — the drift features measure change from the
   session's own start, so a user who begins a session already drowsy has
   drift ~= 0 on exactly the features that flag drowsiness. The dataset cannot
   show this directly (every video has one constant label), so we simulate it:
   drop the first 25% / 50% of each test video's windows, restart the clock,
   recompute trend features from the truncated start, and score with the same
   per-fold models. If drowsy recall collapses as truncation grows, the blind
   spot is real and quantified.

Reproducibility contract (same pattern as A8's anchor run): the probe at
truncation 0.0 must reproduce the saved `trend_logistic` video predictions
before any truncated number is read. The manifest records the check result.

Outputs are written flat into the output dir with an `a9_` prefix so all
A7/A8 artifacts stay untouched for comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training import temporal
from training.artifacts import aggregate_window_predictions, git_sha, package_versions

logger = logging.getLogger("training.failure_analysis")

TREND_RUN_NAME = "trend_logistic"
ANCHOR_RUN_NAME = "logistic_full_anchor"
CLASS_NAMES = {0: "alert", 1: "low_vigilant", 2: "drowsy"}

# Fractions of each test video's windows to drop from the *front* before
# recomputing trend features. 0.0 is the reproduction check; 0.25/0.5 simulate
# a user who opens the aid one-quarter / halfway into a fatigue episode.
TRUNCATION_FRACTIONS = (0.0, 0.25, 0.5)

# Drift features A8's coefficient table identified as the load-bearing ones;
# the miss-slicing table summarizes each video's trajectory on these.
KEY_DRIFT_FEATURES = ("blink_dur_max_drift", "blink_dur_mean_drift", "perclos_drift")


def main() -> None:
    """Command-line entry point for the Assignment 9 failure analysis."""
    parser = argparse.ArgumentParser(description="Run Assignment 9 failure analyses on saved A8 artifacts.")
    parser.add_argument("windows_path", type=Path, nargs="?", default=Path("data/frame_windows.parquet"))
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("--folds-path", type=Path, default=None, help="Defaults to <output_dir>/folds.json.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-cold-start", action="store_true", help="Run only the artifact-slicing analyses.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    run_failure_analysis(
        args.windows_path,
        args.output_dir,
        folds_path=args.folds_path,
        random_seed=args.random_seed,
        include_cold_start=not args.skip_cold_start,
    )


def run_failure_analysis(
    windows_path: str | Path,
    output_dir: str | Path,
    *,
    folds_path: str | Path | None = None,
    random_seed: int = 42,
    include_cold_start: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run all Assignment 9 analyses and write `a9_*` artifacts."""
    output_dir = Path(output_dir)
    folds_path = Path(folds_path) if folds_path is not None else output_dir / "folds.json"
    started_at = datetime.now(timezone.utc).isoformat()

    windows_df = pd.read_parquet(windows_path)
    predictions = pd.read_csv(
        output_dir / "temporal_video_predictions.csv", dtype={"subject_id": str}
    )
    trend_df = temporal.add_trend_features(windows_df)

    tables: dict[str, pd.DataFrame] = {}
    tables["low_vigilant_misses"] = slice_low_vigilant_misses(predictions, trend_df)
    tables["low_vigilant_summary"] = summarize_miss_groups(tables["low_vigilant_misses"])
    tables["corner_migration"] = trace_corner_migration(predictions)
    tables["calibration"] = calibration_by_class(predictions)

    reproduction = None
    if include_cold_start:
        folds = json.loads(folds_path.read_text())
        cold_start, positions = run_cold_start_probe(windows_df, trend_df, folds, random_seed=random_seed)
        tables["cold_start_predictions"] = cold_start
        tables["cold_start_summary"] = summarize_cold_start(cold_start)
        tables["window_position_accuracy"] = positions
        reproduction = check_reproduction(cold_start, predictions)

    _write_tables(output_dir, tables)
    _write_manifest(
        output_dir,
        windows_path=windows_path,
        folds_path=folds_path,
        random_seed=random_seed,
        include_cold_start=include_cold_start,
        reproduction=reproduction,
        started_at=started_at,
    )
    _log_summary(tables, reproduction)
    return tables


# ---------------------------------------------------------------------------
# Analysis 1: who are the still-missed low-vigilant videos?
# ---------------------------------------------------------------------------


def slice_low_vigilant_misses(predictions: pd.DataFrame, trend_df: pd.DataFrame) -> pd.DataFrame:
    """One row per low-vigilant video with slicing covariates for hit/miss comparison.

    Covariates per video: what trend_logistic predicted and how confidently,
    the probability margin the true class lost by, landmark quality
    (`frac_face_missing`), the video's mean and final value on the key drift
    features, and the first-minute elevation of the eyelid signals (how
    fatigued the video already looks at its own start).
    """
    trend = predictions[predictions["run"] == TREND_RUN_NAME]
    low_vig = trend[trend["label"] == 1].copy()
    low_vig["outcome"] = np.select(
        [low_vig["pred_label"] == 1, low_vig["pred_label"] == 0],
        ["hit", "missed_as_alert"],
        default="missed_as_drowsy",
    )
    # How far the true class fell short of the winning class (0 for hits).
    low_vig["true_class_margin"] = low_vig["prob_1"] - low_vig[["prob_0", "prob_1", "prob_2"]].max(axis=1)

    covariates = _per_video_covariates(trend_df)
    return (
        low_vig[["subject_id", "video_id", "outcome", "pred_label", "confidence", "prob_1", "true_class_margin"]]
        .merge(covariates, on="video_id")
        .sort_values(["outcome", "video_id"])
        .reset_index(drop=True)
    )


def summarize_miss_groups(miss_table: pd.DataFrame) -> pd.DataFrame:
    """Compare hits vs. the two miss directions on every slicing covariate."""
    numeric_columns = [
        col for col in miss_table.columns
        if col not in {"subject_id", "video_id", "outcome", "pred_label"}
    ]
    summary = miss_table.groupby("outcome")[numeric_columns].mean()
    summary.insert(0, "n_videos", miss_table.groupby("outcome").size())
    return summary.reset_index()


def _per_video_covariates(trend_df: pd.DataFrame) -> pd.DataFrame:
    """Per-video slicing covariates from the trend-feature table."""
    ordered = trend_df.sort_values(["video_id", "window_idx"])
    grouped = ordered.groupby("video_id")

    covariates = grouped.agg(frac_face_missing=("frac_face_missing", "mean")).reset_index()
    for feature in KEY_DRIFT_FEATURES:
        covariates[f"{feature}_mean"] = grouped[feature].mean().to_numpy()
        covariates[f"{feature}_final"] = grouped[feature].last().to_numpy()

    # First-minute elevation: mean over the first TREND_PAST_WINDOWS windows.
    first_minute = (
        ordered.groupby("video_id", sort=True)
        .head(temporal.TREND_PAST_WINDOWS)
        .groupby("video_id")
        .agg(start_blink_dur_max=("blink_dur_max", "mean"), start_perclos=("perclos", "mean"))
        .reset_index()
    )
    return covariates.merge(first_minute, on="video_id")


# ---------------------------------------------------------------------------
# Analysis 2: did the trend features move the alert<->drowsy corners?
# ---------------------------------------------------------------------------


def trace_corner_migration(predictions: pd.DataFrame) -> pd.DataFrame:
    """Track every video that sits in the alert<->drowsy corner under either model.

    Categories: `persistent` (corner in both), `fixed_by_trends` (corner under
    the anchor only), `introduced_by_trends` (corner under trend_logistic only).
    """
    wide = _pivot_predictions(predictions)
    anchor_corner = (wide["label"] - wide[f"pred_{ANCHOR_RUN_NAME}"]).abs() == 2
    trend_corner = (wide["label"] - wide[f"pred_{TREND_RUN_NAME}"]).abs() == 2

    corners = wide[anchor_corner | trend_corner].copy()
    corners["category"] = np.select(
        [anchor_corner[corners.index] & trend_corner[corners.index], anchor_corner[corners.index]],
        ["persistent", "fixed_by_trends"],
        default="introduced_by_trends",
    )
    corners["true_class"] = corners["label"].map(CLASS_NAMES)
    return corners.sort_values(["category", "video_id"]).reset_index(drop=True)


def _pivot_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per video with each run's prediction and confidence side by side."""
    runs = [ANCHOR_RUN_NAME, TREND_RUN_NAME]
    wide = None
    for run_name in runs:
        run_df = predictions[predictions["run"] == run_name][
            ["subject_id", "video_id", "label", "pred_label", "confidence"]
        ].rename(columns={"pred_label": f"pred_{run_name}", "confidence": f"confidence_{run_name}"})
        wide = run_df if wide is None else wide.merge(run_df, on=["subject_id", "video_id", "label"])
    return wide


# ---------------------------------------------------------------------------
# Analysis 3: does the confidence gap survive per-class inspection?
# ---------------------------------------------------------------------------


def calibration_by_class(predictions: pd.DataFrame) -> pd.DataFrame:
    """Confidence by run, predicted class, and correctness — the warning-threshold view.

    The stakeholder-facing output is "warn when the model is confident the user
    is fatigued," so what matters is confidence *conditioned on the predicted
    class*: when the model says drowsy and is wrong, is it any less sure than
    when it is right?
    """
    rows = []
    for (run_name, pred_label), group in predictions.groupby(["run", "pred_label"]):
        correct = group["label"] == group["pred_label"]
        for is_correct in (True, False):
            subset = group[correct == is_correct]
            rows.append(
                {
                    "run": run_name,
                    "predicted_class": CLASS_NAMES[int(pred_label)],
                    "correct": is_correct,
                    "n_videos": len(subset),
                    "mean_confidence": subset["confidence"].mean() if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 4: the cold-start truncation probe
# ---------------------------------------------------------------------------


def truncate_video_windows(windows_df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Drop the first `fraction` of each video's windows and restart the clock.

    The kept windows are renumbered from 0 so that recomputed trend features
    treat the truncation point as the session start — exactly what a live aid
    would see if the user opened it mid-episode. Raw feature columns are
    untouched; only history is removed.
    """
    ordered = windows_df.sort_values(["video_id", "window_idx"]).copy()
    position = ordered.groupby("video_id").cumcount()
    video_sizes = ordered.groupby("video_id")["window_idx"].transform("size")
    kept = ordered[position >= np.floor(video_sizes * fraction).astype(int)].copy()
    kept["window_idx"] = kept.groupby("video_id").cumcount()
    return kept.reset_index(drop=True)


def run_cold_start_probe(
    windows_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    folds: list[dict[str, list[str]]],
    *,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit trend_logistic per fold and score truncated test sessions.

    Training is identical to A8's `trend_logistic` (full videos, same folds,
    same seed convention), so truncation 0.0 must reproduce the saved
    predictions. Only the *test* videos are truncated: the question is how the
    deployed model behaves on a session that starts mid-episode, not whether a
    model could be trained on truncated data.

    Also returns window-level accuracy by position decile (from the 0.0 run),
    the direct view of whether early-session windows are harder.
    """
    base_columns = temporal.get_sequence_feature_columns(windows_df)
    trend_columns = base_columns + [col for col in trend_df.columns if col not in windows_df.columns]

    prediction_tables: list[pd.DataFrame] = []
    position_tables: list[pd.DataFrame] = []
    for fold_idx, fold in enumerate(folds, start=1):
        logger.debug("Cold-start fold %d/%d", fold_idx, len(folds))
        train_df = trend_df[trend_df["subject_id"].isin(fold["train"])]
        model = _fit_trend_logistic(train_df, trend_columns, random_seed=random_seed + fold_idx)

        raw_test = windows_df[windows_df["subject_id"].isin(fold["test"])]
        for fraction in TRUNCATION_FRACTIONS:
            test_df = temporal.add_trend_features(truncate_video_windows(raw_test, fraction))
            window_predictions = _predict_windows(model, test_df, trend_columns)

            video_predictions = aggregate_window_predictions(window_predictions)
            video_predictions.insert(0, "truncation_fraction", fraction)
            video_predictions.insert(1, "fold", fold_idx)
            prediction_tables.append(video_predictions)

            if fraction == 0.0:
                position_tables.append(_position_correctness(window_predictions))

    cold_start = pd.concat(prediction_tables, ignore_index=True)
    positions = (
        pd.concat(position_tables, ignore_index=True)
        .groupby(["label", "position_decile"], as_index=False)
        .agg(n_windows=("correct", "size"), window_accuracy=("correct", "mean"))
    )
    return cold_start, positions


def summarize_cold_start(cold_start: pd.DataFrame) -> pd.DataFrame:
    """Per-class recall (and overall accuracy) at each truncation fraction."""
    rows = []
    for fraction, group in cold_start.groupby("truncation_fraction"):
        row = {"truncation_fraction": fraction, "accuracy": (group["label"] == group["pred_label"]).mean()}
        for class_idx, class_name in CLASS_NAMES.items():
            class_rows = group[group["label"] == class_idx]
            row[f"{class_name}_recall"] = (class_rows["pred_label"] == class_idx).mean()
            row[f"{class_name}_recall_n"] = int((class_rows["pred_label"] == class_idx).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def check_reproduction(cold_start: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    """Compare the 0.0-truncation run against the saved trend_logistic predictions.

    Returns a graded result rather than a bare boolean: on the original
    environment the probabilities should match to solver precision, while a
    different sklearn/python version can shift them by O(1e-3) without
    changing a single predicted label. Both facts belong in the manifest.
    """
    probe = cold_start[cold_start["truncation_fraction"] == 0.0].sort_values("video_id")
    saved = predictions[predictions["run"] == TREND_RUN_NAME].sort_values("video_id")
    if list(probe["video_id"]) != list(saved["video_id"]):
        return {"videos_match": False, "predictions_identical": False, "max_prob_diff": None}

    prob_columns = ["prob_0", "prob_1", "prob_2"]
    max_prob_diff = float(np.abs(probe[prob_columns].to_numpy() - saved[prob_columns].to_numpy()).max())
    return {
        "videos_match": True,
        "predictions_identical": bool((probe["pred_label"].to_numpy() == saved["pred_label"].to_numpy()).all()),
        "max_prob_diff": max_prob_diff,
        "probabilities_allclose_1e6": bool(max_prob_diff <= 1e-6),
    }


def _fit_trend_logistic(train_df: pd.DataFrame, feature_columns: list[str], *, random_seed: int) -> Pipeline:
    """The exact trend_logistic pipeline from A8 (see baselines._logistic_probabilities)."""
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
    return model


def _predict_windows(model: Pipeline, test_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Window-level class probabilities in the same shape baselines.predict_baseline emits."""
    raw_probabilities = model.predict_proba(test_df.loc[:, feature_columns])
    classifier = model.named_steps["classifier"]

    predictions = test_df[["subject_id", "video_id", "window_idx", "label"]].reset_index(drop=True).copy()
    probabilities = np.zeros((len(test_df), 3))
    for raw_idx, class_label in enumerate(classifier.classes_):
        probabilities[:, int(class_label)] = raw_probabilities[:, raw_idx]
    for class_idx in range(3):
        predictions[f"prob_{class_idx}"] = probabilities[:, class_idx]
    return predictions


def _position_correctness(window_predictions: pd.DataFrame) -> pd.DataFrame:
    """Window-level correctness with each window's position decile in its video."""
    result = window_predictions.copy()
    prob_columns = ["prob_0", "prob_1", "prob_2"]
    result["window_pred"] = result[prob_columns].to_numpy().argmax(axis=1)
    result["correct"] = result["window_pred"] == result["label"]
    video_sizes = result.groupby("video_id")["window_idx"].transform("max").clip(lower=1)
    result["position_decile"] = np.minimum((result["window_idx"] / video_sizes * 10).astype(int), 9)
    return result[["label", "position_decile", "correct"]]


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------

_TABLE_FILENAMES = {
    "low_vigilant_misses": "a9_low_vigilant_miss_slices.csv",
    "low_vigilant_summary": "a9_low_vigilant_miss_summary.csv",
    "corner_migration": "a9_corner_migration.csv",
    "calibration": "a9_calibration_by_class.csv",
    "cold_start_predictions": "a9_cold_start_video_predictions.csv",
    "cold_start_summary": "a9_cold_start_summary.csv",
    "window_position_accuracy": "a9_window_position_accuracy.csv",
}


def _write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(output_dir / _TABLE_FILENAMES[name], index=False)


def _write_manifest(
    output_dir: Path,
    *,
    windows_path: str | Path,
    folds_path: Path,
    random_seed: int,
    include_cold_start: bool,
    reproduction: dict[str, object] | None,
    started_at: str,
) -> None:
    manifest = {
        "experiment": "assignment9_failure_analysis",
        "dataset_path": str(windows_path),
        "folds_path": str(folds_path),
        "seed": random_seed,
        "analyses": list(_TABLE_FILENAMES.keys() if include_cold_start else list(_TABLE_FILENAMES)[:4]),
        "truncation_fractions": list(TRUNCATION_FRACTIONS) if include_cold_start else None,
        "reproduction_check": reproduction,
        "key_drift_features": list(KEY_DRIFT_FEATURES),
        "package_versions": package_versions(used_wandb=False),
        "git_sha": git_sha(),
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "a9_manifest.json").write_text(json.dumps(manifest, indent=2))


def _log_summary(tables: dict[str, pd.DataFrame], reproduction: dict[str, object] | None) -> None:
    misses = tables["low_vigilant_misses"]
    logger.info("Low-vigilant outcomes: %s", misses["outcome"].value_counts().to_dict())
    logger.info("Corner categories: %s", tables["corner_migration"]["category"].value_counts().to_dict())
    if "cold_start_summary" in tables:
        logger.info("Reproduction check: %s", reproduction)
        logger.info("Cold-start summary:\n%s", tables["cold_start_summary"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
