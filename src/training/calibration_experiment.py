"""Assignment 10 calibration experiment: can the low-vigilant confidence inversion be fixed?

Assignment 9 found that `trend_logistic`'s softmax confidence is *inverted* for
the low-vigilant class: when the model predicts low-vigilant and is wrong it is,
on average, MORE confident (0.473) than when it is right (0.437) — a gap of
-0.036. A raw softmax threshold therefore cannot gate a warning on the one class
the aid exists to catch early. A9's handoff named this the single highest-value
experiment left: test whether a calibration method fixes the inversion.

This module runs that test as a bounded head-to-head against the champion, under
the identical LOSO protocol (same saved folds, same seed convention, same
window->video probability averaging). No new representation, no deep net: the
A8/A9 evidence says added capacity loses to a linear model at n=60 subjects, so
both calibration methods stay linear.

Three runs, all on the A8 trend-feature table:

- `trend_logistic` — the champion, raw softmax. Must reproduce the saved A8
  predictions (reproduction anchor) before any new number is read.
- `trend_logistic_temp` — the champion's window logits divided by a temperature
  T before softmax. T is fit leak-free by subject-disjoint cross-fitting on the
  *training* subjects only (a T learned on one subject-half is applied only to
  the other), so the test subject never informs its own calibration.
- `trend_corn` — a linear CORN ordinal head (Shi et al. 2022): two chained
  binary logistic classifiers, P(y>0) and P(y>1 | y>0). Their product is
  rank-monotonic by construction, so the ordinal structure the softmax head
  ignores is enforced. Same impute+scale+logistic front end as the champion.

Temperature scaling changes only confidence, never the argmax — so accuracy /
QWK / recall are identical to the champion by construction, and only the
calibration columns move. CORN changes the decision rule, so its accuracy can
differ. Both are judged on the A9 per-class calibration gap (reused verbatim
from `failure_analysis.calibration_by_class`) and on ECE.

Outputs are written flat with an `a10_` prefix so all prior artifacts stay
untouched for comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training import failure_analysis, temporal
from training.artifacts import aggregate_window_predictions, git_sha, package_versions

logger = logging.getLogger("training.calibration_experiment")

CHAMPION_RUN = "trend_logistic"
TEMP_RUN = "trend_logistic_temp"
CORN_RUN = "trend_corn"
CLASS_NAMES = {0: "alert", 1: "low_vigilant", 2: "drowsy"}
ECE_BINS = 10


def main() -> None:
    """Command-line entry point for the Assignment 10 calibration experiment."""
    parser = argparse.ArgumentParser(description="Run the Assignment 10 calibration head-to-head on saved LOSO folds.")
    parser.add_argument("windows_path", type=Path, nargs="?", default=Path("data/frame_windows.parquet"))
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("--folds-path", type=Path, default=None, help="Defaults to <output_dir>/folds.json.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    run_calibration_experiment(
        args.windows_path,
        args.output_dir,
        folds_path=args.folds_path,
        random_seed=args.random_seed,
    )


def run_calibration_experiment(
    windows_path: str | Path,
    output_dir: str | Path,
    *,
    folds_path: str | Path | None = None,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run the three-way calibration comparison and write `a10_*` artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = Path(folds_path) if folds_path is not None else output_dir / "folds.json"
    started_at = datetime.now(timezone.utc).isoformat()

    windows_df = pd.read_parquet(windows_path)
    trend_df = temporal.add_trend_features(windows_df)
    folds = json.loads(folds_path.read_text())

    base_columns = temporal.get_sequence_feature_columns(windows_df)
    trend_columns = base_columns + [col for col in trend_df.columns if col not in windows_df.columns]

    logger.info("Calibration experiment on %d folds (%d features)", len(folds), len(trend_columns))
    video_predictions, temperatures = _score_all_folds(trend_df, trend_columns, folds, random_seed=random_seed)

    tables: dict[str, pd.DataFrame] = {}
    tables["video_predictions"] = video_predictions
    tables["metric_comparison"] = _metric_comparison(video_predictions)
    tables["calibration_by_class"] = failure_analysis.calibration_by_class(video_predictions)
    tables["calibration_gap"] = _calibration_gap(tables["calibration_by_class"])
    tables["ece"] = _ece_by_run(video_predictions)
    tables["temperatures"] = temperatures

    # Reproduction anchor: the champion run here must reproduce the saved A8 labels.
    saved = pd.read_csv(output_dir / "temporal_video_predictions.csv", dtype={"subject_id": str})
    reproduction = _check_champion_reproduction(video_predictions, saved)

    _write_tables(output_dir, tables)
    _write_manifest(
        output_dir,
        windows_path=windows_path,
        folds_path=folds_path,
        random_seed=random_seed,
        trend_columns=trend_columns,
        reproduction=reproduction,
        temperatures=temperatures,
        started_at=started_at,
    )
    _log_summary(tables, reproduction)
    return tables


# ---------------------------------------------------------------------------
# Per-fold scoring
# ---------------------------------------------------------------------------


def _score_all_folds(
    trend_df: pd.DataFrame,
    trend_columns: list[str],
    folds: list[dict[str, list[str]]],
    *,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all three heads per fold and return video predictions + per-fold temperatures."""
    prediction_tables: list[pd.DataFrame] = []
    temperature_rows: list[dict[str, float]] = []

    for fold_idx, fold in enumerate(folds, start=1):
        logger.debug("Fold %d/%d", fold_idx, len(folds))
        seed = random_seed + fold_idx
        train_df = trend_df[trend_df["subject_id"].isin(fold["train"])]
        test_df = trend_df[trend_df["subject_id"].isin(fold["test"])]

        champion = _fit_logistic(train_df, trend_columns, seed)
        temperature = _fit_temperature(train_df, trend_columns, seed)
        temperature_rows.append({"fold": fold_idx, "temperature": temperature})

        test_logits = champion.decision_function(test_df.loc[:, trend_columns])
        champion_probs = _softmax(test_logits)
        temp_probs = _softmax(test_logits / temperature)
        corn_probs = _predict_corn(_fit_corn(train_df, trend_columns, seed), test_df, trend_columns)

        for run_name, probs in ((CHAMPION_RUN, champion_probs), (TEMP_RUN, temp_probs), (CORN_RUN, corn_probs)):
            video = _aggregate(test_df, probs)
            video.insert(0, "run", run_name)
            video.insert(1, "fold", fold_idx)
            prediction_tables.append(video)

    return pd.concat(prediction_tables, ignore_index=True), pd.DataFrame(temperature_rows)


def _aggregate(test_df: pd.DataFrame, window_probs: np.ndarray) -> pd.DataFrame:
    """Attach window probabilities to metadata and collapse to video level (mean-pool)."""
    window_predictions = test_df[["subject_id", "video_id", "window_idx", "label"]].reset_index(drop=True).copy()
    for class_idx in range(3):
        window_predictions[f"prob_{class_idx}"] = window_probs[:, class_idx]
    return aggregate_window_predictions(window_predictions)


# ---------------------------------------------------------------------------
# The three heads
# ---------------------------------------------------------------------------


def _fit_logistic(train_df: pd.DataFrame, feature_columns: list[str], seed: int) -> Pipeline:
    """The exact champion pipeline from A8 (baselines._logistic_probabilities)."""
    model = _logistic_pipeline(seed)
    model.fit(train_df.loc[:, feature_columns], train_df["label"])
    return model


def _fit_corn(train_df: pd.DataFrame, feature_columns: list[str], seed: int) -> dict[str, Pipeline]:
    """Fit the two chained binary classifiers of a linear CORN ordinal head.

    - `gt0`: P(y>0), trained on all rows with target (label >= 1).
    - `gt1_given_gt0`: P(y>1 | y>0), trained ONLY on rows where label >= 1 with
      target (label == 2). Conditioning on y>0 is what makes P(y>1) <= P(y>0) by
      construction (Shi et al. 2022), so the probabilities stay rank-monotonic.
    """
    gt0 = _logistic_pipeline(seed)
    gt0.fit(train_df.loc[:, feature_columns], (train_df["label"] >= 1).astype(int))

    conditional = train_df[train_df["label"] >= 1]
    gt1 = _logistic_pipeline(seed)
    gt1.fit(conditional.loc[:, feature_columns], (conditional["label"] == 2).astype(int))
    return {"gt0": gt0, "gt1_given_gt0": gt1}


def _predict_corn(corn: dict[str, Pipeline], test_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Chain the two binary heads into a monotone 3-class probability table."""
    features = test_df.loc[:, feature_columns]
    p_gt0 = _positive_proba(corn["gt0"], features)
    p_gt1_given_gt0 = _positive_proba(corn["gt1_given_gt0"], features)

    probs = np.column_stack(
        [
            1.0 - p_gt0,  # P(y=0)
            p_gt0 * (1.0 - p_gt1_given_gt0),  # P(y=1)
            p_gt0 * p_gt1_given_gt0,  # P(y=2)
        ]
    )
    return probs


def _positive_proba(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """P(positive) from a binary logistic pipeline, robust to a degenerate single-class fit."""
    classifier = model.named_steps["classifier"]
    proba = model.predict_proba(features)
    if 1 in classifier.classes_:
        return proba[:, list(classifier.classes_).index(1)]
    # Train fold saw only the negative class: positive probability is 0.
    return np.zeros(len(features))


def _logistic_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ]
    )


# ---------------------------------------------------------------------------
# Temperature scaling (leak-free, subject-disjoint cross-fit)
# ---------------------------------------------------------------------------


def _fit_temperature(train_df: pd.DataFrame, feature_columns: list[str], seed: int) -> float:
    """Fit one temperature T on out-of-fold window logits from the TRAINING subjects only.

    The training subjects are split into two disjoint halves. A model fit on
    half A produces logits for half B and vice versa, so every logit T sees was
    produced by a model that never trained on that subject. T then minimizes the
    window-level negative log-likelihood of those out-of-fold logits. The held-out
    LOSO test subject is never involved, so its calibration is honest.
    """
    subjects = sorted(train_df["subject_id"].unique())
    half = len(subjects) // 2
    split_a, split_b = set(subjects[:half]), set(subjects[half:])

    logits_parts, label_parts = [], []
    for fit_subjects, eval_subjects in ((split_a, split_b), (split_b, split_a)):
        fit_df = train_df[train_df["subject_id"].isin(fit_subjects)]
        eval_df = train_df[train_df["subject_id"].isin(eval_subjects)]
        if fit_df["label"].nunique() < 2 or eval_df.empty:
            continue
        model = _fit_logistic(fit_df, feature_columns, seed)
        logits_parts.append(model.decision_function(eval_df.loc[:, feature_columns]))
        label_parts.append(eval_df["label"].to_numpy())

    if not logits_parts:
        return 1.0  # Not enough class diversity to calibrate; leave logits untouched.

    logits = np.vstack(logits_parts)
    labels = np.concatenate(label_parts)
    result = minimize_scalar(
        lambda t: _nll(logits / t, labels),
        bounds=(0.05, 100.0),
        method="bounded",
    )
    return float(result.x)


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of integer labels under softmax(logits)."""
    probs = _softmax(logits)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------


def _metric_comparison(video_predictions: pd.DataFrame) -> pd.DataFrame:
    """Video-level QWK / rank-MAE / accuracy / macro-F1 and low-vigilant recall per run."""
    from evaluation import metrics

    rows = []
    for run_name, group in video_predictions.groupby("run"):
        summary = metrics.classification_metric_summary(group["label"], group["pred_label"])
        low_vig = group[group["label"] == 1]
        summary["run"] = run_name
        summary["low_vigilant_recall_n"] = int((low_vig["pred_label"] == 1).sum())
        summary["low_vigilant_recall"] = float((low_vig["pred_label"] == 1).mean())
        rows.append(summary)
    return pd.DataFrame(rows)[
        ["run", "qwk", "rank_mae", "accuracy", "macro_f1", "low_vigilant_recall", "low_vigilant_recall_n"]
    ]


def _calibration_gap(calibration_by_class: pd.DataFrame) -> pd.DataFrame:
    """Correct-minus-wrong confidence gap per (run, predicted class) — negative = inverted."""
    pivoted = calibration_by_class.pivot_table(
        index=["run", "predicted_class"], columns="correct", values="mean_confidence"
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={True: "conf_correct", False: "conf_wrong"})
    pivoted["gap"] = pivoted["conf_correct"] - pivoted["conf_wrong"]
    return pivoted.sort_values(["run", "predicted_class"]).reset_index(drop=True)


def _ece_by_run(video_predictions: pd.DataFrame) -> pd.DataFrame:
    """Expected calibration error of video-level confidence (max prob) per run."""
    rows = []
    for run_name, group in video_predictions.groupby("run"):
        confidence = group["confidence"].to_numpy()
        correct = (group["label"] == group["pred_label"]).to_numpy().astype(float)
        rows.append({"run": run_name, "ece": _ece(confidence, correct), "n_videos": len(group)})
    return pd.DataFrame(rows)


def _ece(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Standard binned expected calibration error."""
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, ECE_BINS - 1)
    total = 0.0
    for b in range(ECE_BINS):
        mask = bin_idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def _check_champion_reproduction(video_predictions: pd.DataFrame, saved: pd.DataFrame) -> dict[str, object]:
    """The champion run here must reproduce the saved A8 trend_logistic video labels."""
    probe = video_predictions[video_predictions["run"] == CHAMPION_RUN].sort_values("video_id")
    saved_champ = saved[saved["run"] == CHAMPION_RUN].sort_values("video_id")
    if list(probe["video_id"]) != list(saved_champ["video_id"]):
        return {"videos_match": False, "predictions_identical": False, "max_prob_diff": None}

    prob_columns = ["prob_0", "prob_1", "prob_2"]
    max_prob_diff = float(np.abs(probe[prob_columns].to_numpy() - saved_champ[prob_columns].to_numpy()).max())
    return {
        "videos_match": True,
        "predictions_identical": bool((probe["pred_label"].to_numpy() == saved_champ["pred_label"].to_numpy()).all()),
        "max_prob_diff": max_prob_diff,
        "probabilities_allclose_1e6": bool(max_prob_diff <= 1e-6),
    }


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------

_TABLE_FILENAMES = {
    "video_predictions": "a10_calibration_video_predictions.csv",
    "metric_comparison": "a10_metric_comparison.csv",
    "calibration_by_class": "a10_calibration_by_class.csv",
    "calibration_gap": "a10_calibration_gap.csv",
    "ece": "a10_ece.csv",
    "temperatures": "a10_temperatures.csv",
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
    trend_columns: list[str],
    reproduction: dict[str, object],
    temperatures: pd.DataFrame,
    started_at: str,
) -> None:
    manifest = {
        "experiment": "assignment10_calibration",
        "dataset_path": str(windows_path),
        "folds_path": str(folds_path),
        "seed": random_seed,
        "runs": [CHAMPION_RUN, TEMP_RUN, CORN_RUN],
        "feature_count": len(trend_columns),
        "reproduction_check": reproduction,
        "temperature_summary": {
            "mean": float(temperatures["temperature"].mean()),
            "min": float(temperatures["temperature"].min()),
            "max": float(temperatures["temperature"].max()),
        },
        "ece_bins": ECE_BINS,
        "package_versions": package_versions(used_wandb=False),
        "git_sha": git_sha(),
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "a10_calibration_manifest.json").write_text(json.dumps(manifest, indent=2))


def _log_summary(tables: dict[str, pd.DataFrame], reproduction: dict[str, object]) -> None:
    logger.info("Reproduction check: %s", reproduction)
    logger.info("Metric comparison:\n%s", tables["metric_comparison"].round(3).to_string(index=False))
    low_vig = tables["calibration_gap"][tables["calibration_gap"]["predicted_class"] == "low_vigilant"]
    logger.info("Low-vigilant calibration gap by run:\n%s", low_vig.round(4).to_string(index=False))
    logger.info("ECE by run:\n%s", tables["ece"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
