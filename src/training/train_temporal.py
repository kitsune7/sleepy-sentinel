"""Run the Assignment 8 temporal experiments on the saved LOSO folds.

This is a bounded head-to-head, not a new pipeline. It reuses the frozen
comparison scaffolding — the saved `outputs/folds.json`, the fit-on-train-only
preprocessing, the video-level metric bundle — and adds exactly three runs:

- `logistic_full_anchor`: re-runs the A7 champion on the loaded folds. Its
  numbers must reproduce `outputs/fold_metrics.csv` exactly; if they do, every
  other run in this file is directly comparable to all A5–A7 evidence.
- `trend_logistic`: the same logistic model fed the same window rows, plus the
  causal cross-window trend features from `training.temporal`. If temporal
  signal exists and is simple, this is the cheapest way to capture it.
- `gru_sequence`: a small many-to-one GRU over each video's window sequence.
  If temporal signal exists but is not a simple trend, this is the bounded
  sequence-model probe (echoing the dataset's published HM-LSTM direction).

Outputs are written FLAT into the output dir with a `temporal_` prefix so the
Assignment 7 artifacts stay untouched for comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation import metrics
from training import baselines, temporal
from training.config import RunResult
from training.dataset import fit_preprocessor
from training.train import _git_sha, _package_versions, set_random_seeds

logger = logging.getLogger("training.train_temporal")

GRU_RUN_NAME = "gru_sequence"
TREND_RUN_NAME = "trend_logistic"
ANCHOR_RUN_NAME = "logistic_full_anchor"

GRU_HIDDEN_DIM = 32
GRU_DROPOUT = 0.25
GRU_WEIGHT_DECAY = 1e-4
GRU_LEARNING_RATE = 1e-3
GRU_EPOCHS = 40
GRU_BATCH_SIZE = 16
GRU_PATIENCE = 8


def main() -> None:
    """Command-line entry point for the temporal experiments."""
    parser = argparse.ArgumentParser(description="Run Assignment 8 temporal experiments on saved LOSO folds.")
    parser.add_argument("windows_path", type=Path, nargs="?", default=Path("data/frame_windows.parquet"))
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("--folds-path", type=Path, default=None, help="Defaults to <output_dir>/folds.json.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-gru", action="store_true", help="Run only the logistic-path experiments.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    run_temporal_experiments(
        args.windows_path,
        args.output_dir,
        folds_path=args.folds_path,
        random_seed=args.random_seed,
        include_gru=not args.skip_gru,
    )


def run_temporal_experiments(
    windows_path: str | Path,
    output_dir: str | Path,
    *,
    folds_path: str | Path | None = None,
    random_seed: int = 42,
    include_gru: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the anchor, trend-logistic, and GRU experiments and write artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = Path(folds_path) if folds_path is not None else output_dir / "folds.json"

    started_at = datetime.now(timezone.utc).isoformat()
    windows_df = pd.read_parquet(windows_path)
    folds = json.loads(Path(folds_path).read_text())

    base_columns = temporal.get_sequence_feature_columns(windows_df)
    trend_df = temporal.add_trend_features(windows_df)
    trend_columns = base_columns + [
        col for col in trend_df.columns if col not in windows_df.columns
    ]

    anchor_spec = baselines.BaselineSpec(ANCHOR_RUN_NAME, tuple(base_columns))
    trend_spec = baselines.BaselineSpec(TREND_RUN_NAME, tuple(trend_columns))

    logger.info("Temporal experiments on %d folds (%d videos)", len(folds), windows_df["video_id"].nunique())
    logger.info("Base features: %d, with trends: %d", len(base_columns), len(trend_columns))

    results: list[RunResult] = []
    for fold_idx, fold in enumerate(folds, start=1):
        logger.debug("Fold %d/%d", fold_idx, len(folds))
        results.append(
            _run_logistic(windows_df, fold, anchor_spec, fold_idx, random_seed)
        )
        results.append(
            _run_logistic(trend_df, fold, trend_spec, fold_idx, random_seed)
        )
        if include_gru:
            results.append(
                _run_gru(windows_df, base_columns, fold, fold_idx, random_seed)
            )

    tables = _write_outputs(output_dir, results)
    _write_trend_coefficients(trend_df, trend_columns, folds, random_seed, output_dir)
    _write_manifest(
        output_dir,
        windows_path=windows_path,
        folds_path=folds_path,
        random_seed=random_seed,
        include_gru=include_gru,
        n_folds=len(folds),
        base_columns=base_columns,
        trend_columns=trend_columns,
        started_at=started_at,
    )

    _log_summary(tables["metric_summary"])
    return tables


def _run_logistic(
    table: pd.DataFrame,
    fold: dict[str, list[str]],
    spec: baselines.BaselineSpec,
    fold_idx: int,
    random_seed: int,
) -> RunResult:
    """Fit one logistic-path run for one fold, mirroring the train.py protocol."""
    train_df = table[table["subject_id"].isin(fold["train"])]
    test_df = table[table["subject_id"].isin(fold["test"])]

    test_predictions = baselines.predict_baseline(train_df, test_df, spec, random_seed=random_seed + fold_idx)
    video_predictions = _aggregate_windows(test_predictions)
    video_predictions.insert(0, "run", spec.name)
    video_predictions.insert(1, "fold", fold_idx)

    fold_metrics = metrics.classification_metric_summary(video_predictions["label"], video_predictions["pred_label"])
    fold_metrics.update({"run": spec.name, "fold": fold_idx, "n_videos": len(video_predictions)})
    return RunResult(run_name=spec.name, fold=fold_idx, video_predictions=video_predictions, fold_metrics=fold_metrics)


def _run_gru(
    windows_df: pd.DataFrame,
    feature_columns: list[str],
    fold: dict[str, list[str]],
    fold_idx: int,
    random_seed: int,
) -> RunResult:
    """Train the GRU for one fold with validation-QWK early stopping."""
    set_random_seeds(random_seed + fold_idx)

    train_windows = windows_df[windows_df["subject_id"].isin(fold["train"])]
    preprocessor = fit_preprocessor(train_windows[feature_columns])

    split_sequences = {
        split_name: _prepare_sequences(windows_df, fold[split_name], feature_columns, preprocessor)
        for split_name in ("train", "validation", "test")
    }

    model = temporal.GruVideoClassifier(
        input_dim=len(feature_columns),
        hidden_dim=GRU_HIDDEN_DIM,
        dropout=GRU_DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LEARNING_RATE, weight_decay=GRU_WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss(weight=_balanced_class_weights(split_sequences["train"]["label"]))

    generator = torch.Generator().manual_seed(random_seed + fold_idx)
    history = []
    best_epoch, best_score, best_state = 0, float("-inf"), copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch_idx in range(1, GRU_EPOCHS + 1):
        train_metrics = _train_gru_epoch(model, split_sequences["train"], optimizer, loss_fn, generator)
        validation_metrics = _evaluate_gru(model, split_sequences["validation"], loss_fn)
        history.append({"train": train_metrics, "validation": validation_metrics})

        score = validation_metrics["qwk"]
        score = float("-inf") if np.isnan(score) else score
        if best_epoch == 0 or score > best_score:
            best_epoch, best_score = epoch_idx, score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= GRU_PATIENCE:
            break

    model.load_state_dict(best_state)
    video_predictions = _predict_gru_videos(model, split_sequences["test"])
    video_predictions.insert(0, "run", GRU_RUN_NAME)
    video_predictions.insert(1, "fold", fold_idx)

    fold_metrics = metrics.classification_metric_summary(video_predictions["label"], video_predictions["pred_label"])
    fold_metrics.update(
        {
            "run": GRU_RUN_NAME,
            "fold": fold_idx,
            "n_videos": len(video_predictions),
            "best_epoch": best_epoch,
            "stopped_early": len(history) < GRU_EPOCHS,
        }
    )
    learning_curve = [
        {
            "run": GRU_RUN_NAME,
            "fold": fold_idx,
            "epoch": epoch_idx,
            "selected_checkpoint": epoch_idx == best_epoch,
            **{f"train_{name}": value for name, value in epoch_metrics["train"].items()},
            **{f"validation_{name}": value for name, value in epoch_metrics["validation"].items()},
        }
        for epoch_idx, epoch_metrics in enumerate(history, start=1)
    ]
    return RunResult(
        run_name=GRU_RUN_NAME,
        fold=fold_idx,
        video_predictions=video_predictions,
        fold_metrics=fold_metrics,
        learning_curve=learning_curve,
    )


def _prepare_sequences(
    windows_df: pd.DataFrame,
    subject_ids: list[str],
    feature_columns: list[str],
    preprocessor,
) -> pd.DataFrame:
    """Build standardized per-video sequences for one split."""
    split_df = windows_df[windows_df["subject_id"].astype(str).isin(subject_ids)].copy()
    # Cast first: integer columns like yawn_count would otherwise reject scaled floats.
    split_df[feature_columns] = split_df[feature_columns].astype("float64")
    split_df[feature_columns] = preprocessor.transform(split_df[feature_columns])
    return temporal.build_video_sequences(split_df, feature_columns)


def _balanced_class_weights(labels: pd.Series) -> torch.Tensor:
    counts = labels.value_counts().reindex([0, 1, 2]).fillna(0).to_numpy(dtype=float)
    weights = np.where(counts > 0, len(labels) / (3 * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def _train_gru_epoch(model, sequences_df, optimizer, loss_fn, generator) -> dict[str, float]:
    model.train()
    order = torch.randperm(len(sequences_df), generator=generator).tolist()
    total_loss, total = 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for start in range(0, len(order), GRU_BATCH_SIZE):
        batch = sequences_df.iloc[order[start : start + GRU_BATCH_SIZE]]
        padded, lengths = temporal.pad_sequences(list(batch["features"]))
        targets = torch.tensor(batch["label"].to_numpy(), dtype=torch.long)

        optimizer.zero_grad()
        logits = model(padded, lengths)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(batch)
        total += len(batch)
        y_true.extend(targets.tolist())
        y_pred.extend(logits.argmax(dim=1).tolist())

    return {"loss": total_loss / total, **metrics.classification_metric_summary(y_true, y_pred)}


def _evaluate_gru(model, sequences_df, loss_fn) -> dict[str, float]:
    model.eval()
    padded, lengths = temporal.pad_sequences(list(sequences_df["features"]))
    targets = torch.tensor(sequences_df["label"].to_numpy(), dtype=torch.long)
    with torch.no_grad():
        logits = model(padded, lengths)
        loss = loss_fn(logits, targets)
    return {
        "loss": float(loss.item()),
        **metrics.classification_metric_summary(targets.tolist(), logits.argmax(dim=1).tolist()),
    }


def _predict_gru_videos(model, sequences_df) -> pd.DataFrame:
    model.eval()
    padded, lengths = temporal.pad_sequences(list(sequences_df["features"]))
    with torch.no_grad():
        probabilities = torch.softmax(model(padded, lengths), dim=1).numpy()

    video_predictions = sequences_df[["subject_id", "video_id", "label"]].reset_index(drop=True).copy()
    video_predictions["window_count"] = sequences_df["length"].to_numpy()
    for class_idx in range(probabilities.shape[1]):
        video_predictions[f"prob_{class_idx}"] = probabilities[:, class_idx]
    video_predictions["pred_label"] = probabilities.argmax(axis=1)
    video_predictions["confidence"] = probabilities.max(axis=1)
    return video_predictions.sort_values("video_id").reset_index(drop=True)


def _aggregate_windows(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse window predictions to video predictions (mirrors train.py)."""
    from training.train import aggregate_window_predictions

    return aggregate_window_predictions(predictions_df)


def _write_outputs(output_dir: Path, results: list[RunResult]) -> dict[str, pd.DataFrame]:
    video_predictions = pd.concat([r.video_predictions for r in results], ignore_index=True)
    fold_metrics = pd.DataFrame([r.fold_metrics for r in results])
    metric_summary = fold_metrics.groupby("run")[["qwk", "rank_mae", "accuracy", "macro_f1"]].agg(["mean", "std"])

    confusion_tables = []
    for result in results:
        table = metrics.confusion_long_form(result.video_predictions["label"], result.video_predictions["pred_label"])
        table.insert(0, "run", result.run_name)
        table.insert(1, "fold", result.fold)
        confusion_tables.append(table)
    confusion = pd.concat(confusion_tables, ignore_index=True)

    diagnostics_tables = []
    for (run_name, fold), group in video_predictions.groupby(["run", "fold"]):
        confidence = metrics.confidence_summary(group)
        confidence.insert(0, "diagnostic", "confidence_by_correctness")
        confidence.insert(1, "run", run_name)
        confidence.insert(2, "fold", fold)
        diagnostics_tables.append(confidence)
    diagnostics = pd.concat(diagnostics_tables, ignore_index=True)

    learning_curves = pd.DataFrame([row for result in results for row in result.learning_curve])

    fold_metrics.to_csv(output_dir / "temporal_fold_metrics.csv", index=False)
    video_predictions.to_csv(output_dir / "temporal_video_predictions.csv", index=False)
    metric_summary.to_csv(output_dir / "temporal_metric_summary.csv")
    confusion.to_csv(output_dir / "temporal_confusion_matrices.csv", index=False)
    diagnostics.to_csv(output_dir / "temporal_diagnostics.csv", index=False)
    learning_curves.to_csv(output_dir / "temporal_learning_curves.csv", index=False)

    return {
        "fold_metrics": fold_metrics,
        "video_predictions": video_predictions,
        "metric_summary": metric_summary,
        "confusion_matrices": confusion,
        "diagnostics": diagnostics,
        "learning_curves": learning_curves,
    }


def _write_trend_coefficients(
    trend_df: pd.DataFrame,
    trend_columns: list[str],
    folds: list[dict[str, list[str]]],
    random_seed: int,
    output_dir: Path,
) -> None:
    """Refit trend_logistic per fold and save standardized coefficient summaries."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    class_names = {0: "alert", 1: "low_vigilant", 2: "drowsy"}
    per_fold_coefs = []
    for fold_idx, fold in enumerate(folds, start=1):
        train_df = trend_df[trend_df["subject_id"].isin(fold["train"])]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed + fold_idx),
                ),
            ]
        )
        model.fit(train_df.loc[:, trend_columns], train_df["label"])
        classifier = model.named_steps["classifier"]
        coefs = np.zeros((3, len(trend_columns)))
        for raw_idx, class_label in enumerate(classifier.classes_):
            coefs[int(class_label)] = classifier.coef_[raw_idx]
        per_fold_coefs.append(coefs)

    stacked = np.stack(per_fold_coefs)
    rows = [
        {
            "class": class_name,
            "feature": feature,
            "coef_mean": stacked[:, class_idx, feat_idx].mean(),
            "coef_std": stacked[:, class_idx, feat_idx].std(),
            "abs_coef_mean": np.abs(stacked[:, class_idx, feat_idx]).mean(),
        }
        for class_idx, class_name in class_names.items()
        for feat_idx, feature in enumerate(trend_columns)
    ]
    pd.DataFrame(rows).to_csv(output_dir / "temporal_trend_coefficients.csv", index=False)


def _write_manifest(
    output_dir: Path,
    *,
    windows_path: str | Path,
    folds_path: Path,
    random_seed: int,
    include_gru: bool,
    n_folds: int,
    base_columns: list[str],
    trend_columns: list[str],
    started_at: str,
) -> None:
    manifest = {
        "experiment": "assignment8_temporal",
        "dataset_path": str(windows_path),
        "folds_path": str(folds_path),
        "n_folds": n_folds,
        "seed": random_seed,
        "runs": [ANCHOR_RUN_NAME, TREND_RUN_NAME] + ([GRU_RUN_NAME] if include_gru else []),
        "base_feature_count": len(base_columns),
        "trend_feature_count": len(trend_columns),
        "trend_config": {
            "trend_columns": list(temporal.TREND_COLUMNS),
            "past_windows": temporal.TREND_PAST_WINDOWS,
        },
        "gru_config": {
            "hidden_dim": GRU_HIDDEN_DIM,
            "dropout": GRU_DROPOUT,
            "weight_decay": GRU_WEIGHT_DECAY,
            "learning_rate": GRU_LEARNING_RATE,
            "epochs": GRU_EPOCHS,
            "batch_size": GRU_BATCH_SIZE,
            "early_stopping_patience": GRU_PATIENCE,
        },
        "package_versions": _package_versions(used_wandb=False),
        "git_sha": _git_sha(),
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "temporal_manifest.json").write_text(json.dumps(manifest, indent=2))


def _log_summary(metric_summary: pd.DataFrame) -> None:
    logger.info("Final video-level metrics:")
    for run_name, row in metric_summary.iterrows():
        logger.info(
            "  %-22s QWK %.3f +/- %.3f   rank MAE %.3f",
            run_name,
            row[("qwk", "mean")],
            row[("qwk", "std")],
            row[("rank_mae", "mean")],
        )


if __name__ == "__main__":
    main()
