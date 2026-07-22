"""Train and evaluate the alertness baseline across subject-wise folds.

This file owns the Stage 5 and Stage 6 orchestration:
- load the window-level dataset
- ask `training.splits` for subject-wise CV folds
- ask `training.dataset` to prepare fold-specific train/validation/test objects
- run the diagnostic baselines and, on request, one regularized MLP
- aggregate window predictions up to video-level predictions
- hand structured results to `training.artifacts` for writing

Separation of concerns:
- feature engineering -> `data_prep.windows`
- split logic -> `training.splits`
- preprocessing -> `training.dataset`
- metric math -> `evaluation.metrics`
- experiment tracking -> `training.tracking` (never `wandb.*` directly here)
- output writing -> `training.artifacts`

Output layout: results are written FLAT into the provided `output_dir` (default
`outputs/`). We deliberately do NOT split into `outputs/latest/` plus
`outputs/runs/<id>/`; a single directory per command is simpler to diff and
reason about, and run identity lives in `manifest.json`.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluation import metrics
from training import artifacts, baselines, dataset, models, splits, tracking
from training.artifacts import aggregate_window_predictions
from training.config import (
    CrossValidationConfig,
    ModelConfig,
    RunResult,
    TrainingConfig,
)
from training.dataset import FoldDatasets
from training.tracking import NullTracker, Tracker

logger = logging.getLogger("training.train")

MLP_RUN_NAME = "mlp_regularized"
HEADLINE_RUNS = ("perclos", MLP_RUN_NAME)


def main() -> None:
    """Command-line entry point for training and evaluation."""
    parser = argparse.ArgumentParser(description="Run subject-wise alertness cross-validation.")
    parser.add_argument("windows_path", type=Path, nargs="?", default=Path("data/frame_windows.parquet"))
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-splits", type=int, default=None, help="Override LOSO with grouped K-fold CV.")
    parser.add_argument("--validation-subject-count", type=int, default=9)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Validation-QWK patience; negative disables.",
    )
    parser.add_argument("--include-mlp", action="store_true", help="Train the regularized MLP run as well.")
    parser.add_argument("--verbose", action="store_true", help="Log fold-by-fold progress.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Enable W&B logging for the CV experiment.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B team or username.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group name for the CV run.")
    parser.add_argument(
        "--wandb-per-fold-runs",
        action="store_true",
        help="Restore one separate W&B run per fold (default: one run for the whole command).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    set_random_seeds(args.random_seed)

    tracker = tracking.build_tracker(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        group=args.wandb_group,
        per_fold_runs=args.wandb_per_fold_runs,
    )

    run_cross_validation(
        args.windows_path,
        args.output_dir,
        n_splits=args.n_splits,
        validation_subject_count=args.validation_subject_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        random_seed=args.random_seed,
        include_mlp=args.include_mlp,
        tracker=tracker,
    )


def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_cross_validation(
    windows_path: str | Path,
    output_dir: str | Path,
    *,
    n_splits: int | None = None,
    validation_subject_count: int = 9,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    early_stopping_patience: int | None = 8,
    random_seed: int = 42,
    include_mlp: bool = False,
    tracker: Tracker | None = None,
) -> dict[str, object]:
    """Run the full CV experiment and write the flat output contract."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = tracker if tracker is not None else NullTracker()

    started_at = datetime.now(timezone.utc).isoformat()
    dataset_path = str(windows_path)
    windows_df = pd.read_parquet(windows_path)

    cv_config = CrossValidationConfig(
        n_splits=n_splits,
        validation_subject_count=validation_subject_count,
        random_seed=random_seed,
    )
    model_config = ModelConfig()
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        seed=random_seed,
    )

    base_folds = (
        splits.make_loso_folds(windows_df)
        if cv_config.n_splits is None
        else splits.make_group_folds(windows_df, n_splits=cv_config.n_splits, random_seed=cv_config.random_seed)
    )
    folds = [
        splits.add_validation_subjects(
            fold,
            validation_subject_count=cv_config.validation_subject_count,
            random_seed=cv_config.random_seed + fold_idx,
        )
        for fold_idx, fold in enumerate(base_folds)
    ]
    splits.save_fold_assignments(folds, output_dir / "folds.json")

    baseline_specs = baselines.available_baselines(windows_df)
    run_names = [spec.name for spec in baseline_specs] + ([MLP_RUN_NAME] if include_mlp else [])
    _log_run_header(windows_df, dataset_path, cv_config, run_names, output_dir, len(folds))

    tracker.start_experiment(
        {
            "dataset_path": dataset_path,
            "cv_strategy": cv_config.strategy,
            "n_splits": len(folds),
            "validation_subject_count": cv_config.validation_subject_count,
            "seed": random_seed,
            "include_mlp": include_mlp,
            **_dataclass_dict(model_config),
            **_dataclass_dict(training_config),
        }
    )

    results: list[RunResult] = []
    split_summaries = []

    for fold_idx, fold in enumerate(folds, start=1):
        splits.assert_disjoint_subjects(fold)
        split_summary = splits.describe_split(windows_df, fold)
        split_summary.insert(0, "fold", fold_idx)
        split_summaries.append(split_summary)
        logger.debug("Fold %d/%d", fold_idx, len(folds))

        train_df = splits.filter_split(windows_df, fold["train"])
        validation_df = splits.filter_split(windows_df, fold["validation"])
        test_df = splits.filter_split(windows_df, fold["test"])

        for baseline_spec in baseline_specs:
            result = _run_baseline(baseline_spec, train_df, test_df, fold_idx, random_seed)
            tracker.log_fold(result.run_name, fold_idx, result.fold_metrics)
            results.append(result)

        if include_mlp:
            fold_data = dataset.prepare_fold_datasets(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
            )
            result = _run_mlp(fold_data, fold_idx, model_config, training_config, tracker)
            tracker.log_fold(result.run_name, fold_idx, result.fold_metrics)
            results.append(result)

    split_summaries_df = pd.concat(split_summaries, ignore_index=True)
    manifest = _build_manifest(
        windows_path=windows_path,
        cv_config=cv_config,
        model_config=model_config,
        training_config=training_config,
        include_mlp=include_mlp,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc).isoformat(),
        used_wandb=not isinstance(tracker, NullTracker),
    )

    tables = artifacts.write_run_outputs(
        output_dir,
        results=results,
        split_summaries=split_summaries_df,
        manifest=manifest,
    )

    tracker.log_summary(_summary_metrics(tables["metric_summary"]))
    tracker.finish()

    _log_run_footer(tables, output_dir)
    return {"folds": folds, "manifest": manifest, **tables}


def _run_baseline(
    spec: baselines.BaselineSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_idx: int,
    random_seed: int,
) -> RunResult:
    """Fit one baseline for one fold and package it as a RunResult."""
    test_predictions = baselines.predict_baseline(train_df, test_df, spec, random_seed=random_seed + fold_idx)
    video_predictions = aggregate_window_predictions(test_predictions)
    video_predictions.insert(0, "run", spec.name)
    video_predictions.insert(1, "fold", fold_idx)

    fold_metrics = metrics.classification_metric_summary(video_predictions["label"], video_predictions["pred_label"])
    fold_metrics.update({"run": spec.name, "fold": fold_idx, "n_videos": len(video_predictions)})
    return RunResult(run_name=spec.name, fold=fold_idx, video_predictions=video_predictions, fold_metrics=fold_metrics)


def _run_mlp(
    fold_data: FoldDatasets,
    fold_idx: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tracker: Tracker,
) -> RunResult:
    """Train the MLP for one fold and package it as a RunResult."""
    fold_training_config = TrainingConfig(
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        early_stopping_patience=training_config.early_stopping_patience,
        early_stopping_metric=training_config.early_stopping_metric,
        early_stopping_min_delta=training_config.early_stopping_min_delta,
        seed=training_config.seed + fold_idx,
    )
    model_result = train_fold(
        fold_data,
        model_config,
        fold_training_config,
        tracker=tracker,
        run_name=MLP_RUN_NAME,
        fold=fold_idx,
    )

    learning_curve = [
        {
            "run": MLP_RUN_NAME,
            "fold": fold_idx,
            "epoch": epoch_idx,
            "selected_checkpoint": epoch_idx == model_result["best_epoch"],
            **{f"train_{name}": value for name, value in epoch_metrics["train"].items()},
            **{f"validation_{name}": value for name, value in epoch_metrics["validation"].items()},
        }
        for epoch_idx, epoch_metrics in enumerate(model_result["history"], start=1)
    ]

    test_predictions = predict_split(model_result["model"], fold_data, "test")
    video_predictions = aggregate_window_predictions(test_predictions)
    video_predictions.insert(0, "run", MLP_RUN_NAME)
    video_predictions.insert(1, "fold", fold_idx)

    fold_metrics = metrics.classification_metric_summary(video_predictions["label"], video_predictions["pred_label"])
    fold_metrics.update(
        {
            "run": MLP_RUN_NAME,
            "fold": fold_idx,
            "n_videos": len(video_predictions),
            "best_epoch": model_result["best_epoch"],
            "stopped_early": model_result["stopped_early"],
        }
    )
    return RunResult(
        run_name=MLP_RUN_NAME,
        fold=fold_idx,
        video_predictions=video_predictions,
        fold_metrics=fold_metrics,
        learning_curve=learning_curve,
    )


def _epoch_metric_summary(total_loss: float, total: int, y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Build the metric bundle logged for one train/validation epoch."""
    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "qwk": 0.0, "rank_mae": 0.0, "macro_f1": 0.0}

    metric_summary = metrics.classification_metric_summary(y_true, y_pred)
    return {"loss": total_loss / total, **metric_summary}


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
) -> dict[str, float]:
    """Run one training epoch and return training metrics."""
    model.train()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = len(batch_y)
        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item()) * batch_size
        total += batch_size
        y_true.extend(batch_y.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    return _epoch_metric_summary(total_loss, total, y_true, y_pred)


def evaluate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
) -> dict[str, float]:
    """Run one validation or test pass and return metrics."""
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            batch_size = len(batch_y)
            predictions = logits.argmax(dim=1)
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            y_true.extend(batch_y.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    return _epoch_metric_summary(total_loss, total, y_true, y_pred)


def train_fold(
    fold_data: FoldDatasets,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    tracker: Tracker | None = None,
    run_name: str = MLP_RUN_NAME,
    fold: int = 0,
) -> dict[str, object]:
    """Train one model for one subject-wise fold."""
    tracker = tracker if tracker is not None else NullTracker()
    set_random_seeds(training_config.seed)
    dataloaders = dataset.make_dataloaders(fold_data, batch_size=training_config.batch_size)
    model = models.build_cross_entropy_mlp(
        input_dim=len(fold_data["feature_columns"]),
        hidden_dims=model_config.hidden_dims,
        dropout=model_config.dropout,
        num_classes=model_config.num_classes,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    history: list[dict[str, dict[str, float]]] = []
    best_epoch = 0
    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    patience = training_config.early_stopping_patience
    use_early_stopping = patience is not None and patience >= 0
    min_delta = training_config.early_stopping_min_delta

    for epoch_idx in range(1, training_config.epochs + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], optimizer, loss_fn)
        validation_metrics = evaluate_one_epoch(model, dataloaders["validation"], loss_fn)
        epoch_metrics = {"train": train_metrics, "validation": validation_metrics}
        history.append(epoch_metrics)
        tracker.log_epoch(
            run_name,
            fold,
            epoch_idx,
            {
                **{f"train/{name}": value for name, value in train_metrics.items()},
                **{f"validation/{name}": value for name, value in validation_metrics.items()},
            },
        )

        selection_score = _selection_score(validation_metrics)
        if best_epoch == 0 or selection_score > best_score + min_delta:
            best_epoch = epoch_idx
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if use_early_stopping and epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_qwk": best_score,
        "stopped_early": use_early_stopping and len(history) < training_config.epochs,
    }


def _selection_score(validation_metrics: dict[str, float]) -> float:
    """Return the validation metric used for checkpoint selection."""
    score = validation_metrics.get("qwk", float("-inf"))
    if math.isnan(score):
        return float("-inf")
    return score


def predict_split(model: torch.nn.Module, fold_data: FoldDatasets, split_name: str) -> pd.DataFrame:
    """Return window-level class probabilities with split metadata."""
    split = fold_data[split_name]
    split_x = torch.as_tensor(split["x"], dtype=torch.float32)
    probabilities = models.predict_probabilities(model, split_x).numpy()
    predictions_df = split["metadata"].reset_index(drop=True).copy()

    for class_idx in range(probabilities.shape[1]):
        predictions_df[f"prob_{class_idx}"] = probabilities[:, class_idx]

    return predictions_df


def _dataclass_dict(config: object) -> dict[str, object]:
    """Shallow dataclass-to-dict that keeps tuples JSON-friendly as lists."""
    from dataclasses import asdict

    return {key: (list(value) if isinstance(value, tuple) else value) for key, value in asdict(config).items()}


# Moved to training.artifacts so torch-free callers can build manifests too;
# re-exported under the old names for existing callers (train_temporal).
_git_sha = artifacts.git_sha
_package_versions = artifacts.package_versions


def _build_manifest(
    *,
    windows_path: str | Path,
    cv_config: CrossValidationConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    include_mlp: bool,
    started_at: str,
    ended_at: str,
    used_wandb: bool,
) -> dict[str, object]:
    return {
        "dataset_path": str(windows_path),
        "cv_strategy": cv_config.strategy,
        "seed": cv_config.random_seed,
        "include_mlp": include_mlp,
        "config": {
            "model": _dataclass_dict(model_config),
            "training": _dataclass_dict(training_config),
            "cross_validation": _dataclass_dict(cv_config),
        },
        "package_versions": _package_versions(used_wandb),
        "git_sha": _git_sha(),
        "started_at": started_at,
        "ended_at": ended_at,
    }


def _summary_metrics(metric_summary: pd.DataFrame) -> dict[str, object]:
    """Flatten the metric-summary table into scalar W&B summary values."""
    if metric_summary.empty:
        return {}
    summary: dict[str, object] = {}
    for run_name, row in metric_summary.iterrows():
        for (metric_name, stat), value in row.items():
            summary[f"{run_name}/{metric_name}_{stat}"] = float(value)
    return summary


def _log_run_header(
    windows_df: pd.DataFrame,
    dataset_path: str,
    cv_config: CrossValidationConfig,
    run_names: list[str],
    output_dir: Path,
    n_folds: int,
) -> None:
    logger.info("Training alertness classifier")
    logger.info(
        "Dataset: %s (%d subjects, %d videos, %d windows)",
        dataset_path,
        windows_df["subject_id"].nunique(),
        windows_df["video_id"].nunique(),
        len(windows_df),
    )
    logger.info(
        "CV: %s, %d folds, validation subjects per fold: %d",
        cv_config.strategy.upper(),
        n_folds,
        cv_config.validation_subject_count,
    )
    logger.info("Runs: %s", ", ".join(run_names))
    logger.info("Output: %s", output_dir)


def _log_run_footer(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    summary = tables["metric_summary"]
    logger.info("Final video-level metrics:")
    if not summary.empty:
        for run_name, row in summary.iterrows():
            logger.info(
                "  %-16s QWK %.3f +/- %.3f   rank MAE %.3f",
                run_name,
                row[("qwk", "mean")],
                row[("qwk", "std")],
                row[("rank_mae", "mean")],
            )
    logger.info("Diagnostics written:")
    for name in ("fold_metrics.csv", "confusion_matrices.csv", "diagnostics.csv"):
        logger.info("  %s", output_dir / name)


if __name__ == "__main__":
    main()
