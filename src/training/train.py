"""Train and evaluate the alertness baseline across subject-wise folds.

This file owns the Stage 5 and Stage 6 orchestration:
- load the window-level dataset
- ask `training.splits` for subject-wise CV folds
- ask `training.dataset` to prepare fold-specific train/validation/test objects
- train one baseline model and one focused generalization intervention
- track training and validation metrics over epochs
- aggregate window predictions up to video-level predictions
- save fold metrics, confusion matrices, and traces for the writeup

Keep this file as the coordinator. Feature engineering belongs in
`data_prep.windows`, split logic in `training.splits`, preprocessing in
`training.dataset`, and metric math in `evaluation.metrics`.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluation import metrics
from training import baselines, dataset, models, splits

try:
    import wandb
except ImportError:  # pragma: no cover - dependency is optional unless W&B is enabled
    wandb = None


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
    parser.add_argument("--wandb-project", type=str, default=None, help="Enable W&B logging for each model/fold run.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B team or username.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group name for the CV run.")
    args = parser.parse_args()

    set_random_seeds(args.random_seed)

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
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_group=args.wandb_group,
    )
    # print(results["metric_summary"])


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
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
    wandb_group: str | None = None,
) -> dict[str, Any]:
    """Run the full CV experiment and save fold-level outputs."""
    output_dir = Path(output_dir)
    windows_df = pd.read_parquet(windows_path)
    base_folds = (
        splits.make_loso_folds(windows_df)
        if n_splits is None
        else splits.make_group_folds(windows_df, n_splits=n_splits, random_seed=random_seed)
    )
    folds = [
        splits.add_validation_subjects(
            fold,
            validation_subject_count=validation_subject_count,
            random_seed=random_seed + fold_idx,
        )
        for fold_idx, fold in enumerate(base_folds)
    ]
    splits.save_fold_assignments(folds, output_dir / "folds.json")

    baseline_specs = baselines.available_baselines(windows_df)
    model_config = {"hidden_dims": (64, 32), "dropout": 0.25, "weight_decay": 1e-4}
    mlp_run_name = "mlp_regularized"
    split_summaries = []
    learning_curves = []
    prediction_tables = []
    fold_metrics = []

    for fold_idx, fold in enumerate(folds, start=1):
        splits.assert_disjoint_subjects(fold)
        split_summary = splits.describe_split(windows_df, fold)
        split_summary.insert(0, "fold", fold_idx)
        split_summaries.append(split_summary)

        train_df = splits.filter_split(windows_df, fold["train"])
        validation_df = splits.filter_split(windows_df, fold["validation"])
        test_df = splits.filter_split(windows_df, fold["test"])

        for baseline_spec in baseline_specs:
            test_predictions = baselines.predict_baseline(
                train_df,
                test_df,
                baseline_spec,
                random_seed=random_seed + fold_idx,
            )
            video_predictions = aggregate_window_predictions(test_predictions)
            video_predictions.insert(0, "run", baseline_spec.name)
            video_predictions.insert(1, "fold", fold_idx)
            prediction_tables.append(video_predictions)

            metric_row = metrics.classification_metric_summary(
                video_predictions["label"],
                video_predictions["pred_label"],
            )
            metric_row.update({"run": baseline_spec.name, "fold": fold_idx, "n_videos": len(video_predictions)})
            fold_metrics.append(metric_row)

            metrics.confusion_matrix_table(video_predictions["label"], video_predictions["pred_label"]).to_csv(
                output_dir / f"{baseline_spec.name}_fold{fold_idx}_confusion_matrix.csv"
            )

        fold_data = dataset.prepare_fold_datasets(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
        )

        fold_run_name = f"{mlp_run_name}-fold{fold_idx}"
        fold_model_config = {
            "hidden_dims": model_config["hidden_dims"],
            "dropout": model_config["dropout"],
            "num_classes": 3,
        }
        fold_training_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": model_config["weight_decay"],
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_metric": "validation_qwk",
            "seed": random_seed + fold_idx,
        }
        wandb_run = _init_wandb_run(
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
            group=wandb_group,
            run_name=fold_run_name,
            config={
                "fold": fold_idx,
                "n_splits": n_splits if n_splits is not None else len(base_folds),
                "cv_strategy": "loso" if n_splits is None else "group_kfold",
                "validation_subject_count": validation_subject_count,
                **fold_model_config,
                **fold_training_config,
            },
        )
        try:
            model_result = train_fold(
                fold_data,
                model_config=fold_model_config,
                training_config=fold_training_config,
                wandb_run=wandb_run,
            )

            for epoch_idx, epoch_metrics in enumerate(model_result["history"], start=1):
                learning_curves.append(
                    {
                        "run": mlp_run_name,
                        "fold": fold_idx,
                        "epoch": epoch_idx,
                        "selected_checkpoint": epoch_idx == model_result["best_epoch"],
                        **{f"train_{metric_name}": value for metric_name, value in epoch_metrics["train"].items()},
                        **{
                            f"validation_{metric_name}": value
                            for metric_name, value in epoch_metrics["validation"].items()
                        },
                    }
                )

            test_predictions = predict_split(model_result["model"], fold_data, "test")
            video_predictions = aggregate_window_predictions(test_predictions)
            video_predictions.insert(0, "run", mlp_run_name)
            video_predictions.insert(1, "fold", fold_idx)
            prediction_tables.append(video_predictions)

            metric_row = metrics.classification_metric_summary(
                video_predictions["label"],
                video_predictions["pred_label"],
            )
            _log_final_test_metrics(wandb_run, metric_row, len(video_predictions))
            metric_row.update(
                {
                    "run": mlp_run_name,
                    "fold": fold_idx,
                    "n_videos": len(video_predictions),
                    "best_epoch": model_result["best_epoch"],
                    "stopped_early": model_result["stopped_early"],
                }
            )
            fold_metrics.append(metric_row)

            metrics.confusion_matrix_table(video_predictions["label"], video_predictions["pred_label"]).to_csv(
                output_dir / f"{mlp_run_name}_fold{fold_idx}_confusion_matrix.csv"
            )
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    split_summaries_df = pd.concat(split_summaries, ignore_index=True)
    learning_curves_df = pd.DataFrame(learning_curves)
    video_predictions_df = pd.concat(prediction_tables, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics)
    metric_summary_df = fold_metrics_df.groupby("run")[["qwk", "rank_mae", "accuracy", "macro_f1"]].agg(
        ["mean", "std"]
    )

    confidence_tables = []
    error_tables = []
    for (run_name, fold_idx), fold_predictions in video_predictions_df.groupby(["run", "fold"]):
        confidence_table = metrics.confidence_summary(fold_predictions)
        confidence_table.insert(0, "run", run_name)
        confidence_table.insert(1, "fold", fold_idx)
        confidence_tables.append(confidence_table)

        error_table = metrics.error_slice_summary(fold_predictions, "label")
        error_table.insert(0, "run", run_name)
        error_table.insert(1, "fold", fold_idx)
        error_tables.append(error_table)

    split_summaries_df.to_csv(output_dir / "split_summaries.csv", index=False)
    learning_curves_df.to_csv(output_dir / "learning_curves.csv", index=False)
    video_predictions_df.to_csv(output_dir / "video_predictions.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    metric_summary_df.to_csv(output_dir / "metric_summary.csv")
    pd.concat(confidence_tables, ignore_index=True).to_csv(output_dir / "confidence_by_correctness.csv", index=False)
    pd.concat(error_tables, ignore_index=True).to_csv(output_dir / "error_by_true_label.csv", index=False)

    return {
        "folds": folds,
        "split_summaries": split_summaries_df,
        "learning_curves": learning_curves_df,
        "video_predictions": video_predictions_df,
        "fold_metrics": fold_metrics_df,
        "metric_summary": metric_summary_df,
    }


def _epoch_metric_summary(total_loss: float, total: int, y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Build the metric bundle logged for one train/validation epoch."""
    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "qwk": 0.0, "rank_mae": 0.0, "macro_f1": 0.0}

    metric_summary = metrics.classification_metric_summary(y_true, y_pred)
    return {"loss": total_loss / total, **metric_summary}


def train_one_epoch(model: Any, dataloader: Any, optimizer: Any, loss_fn: Any) -> dict[str, float]:
    """Run one training epoch and return training metrics."""
    model.train()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

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


def evaluate_one_epoch(model: Any, dataloader: Any, loss_fn: Any) -> dict[str, float]:
    """Run one validation or test pass and return metrics."""
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

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


def _init_wandb_run(
    *,
    project: str | None,
    entity: str | None,
    mode: str | None,
    group: str,
    run_name: str,
    config: dict[str, Any],
) -> Any | None:
    """Create a W&B run when tracking is enabled."""
    if project is None:
        return None

    if wandb is None:
        raise RuntimeError("wandb is not installed. Install dependencies or omit --wandb-project.")

    init_kwargs = {
        "project": project,
        "name": run_name,
        "group": group,
        "config": config,
        "tags": ["cross-validation", run_name.split("-fold")[0]],
    }
    if entity is not None:
        init_kwargs["entity"] = entity
    if mode is not None:
        init_kwargs["mode"] = mode

    return wandb.init(**init_kwargs)


def _log_epoch_metrics(wandb_run: Any | None, epoch: int, epoch_metrics: dict[str, dict[str, float]]) -> None:
    """Log one epoch of train/validation metrics to W&B."""
    if wandb_run is None:
        return

    wandb_run.log(
        {
            "epoch": epoch,
            **{f"train/{metric_name}": value for metric_name, value in epoch_metrics["train"].items()},
            **{f"validation/{metric_name}": value for metric_name, value in epoch_metrics["validation"].items()},
        },
        step=epoch,
    )


def _log_final_test_metrics(wandb_run: Any | None, metric_row: dict[str, float], n_videos: int) -> None:
    """Log fold-level video metrics after training finishes."""
    if wandb_run is None:
        return

    wandb_run.log({**{f"test/{name}": value for name, value in metric_row.items()}, "test/n_videos": n_videos})


def train_fold(
    fold_data: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """Train one model for one subject-wise fold."""
    set_random_seeds(training_config.get("seed", 0))
    dataloaders = dataset.make_dataloaders(fold_data, batch_size=training_config.get("batch_size", 64))
    model = models.build_cross_entropy_mlp(
        input_dim=len(fold_data["feature_columns"]),
        hidden_dims=model_config.get("hidden_dims", (64, 32)),
        dropout=model_config.get("dropout", 0.0),
        num_classes=model_config.get("num_classes", 3),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-3),
        weight_decay=training_config.get("weight_decay", 0.0),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    best_epoch = 0
    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    patience = training_config.get("early_stopping_patience")
    use_early_stopping = patience is not None and patience >= 0
    min_delta = training_config.get("early_stopping_min_delta", 0.0)

    for epoch_idx in range(1, training_config.get("epochs", 1) + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], optimizer, loss_fn)
        validation_metrics = evaluate_one_epoch(model, dataloaders["validation"], loss_fn)
        epoch_metrics = {"train": train_metrics, "validation": validation_metrics}
        history.append(epoch_metrics)
        _log_epoch_metrics(wandb_run, epoch_idx, epoch_metrics)

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
        "stopped_early": use_early_stopping and len(history) < training_config.get("epochs", 1),
    }


def _selection_score(validation_metrics: dict[str, float]) -> float:
    """Return the validation metric used for checkpoint selection."""
    score = validation_metrics.get("qwk", float("-inf"))
    if math.isnan(score):
        return float("-inf")
    return score


def aggregate_window_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse window-level predictions into video-level predictions."""
    prob_columns = [col for col in predictions_df.columns if col.startswith("prob_")]
    aggregated = (
        predictions_df.groupby(["subject_id", "video_id"], as_index=False)
        .agg(
            label=("label", "first"),
            window_count=("window_idx", "count"),
            **{prob_col: (prob_col, "mean") for prob_col in prob_columns},
        )
        .sort_values("video_id")
        .reset_index(drop=True)
    )

    probabilities = aggregated[prob_columns]
    aggregated["pred_label"] = probabilities.to_numpy().argmax(axis=1)
    aggregated["confidence"] = probabilities.max(axis=1)
    return aggregated


def predict_split(model: Any, fold_data: dict[str, Any], split_name: str) -> pd.DataFrame:
    """Return window-level class probabilities with split metadata."""
    split = fold_data[split_name]
    split_x = torch.as_tensor(split["x"], dtype=torch.float32)
    probabilities = models.predict_probabilities(model, split_x).numpy()
    predictions_df = split["metadata"].reset_index(drop=True).copy()

    for class_idx in range(probabilities.shape[1]):
        predictions_df[f"prob_{class_idx}"] = probabilities[:, class_idx]

    return predictions_df


if __name__ == "__main__":
    main()
