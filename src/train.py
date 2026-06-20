"""Train and evaluate the alertness baseline across subject-wise folds.

This file owns the Stage 5 and Stage 6 orchestration:
- load the window-level dataset
- ask `splits.py` for subject-wise CV folds
- ask `dataset.py` to prepare fold-specific train/validation/test objects
- train one baseline model and one focused generalization intervention
- track training and validation metrics over epochs
- aggregate window predictions up to video-level predictions
- save fold metrics, confusion matrices, and traces for the writeup

Keep this file as the coordinator. Feature engineering belongs in `windows.py`,
split logic in `splits.py`, preprocessing in `dataset.py`, and metric math in
`metrics.py`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import dataset
import metrics
import models
import splits


def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model: Any, dataloader: Any, optimizer: Any, loss_fn: Any) -> dict[str, float]:
    """Run one training epoch and return training metrics."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = len(batch_y)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total += batch_size

    return {"loss": total_loss / total if total else 0.0, "accuracy": correct / total if total else 0.0}


def evaluate_one_epoch(model: Any, dataloader: Any, loss_fn: Any) -> dict[str, float]:
    """Run one validation or test pass and return metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            batch_size = len(batch_y)
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total += batch_size

    return {"loss": total_loss / total if total else 0.0, "accuracy": correct / total if total else 0.0}


def train_fold(
    fold_data: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
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

    for _ in range(training_config.get("epochs", 1)):
        train_metrics = train_one_epoch(model, dataloaders["train"], optimizer, loss_fn)
        validation_metrics = evaluate_one_epoch(model, dataloaders["validation"], loss_fn)
        history.append({"train": train_metrics, "validation": validation_metrics})

    return {"model": model, "history": history}


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


def run_cross_validation(
    windows_path: str | Path,
    output_dir: str | Path,
    *,
    n_splits: int = 5,
    validation_subject_count: int = 9,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run the full CV experiment and save fold-level outputs."""
    windows_df = pd.read_parquet(windows_path)
    windows_df["subject_id"] = windows_df["subject_id"].astype(str)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_folds = splits.make_group_folds(windows_df, n_splits=n_splits, random_seed=random_seed)
    folds = [
        splits.add_validation_subjects(
            fold,
            validation_subject_count=validation_subject_count,
            random_seed=random_seed + fold_idx,
        )
        for fold_idx, fold in enumerate(base_folds)
    ]
    splits.save_fold_assignments(folds, output_dir / "folds.json")

    model_runs = {
        "baseline": {"hidden_dims": (64, 32), "dropout": 0.0, "weight_decay": 0.0},
        "regularized": {"hidden_dims": (64, 32), "dropout": 0.25, "weight_decay": 1e-4},
    }
    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    split_summaries = []
    learning_curves = []
    prediction_tables = []
    fold_metrics = []

    for fold_idx, fold in enumerate(folds, start=1):
        splits.assert_disjoint_subjects(fold)
        split_summary = splits.describe_split(windows_df, fold)
        split_summary.insert(0, "fold", fold_idx)
        split_summaries.append(split_summary)

        fold_data = dataset.prepare_fold_datasets(
            train_df=splits.filter_split(windows_df, fold["train"]),
            validation_df=splits.filter_split(windows_df, fold["validation"]),
            test_df=splits.filter_split(windows_df, fold["test"]),
        )

        for run_name, model_config in model_runs.items():
            model_result = train_fold(
                fold_data,
                model_config={
                    "hidden_dims": model_config["hidden_dims"],
                    "dropout": model_config["dropout"],
                    "num_classes": 3,
                },
                training_config={
                    **training_config,
                    "weight_decay": model_config["weight_decay"],
                    "seed": random_seed + fold_idx,
                },
            )

            for epoch_idx, epoch_metrics in enumerate(model_result["history"], start=1):
                learning_curves.append(
                    {
                        "run": run_name,
                        "fold": fold_idx,
                        "epoch": epoch_idx,
                        "train_loss": epoch_metrics["train"]["loss"],
                        "train_accuracy": epoch_metrics["train"]["accuracy"],
                        "validation_loss": epoch_metrics["validation"]["loss"],
                        "validation_accuracy": epoch_metrics["validation"]["accuracy"],
                    }
                )

            test_predictions = predict_split(model_result["model"], fold_data, "test")
            video_predictions = aggregate_window_predictions(test_predictions)
            video_predictions.insert(0, "run", run_name)
            video_predictions.insert(1, "fold", fold_idx)
            prediction_tables.append(video_predictions)

            metric_row = metrics.classification_metric_summary(
                video_predictions["label"],
                video_predictions["pred_label"],
            )
            metric_row.update({"run": run_name, "fold": fold_idx, "n_videos": len(video_predictions)})
            fold_metrics.append(metric_row)

            metrics.confusion_matrix_table(video_predictions["label"], video_predictions["pred_label"]).to_csv(
                output_dir / f"{run_name}_fold{fold_idx}_confusion_matrix.csv"
            )

    split_summaries_df = pd.concat(split_summaries, ignore_index=True)
    learning_curves_df = pd.DataFrame(learning_curves)
    video_predictions_df = pd.concat(prediction_tables, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics)
    metric_summary_df = fold_metrics_df.groupby("run")[["qwk", "rank_mae", "accuracy", "macro_f1"]].agg(["mean", "std"])

    confidence_tables = []
    error_tables = []
    for run_name, run_predictions in video_predictions_df.groupby("run"):
        confidence_table = metrics.confidence_summary(run_predictions)
        confidence_table.insert(0, "run", run_name)
        confidence_tables.append(confidence_table)

        error_table = metrics.error_slice_summary(run_predictions, "label")
        error_table.insert(0, "run", run_name)
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


def main() -> None:
    """Command-line entry point for training and evaluation."""
    parser = argparse.ArgumentParser(description="Run subject-wise alertness cross-validation.")
    parser.add_argument("windows_path", type=Path, nargs="?", default=Path("data/frame_windows.parquet"))
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("outputs/assignment5_initial_eval"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--validation-subject-count", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    results = run_cross_validation(
        args.windows_path,
        args.output_dir,
        n_splits=args.n_splits,
        validation_subject_count=args.validation_subject_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )
    print(f"Wrote train/eval artifacts to {args.output_dir}")
    print(results["metric_summary"])


if __name__ == "__main__":
    main()
