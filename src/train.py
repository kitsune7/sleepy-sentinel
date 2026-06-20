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


def run_cross_validation(windows_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run the full CV experiment and save fold-level outputs."""
    windows_df = pd.read_parquet(windows_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = splits.make_group_folds(windows_df, n_splits=5, random_seed=42)
    splits.save_fold_assignments(folds, output_dir / "folds.json")
    return {"folds": folds}


def main() -> None:
    """Command-line entry point for training and evaluation."""
    parser = argparse.ArgumentParser(description="Run subject-wise alertness cross-validation.")
    parser.add_argument("windows_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    run_cross_validation(args.windows_path, args.output_dir)
