"""Typed run/config objects for the alertness cross-validation experiment.

These replace the stringly-typed `model_config` / `training_config` dicts that
used to float through `train.py`. Keeping them here gives every call site
editor support and one obvious place to read the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and regularization knobs for the MLP."""

    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.25
    weight_decay: float = 1e-4
    num_classes: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and early-stopping knobs for one fold's training run."""

    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int | None = 8
    early_stopping_metric: str = "validation_qwk"
    early_stopping_min_delta: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class CrossValidationConfig:
    """How folds are built across subjects."""

    n_splits: int | None = None
    validation_subject_count: int = 9
    random_seed: int = 42

    @property
    def strategy(self) -> str:
        """Name the CV strategy: LOSO when `n_splits` is unset, else grouped K-fold."""
        return "loso" if self.n_splits is None else "group_kfold"


@dataclass
class RunResult:
    """One (run_name, fold) outcome, shared by baselines and the MLP.

    `learning_curve` is empty for baselines; the orchestration loop can then
    treat every run identically instead of branching on model type.
    """

    run_name: str
    fold: int
    video_predictions: pd.DataFrame
    fold_metrics: dict[str, float | int | bool | str]
    learning_curve: list[dict[str, float | int | bool | str]] = field(default_factory=list)
