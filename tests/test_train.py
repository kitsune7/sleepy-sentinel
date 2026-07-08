from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from training import train
from training.config import ModelConfig, TrainingConfig

# The exact flat output contract: nine files, nothing else. The new "quieter"
# pipeline must NOT spawn per-fold confusion files or narrow diagnostic files.
CONTRACT_FILES = {
    "manifest.json",
    "metric_summary.csv",
    "fold_metrics.csv",
    "video_predictions.csv",
    "learning_curves.csv",
    "confusion_matrices.csv",
    "diagnostics.csv",
    "folds.json",
    "split_summaries.csv",
}


def _write_windows(tmp_path) -> tuple:
    """Write the small synthetic dataset (4 subjects x 3 labels) used across tests."""
    rows = []
    for subject_idx in range(4):
        subject_id = f"{subject_idx + 1:02d}"
        for label in [0, 1, 2]:
            rows.append(
                {
                    "subject_id": subject_id,
                    "video_id": f"{subject_id}/{label}",
                    "label": label,
                    "window_idx": 0,
                    "frac_face_missing": 0.0,
                    "perclos": label / 2,
                    "blink_rate": 10 + subject_idx,
                    "ear_mean": 0.3 - label / 20,
                    "bright_mean": 100 + label,
                    "warmth_mean": 1.0 + label / 10,
                }
            )
    windows_path = tmp_path / "windows.parquet"
    output_dir = tmp_path / "eval"
    pd.DataFrame(rows).to_parquet(windows_path, index=False)
    return windows_path, output_dir


def _run(windows_path, output_dir, *, include_mlp=False, tracker=None) -> dict:
    """Run a fast, deterministic 2-fold CV over the synthetic dataset."""
    return train.run_cross_validation(
        windows_path,
        output_dir,
        n_splits=2,
        validation_subject_count=1,
        epochs=1,
        batch_size=4,
        early_stopping_patience=-1,
        random_seed=1,
        include_mlp=include_mlp,
        tracker=tracker,
    )


class RecordingTracker:
    """Fake Tracker capturing call counts/args, satisfying the Tracker protocol."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.finish_calls = 0
        self.summary_calls = 0
        self.fold_calls: list[tuple[str, int]] = []
        self.epoch_calls: list[tuple[str, int, int]] = []

    def start_experiment(self, config) -> None:
        self.start_calls += 1

    def log_epoch(self, run_name, fold, epoch, metrics) -> None:
        self.epoch_calls.append((run_name, fold, epoch))

    def log_fold(self, run_name, fold, metrics) -> None:
        self.fold_calls.append((run_name, fold))

    def log_summary(self, summary) -> None:
        self.summary_calls += 1

    def finish(self) -> None:
        self.finish_calls += 1


def test_set_random_seeds_makes_python_numpy_and_torch_reproducible() -> None:
    train.set_random_seeds(123)
    first = (random.random(), np.random.random(), torch.rand(1).item())

    train.set_random_seeds(123)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert second == first


def test_aggregate_window_predictions_averages_probabilities_to_video_level() -> None:
    predictions_df = pd.DataFrame(
        {
            "subject_id": ["01", "01", "01", "02"],
            "video_id": ["01/0", "01/0", "01/5", "02/10"],
            "label": [0, 0, 1, 2],
            "window_idx": [0, 1, 0, 0],
            "prob_0": [0.8, 0.6, 0.2, 0.1],
            "prob_1": [0.1, 0.3, 0.7, 0.2],
            "prob_2": [0.1, 0.1, 0.1, 0.7],
        }
    )

    video_predictions = train.aggregate_window_predictions(predictions_df)

    assert list(video_predictions["video_id"]) == ["01/0", "01/5", "02/10"]
    assert video_predictions["label"].tolist() == [0, 1, 2]
    assert video_predictions["pred_label"].tolist() == [0, 1, 2]
    assert video_predictions.loc[video_predictions["video_id"] == "01/0", "prob_0"].item() == pytest.approx(0.7)


def test_run_cross_validation_writes_assignment_artifacts(tmp_path) -> None:
    rows = []
    for subject_idx in range(4):
        subject_id = f"{subject_idx + 1:02d}"
        for label in [0, 1, 2]:
            rows.append(
                {
                    "subject_id": subject_id,
                    "video_id": f"{subject_id}/{label}",
                    "label": label,
                    "window_idx": 0,
                    "frac_face_missing": 0.0,
                    "perclos": label / 2,
                    "blink_rate": 10 + subject_idx,
                    "ear_mean": 0.3 - label / 20,
                    "bright_mean": 100 + label,
                    "warmth_mean": 1.0 + label / 10,
                }
            )
    windows_path = tmp_path / "windows.parquet"
    output_dir = tmp_path / "eval"
    pd.DataFrame(rows).to_parquet(windows_path, index=False)

    results = train.run_cross_validation(
        windows_path,
        output_dir,
        n_splits=2,
        validation_subject_count=1,
        epochs=1,
        batch_size=4,
        early_stopping_patience=-1,
        random_seed=1,
        include_mlp=True,
    )

    expected_outputs = {
        "manifest.json",
        "folds.json",
        "split_summaries.csv",
        "learning_curves.csv",
        "video_predictions.csv",
        "fold_metrics.csv",
        "metric_summary.csv",
        "confusion_matrices.csv",
        "diagnostics.csv",
    }
    output_names = {path.name for path in output_dir.iterdir()}
    assert expected_outputs.issubset(output_names)
    assert not any(name.endswith("_confusion_matrix.csv") for name in output_names)
    assert set(results["fold_metrics"]["run"]) == {
        "majority",
        "perclos",
        "luminance",
        "logistic_full",
        "mlp_regularized",
    }
    assert {
        "run",
        "train_loss",
        "train_accuracy",
        "train_qwk",
        "train_rank_mae",
        "train_macro_f1",
        "validation_loss",
        "validation_accuracy",
        "validation_qwk",
        "validation_rank_mae",
        "validation_macro_f1",
        "selected_checkpoint",
    }.issubset(results["learning_curves"].columns)


def test_train_fold_stops_after_validation_qwk_patience(monkeypatch) -> None:
    validation_scores = iter([0.1, 0.2, 0.15])

    def fake_train_one_epoch(*_args):
        return {"loss": 1.0, "accuracy": 0.0, "qwk": 0.0, "rank_mae": 1.0, "macro_f1": 0.0}

    def fake_evaluate_one_epoch(*_args):
        score = next(validation_scores)
        return {"loss": 1.0 - score, "accuracy": score, "qwk": score, "rank_mae": 1.0, "macro_f1": score}

    monkeypatch.setattr(train, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train, "evaluate_one_epoch", fake_evaluate_one_epoch)
    fold_data = {
        "feature_columns": ["perclos", "blink_rate"],
        "train": {"x": np.array([[0.0, 1.0], [1.0, 0.0]]), "y": np.array([0, 1])},
        "validation": {"x": np.array([[0.5, 0.5]]), "y": np.array([1])},
        "test": {"x": np.array([[0.2, 0.8]]), "y": np.array([0])},
    }

    result = train.train_fold(
        fold_data,
        ModelConfig(hidden_dims=(4,), dropout=0.0, weight_decay=0.0, num_classes=3),
        TrainingConfig(epochs=5, batch_size=2, early_stopping_patience=1),
    )

    assert result["best_epoch"] == 2
    assert len(result["history"]) == 3
    assert result["stopped_early"] is True


def test_one_combined_confusion_matrix(tmp_path) -> None:
    windows_path, output_dir = _write_windows(tmp_path)
    _run(windows_path, output_dir)

    confusion = pd.read_csv(output_dir / "confusion_matrices.csv")
    assert list(confusion.columns) == ["run", "fold", "true_label", "pred_label", "count"]
    # No per-fold confusion files -- it all lives in the one combined table.
    assert list(output_dir.glob("*_confusion_matrix.csv")) == []

    # 3-class problem -> each (run, fold) contributes a full 3x3 = 9 rows.
    per_group = confusion.groupby(["run", "fold"]).size()
    assert (per_group == 9).all()


def test_manifest_has_expected_keys(tmp_path) -> None:
    windows_path, output_dir = _write_windows(tmp_path)
    _run(windows_path, output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert {
        "config",
        "dataset_path",
        "cv_strategy",
        "seed",
        "package_versions",
        "git_sha",
        "started_at",
        "ended_at",
    }.issubset(manifest)
    assert {"model", "training", "cross_validation"}.issubset(manifest["config"])


def test_no_per_fold_file_explosion(tmp_path) -> None:
    windows_path, output_dir = _write_windows(tmp_path)
    _run(windows_path, output_dir, include_mlp=True)

    produced = {path.name for path in output_dir.iterdir()}
    # Exactly the flat contract -- no extra narrow/per-fold files.
    assert produced == CONTRACT_FILES
    assert not any(name.endswith("_confusion_matrix.csv") for name in produced)
    assert "confidence_by_correctness.csv" not in produced
    assert "error_by_true_label.csv" not in produced


def test_one_grouped_diagnostics_table(tmp_path) -> None:
    windows_path, output_dir = _write_windows(tmp_path)
    _run(windows_path, output_dir)

    diagnostics = pd.read_csv(output_dir / "diagnostics.csv")
    assert "diagnostic" in diagnostics.columns
    assert {"confidence_by_correctness", "error_by_true_label"}.issubset(set(diagnostics["diagnostic"]))
    # The grouped table replaces the separate narrow files.
    assert not (output_dir / "confidence_by_correctness.csv").exists()
    assert not (output_dir / "error_by_true_label.csv").exists()


def test_tracker_receives_one_experiment_lifecycle(tmp_path) -> None:
    windows_path, output_dir = _write_windows(tmp_path)
    tracker = RecordingTracker()
    _run(windows_path, output_dir, include_mlp=True, tracker=tracker)

    # ONE lifecycle for the whole command -- not once per fold.
    assert tracker.start_calls == 1
    assert tracker.finish_calls == 1
    assert tracker.summary_calls == 1

    # log_fold once per (run, fold): 2 folds x (4 baseline-path runs + mlp) = 10.
    assert len(tracker.fold_calls) == 10
    assert len(tracker.fold_calls) == len(set(tracker.fold_calls))
    assert {run for run, _ in tracker.fold_calls} == {
        "majority",
        "perclos",
        "luminance",
        "logistic_full",
        "mlp_regularized",
    }


def test_include_mlp_false_has_no_mlp_and_empty_learning_curves(tmp_path) -> None:
    windows_path, output_dir = _write_windows(tmp_path)
    results = _run(windows_path, output_dir, include_mlp=False)

    assert set(results["fold_metrics"]["run"]) == {"majority", "perclos", "luminance", "logistic_full"}
    assert "mlp_regularized" not in set(results["fold_metrics"]["run"])
    # Header-only learning curves keep the contract stable when no MLP ran.
    learning_curves = pd.read_csv(output_dir / "learning_curves.csv")
    assert learning_curves.empty


def test_include_mlp_true_has_mlp_and_learning_curve_rows(tmp_path) -> None:
    windows_path, output_dir = _write_windows(tmp_path)
    results = _run(windows_path, output_dir, include_mlp=True)

    assert "mlp_regularized" in set(results["fold_metrics"]["run"])
    learning_curves = pd.read_csv(output_dir / "learning_curves.csv")
    assert not learning_curves.empty
