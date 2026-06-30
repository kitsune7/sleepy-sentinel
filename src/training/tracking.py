"""Experiment-tracking adapters, kept separate from the training loop.

The training code talks to a `Tracker`; it never touches `wandb` directly. The
default is `NullTracker` (no-op), so a plain run never needs wandb installed.

`WandbTracker` creates exactly ONE W&B run per training command -- the
command-level CV experiment IS the run. Per-fold and per-epoch numbers go into
W&B Tables under that single run rather than spawning a run per fold. The old
"one run per fold" behavior is opt-in via `WandbPerFoldTracker`.
"""

from __future__ import annotations

from typing import Any, Protocol


class Tracker(Protocol):
    """Minimal tracking surface the training loop depends on."""

    def start_experiment(self, config: dict[str, Any]) -> None: ...

    def log_epoch(self, run_name: str, fold: int, epoch: int, metrics: dict[str, float]) -> None: ...

    def log_fold(self, run_name: str, fold: int, metrics: dict[str, Any]) -> None: ...

    def log_summary(self, summary: dict[str, Any]) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    """Default no-op tracker. Requires no dependencies."""

    def start_experiment(self, config: dict[str, Any]) -> None:
        return None

    def log_epoch(self, run_name: str, fold: int, epoch: int, metrics: dict[str, float]) -> None:
        return None

    def log_fold(self, run_name: str, fold: int, metrics: dict[str, Any]) -> None:
        return None

    def log_summary(self, summary: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


def _require_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - only hit when wandb missing
        raise RuntimeError("wandb is not installed. Install dependencies or omit --wandb-project.") from exc
    return wandb


class WandbTracker:
    """Single W&B run for the whole CV command; per-fold/epoch rows go to Tables."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        mode: str | None = None,
        group: str | None = None,
    ) -> None:
        self._wandb = _require_wandb()
        self._project = project
        self._entity = entity
        self._mode = mode
        self._group = group
        self._run: Any | None = None
        self._epoch_rows: list[dict[str, Any]] = []
        self._fold_rows: list[dict[str, Any]] = []

    def start_experiment(self, config: dict[str, Any]) -> None:
        init_kwargs: dict[str, Any] = {
            "project": self._project,
            "group": self._group,
            "config": config,
            "tags": ["cross-validation"],
        }
        if self._entity is not None:
            init_kwargs["entity"] = self._entity
        if self._mode is not None:
            init_kwargs["mode"] = self._mode
        self._run = self._wandb.init(**init_kwargs)

    def log_epoch(self, run_name: str, fold: int, epoch: int, metrics: dict[str, float]) -> None:
        self._epoch_rows.append({"run": run_name, "fold": fold, "epoch": epoch, **metrics})

    def log_fold(self, run_name: str, fold: int, metrics: dict[str, Any]) -> None:
        self._fold_rows.append({"run": run_name, "fold": fold, **metrics})

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self._run is None:
            return
        if self._epoch_rows:
            self._run.log({"learning_curves": self._table(self._epoch_rows)})
        if self._fold_rows:
            self._run.log({"fold_metrics": self._table(self._fold_rows)})
        self._run.summary.update(summary)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def _table(self, rows: list[dict[str, Any]]) -> Any:
        columns = list({key for row in rows for key in row})
        data = [[row.get(col) for col in columns] for row in rows]
        return self._wandb.Table(columns=columns, data=data)


class WandbPerFoldTracker:
    """Opt-in legacy behavior: a separate W&B run per (run_name, fold)."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        mode: str | None = None,
        group: str | None = None,
    ) -> None:
        self._wandb = _require_wandb()
        self._project = project
        self._entity = entity
        self._mode = mode
        self._group = group
        self._config: dict[str, Any] = {}
        self._runs: dict[tuple[str, int], Any] = {}

    def start_experiment(self, config: dict[str, Any]) -> None:
        self._config = config

    def _run_for(self, run_name: str, fold: int) -> Any:
        key = (run_name, fold)
        if key not in self._runs:
            init_kwargs: dict[str, Any] = {
                "project": self._project,
                "name": f"{run_name}-fold{fold}",
                "group": self._group,
                "config": {**self._config, "run": run_name, "fold": fold},
                "tags": ["cross-validation", run_name],
                "reinit": True,
            }
            if self._entity is not None:
                init_kwargs["entity"] = self._entity
            if self._mode is not None:
                init_kwargs["mode"] = self._mode
            self._runs[key] = self._wandb.init(**init_kwargs)
        return self._runs[key]

    def log_epoch(self, run_name: str, fold: int, epoch: int, metrics: dict[str, float]) -> None:
        self._run_for(run_name, fold).log({"epoch": epoch, **metrics}, step=epoch)

    def log_fold(self, run_name: str, fold: int, metrics: dict[str, Any]) -> None:
        self._run_for(run_name, fold).summary.update(metrics)

    def log_summary(self, summary: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        for run in self._runs.values():
            run.finish()
        self._runs.clear()


def build_tracker(
    *,
    project: str | None,
    entity: str | None,
    mode: str | None,
    group: str | None,
    per_fold_runs: bool,
) -> Tracker:
    """Pick the tracker for the requested CLI flags. No project -> NullTracker."""
    if project is None:
        return NullTracker()
    tracker_cls = WandbPerFoldTracker if per_fold_runs else WandbTracker
    return tracker_cls(project=project, entity=entity, mode=mode, group=group)
