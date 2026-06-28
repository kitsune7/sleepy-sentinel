# Technical Debt Audit

Last reviewed: 2026-06-27

This audit focuses on reducing accidental complexity in the current training
pipeline, especially the noisy experiment output behavior. The repo is small
enough that most fixes should simplify code rather than add new framework.

## Priority Checklist

- [ ] **Collapse W&B tracking to one run per training command.**
  `uv run train_alertness --wandb-project sleepy-sentinel` currently initializes
  a separate W&B run for every LOSO fold of the MLP. With the default 60-fold
  LOSO setup, that produces 60 independent W&B runs before counting local W&B
  metadata files. Make the command-level CV experiment the W&B run, and log fold
  metrics as rows in a W&B table or as metrics with `fold`/`run` dimensions. If
  per-fold detail is still useful, make it opt-in with a flag such as
  `--wandb-per-fold-runs`.

- [ ] **Replace per-fold confusion matrix files with one combined artifact.**
  `training.train` writes one confusion-matrix CSV per baseline/fold and one per
  MLP fold. Under default LOSO with majority, PERCLOS, luminance, and MLP runs,
  that is 240 small CSVs. Store these as a single
  `confusion_matrices.csv` with columns like `run`, `fold`, `true_label`,
  `pred_label`, and `count`. This keeps all information while turning hundreds
  of files into one grouped table.

- [ ] **Introduce a run output contract.**
  The default output directory is a fixed `outputs/` path with several summary
  files and many generated fold files. Move toward a small, predictable layout:
  `manifest.json`, `metric_summary.csv`, `fold_metrics.csv`,
  `video_predictions.csv`, `learning_curves.csv`, and
  `diagnostics.csv`/`confusion_matrices.csv`. Include config, dataset path, CV
  strategy, seed, package versions, start/end time, and git SHA in the manifest.
  Decide whether each command overwrites `outputs/latest/` or writes
  `outputs/runs/<timestamp-or-run-id>/`; do not mix both behaviors silently.

- [ ] **Make logs high-level by default.**
  Training currently has almost no terminal summary, while W&B/local artifacts
  are overly detailed. Add concise command-line output: dataset summary, CV
  strategy, number of folds, runs evaluated, output directory, final metric
  summary, and where diagnostics were written. Avoid fold-by-fold chatter unless
  `--verbose` is set.

- [ ] **Separate artifact writing from experiment orchestration.**
  `training.train.run_cross_validation` currently owns split creation, baseline
  execution, MLP config construction, W&B lifecycle, metric calculation,
  prediction aggregation, and file writing. Extract a small artifact writer
  module or class so the training loop returns structured results and one place
  decides how those results become files and logs.

- [ ] **Separate tracking from training.**
  W&B calls are embedded in the fold loop and epoch loop. Introduce a tiny
  tracking adapter with methods such as `start_experiment`, `log_epoch`,
  `log_fold`, `log_summary`, and `finish`. A no-op implementation should be the
  default. This keeps model training independent of W&B's run model and makes it
  easier to test the intended one-run-per-command behavior.

- [ ] **Create explicit run/config objects.**
  The training path passes loose dictionaries (`model_config`,
  `training_config`, `fold_data`) and many top-level arguments. Replace the
  most important ones with dataclasses such as `TrainingConfig`, `ModelConfig`,
  `CrossValidationConfig`, and `FoldData`. This would reduce stringly typed keys,
  make defaults easier to inspect, and avoid scattering magic values across
  `main`, `run_cross_validation`, and `train_fold`.

- [ ] **Decide whether the neural model earns default status.**
  The current spec says PERCLOS-only logistic regression is the best candidate
  and the regularized MLP does not beat it. If that remains true, make the
  simpler baseline the default headline model and move the MLP behind an
  explicit `--include-mlp` or `--models` selection. This would make the default
  command faster, quieter, and better aligned with the evidence.

- [ ] **Remove or finish the ordinal-model stub.**
  `models.build_ordinal_mlp` exists and has tests, but CORN/ordinal training is
  not wired into `training.train`. Either implement the complete ordinal path
  behind a model-selection flag or delete the stub until the experiment is ready.
  Keeping a tested but unused model path suggests capability the CLI does not
  actually provide.

- [ ] **Give baseline/model runs a shared evaluation interface.**
  Baselines and the MLP both eventually produce window-level probabilities, but
  the orchestration treats them differently. Define a small shared result shape
  for `run_name`, `fold`, window predictions, video predictions, fold metrics,
  and optional learning curves. That would make adding or removing models a data
  change instead of another branch in `run_cross_validation`.

- [ ] **Keep diagnostics grouped, not fragmented.**
  `confidence_by_correctness.csv` and `error_by_true_label.csv` are useful, but
  they are narrow one-off diagnostics. Prefer a grouped diagnostics table with a
  `diagnostic` column, or a `diagnostics/` directory containing a small number of
  intentionally named files. The goal is that a reader can understand the run
  from a few files without hunting through hundreds of fold artifacts.

- [ ] **Replace extraction `print` calls with level-based logging.**
  `data_prep.extract_features` prints progress, skips, removals, full
  tracebacks, and start/end messages directly. Use `logging` with concise
  default output: total videos discovered, processed/skipped/failed counts, and
  the output root. Keep per-video progress and full tracebacks for `--verbose`
  or write failures to an error report file.

- [ ] **Fix extraction CLI validation flow.**
  `extract_features.main` checks `if not args.video and not args.root` twice,
  and the first check returns before validating `--out`. Remove the dead branch
  so missing required arguments always produce one clear parser error.

- [ ] **Reduce broad `Any` usage where the data shape is stable.**
  `training.train`, `training.dataset`, `training.models`, `evaluation.metrics`,
  and `data_prep.windows` use `Any` for stable values like fold data,
  preprocessors, dataloaders, models, and metric inputs. Do not type everything
  aggressively, but add named types for stable internal payloads so refactors can
  rely on the editor and tests.

- [ ] **Update tests so they protect the quieter contract.**
  `tests/test_train.py` currently expects the existing summary files but does
  not assert that per-fold confusion matrices are consolidated or that W&B uses
  one run. Add tests for the new artifact contract: one combined confusion
  matrix table, one manifest, no per-fold file explosion, and a mock tracker
  receiving one experiment lifecycle.

- [ ] **Align docs with the simplified default command.**
  `docs/alertness_classifier_spec.md` and `docs/Assignment 6 Notes.md` document
  the current output behavior and W&B flags. After the cleanup, update them to
  describe the concise output layout, the expected W&B grouping, and which files
  are the canonical evidence for writeups.

## Suggested Output Shape

The default command should produce enough evidence to support a writeup without
creating hundreds of files:

```text
outputs/
  latest/
    manifest.json
    metric_summary.csv
    fold_metrics.csv
    video_predictions.csv
    learning_curves.csv
    confusion_matrices.csv
    diagnostics.csv
```

Optional deep artifacts can live under a separate opt-in path:

```text
outputs/
  latest/
    fold_artifacts/
      ...
```

That keeps the common workflow simple and leaves room for detailed debugging
when there is a real reason to inspect one fold in isolation.

## Suggested Terminal Summary

Default logs should be plain and grouped:

```text
Training alertness classifier
Dataset: data/frame_windows.parquet (60 subjects, 180 videos, N windows)
CV: LOSO, 60 folds, validation subjects per fold: 9
Runs: majority, perclos, luminance, mlp_regularized
Output: outputs/latest

Final video-level metrics:
  perclos         QWK 0.410 +/- ...  rank MAE ...
  mlp_regularized QWK ...            rank MAE ...

Diagnostics written:
  outputs/latest/fold_metrics.csv
  outputs/latest/confusion_matrices.csv
  outputs/latest/diagnostics.csv
```

This gives the operator what they need at the end of the command without
turning every fold or diagnostic into its own local artifact.

## Ranked Simplification Targets

1. **shrink:** replace 240 per-fold confusion CSVs with one grouped
   `confusion_matrices.csv`. Replacement: long-form table keyed by
   `run`/`fold`/`true_label`/`pred_label`.
2. **shrink:** replace 60 W&B runs per default training command with one W&B run
   containing fold tables and aggregate metrics. Replacement: experiment-level
   run plus optional per-fold mode.
3. **yagni:** remove or finish `build_ordinal_mlp`; a tested stub with no
   training path is misleading. Replacement: complete model-selection support or
   deletion until needed.
4. **shrink:** move artifact writing and W&B lifecycle out of
   `run_cross_validation`. Replacement: small result objects plus artifact and
   tracking adapters.
5. **shrink:** convert direct extraction `print` calls into concise logging.
   Replacement: summary logs by default, progress/errors behind verbosity.

