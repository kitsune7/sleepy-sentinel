# Part A — Portfolio tabular representation and model comparison

> Scope boundary: one focused comparison, not a model tournament. A small number
> of diagnostic variants is fine; no exhaustive hyperparameter/architecture search.

## Problem, stakeholder, dataset, target, inputs — and data-type declaration

> State the portfolio problem, client/stakeholder scenario, dataset, prediction
> target, candidate inputs, and whether Part A uses native tabular portfolio
> data, a portfolio-derived proxy, or approved fallback case-study materials.

- **Problem / stakeholder:** The portfolio problem is a personal early-fatigue aid: given a short live-video window, estimate whether the user appears alert, low-vigilant, or drowsy. The immediate stakeholder is me / a future self-monitoring user, not an employer, clinician, or safety-critical operator.
- **Dataset:** UTA-RLDD, 60 subjects × 3 videos → per-window feature CSVs in `data/<subject>/{0,5,10}.csv`. The LOSO outputs cover 180 held-out videos, with each test fold holding out one subject's three videos. Each video contributes roughly 50-116 windows depending on frame availability.
- **Target:** ordinal `y ∈ {0,1,2}` = {alert, low-vigilant, drowsy}, scored at video level.
- **Candidate inputs:** numeric geometric/temporal summaries extracted from the video windows, including eye-aspect-ratio features, blink/PERCLOS-style eye closure summaries, mouth/yawn-related features, head-pose summaries, and quality/confound fields such as face-missing fraction and luminance.
- **Data-type:** Native tabular portfolio data. The raw modality is video, but our portfolio representation _is already_ a tabular feature table extracted in A-prior work — so I use the portfolio dataset directly.

## Proxy / fallback notes

> If using a proxy, describe the transformation and what it preserves/loses.
> If using fallback materials, explain the blocker.

- Technically, the tabular data is a proxy for the original video stream: it preserves selected fatigue-relevant geometric, eye-closure, yawn, head-pose, and quality cues, while losing subtle video cues that were not extracted into the window table. This proxy is not new for Assignment 6; it is the portfolio's working representation.
- No fallback case study is needed because the portfolio data is downloaded, labeled, and usable for research.

## Feature preparation (leakage + prediction-time availability)

> Prepare numeric/categorical/missing/rare/high-cardinality features without
> leakage; keep prediction-time availability clear.

- **All features numeric** → no categorical encoding / no high-cardinality handling needed. This is also why embeddings are N/A.
- **Leakage controls:**
  - Train-only `StandardScaler` — `dataset.prepare_fold_datasets`.
  - Subject-wise split, disjointness asserted — `splits.assert_disjoint_subjects`.
  - Per-subject EAR normalization uses only that subject's own frames; alert video never used as baseline.
- **Missingness:** low-quality windows are controlled through the
  `frac_face_missing` feature/gate and the existing interpolation rule from the feature pipeline, so both the MLP and the baselines see the same usable-window population.
- **Prediction-time availability:** every feature computable from one live window. PERCLOS, luminance, and the full MLP feature set are therefore available at inference time without using future frames or another subject's data.
- **Baseline-prep note:** the simple baselines use narrow feature slices (majority, luminance-only, PERCLOS-only), so they do not need the full MLP feature scaling path. The MLP keeps the training-fold-only scaler.

## Simple baseline model

> Train at least one simple baseline (logistic/linear, tree-based, or other
> classical model).

- **Chosen baseline:** A logistic regression using only PERCLOS is the primary simple baseline because it is clinically/task-relevant, prediction-time available, and much easier to explain than the full MLP. I also ran majority and luminance-only floors.
- **Key settings:** the baseline models use the same subject-wise LOSO folds, the same window→video aggregation, and the same video-level QWK/rank-MAE/
  accuracy/macro-F1 reporting. The majority baseline is a floor; luminance-only is the confound check; PERCLOS-only is the serious simple-model comparator.
- **Ordinal handling:** predictions are still evaluated as the same three ordered classes `{0,1,2}`. QWK and rank-MAE therefore preserve the ordinal penalty even when the model itself is a simple classifier/baseline.
- **Outputs:** root `outputs/metric_summary.csv`, `outputs/fold_metrics.csv`,
  `outputs/video_predictions.csv`, and per-fold confusion matrices.

## Neural tabular model

> Train at least one neural model appropriate for tabular data (MLP / embeddings).

- **Reuse/re-run the assignment 5 MLP** (`(64,32)` ReLU, the regularized variant: dropout 0.25 + wd 1e-4), but evaluate it in the cleaner assignment 6 setup: LOSO rather than 5-fold GroupKFold, and validation-aware checkpoint selection.
- **Generalization practices (required for the MLP):**
  - Training-split normalization — already done with the scaler fit only on the training fold.
  - **Validation-aware early stopping / checkpoint selection** — added for this assignment. The MLP stopped early in 58/60 folds; best epoch had mean 6.0, median 4.0, min 1, max 36. This confirms the assignment 5 diagnosis that fixed 100-epoch training was overtraining.
  - One regularization control — dropout + weight decay already in place.
- **Outputs:** root `outputs/metric_summary.csv`, `outputs/fold_metrics.csv`,
  `outputs/learning_curves.csv`, and `outputs/video_predictions.csv`.

## Categorical embeddings — appropriate or not?

> Use categorical embeddings if appropriate; if not, briefly explain why.

- **Not appropriate.** The model inputs are continuous numeric geometric/temporal summaries, not categorical tokens or high-cardinality IDs. Adding categorical embeddings would either require inventing categorical features that are not part of the intended inference signal, or embedding subject identity, which would be leakage for the actual goal of generalizing to a new person.

## Fair comparison — same target, split, metrics

> Compare using the same target, split/eval procedure, and task-relevant
> metrics. Validation evidence for selection; test reserved for final reporting.

- All runs use the same LOSO `folds.json`, same window→video aggregation, and same video-level metrics.
- For the MLP, validation subjects inside each training fold select the checkpoint / early-stopping epoch. The reported numbers are held-out test videos: one unseen subject and three videos per fold, repeated across 60 folds.
- Because each LOSO test fold contains only three videos, fold-level metrics are very quantized. The mean ± std across folds is still the honest LOSO report, but pooled 180-video summaries and summed confusion matrices are useful interpretation aids.

## Results

> Compact table: each model, key settings, metrics, practical notes.

| Model             | Key settings                                | QWK (mean±std) | rank-MAE (mean±std) | Accuracy (mean±std) | macro-F1 (mean±std) | Practical notes |
| ----------------- | ------------------------------------------- | -------------- | ------------------- | ------------------- | ------------------- | --------------- |
| Majority          | predicts the majority class                 | 0.000 ± 0.000  | 0.944 ± 0.125       | 0.333 ± 0.000       | 0.167 ± 0.000       | sanity floor |
| Luminance-only    | confound baseline                           | 0.048 ± 0.502  | 0.844 ± 0.400       | 0.361 ± 0.248       | 0.258 ± 0.244       | near chance; brightness is not enough |
| PERCLOS-only      | simple task-specific tabular baseline       | **0.410 ± 0.414** | **0.617 ± 0.357** | **0.517 ± 0.241** | **0.405 ± 0.272** | simplest serious model; slightly best |
| MLP (regularized) | `(64,32)`, dropout .25, wd 1e-4, early stop | 0.363 ± 0.382  | 0.650 ± 0.339       | 0.483 ± 0.233       | 0.370 ± 0.254       | more complex; does not beat PERCLOS |

Pooled across the 180 held-out videos, the same pattern holds: PERCLOS-only scores QWK 0.409 / accuracy 0.517, while the regularized MLP scores QWK 0.353 / accuracy 0.483. Both reduce the dangerous off-by-two alert↔drowsy errors to 24/180, compared with 37/180 for luminance and 50/180 for majority.

Summed-over-folds confusion matrices (rows = true, columns = predicted):

```
PERCLOS-only                      MLP regularized
         pred a  l  d                      pred a  l  d
true alert   48  7  5             true alert   36 13 11
true low     32 14 14             true low     25 18 17
true drowsy  19 10 31             true drowsy  13 14 33
```

PERCLOS is much better on true alert videos (48/60 correct vs. 36/60) and the MLP is slightly better on true drowsy videos (33/60 vs. 31/60). Both struggle with the low-vigilant middle class, which is expected for an ordinal midpoint.

## Commands and Logs

I ran `uv run train_alertness --wandb-project sleepy-sentinel` to get the output logs in `./outputs` and W&B. Without W&B, run `uv run train_alertness`. This runs the full pipeline with the default parameters below.

### Default parameters

| Flag | Type | Default | Help |
| --- | --- | --- | --- |
| `windows_path` | `Path` | `data/frame_windows.parquet` | |
| `output_dir` | `Path` | `outputs` | |
| `--epochs` | `int` | `40` | |
| `--batch-size` | `int` | `64` | |
| `--random-seed` | `int` | `42` | |
| `--learning-rate` | `float` | `1e-3` | |
| `--n-splits` | `int` | `None` | Override LOSO with grouped K-fold CV. |
| `--validation-subject-count` | `int` | `9` | |
| `--early-stopping-patience` | `int` | `8` | Validation-QWK patience; negative disables. |
| `--wandb-project` | `str` | `None` | Enable W&B logging for each model/fold run. |
| `--wandb-entity` | `str` | `None` | Optional W&B team or username. |
| `--wandb-mode` | `online`, `offline`, or `disabled` | `None` | |
| `--wandb-group` | `str` | `None` | Optional W&B group name for the CV run. |

The seed is fixed at `42` by default. PyTorch training can still have small hardware/library nondeterminism, so the saved CSVs in `outputs/` are the specific run evidence used for this writeup.

## Dataset citation

Ghoddoosian, Galib, and Athitsos, "A Realistic Dataset and Baseline Temporal
Model for Early Drowsiness Detection," CVPR Workshops 2019.

## Practical constraints

> Interpretability, cost, training/inference complexity, maintainability, data
> size, ease of monitoring.

- The practical comparison favors PERCLOS-only right now. It is transparent, cheap to compute, easy to monitor, and much easier to explain: the system is mostly reacting to eye-closure behavior rather than an opaque mix of 17 features.
- The MLP has higher operational complexity: a scaler artifact, a learned checkpoint, validation-aware training, and more monitoring burden. With only 60 subjects, that complexity would be justified only if it clearly beat the simpler baseline. It does not.
- The small dataset remains the limiting factor. LOSO is stricter and more appropriate than the earlier 5-fold report, but each test fold has only one subject / three videos, so the across-fold standard deviations are large.

## Responsible-use concern (tabular-specific)

> At least one: sensitive features, proxy variables, fairness across groups,
> automation bias, human-review needs.

- **Proxy-variable / lighting confound:** this is now partly tested. The luminance-only baseline is near chance (QWK 0.048, accuracy 0.361), so the current signal is probably not just brightness. That reduces, but does not eliminate, the confound risk because lighting can still interact with face and eye tracking quality.
- **Fairness and representativeness:** UTA-RLDD has limited subject diversity, and the model is evaluated on only 60 people. A face/eye-based tabular model could behave differently across glasses, facial hair, skin tone, camera angle, and lighting conditions even if those attributes are not explicit features.
- **Automation bias:** confidence remains weakly informative. In the assignment 6 outputs, MLP mean confidence is 0.480 on correct predictions vs. 0.435 on incorrect predictions; PERCLOS is 0.449 vs. 0.410. Those small gaps are not enough to support a high-stakes automated warning threshold.

## Recommendation

> Is a tabular approach justified? Is the neural model justified over the baseline?

- **Tabular approach justified?** Yes. It is the native representation for this portfolio project, and both PERCLOS-only and the MLP beat the majority and luminance floors.
- **Neural over baseline?** No, not yet. The regularized MLP does not beat the simpler PERCLOS-only baseline under LOSO: QWK 0.363 vs. 0.410, accuracy 0.483 vs. 0.517, and rank-MAE 0.650 vs. 0.617. The bands are wide and overlapping, but the burden of proof is on the more complex model. By the charter's simpler-and-equal rule, the current recommendation is to treat PERCLOS-only as the stronger candidate until a richer model clearly improves the dangerous off-by-two errors or the drowsy-class recall without sacrificing simplicity.

---

# Part B — Portfolio checkpoint and model-choice note

> Keep concise but specific.

- **Current data readiness:** features are extracted, subject-wise folds are saved, leakage checks are in place, and the root `outputs/` directory now has LOSO metrics, video predictions, learning curves, and confusion matrices for majority, luminance, PERCLOS, and the regularized MLP.
- **Current baseline / model status:** the assignment 5 regularized MLP is re-evaluated with early stopping under LOSO. Stage 0 floors are now implemented. PERCLOS is the best current candidate; luminance is near chance; the MLP is useful but has not earned its extra complexity.
- **One concrete next experiment:** test whether the full feature set can beat PERCLOS with a simpler interpretable tabular model (for example logistic/
  ordinal regression or a shallow tree model), using the exact same LOSO folds. If it cannot beat PERCLOS, keep PERCLOS.
- **Expected staged improvement before final package:** compare CORN or another ordinal head against both the regularized MLP and PERCLOS-only, but require a real gain on QWK/rank-MAE and the alert↔drowsy off-by-two corners. A higher mean alone is not enough if the fold bands remain this wide.
- **How Week 6 evidence affects the final model-choice argument:** the final argument now leans simpler. The tabular representation is validated, but the neural model is not yet justified over the PERCLOS-only baseline.
- **Charter/audit updates, emphasis, or still-untested:** lighting as a sole explanation is reduced by the luminance baseline, but subgroup robustness and confidence calibration remain unproven.
- **Relevance of tabular methods / embeddings / simpler baselines:** tabular =
  **directly relevant** (native rep); embeddings = **not relevant** (no categoricals); simpler baselines = **directly relevant** (the whole point).
