# Alertness Classifier — Build Specification

A complete, self-contained spec for building a subject-independent, ordinal
3-class alertness classifier from per-subject video. This document describes the
intended design **and** the current project state as of Assignment 6 (June 2026).

---

## 0. Current project status

| Stage | Component | Status |
|-------|-----------|--------|
| 1 | Per-frame feature extraction (`data_prep.extract_features`) | **Done** |
| 2 | Windowing + summary features (`data_prep.windows` → `data/frame_windows.parquet`) | **Done** |
| 3 | Per-subject EAR norm + train-only scaler (`training.dataset`) | **Done** |
| 4 | Subject-wise CV splits (`training.splits`, saved to `outputs/folds.json`) | **Done** — **LOSO is the primary evaluation** |
| 5 | Neural model training (`training.train`, `training.models`) | **Done** — cross-entropy MLP, opt-in via `--include-mlp`; CORN head **not implemented** |
| 6 | Window→video aggregation + metrics | **Done** |
| 7 | Evaluation metrics (`evaluation.metrics`) | **Done** |
| 8 | Baselines (`training.baselines`) | **Done** — majority, luminance-only, PERCLOS-only |

**Current best candidate:** PERCLOS-only logistic regression (QWK 0.410 under LOSO).
The regularized cross-entropy MLP does not beat it and has not earned its extra
complexity.

**Primary evaluation:** Leave-One-Subject-Out (60 folds, 180 held-out videos).
Assignment 5 used 5-fold GroupKFold as an intermediate step; LOSO is the
gold-standard estimate going forward.

**Run training:**

```bash
uv run train_alertness                          # baselines only → writes to outputs/
uv run train_alertness --include-mlp            # also train the regularized MLP
uv run train_alertness --wandb-project sleepy-sentinel   # optional W&B (one run per command)
```

The default run trains only the baselines (majority, luminance-only,
PERCLOS-only), since PERCLOS-only is the strongest candidate and the MLP does
not beat it. Pass `--include-mlp` to add the regularized MLP. Every run writes a
fixed, flat set of files into `outputs/` (see §10/§15).

---

## 1. Objective

Classify a person's alertness from short video into three **ordinal** states:

| Class | Name          | Source file |
|-------|---------------|-------------|
| 0     | alert         | `0.mov`     |
| 1     | low-vigilant  | `5.mov`     |
| 2     | drowsy        | `10.mov`    |

The order matters: `alert < low-vigilant < drowsy`. Mislabeling alert as drowsy
(off-by-two) is a worse error than alert as low-vigilant (off-by-one), and the
model, loss, and metrics must all respect that ordering.

We do **not** model raw pixels. Each frame is reduced to a small set of
geometric/physiological signals via MediaPipe FaceLandmarker, those are
aggregated into interpretable per-window summary features, and a small ordinal
model is trained on the summaries. This sidesteps lighting/appearance confounds
(a warm or dim room can't leak into a geometric feature) and shrinks ~111 GB of
video to a few hundred MB of text.

**Stakeholder context:** personal early-fatigue aid for self-monitoring — not a
clinical, employer, or safety-critical system.

---

## 2. Dataset

- 60 subjects. One folder per subject, three videos each.
- Layout: `<root>/<subject>/{0,5,10}.mov`.
- ~30 fps. Resolutions and orientations are mixed (some portrait, some
  landscape); this is irrelevant after landmark extraction because MediaPipe
  landmarks are normalized and blendshapes are scale-invariant.
- **Balance is automatic**: every subject contributes exactly one video per
  class, so any subject-wise split is class-balanced. No stratification by class
  is required for the MLP; baselines use `class_weight="balanced"` in logistic
  regression as a minor safeguard.
- Each video contributes roughly 50–116 windows depending on frame availability.
- **Citation:** Ghoddoosian, Galib, and Athitsos, "A Realistic Dataset and
  Baseline Temporal Model for Early Drowsiness Detection," CVPR Workshops 2019.

---

## 3. Pipeline overview

```
.mov  --[Stage 1: extract_features.py]-->  per-frame CSV  (one per video)
      --[Stage 2: windowing + summaries]-->  windows table (one row per window)
      --[Stage 3: encode + per-subject normalize]-->  model-ready arrays
      --[Stage 4: subject-wise CV split]-->  train/val/test folds
      --[Stage 5: train ordinal model]-->  fitted model per fold
      --[Stage 6: predict windows -> aggregate to video]
      --[Stage 7: metrics]  vs  [Stage 8: baselines]
```

All stages are implemented. Stage 5 trains a cross-entropy MLP (opt-in via
`--include-mlp`); the CORN ordinal head is not implemented (see §15).

---

## 4. Stage 1 — Per-frame extraction

Implemented by `data_prep.extract_features`. It streams each video frame by
frame (RAM bounded by a single frame), runs MediaPipe FaceLandmarker with
blendshapes + transformation matrix enabled, and writes one small CSV per video.

**Per-frame CSV columns:**

| Column                     | Meaning |
|----------------------------|---------|
| `frame_idx`, `t_ms`        | frame number and timestamp (ms) |
| `face`                     | 1 if a face was detected, else 0 |
| `eye_blink_l`,`eye_blink_r`| blendshape eye-closure 0–1 (1 = closed) |
| `ear`                      | eye aspect ratio (geometric cross-check) |
| `jaw_open`                 | blendshape jaw-open 0–1 (yawn signal) |
| `pitch`,`yaw`,`roll`       | head pose in degrees |
| `bright_mean`,`warmth`     | whole-frame brightness and R/B ratio (for the luminance baseline only) |

Frames with no detected face have `face=0` and `NaN` feature values — never
treat a missing face as "eyes closed."

---

## 5. Stage 2 — Windowing and summary features

Slide a fixed window over each video's per-frame CSV.

**Window parameters (configurable; these defaults assume multi-minute clips):**
- `WINDOW_SEC = 15`, `STRIDE_SEC = 7.5` (50% overlap).
- At ~30 fps that is 450 frames, stride 225.
- Each window inherits its **video's label**.
- **Validity gate:** drop any window where `face == 0` for more than 30% of its
  frames. If a clip is so short it yields fewer than ~3 valid windows, reduce
  `WINDOW_SEC`/`STRIDE_SEC` rather than accept one-window videos.

**Pre-window cleaning, per video:**
- Apply a short median filter (≈5 frames) to `eye_blink_*`, `ear`, and
  `jaw_open` to suppress single-frame landmark jitter before event detection.
- Linearly interpolate feature gaps shorter than ~0.3 s; leave longer gaps as
  NaN and let the validity gate handle them.

**Eye-closed and yawn definitions (used by the summaries):**
- `eye_closed[t] = mean(eye_blink_l[t], eye_blink_r[t]) > 0.5`. The blendshape
  is already person-normalized, which mostly avoids the per-person EAR-baseline
  problem; EAR is retained only as a cross-check feature.
- A **blink** = a maximal run of `eye_closed` frames; record its duration.
- A **yawn** = `jaw_open > 0.5` sustained for ≥ 0.5 s.

**Per-window summary features (the model input vector):**

| Feature | Definition |
|---|---|
| `perclos` | fraction of frames with `eye_closed` (proxy for PERCLOS) |
| `blink_rate` | blinks per minute |
| `blink_dur_mean`, `blink_dur_max` | blink duration stats (seconds) |
| `eye_blink_mean`, `eye_blink_std` | mean/std of eye-closure signal |
| `ear_mean`, `ear_std`, `ear_min` | EAR stats (after per-subject normalization, §6) |
| `jaw_open_mean`, `jaw_open_max` | jaw-open stats |
| `yawn_count` | yawns in the window |
| `pitch_mean`, `pitch_std`, `pitch_range` | head pitch (nodding) |
| `frac_head_down` | fraction of frames with pitch beyond a downward threshold |
| `yaw_std`, `roll_std` | head stability |
| `frac_face_missing` | data-quality covariate (recorded; **not** fed to the MLP) |

Luminance columns (`bright_mean`, `bright_std`, `warmth_mean`, `warmth_std`,
`warmth`) may appear in the window table when present in the frame CSVs; they
are used **only** by the luminance-only baseline, not by the MLP.

**Output of Stage 2:** `data/frame_windows.parquet` with the summary features
above + `subject_id` + `label` + `window_idx` + `video_id`.

---

## 6. Stage 3 — Encoding and normalization

**Labels:** integer ordinal `y ∈ {0,1,2}`. No one-hot. The CORN loss (when
implemented) derives its own extended-binary targets from the integer label.

**Per-subject normalization (leakage-safe):** baseline EAR differs by face, so
compute each subject's normalization stats from *that subject's own frames
only*, across all three of their videos, and apply within-subject. Because the
stats never cross from one subject to another, this is safe regardless of which
fold a subject lands in. Recommended: subtract the subject's median EAR (and,
optionally, divide by the subject's open-eye EAR scale). Do **not** use the
alert video as a baseline — that wouldn't exist at deployment time.

**Global feature scaling:** after per-subject normalization, fit a
`StandardScaler` **on the training fold only** and apply it to val/test. This is
the one normalization that must respect the train/test boundary. Implemented in
`dataset.prepare_fold_datasets`. Simple baselines (majority, luminance, PERCLOS)
fit their own per-fold imputer/scaler on their narrow feature slices and do not
use the MLP's full-feature scaling path.

**Categorical embeddings:** not appropriate. All model inputs are continuous
numeric summaries. Embedding subject identity would be leakage; inventing
categorical tokens would add features not available at inference.

---

## 7. Stage 4 — Data split (subject-wise; this is critical)

The unit of generalization is the **subject**, not the window or the frame.
Windows from one person are highly correlated; if any of a subject's windows
appear in both train and test, the model learns to recognize the face and test
scores become meaningless.

**Primary (current): Leave-One-Subject-Out (LOSO).** 60 folds, one held-out
subject (3 videos) per fold. This is the gold-standard estimate and the basis
for all Assignment 6 results. Each LOSO test fold has only three videos, so
fold-level metrics are very quantized; pooled 180-video summaries and summed
confusion matrices are useful interpretation aids alongside mean ± std.

**Secondary (historical): 5-fold GroupKFold.** Used in Assignment 5
(12 held-out subjects per fold). Still available via `--n-splits 5`. Useful for
faster iteration but less strict than LOSO.

Within each training fold, carve out **9 subjects** (by group) as a validation
set for early stopping and checkpoint selection. Report every metric as **mean ±
std across folds** — the std is the headline measure of stability with only 60
subjects.

**Optional fixed holdout** for a single clean final number: 42 train / 9 val /
9 test subjects (seeded). Touch the test split exactly once. Not yet run; with
only 9 test subjects this number is high-variance — CV is the trustworthy
estimate.

**Invariant enforced in code:** `splits.assert_disjoint_subjects` asserts that
train/val/test `subject_id` sets are disjoint in every fold. Fold assignments
are saved to `outputs/folds.json`.

---

## 8. Stage 5 — Model, loss, training

### Implemented: regularized cross-entropy MLP

This model is trained when you pass `--include-mlp`; the default run is
baselines only.

- **Architecture:** input ~17 summary features (all columns except IDs, label,
  and `frac_face_missing`) → hidden `(64, 32)` → ReLU → dropout 0.25 → 3-way
  softmax logits.
- **Loss:** plain cross-entropy.
- **Optimizer:** Adam (lr 1e-3, weight_decay 1e-4).
- **Batch size:** 64.
- **Early stopping:** validation QWK, patience 8 (default). Disabled with
  `--early-stopping-patience -1`.
- **Max epochs:** 40 (default; was 100 in Assignment 5 before early stopping
  was added).
- **Seeds:** fixed at 42 by default; per-fold seed offset for training.

**Assignment 5 finding (5-fold, fixed 100 epochs):** dropout 0.25 + weight
decay 1e-4 improved QWK from 0.279 ± 0.145 to 0.400 ± 0.158, but train/val
learning curves showed clear overfitting — validation loss climbed while
training loss kept falling.

**Assignment 6 finding (LOSO, early stopping):** early stopping fired in 58/60
folds; best epoch mean 6.0, median 4.0. This confirmed that fixed long training
was overfitting. Under LOSO the regularized MLP scores QWK 0.363 ± 0.382 — below
PERCLOS-only (0.410 ± 0.414).

### Planned: CORN ordinal head

The original primary design — not yet implemented in the training loop:

- Same MLP body, but output `K-1 = 2` logits (CORN reformulates 3-class ordinal
  as two cumulative binary questions).
- **Loss:** CORN loss from `coral_pytorch`:

```python
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

# logits: (batch, K-1=2); y: (batch,) integer labels in {0,1,2}
loss = corn_loss(logits, y, num_classes=3)
pred_rank = corn_label_from_logits(logits)   # -> {0,1,2}
```

The CORN ordinal head is **not implemented** — there is no ordinal-MLP builder
in `training.models`. The CORN vs. cross-entropy A/B remains an open experiment
(§15).

### Planned: alternative tabular models

- **Full-feature logistic / ordinal regression** — interpretable test of whether
  the ~17 features add signal beyond PERCLOS alone.
- **Shallow tree model** (e.g. LightGBM with CORAL-decomposed cumulative
  binary classifiers) — optional comparison if simpler models don't win.

---

## 9. Stage 6 — Inference and window→video aggregation

Train on windows; evaluate on **videos** (the real unit of interest).

**Current implementation:**
1. Predict each window: softmax probabilities for classes `{0,1,2}`.
2. Aggregate a video's windows by **averaging class probabilities** across
   windows, then take the argmax for the video-level prediction.
3. Metrics computed over held-out videos in each fold, then averaged across folds.

**Planned for CORN:** average cumulative probabilities `P(y>0)`, `P(y>1)` across
windows, then apply the CORN decision rule (count how many averaged cumulative
probs exceed 0.5).

All models and baselines share the same aggregation path in
`train.aggregate_window_predictions`, so comparisons stay fair.

---

## 10. Stage 7 — Metrics

Do **not** lead with accuracy; it ignores the ordering. Headline trio:

- **Quadratic Weighted Kappa (QWK)** — primary single number.
  `sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")`.
  Penalizes far-apart disagreements quadratically, corrects for chance.
- **MAE on ranks** — `mean(|y_pred - y_true|)` with labels as 0/1/2. An
  off-by-two costs double an off-by-one; directly interpretable.
- **Confusion matrix** (3×3) — shows *where* errors land. Mass on/near the
  diagonal (adjacent confusions) is acceptable; mass in the top-right /
  bottom-left corners (alert↔drowsy) is the real failure.

Secondary (report but don't optimize): macro-F1, accuracy, Spearman correlation.

**Reporting format:** every metric as **mean ± std across CV folds**, at the
**video level**. Include the aggregated confusion matrix summed over folds.
Additional diagnostic outputs: `confusion_matrices.csv` (one long-form table,
columns `run, fold, true_label, pred_label, count`), `diagnostics.csv` (one
grouped table with a `diagnostic` column covering confidence-by-correctness and
error-by-true-label), and `learning_curves.csv` (MLP only).

---

## 11. Stage 8 — Baselines

Implemented in `training.baselines`. All use the same LOSO folds, window→video
aggregation, and video-level metrics as the MLP.

| Baseline | Implementation | Purpose |
|----------|----------------|---------|
| **Majority class** | always predict training majority | sanity floor |
| **Luminance-only** | logistic regression on available luminance columns | confound check — is the signal just brightness? |
| **PERCLOS-only** | logistic regression on `perclos` alone | interpretable task-specific reference |

Logistic baselines use median imputation + StandardScaler + balanced class
weights, fit per fold on training rows only.

**Confound check result (Assignment 6):** luminance-only is near chance (QWK
0.048 ± 0.502, accuracy 0.361 ± 0.248). The current signal is probably not
*just* brightness, though lighting can still interact with face/eye tracking
quality.

**Acceptance criteria (updated with evidence):**
- Beat majority and luminance-only by a clear margin → **met** by PERCLOS and MLP.
- Beat PERCLOS-only to justify the full feature set → **not met** by the MLP.
- Confusion-matrix errors concentrate on diagonal/off-by-one, not alert↔drowsy
  corners → **partially met** (24/180 off-by-two for PERCLOS and MLP vs. 50/180
  for majority).

By the charter's **simpler-and-equal rule**, PERCLOS-only is the current
recommended candidate until a richer model clearly improves QWK/rank-MAE and the
alert↔drowsy off-by-two corners.

---

## 12. Results to date

### Assignment 5 — 5-fold GroupKFold, cross-entropy MLP, fixed 100 epochs

| Model | QWK (mean±std) | rank-MAE (mean±std) | Accuracy (mean±std) | macro-F1 (mean±std) |
|-------|----------------|---------------------|---------------------|---------------------|
| MLP baseline (no reg) | 0.279 ± 0.145 | 0.694 ± 0.119 | 0.433 ± 0.061 | 0.428 ± 0.068 |
| MLP regularized (dropout 0.25 + wd 1e-4) | **0.400 ± 0.158** | **0.611 ± 0.124** | **0.489 ± 0.082** | **0.483 ± 0.082** |

Key learnings: regularization helped but fold bands overlap heavily; learning
curves showed train/val divergence → early stopping needed. Luminance and
PERCLOS baselines were not yet implemented.

### Assignment 6 — LOSO, baselines + regularized MLP with early stopping

| Model | QWK (mean±std) | rank-MAE (mean±std) | Accuracy (mean±std) | macro-F1 (mean±std) | Notes |
|-------|----------------|---------------------|---------------------|---------------------|-------|
| Majority | 0.000 ± 0.000 | 0.944 ± 0.125 | 0.333 ± 0.000 | 0.167 ± 0.000 | sanity floor |
| Luminance-only | 0.048 ± 0.502 | 0.844 ± 0.400 | 0.361 ± 0.248 | 0.258 ± 0.244 | near chance |
| **PERCLOS-only** | **0.410 ± 0.414** | **0.617 ± 0.357** | **0.517 ± 0.241** | **0.405 ± 0.272** | **current best** |
| MLP regularized | 0.363 ± 0.382 | 0.650 ± 0.339 | 0.483 ± 0.233 | 0.370 ± 0.254 | does not beat PERCLOS |

Pooled across 180 held-out videos: PERCLOS QWK 0.409 / accuracy 0.517; MLP QWK
0.353 / accuracy 0.483. Both reduce alert↔drowsy off-by-two errors to 24/180
(vs. 37/180 luminance, 50/180 majority).

Summed-over-folds confusion matrices (rows = true, columns = predicted):

```
PERCLOS-only                      MLP regularized
         pred a  l  d                      pred a  l  d
true alert   48  7  5             true alert   36 13 11
true low     32 14 14             true low     25 18 17
true drowsy  19 10 31             true drowsy  13 14 33
```

**Error-pattern notes:**
- PERCLOS is much better on true alert videos (48/60 vs. 36/60).
- MLP is slightly better on true drowsy videos (33/60 vs. 31/60).
- Both struggle with the low-vigilant middle class (expected for an ordinal midpoint).
- Confidence is weakly informative: MLP mean confidence 0.480 on correct vs.
  0.435 on incorrect; PERCLOS 0.449 vs. 0.410. Not sufficient for a high-stakes
  automated warning threshold without calibration.

**Practical recommendation:** tabular approach is validated; neural model is not
yet justified over PERCLOS-only. PERCLOS is transparent, cheap, easy to monitor,
and easier to explain.

---

## 13. Build order and implementation map

| Step | Module | Status |
|------|--------|--------|
| 1. Extract features | `data_prep.extract_features` | Done |
| 2. Build windows | `data_prep.windows` → `data/frame_windows.parquet` | Done |
| 3. Dataset + splits | `training.dataset`, `training.splits` | Done |
| 4. Baselines | `training.baselines` | Done |
| 5. Neural model + training | `training.models`, `training.train` | Done (cross-entropy only) |
| 6. Metrics | `evaluation.metrics` | Done |
| 7. CORN ordinal head | `training.models` + `training.train` | **Not started** |
| 8. Full-feature interpretable model | extend `training.baselines` or new module | **Not started** |
| 9. LightGBM comparison | new module | **Not started** (optional) |
| 10. Confidence calibration | post-hoc on saved predictions | **Not started** |
| 11. Subgroup error analysis | slice `diagnostics.csv` (error-by-true-label rows) by covariates | **Not started** |

---

## 14. Open experiments and model-choice criteria

These are the remaining experiments before a final model recommendation. Scope
boundary: one focused comparison, not a model tournament.

### Priority 1 — Full-feature interpretable model vs. PERCLOS

Test whether the full ~17-feature set beats PERCLOS with a simpler interpretable
tabular model (logistic/ordinal regression or a shallow tree), using the exact
same LOSO folds and aggregation. **If it cannot beat PERCLOS, keep PERCLOS.**

### Priority 2 — CORN ordinal head vs. cross-entropy MLP

Add an ordinal-MLP head + CORN loss and compare against the regularized
cross-entropy MLP and PERCLOS-only. Require a real gain on QWK, rank-MAE, and
the alert↔drowsy off-by-two corners — a higher mean alone is not enough if fold
bands remain this wide.

### Priority 3 — Confidence calibration

Temperature scaling or reliability-diagram analysis on saved predictions.
Possibly evaluate CORN cumulative probabilities as a confidence signal. Needed
before any confidence threshold could drive a warning.

### Priority 4 — Subgroup robustness (if feasible)

Break out errors by glasses, facial hair, gender, or other available covariates
where sample size permits. UTA-RLDD has limited diversity (51M/9F; skewed
ethnicity); a face/eye-based model may behave differently across groups even
when those attributes are not explicit features.

### Optional — LightGBM with CORAL decomposition

Gradient boosting with two cumulative binary classifiers (`P(y>0)`, `P(y>1)`)
as an alternative tabular model if simpler options don't win.

### Model-choice bar (unchanged from charter)

A richer model must **clearly** beat PERCLOS on QWK/rank-MAE **and** reduce
dangerous off-by-two errors, with tighter fold bands — not just a higher mean
with overlapping std. If statistically indistinguishable, **simpler wins**.

---

## 15. Reproducibility

Fix and log seeds for `random`, `numpy`, and `torch`; record fold assignments
to `outputs/folds.json` so every run is reproducible. Pin package versions in
`uv.lock`.

**Default CLI parameters** (`uv run train_alertness`):

| Flag | Default |
|------|---------|
| `windows_path` | `data/frame_windows.parquet` |
| `output_dir` | `outputs` |
| `--epochs` | 40 |
| `--batch-size` | 64 |
| `--random-seed` | 42 |
| `--learning-rate` | 1e-3 |
| `--n-splits` | `None` (LOSO) |
| `--validation-subject-count` | 9 |
| `--early-stopping-patience` | 8 |
| `--include-mlp` | off (baselines only) |
| `--verbose` | off (concise summary) |

W&B is optional and off unless `--wandb-project` is given. The command-level CV
experiment is logged as a **single** W&B run, with per-fold (`fold_metrics`) and
per-epoch (`learning_curves`) numbers as W&B Tables under it. The legacy
one-run-per-model/fold behavior is available via `--wandb-per-fold-runs`.

PyTorch training can still have small hardware/library nondeterminism; saved
CSVs in `outputs/` are the run evidence for writeups.

**Output artifacts (flat, fixed contract):** every run writes exactly these
files into `outputs/` and nothing else — `manifest.json` (config, dataset path,
cv_strategy, seed, package_versions, git_sha, started_at/ended_at),
`metric_summary.csv`, `fold_metrics.csv`, `video_predictions.csv`,
`learning_curves.csv` (header-only unless the MLP ran), `confusion_matrices.csv`
(one long-form table: `run, fold, true_label, pred_label, count`),
`diagnostics.csv` (one grouped table with a `diagnostic` column),
`folds.json`, and `split_summaries.csv`.

---

## 16. Responsible-use limitations

- **Not safety-critical.** Personal awareness aid only.
- **Proxy-variable / lighting confound:** partly tested — luminance baseline near
  chance, but lighting can still interact with tracking quality.
- **Fairness and representativeness:** 60 subjects, limited demographic diversity.
  Subgroup error rates may differ; re-validate on the target user before any
  deployment.
- **Automation bias:** confidence gaps between correct and incorrect predictions
  are small (~0.04–0.05). Raw softmax/logistic confidence is not yet a
  trustworthy gate.
- **Privacy:** raw video stays local; only aggregate features/metrics are shared.
