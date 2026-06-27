# Alertness Classifier — Build Specification

A complete, self-contained spec for building a subject-independent, ordinal
3-class alertness classifier from per-subject video.

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

---

## 2. Dataset

- 60 subjects. One folder per subject, three videos each.
- Layout: `<root>/<subject>/{0,5,10}.mov`.
- ~30 fps. Resolutions and orientations are mixed (some portrait, some
  landscape); this is irrelevant after landmark extraction because MediaPipe
  landmarks are normalized and blendshapes are scale-invariant.
- **Balance is automatic**: every subject contributes exactly one video per
  class, so any subject-wise split is class-balanced. No stratification by class
  and no class weighting are required.

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

---

## 4. Stage 1 — Per-frame extraction

Implemented by the accompanying `extract_features.py` (provided). It streams
each video frame by frame (RAM bounded by a single frame), runs MediaPipe
FaceLandmarker with blendshapes + transformation matrix enabled, and writes one
small CSV per video. It can run on a single video and optionally delete the
source afterward, so the full dataset never needs to be on disk at once.

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
| `frac_face_missing` | data-quality covariate (recorded; **not** fed to the model) |

**Output of Stage 2:** a single table `windows.parquet` with columns =
the summary features above + `subject_id` + `label` + `window_idx` + `video_id`.

---

## 6. Stage 3 — Encoding and normalization

**Labels:** integer ordinal `y ∈ {0,1,2}`. No one-hot. The CORN loss derives its
own extended-binary targets from the integer label.

**Per-subject normalization (leakage-safe):** baseline EAR differs by face, so
compute each subject's normalization stats from *that subject's own frames
only*, across all three of their videos, and apply within-subject. Because the
stats never cross from one subject to another, this is safe regardless of which
fold a subject lands in. Recommended: subtract the subject's median EAR (and,
optionally, divide by the subject's open-eye EAR scale). Do **not** use the
alert video as a baseline — that wouldn't exist at deployment time.

**Global feature scaling:** after per-subject normalization, fit a
`StandardScaler` **on the training fold only** and apply it to val/test. This is
the one normalization that must respect the train/test boundary.

---

## 7. Stage 4 — Data split (subject-wise; this is critical)

The unit of generalization is the **subject**, not the window or the frame.
Windows from one person are highly correlated; if any of a subject's windows
appear in both train and test, the model learns to recognize the face and test
scores become meaningless.

**Primary: subject-wise cross-validation.** Use `GroupKFold` (groups =
`subject_id`). Because the models are tiny and train in seconds on summary
features, prefer one of:
- **5-fold** (12 held-out subjects per fold), or
- **Leave-One-Subject-Out (LOSO)** (60 folds) for the gold-standard estimate
  drowsiness papers report.

Within each training fold, carve out a few subjects (also by group) as a
validation set for early stopping. Report every metric as **mean ± std across
folds** — the std is the headline measure of stability with only 60 subjects.

**Optional fixed holdout** for a single clean final number: 42 train / 9 val /
9 test subjects (seeded). Touch the test split exactly once. Note that with only
9 test subjects this number is high-variance; CV is the trustworthy estimate.

**Invariant to enforce in code:** assert that the intersection of `subject_id`
sets across train/val/test is empty in every fold.

---

## 8. Stage 5 — Model, loss, training

**Model (primary): small MLP with a CORN head.**
- Input: the ~17 summary features (drop `frac_face_missing`).
- Body: 2 hidden layers (e.g., 64 → 32), ReLU, dropout ≈ 0.3, optional batchnorm.
- Output: `K-1 = 2` logits (CORN reformulates 3-class ordinal as two cumulative
  binary questions: "past alert?" and "past low-vigilant?", with rank-consistent
  predictions).

**Loss (primary): CORN loss** from `coral_pytorch`:

```python
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

# logits: (batch, K-1=2); y: (batch,) integer labels in {0,1,2}
loss = corn_loss(logits, y, num_classes=3)
# inference:
pred_rank = corn_label_from_logits(logits)   # -> {0,1,2}
```

**Training:** Adam (lr 1e-3, weight_decay 1e-4), batch size 64, max ~200 epochs,
early stopping on **validation QWK** (patience ~20). No class weights (balanced).
Set and log all seeds (`numpy`, `torch`, Python `random`).

**Required baseline-to-beat model:** the same MLP with a 3-way softmax head and
plain cross-entropy. With only 3 classes the ordinal loss's edge is often small,
so A/B CORN vs. cross-entropy and report both — do not assume CORN wins.

**Alternative model (optional comparison):** gradient boosting (LightGBM). To
keep it ordinal, use the CORN/CORAL decomposition — train two cumulative binary
classifiers, `P(y>0)` and `P(y>1)` — rather than a plain multiclass objective.

---

## 9. Stage 6 — Inference and window→video aggregation

Train on windows; evaluate on **videos** (the real unit of interest) and report
subject-level too.

1. Predict each window: from CORN, take the per-window cumulative probabilities
   `P(y>0)`, `P(y>1)`.
2. Aggregate a video's windows by **averaging those cumulative probabilities**
   across the video's windows, then apply the CORN decision rule (count how many
   averaged cumulative probs exceed 0.5) to get the video's predicted rank.
3. There are 180 videos total → metrics are computed over the held-out videos in
   each fold, then averaged across folds.

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

---

## 11. Stage 8 — Baselines

The model's numbers are only meaningful against floors:

1. **Majority class** — always predict the most frequent class. Accuracy ≈ 33%,
   QWK ≈ 0. Sanity floor.
2. **Luminance-only (confound check)** — per-video mean of `bright_mean` and
   `warmth`; fit logistic/ordinal regression on just those two, same subject-wise
   CV. **If the real model doesn't clearly beat this, lighting is leaking** and
   the model is partly a brightness detector, not an alertness detector.
3. **PERCLOS-only** — a one-feature threshold/ordinal model on `perclos` alone.
   A strong, interpretable reference; the full model should beat it to justify
   the extra features.

**Acceptance criteria:** the chosen model must beat majority and luminance-only
by a clear margin on QWK (mean across folds, with non-overlapping std bands), and
its confusion-matrix errors should concentrate on the diagonal/off-by-one cells.

---

## 12. Build order

1. Run `data_prep.extract_features` over the dataset → `features/`.
2. `data_prep.windows` → `windows.parquet`.
3. `training.dataset` (per-subject norm + train-only scaler) and `training.splits`.
4. `baselines.py` — establish the floors first.
5. `training.models` + `training.train` — CORN model and the cross-entropy baseline, under
   subject-wise CV, aggregating windows → video.
6. `evaluation.metrics` — report mean ± std across folds vs. the baselines.

## 13. Reproducibility

Fix and log seeds for `random`, `numpy`, and `torch`; record fold assignments
(subject→fold) to disk so every run is reproducible. Pin package versions in a
lockfile.
