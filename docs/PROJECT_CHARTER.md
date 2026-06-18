# Portfolio Project Charter — Sleepy Sentinel

## Portfolio problem statement

Some models exist that detect sleepiness, but they do so at an individual frame level, can be fooled easily, and they fail to detect subtle signs of fatigue that could give people a chance to respond appropriately when caught early.

## Use case

A personal early-warning aid for *my own* fatigue — a tool that could eventually flag drowsiness on my laptop during long work or study sessions, when my own judgment of how tired I am is least reliable.

## Dataset source, intended use, and access status

- **Source:** UTA Real-Life Drowsiness Dataset (**UTA-RLDD**), Ghoddoosian, Galib & Athitsos, UT Arlington. Project page: <https://sites.google.com/view/utarldd/home>. Associated paper: *"A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection,"* CVPR Workshops 2019 (arXiv:1904.07312).
- **Composition:** 60 healthy subjects × 3 videos (one per class) = **180 RGB videos**, ~10 min each, ~30 hours / **111.3 GB** total. Recorded by participants on their own webcams/phones, so resolution, orientation, and quality vary; frame rate ≤ 30 fps. Labels follow the **Karolinska Sleepiness Scale (KSS)**: alert ≈ KSS 1–3, low-vigilant ≈ KSS 6–7, drowsy ≈ KSS 8–9.
- **Intended use:** academic research on drowsiness/early-fatigue detection — exactly this project's purpose.
- **Access status:** **Freely downloadable in full** from the project's public Google Drive (no request form or signed agreement). I have it **partially downloaded**. Because it's large, I intend to run some preprocessing on the videos to transform them into CSVs of the features I'd want the model to learn from the images in a raw analysis. This drastically reduces the size of the dataset so I don't need to store all 111.3 GB after the initial processing is done.

## Access, licensing, consent, and usage constraints

- **License:** the project page states **no explicit license**. The only stated obligation is to **cite the CVPRW 2019 paper** when reporting results. The conservative, charter-binding interpretation is therefore **research/academic use with attribution only** — I claim no commercial or redistribution rights, and I will not republish the raw videos.
- **Consent:** participation was voluntary (some students received extra credit); all subjects were over 18. **36 of 60** participants consented to having their faces published; subject identities are not provided and per the authors cannot be revealed. The dataset ships pre-anonymized as numbered subjects.
- **No formal IRB number** is published on the project page. I'll treat the videos as sensitive biometric data regardless: keep them local, never commit them or extracted faces to git, and publish only **aggregate, non-identifying** features/metrics.

## Prediction target

Ordinal integer label **y ∈ {0, 1, 2}** = {alert, low-vigilant, drowsy}, taken from the source video each window came from (`0.mov` → 0, `5.mov` → 1, `10.MOV` → 2). Ordering is load-bearing: an alert↔drowsy (off-by-two) error is strictly worse than an adjacent (off-by-one) error, and the loss and metrics must respect that.

The real unit of interest at evaluation time is the **video** (and the **subject**), so window-level predictions are aggregated up to video-level predictions before scoring.

## Candidate input features

Per-window summaries computed from per-frame MediaPipe signals. The model-input vector (~17 features):

- **Eyes / PERCLOS:** `perclos` (**per**centage of eye **clos**ure), `blink_rate`, `blink_dur_mean`, `blink_dur_max`, `eye_blink_mean`, `eye_blink_std`, `ear_mean` (EAR means "Eye Aspect Ratio"), `ear_std`, `ear_min`.
- **Yawning:** `jaw_open_mean`, `jaw_open_max`, `yawn_count`.
- **Head pose / stability:** `pitch_mean`, `pitch_std`, `pitch_range`, `frac_head_down`, `yaw_std`, `roll_std`.
- **Recorded but NOT fed to the model:** `frac_face_missing` (data-quality covariate), and `bright_mean` / `warmth` (reserved exclusively for the luminance confound baseline).

## Prediction-time availability and leakage risks

Every model feature is computable from a single window of live video at inference time, so there is **no temporal/feature leakage from the future**. The risks that *do* matter here are identity and normalization leakage:

- **Subject leakage (the dominant risk):** windows from one person are highly correlated; if any of a subject's windows land in both train and test, the model recognizes the *face*, not the *state*. **Mitigation:** all splits are **subject-wise** (GroupKFold / LOSO on `subject_id`), with a coded assertion that train/val/test subject sets are disjoint in every fold.
- **Per-subject normalization leakage:** baseline EAR differs by face. Per-subject normalization stats are computed from **that subject's own frames only** (across all 3 of their videos), so nothing crosses between subjects — safe in any fold. Crucially, **the alert video is never used as a per-subject baseline**, because that baseline would not exist at deployment time.
- **Global scaling leakage:** the `StandardScaler` is fit on the **training fold only**, then applied to val/test.
- **Appearance/lighting leakage:** addressed by design (geometric features, not pixels) and *empirically checked* via the luminance-only baseline.

## Likely data quality / missingness / imbalance / representativeness concerns

- **Class balance:** Every subject contributes exactly one video per class, so any subject-wise split is class-balanced. No stratification or class weighting needed.
- **Missingness:** frames with no detected face get `face=0` / NaN features — *never* treated as "eyes closed." A validity gate drops any window with >30% missing-face frames; short gaps (<0.3 s) are interpolated, longer gaps left NaN.
- **Quality variance:** participant-recorded video means mixed resolution, orientation, lighting, and ≤30 fps. Geometric/normalized landmarks absorb most of this; `frac_face_missing` is tracked as a covariate to detect quality-driven artifacts.
- **Representativeness:** the cohort is **skewed — 51 men / 9 women**, ethnicity-skewed (30 of 60 Indo-Aryan/Dravidian), age 20–59 (mean 25), glasses in only 21/180 videos, facial hair in 72/180. A model tuned here may generalize poorly to under-represented groups, though it should work fine for me.
- **Acted vs. real fatigue:** states are self-induced/self-labeled in a recording session, not naturalistic continuous drowsiness. This is the core domain gap to the live use case.

## Responsible-use limitations

- **Not safety-critical.** This is a personal awareness aid, **not** a driver-monitoring or medical device. I would want to test under a wider variety of scenarios and have more video in a driving setting be used for training in order to use it for this.
- **Demographic caution:** given the cohort skew, per-group error rates are expected to differ; any future deployment must re-validate on the target user.
- **Privacy:** raw video and any face crops stay local and out of version control; only aggregate features/metrics are shared. Live use would process video on-device with nothing recorded or transmitted.
- **No deception/surveillance use:** intended for self-monitoring with consent of the person on camera, not for monitoring others.

## Baseline model plan

Three baselines establish the floors the real model must clear:

1. **Majority class** — always predict the most frequent class. Accuracy ≈ 33%, QWK ≈ 0. Sanity floor.
2. **Luminance-only** — ordinal/logistic regression on just per-video `bright_mean` + `warmth`. **If the real model doesn't clearly beat this, lighting is leaking** and I'm partly training a brightness detector. While most videos have consistent lighting, some have clear nighttime light conditions that the model could learn from if it shows through in the features extracted in each window.
3. **PERCLOS-only** — a one-feature ordinal/threshold model on `perclos` alone. A strong, interpretable reference; the full feature set must beat it to earn its complexity.

## Initial model candidate

A small **MLP with a CORN ordinal head** (input ~17 features → 64 → 32, ReLU, dropout ≈0.3 → K−1=2 cumulative logits), trained with **CORN loss** (`coral_pytorch`), Adam (lr 1e-3, wd 1e-4), early stopping on validation QWK.

Paired head-to-head with the **required baseline-to-beat**: the *same* MLP with a 3-way softmax head + plain cross-entropy. With only 3 classes the ordinal loss's edge is often small, so I will **A/B CORN vs. cross-entropy and report both** — not assume CORN wins.

## Evaluation metrics and why they fit

Accuracy is explicitly **not** the headline — it ignores the ordering that defines this problem. Headline trio (all at the **video level**, **mean ± std across folds**):

- **Quadratic Weighted Kappa (QWK)** — primary single number. Penalizes far-apart disagreements quadratically and corrects for chance, so it directly rewards respecting the alert<low<drowsy order.
- **MAE on ranks** (labels as 0/1/2) — an off-by-two costs double an off-by-one; directly interpretable as "how many steps off, on average."
- **3×3 confusion matrix** — shows *where* errors land. Adjacent (off-by-one) confusion is acceptable; mass in the alert↔drowsy corners is the real failure.

Secondary (reported, not optimized): macro-F1, accuracy, Spearman. The **std across folds is itself a headline result** — with only 60 subjects, stability matters as much as the mean.

## Train / validation / test strategy

The unit of generalization is the **subject**. Plan:

- **Primary: subject-wise cross-validation** via `GroupKFold` (groups = `subject_id`). Start with **5-fold** (12 held-out subjects/fold — this mirrors the dataset's own published 5-fold structure); escalate to **Leave-One-Subject-Out (60 folds)** for the gold-standard estimate if time allows, since models train in seconds.
- Within each training fold, hold out a few subjects (by group) as a **validation set** for early stopping.
- **Train on windows, evaluate on videos:** aggregate a video's window-level CORN cumulative probabilities, then apply the decision rule to get the video's rank. Metrics computed over held-out videos per fold, averaged across folds.
- **Optional fixed holdout** (42 train / 9 val / 9 test, seeded, touched once) for a single clean headline number — but with only 9 test subjects this is high-variance, so **CV is the trustworthy estimate**.
- Fold assignments and all seeds (`random`, `numpy`, `torch`) are logged to disk; package versions pinned in `uv.lock` for reproducibility.

## Scope limits — what this project will and will not attempt

**Will:**
- Build the Stage 1–8 pipeline (feature extraction → windowing → subject-wise CV → ordinal model → metrics vs. baselines).
- Deliver an **offline, rigorously evaluated** subject-independent 3-class ordinal classifier with honest mean ± std reporting against baselines.
- A/B test the ordinal head vs. cross-entropy.

**Will NOT:**
- Build or evaluate a **real-time / live laptop monitor** — named as motivation and future work, out of scope for the graded project.
- Model raw pixels / train a CNN end-to-end on video.
- Predict KSS (Karolinska Sleepiness Scale) as a continuous value or do frame-level (sub-window) temporal modeling beyond the windowing scheme described.
- Claim fitness for any safety-critical or medical use.
- Re-balance, augment, or correct the cohort's demographic skew — it is documented as a limitation, not solved here.

## Success criteria

A **useful course-project result** is defined by *clearing the floors with margin*, not by hitting an arbitrary absolute number (honest given the high subject-to-subject variance with n=60):

- ✅ **Success:** the chosen model **beats both the majority and luminance-only baselines on QWK by a clear margin, with non-overlapping mean ± std bands across folds**, AND its confusion-matrix errors concentrate on the diagonal / off-by-one cells (little alert↔drowsy mass). Bonus: it also beats PERCLOS-only, justifying the full feature set.
- ⚠️ **Partial / informative:** it beats majority but **not** luminance-only by a clean margin → strong evidence of a lighting confound; the finding itself is a valid, reportable result that redirects the next stage.
- ❌ **Unsuccessful:** it cannot separate from the majority baseline, or QWK std bands overlap zero across folds (no reliable signal beyond chance).

## Fallback plan

| If this fails… | Fallback |
|---|---|
| Google Drive link dies / download breaks | Email the authors (contact on project page); failing that, pivot to a comparable public drowsiness set (e.g., NTHU-DDD, requires request) and re-scope — flagged as a **risk to resolve in week 1**. |
| MediaPipe extraction unreliable on the participant-recorded video | Fall back to the geometric **EAR / MAR** computed directly from landmarks (drop blendshape-dependent features); worst case, reduce to the PERCLOS + blink + yaw feature subset. |
| CORN / `coral_pytorch` gives no edge or is unstable | Ship the **cross-entropy MLP** or **LightGBM** as the primary model — the A/B is designed so the baseline-to-beat is already a fully valid deliverable. |
| Full pipeline too heavy for the timeline | Drop LOSO (use 5-fold only) and the optional fixed holdout; the 5-fold CV result stands alone. |
| Lighting confound confirmed (model ≈ luminance baseline) | Report it as the finding, add stricter geometric-only ablation, and present the confound analysis as the project's contribution. |

## Staged model-improvement plan (across the remaining assignments)

This charter does **not** claim the first model is final. The work develops in
stages across the remaining assignments:

1. **Stage 0 — Baselines (establish floors first).** Implement and score majority, luminance-only, and PERCLOS-only under subject-wise CV. *Deliverable:* the floor numbers every later model is judged against.
2. **Stage 1 — Initial model candidate (the A5 core).** MLP+CORN **vs.** the required MLP+cross-entropy baseline-to-beat, under 5-fold GroupKFold, window→video aggregation, headline metrics with mean ± std. *Deliverable:* a first honest QWK/MAE/confusion table vs. baselines.
3. **Stage 2 — Revised model or justified simpler alternative.** Driven by Stage 1 evidence: a LightGBM (CORN/CORAL-decomposed) comparison, light feature-set ablations (e.g., does the full set beat PERCLOS-only?), threshold/aggregation tuning, and escalation to LOSO for the gold-standard estimate. **If the simpler model (cross-entropy or PERCLOS-only) is statistically indistinguishable, the justified choice is to keep it** and say so — simpler-and-equal wins.
4. **Stage 3 — Final recommendation + evidence (presentation).** A single defended recommendation, including the limitations and confound analysis, and an explicit statement of what would be needed before any real-time/personal deployment.

## Evidence needed to support the final recommendation

To defend the final pick at the presentation, I plan to collect:

- **Headline metrics table:** QWK, rank-MAE (mean ± std across folds, video-level) for every model **and** all three baselines, side by side.
- **Confusion matrices** (summed over folds) for the chosen model and the key baselines — showing error mass is on the diagonal/off-by-one, not in the alert↔drowsy corners.
- **Confound check result:** explicit chosen-model-vs-luminance-only comparison demonstrating the model is an alertness detector, not a brightness detector.
- **Ordinal-loss ablation:** CORN vs. cross-entropy numbers proving (or refuting) that the ordinal head earns its place.
- **Feature-value evidence:** full-feature vs. PERCLOS-only to justify (or retire) the extra features.
- **Stability evidence:** the across-fold std and, ideally, LOSO results — the honest measure of generalization with only 60 subjects.
- **Subgroup sanity check (if feasible):** error broken out by glasses / facial-hair / gender where sample size permits, to surface representativeness limits.
- **Reproducibility artifacts:** logged seeds, saved subject→fold assignments, pinned dependency lockfile.
