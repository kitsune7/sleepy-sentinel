# Assignment 7 — Focus Checklist and Plan

Assignment: Portfolio Modeling Progress and Preliminary Transfer Learning Relevance.
Budget: 4–6 hours total (Part A ~3–4 h, Part B ~1–2 h).

## The one-sentence plan

Run the exact next experiment declared in Assignment 6 — a full-feature
interpretable tabular model (multinomial logistic regression, optionally one
shallow-tree diagnostic) on the same LOSO folds, judged against PERCLOS-only —
then write a Part B note arguing Week 7 vision methods are **indirectly
relevant** because the pipeline already uses a pretrained vision model
(MediaPipe face landmarker) as a frozen feature extractor, while end-to-end
CNNs remain scoped out by the charter.

## What carries forward unchanged (state, don't redo)

- **Problem/stakeholder:** personal early-fatigue aid; stakeholder is me/a future self-monitoring user.
- **Dataset/target:** UTA-RLDD, 60 subjects × 3 videos; ordinal y ∈ {0,1,2}, scored at video level.
- **Split:** LOSO on `subject_id`, saved `folds.json`, disjointness asserted, train-only scaler, window→video aggregation. Do not change it.
- **Metrics:** QWK (headline), rank-MAE, accuracy, macro-F1, mean ± std across folds; summed confusion matrices; off-by-two corner counts.
- **Current model status (the comparison anchor):** PERCLOS-only QWK 0.410 ± 0.414 is the model to beat; regularized MLP 0.363 ± 0.382 did not beat it; luminance near chance (0.048); majority floor 0.
- **A6-declared next experiment:** "test whether the full feature set can beat PERCLOS with a simpler interpretable tabular model … using the exact same LOSO folds. If it cannot beat PERCLOS, keep PERCLOS." Part A executes exactly this.

## Part A checklist — full-feature interpretable model (~3–4 h)

**Scope guard: one focused experiment. No hyperparameter search, no tournament, no CORN this week (that's the pre-final-package comparison), no new features.**

- [ ] **(15 min) Freeze the comparison.** Confirm `folds.json`, seed 42, and the A6 `outputs/` metrics are the baseline evidence. Note the exact A6 numbers in the memo before running anything.
- [ ] **(60–90 min) Implement the run.** Add a `logistic_full` run: multinomial `LogisticRegression` on all ~17 model-input features (exclude `frac_face_missing`, `bright_mean`, `warmth` per charter), reusing the existing baseline path in `src/training/baselines.py` (it already has the logistic + feature-slice machinery — this should be a new `spec`, not new architecture). Same train-only scaling, same folds, same aggregation.
  - Optional bounded diagnostic if time allows: one shallow depth-limited tree or small LightGBM with defaults, same folds — one config, not a sweep.
- [ ] **(15 min) Run it.** `uv run train_alertness` (baselines + new run; record the exact command, seed, and that outputs land in `outputs/`). W&B optional via `--wandb-project sleepy-sentinel`.
- [ ] **(45–60 min) Evidence beyond the aggregate.** From `metric_summary.csv`, `fold_metrics.csv`, `confusion_matrices.csv`, `diagnostics.csv`:
  - Headline table: logistic_full vs PERCLOS-only vs MLP vs floors (QWK, rank-MAE, acc, macro-F1, mean ± std).
  - Summed confusion matrix; count the alert↔drowsy off-by-two corners (PERCLOS/MLP currently 24/180 — did it improve?).
  - Per-class recall, especially the low-vigilant middle class (both current models ≈ chance there).
  - Coefficient inspection: which features the logistic model actually leans on — is it mostly re-deriving PERCLOS? This is the interpretability payoff and feeds Part B.
- [ ] **(15 min) Decide per the charter rule.** If logistic_full doesn't clearly beat PERCLOS (mean gap small relative to the wide fold bands, no off-by-two improvement), the reportable result is "keep PERCLOS" — that is a valid, strong outcome, not a failure.
- [ ] **(15 min) Name the blocker/uncertainty.** Best candidates from existing evidence: fold-band width with n=60 makes model separation nearly undecidable at this sample size; the low-vigilant class remains unresolved; confidence is still weakly calibrated (0.44–0.48 either way).
- [ ] **(10 min) State the next staged experiment.** Per A6: the CORN/ordinal-head comparison before the final package, with the explicit bar of improving QWK/rank-MAE or the off-by-two corners with comparable band width — plus confidence calibration (temperature scaling) as the runner-up candidate.

## Part B checklist — transfer-learning relevance note (~1–2 h)

**Decision: indirectly relevant.** Structure the note around these points:

- [ ] **The project already uses pretrained vision feature extraction.** MediaPipe's face landmarker is a pretrained CNN-based model used as a *frozen feature extractor*; the tabular features are derived from its outputs. So the Week 7 concept is embodied in the pipeline — just extracting geometric semantics instead of generic embeddings. That's the "indirect" core.
- [ ] **Why not directly relevant.** Charter explicitly scopes out end-to-end CNNs / raw-pixel modeling; 60 subjects is far too few to fine-tune anything without severe subject memorization risk; A6 showed even a small MLP on 17 features can't beat one feature — generic high-dimensional embeddings would worsen the capacity/data mismatch and reintroduce appearance/lighting leakage the geometric features were designed to avoid.
- [ ] **Evidence already in hand.** Luminance baseline near chance (appearance confound controlled by design), PERCLOS ≈ full-feature MLP (more representation capacity is not currently the bottleneck), subject-wise LOSO scaffolding proven.
- [ ] **Missing evidence before the decision is final (for Assignment 8).** The only credible pretrained-vision test would be a *bounded* comparison: frozen generic face/image embeddings (per-window pooled) vs the geometric features, same LOSO folds, same metrics. Not run this week; name it as the A8 candidate test and its acceptance bar.
- [ ] **Charter/audit updates.** Add MediaPipe explicitly to the audit as a pretrained dependency: its own training data is unknown/likely demographically skewed, which compounds the documented cohort skew; landmark quality is an unquantified per-subgroup failure mode (`frac_face_missing` partially tracks it).
- [ ] **New risks introduced by pretrained models (list even though not adopting).** Licensing (MediaPipe Apache-2.0 is fine; generic embedding models vary), pretraining–portfolio domain shift (webcam quality, lighting), spurious visual cues/identity memorization with n=60, compute and maintainability cost vs a logistic model, and overclaiming risk if embeddings "win" via appearance leakage.

## Deliverable checklist

- [ ] Code/experiment artifacts: the new run spec in `src/training/`, `outputs/` files (`manifest.json`, `metric_summary.csv`, `fold_metrics.csv`, `confusion_matrices.csv`, `diagnostics.csv`, `folds.json`).
- [ ] Run evidence: exact command, LOSO description, seed 42 + nondeterminism note (sklearn logistic is deterministic given seed — say so), saved metric files.
- [ ] `docs/Assignment 7 Notes.md` memo with labeled Part A and Part B (skeleton already created).
- [ ] Dataset citation (Ghoddoosian et al., CVPRW 2019) — required by the data source.

## Out-of-scope traps to avoid

Forcing an image model in (assignment explicitly says don't), the CORN rebuild (removed stub; it's next stage, not this week), any hyperparameter/architecture sweep, changing the split or metrics, chasing a higher score if the honest answer is "keep PERCLOS," and any scope change to the charter.
