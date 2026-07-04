# Part A — Portfolio modeling progress

> Scope boundary: one focused experiment or refinement, not a model tournament.
> No exhaustive hyperparameter or architecture search. No final model claim.

## Problem, stakeholder, dataset, target, current status, evaluation setup

- **Problem / stakeholder:** Personal early-fatigue aid — given a short live-video window, estimate whether the user appears alert, low-vigilant, or drowsy. Stakeholder is me / a future self-monitoring user; not safety-critical or client-facing.
- **Dataset:** UTA-RLDD, 60 subjects × 3 videos → per-window tabular feature CSVs (~17 geometric/temporal features from MediaPipe landmarks). Native portfolio data; no fallback needed.
- **Target:** ordinal `y ∈ {0,1,2}` = {alert, low-vigilant, drowsy}, scored at video level after window→video aggregation.
- **Current model status (from Assignment 6, the comparison anchor):** under LOSO, PERCLOS-only logistic regression is the best candidate (QWK 0.410 ± 0.414); the regularized MLP does not beat it (0.363 ± 0.382); luminance-only is near chance (0.048 ± 0.502); majority floor QWK 0.
- **Evaluation setup (unchanged from A5/A6):** LOSO on `subject_id` from saved `folds.json`, disjointness asserted per fold, train-fold-only StandardScaler, grouped validation subjects for the MLP, video-level QWK / rank-MAE / accuracy / macro-F1 as mean ± std across 60 folds.

## Experiment performed

> A6 committed to: "test whether the full feature set can beat PERCLOS with a
> simpler interpretable tabular model … using the exact same LOSO folds. If it
> cannot beat PERCLOS, keep PERCLOS." This section executes that experiment.

- **Model:** multinomial logistic regression on the full ~17-feature model-input set (excluding `frac_face_missing`, `bright_mean`, `warmth` per charter), implemented as a new run spec in the existing `baselines.py` logistic path.
- **Held fixed:** folds, seed, scaling, aggregation, metrics — so the comparison to A6 evidence is direct.
- TODO: note the optional shallow-tree diagnostic if run (one config only), or state it was skipped for scope.

## Split / evaluation procedure

- Same LOSO procedure and `folds.json` as Assignment 6; no change, so results are directly comparable to prior evidence. (If anything changed, justify here.)

## Updated results

TODO: fill from `outputs/metric_summary.csv`.

| Model | QWK (mean±std) | rank-MAE (mean±std) | Accuracy (mean±std) | macro-F1 (mean±std) | Notes |
| --- | --- | --- | --- | --- | --- |
| Majority (A6) | 0.000 ± 0.000 | 0.944 ± 0.125 | 0.333 ± 0.000 | 0.167 ± 0.000 | floor |
| Luminance-only (A6) | 0.048 ± 0.502 | 0.844 ± 0.400 | 0.361 ± 0.248 | 0.258 ± 0.244 | confound check |
| PERCLOS-only (A6) | 0.410 ± 0.414 | 0.617 ± 0.357 | 0.517 ± 0.241 | 0.405 ± 0.272 | model to beat |
| MLP regularized (A6) | 0.363 ± 0.382 | 0.650 ± 0.339 | 0.483 ± 0.233 | 0.370 ± 0.254 | prior neural result |
| **Logistic, full features (A7)** | TODO | TODO | TODO | TODO | this week's experiment |

## Comparison to earlier evidence

TODO after the run:

- QWK / rank-MAE vs PERCLOS-only, read against the fold-band width.
- Off-by-two alert↔drowsy corner count vs 24/180 (PERCLOS and MLP in A6).
- Low-vigilant (middle class) recall vs A6 (PERCLOS 14/60, MLP 18/60).

## Evidence beyond the aggregate score

TODO after the run:

- Summed-over-folds confusion matrix.
- Per-class metrics, with the low-vigilant class called out.
- Coefficient inspection: which features carry weight — is the full-feature model mostly re-deriving PERCLOS, or do yawn/head-pose features contribute?
- Confidence-by-correctness from `diagnostics.csv` vs A6 gaps.

## Blocker / failure mode / uncertainty

TODO — pick the one the evidence best supports. Standing candidates:

- Fold-band width at n=60 makes small model-vs-model gaps undecidable; band overlap is the persistent obstacle to any "X beats Y" claim.
- The low-vigilant middle class remains near chance for every model tried.
- Confidence remains weakly separated between correct and incorrect predictions, blocking any warning-threshold use.

## Implication for the next staged experiment

TODO — decide per the charter's simpler-and-equal rule:

- If the full-feature logistic does not clearly beat PERCLOS: keep PERCLOS as the leading candidate; the pre-final-package experiment remains the ordinal-head (CORN) comparison, with the explicit bar of improving QWK/rank-MAE or off-by-two corners without wider bands.
- If it does beat PERCLOS: the next experiment is confirming which features drive the gain and whether the MLP's failure was capacity vs representation.

## Commands and logs

TODO after the run. Template:

- Command: `uv run train_alertness ...` (record exact flags).
- Split: LOSO, 60 folds, from saved `folds.json`; disjointness asserted.
- Seed: 42; sklearn logistic regression is deterministic given the seed (note any torch nondeterminism only if the MLP re-ran).
- Evidence: `outputs/manifest.json`, `metric_summary.csv`, `fold_metrics.csv`, `video_predictions.csv`, `confusion_matrices.csv`, `diagnostics.csv`, `folds.json`.

## Dataset citation

Ghoddoosian, Galib, and Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection," CVPR Workshops 2019.

---

# Part B — Preliminary transfer-learning relevance decision

## Decision

**Indirectly relevant.** Transfer learning and pretrained vision models are not a direct candidate for the staged model plan, but the core Week 7 idea — using a pretrained vision model as a frozen feature extractor — is already embodied in this project: MediaPipe's face landmarker is a pretrained CNN-based model whose outputs are the source of every tabular feature the models consume. The project's representation *is* pretrained feature extraction, specialized to geometric fatigue cues rather than generic embeddings.

## Why not directly relevant

- The Assignment 4 charter explicitly scopes out raw-pixel / end-to-end CNN modeling, and that scope has not become infeasible.
- With 60 subjects and subject-wise evaluation, fine-tuning or training a CNN would risk exactly the identity/appearance leakage the geometric feature design exists to prevent, and there is far too little data to fine-tune responsibly.
- Assignment 6 evidence points the opposite direction from added capacity: a 17-feature MLP could not beat a one-feature logistic model. High-dimensional generic embeddings would worsen the capacity-to-data mismatch.

## Evidence already supporting this decision

- Luminance-only baseline near chance (QWK 0.048) — the appearance/lighting confound is controlled by the geometric design; pixel-level models would reopen it.
- PERCLOS-only ≥ full-feature MLP under LOSO — representation richness is not the current bottleneck.
- The LOSO scaffolding, leakage assertions, and window→video aggregation transfer unchanged to any future embedding test, so the comparison infrastructure already exists.

## Missing evidence before the decision is final (for Assignment 8)

- The one credible test not yet run: frozen generic image/face embeddings (per-window pooled), same LOSO folds and metrics, head-to-head against the geometric features. Acceptance bar: clear QWK/off-by-two improvement with comparable fold bands — and a check that gains aren't appearance leakage (e.g., the luminance baseline logic extended to embedding space).
- Per-subgroup landmark quality (glasses / facial hair / demographics) — whether the pretrained extractor itself fails unevenly, which `frac_face_missing` only partially captures.

## Updates to Assignment 4 audit / charter assumptions

- The audit should name MediaPipe explicitly as a **pretrained upstream dependency**: its training data is not documented, so its own demographic skew compounds the documented UTA-RLDD cohort skew (51M/9F, ethnicity-skewed). Landmark quality is an unquantified per-subgroup failure mode.
- No change to scope, split, metrics, or success criteria.

## Risks introduced or avoided

- **Avoided by non-adoption:** appearance/identity memorization from pixel models at n=60; domain shift between web-scale pretraining data and participant-recorded webcam video; spurious visual cues (lighting, background, camera angle) re-entering the signal; compute/maintainability cost of an embedding pipeline; overclaiming from an embedding "win" that is really appearance leakage.
- **Already carried (and now documented):** dependence on MediaPipe's pretrained model — licensing is permissive (Apache-2.0), but version drift and subgroup performance are maintenance risks for any live use.

## Checkpoint status

This is a preliminary, evidence-based non-adoption decision, to be revisited in Assignment 8 after fine-tuning across domains is covered. What Assignment 8 should test is narrowed to one bounded question: do frozen generic embeddings beat the geometric features under the identical LOSO protocol?
