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
- **Held fixed:** folds, seed, scaling, aggregation, metrics — so the comparison to A6 evidence is direct. The regenerated `folds.json` and the majority / luminance / PERCLOS rows reproduce the A6 numbers exactly, confirming the comparison is apples-to-apples.
- The optional shallow-tree diagnostic was skipped for scope; one focused experiment was run.

## Split / evaluation procedure

- Same LOSO procedure and `folds.json` as Assignment 6; no change, so results are directly comparable to prior evidence.

## Updated results

From `outputs/metric_summary.csv`:

| Model | QWK (mean±std) | rank-MAE (mean±std) | Accuracy (mean±std) | macro-F1 (mean±std) | Notes |
| --- | --- | --- | --- | --- | --- |
| Majority (A6) | 0.000 ± 0.000 | 0.944 ± 0.125 | 0.333 ± 0.000 | 0.167 ± 0.000 | floor |
| Luminance-only (A6) | 0.048 ± 0.502 | 0.844 ± 0.400 | 0.361 ± 0.248 | 0.258 ± 0.244 | confound check |
| PERCLOS-only (A6) | 0.410 ± 0.414 | 0.617 ± 0.357 | 0.517 ± 0.241 | 0.405 ± 0.272 | model to beat |
| MLP regularized (A6) | 0.363 ± 0.382 | 0.650 ± 0.339 | 0.483 ± 0.233 | 0.370 ± 0.254 | prior neural result |
| **Logistic, full features (A7)** | **0.481 ± 0.378** | **0.567 ± 0.326** | **0.528 ± 0.224** | **0.414 ± 0.255** | this week's experiment |

Pooled over the 180 held-out videos: logistic_full QWK 0.485 vs PERCLOS 0.409.

## Comparison to earlier evidence

- **vs PERCLOS-only:** the full-feature logistic improves every headline metric — QWK +0.071 mean (0.481 vs 0.410) with a slightly *narrower* fold band (±0.378 vs ±0.414), rank-MAE 0.567 vs 0.617, accuracy 0.528 vs 0.517, macro-F1 0.414 vs 0.405. Read honestly against band width, the bands still overlap heavily; a paired per-fold read is more informative: logistic_full beats PERCLOS in 18 folds, ties in 31, and loses in 11.
- **Off-by-two corners:** 17/180 alert↔drowsy errors vs 24/180 for both PERCLOS and the A6 MLP — a ~29% reduction in exactly the dangerous corner errors the charter's staged-improvement bar names.
- **Low-vigilant recall:** unchanged at 14/60 (PERCLOS 14/60, MLP 18/60). The middle class remains the unsolved failure mode.
- **vs the A6 MLP:** logistic_full beats it on all four headline metrics with a far simpler, fully interpretable model — reinforcing the A6 conclusion that added capacity was not the missing ingredient; the representation fed to a properly regularized linear model was.

## Evidence beyond the aggregate score

Summed-over-folds confusion matrix (rows = true, columns = predicted):

```
logistic_full                     PERCLOS-only (A6, for reference)
         pred a  l  d                      pred a  l  d
true alert   46 10  4             true alert   48  7  5
true low     29 14 17             true low     32 14 14
true drowsy  13 12 35             true drowsy  19 10 31
```

- **Per-class:** drowsy recall improves 31→35/60 and true-drowsy-predicted-alert errors drop 19→13; alert recall gives up two videos (48→46). Low-vigilant stays at 14/60 — near chance for every model tried.
- **Coefficient inspection** (`outputs/logistic_full_coefficients.csv`, from `scripts/inspect_logistic_full.py`; per-LOSO-fold refits on the standardized scale): PERCLOS carries the largest weight (mean |coef| 0.797), but the model is not merely re-deriving PERCLOS — the blink-dynamics cluster contributes real signal (blink_dur_max 0.369, eye_blink_mean 0.363, blink_rate 0.329, blink_dur_mean 0.327, ear_std 0.237). Yawn/jaw and head-pose features contribute little (all ≤ 0.134). The gains over PERCLOS-only appear to come from *eyelid temporal dynamics*, not from the wider feature families.
- **Confidence-by-correctness** (`diagnostics.csv`, n-weighted across folds): 0.489 on correct vs 0.446 on incorrect predictions — a slightly wider gap than A6 (PERCLOS 0.449/0.410) but still far too small to support a warning threshold without calibration work.

## Blocker / failure mode / uncertainty

- **Primary failure mode: the low-vigilant middle class remains at chance (14/60) for every model tried.** The corner errors improved this week, but no model yet distinguishes the ordinal midpoint, and it is exactly the class an *early*-fatigue aid exists to catch. The coefficient evidence suggests why: the discriminative signal lives in eyelid dynamics summarized per-window, and low-vigilance may only be separable in how those dynamics evolve *across* windows — structure the current window-independent representation discards.
- **Process finding (uncertainty to carry):** while implementing this run I found the A6 MLP's feature path (`dataset.get_feature_columns`) excludes only IDs/label/`frac_face_missing`, so the MLP silently consumed the four luminance columns the charter reserves for the confound baseline. The A6 MLP comparison should be read with that caveat (its inputs were 22 features, not the charter's ~18); this week's logistic_full excludes them per charter. Since luminance-only is near chance, the practical effect is likely noise-dilution rather than leakage, but it should be fixed before any future MLP/CORN run.
- Fold-band width at n=60 remains the standing obstacle: even this week's across-the-board improvement shows overlapping bands (18 wins / 31 ties / 11 losses per fold).

## Implication for the next staged experiment

Per the charter's simpler-and-equal rule, **logistic_full becomes the leading candidate**: it improves every headline metric with a comparable-or-narrower fold band and cuts the dangerous off-by-two corners 24→17 — the specific bar the charter set for justifying the full feature set over PERCLOS-only. The claim is stated modestly (bands overlap; 31 of 60 folds tie), but the burden-of-proof standard the charter set has been met on the corner errors.

The coefficient inspection already answers part of the "what drives the gain" question: eyelid/blink dynamics, not yawn or head pose. That points the next staged experiment at *temporal structure over the existing window features* — the per-window tabularization keeps within-window summaries but discards how blink behavior evolves across a video, which is exactly where the dataset's published baseline (an HM-LSTM over sequential blink features; Ghoddoosian et al. 2019) found its signal. A bounded A8 experiment — one small sequence model or richer cross-window trend features on the same LOSO folds, judged against logistic_full — requires no new data and directly tests whether the representation (not model capacity) is the remaining bottleneck. The CORN ordinal-head comparison remains the pre-final-package experiment, with the same bar: improve QWK/rank-MAE or off-by-two corners without wider bands.

## Commands and logs

- Command: `uv run train_alertness` (no flags; runs the four baseline-path models including the new `logistic_full`; the MLP was not re-run — A6 numbers are cited for it). Coefficient diagnostic: `uv run python scripts/inspect_logistic_full.py`.
- Split: LOSO, 60 folds, regenerated deterministically to `outputs/folds.json`; per-fold subject disjointness asserted. The majority / luminance / PERCLOS rows reproduce the A6 values exactly, which confirms folds and pipeline are unchanged and the A7-vs-A6 comparison is direct.
- Seed: 42; sklearn logistic regression is deterministic given the seed (no torch training this week, so no torch nondeterminism note applies). Library versions recorded in `outputs/manifest.json`.
- Evidence: `outputs/manifest.json`, `metric_summary.csv`, `fold_metrics.csv`, `video_predictions.csv`, `confusion_matrices.csv`, `diagnostics.csv`, `folds.json`, `logistic_full_coefficients.csv`. The pre-run A6 artifacts are preserved unchanged in `outputs_a6_backup/`.

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
- The geometric features are sufficient to beat PERCLOS-only when used well: this week's full-feature logistic (QWK 0.481) outperforms both PERCLOS-only (0.410) and the A6 MLP (0.363). The signal the extraction preserves is real and improvable — the open question is temporal structure across windows, not a richer per-frame appearance representation.
- The LOSO scaffolding, leakage assertions, and window→video aggregation transfer unchanged to any future embedding test, so the comparison infrastructure already exists.

## Missing evidence before the decision is final (for Assignment 8)

- The one credible pretrained-vision test not yet run: frozen generic image/face embeddings (per-window pooled), same LOSO folds and metrics, head-to-head against the geometric features. Acceptance bar: clear QWK/off-by-two improvement with comparable fold bands — and a check that gains aren't appearance leakage (e.g., the luminance baseline logic extended to embedding space). **Feasibility constraint:** the raw videos (~111 GB) are no longer stored locally per the charter's storage plan, so this test requires many hours of re-downloading and staged processing. It stays on the list as the test that would finalize the decision, but its cost must be weighed against its expected value.
- A cheaper, better-motivated representation test now exists: this week's coefficient evidence shows the gains over PERCLOS come from blink-dynamics features, and the dataset's published baseline found its signal in *sequential* blink features (HM-LSTM). Temporal modeling over the already-extracted window sequences — same LOSO folds, no video re-download — is the missing evidence most likely to change the modality decision, because it tests whether the tabularization's real loss was temporal structure rather than pixel appearance.
- Per-subgroup landmark quality (glasses / facial hair / demographics) — whether the pretrained extractor itself fails unevenly, which `frac_face_missing` only partially captures.

## Updates to Assignment 4 audit / charter assumptions

- The audit should name MediaPipe explicitly as a **pretrained upstream dependency**: its training data is not documented, so its own demographic skew compounds the documented UTA-RLDD cohort skew (51M/9F, ethnicity-skewed). Landmark quality is an unquantified per-subgroup failure mode.
- No change to scope, split, metrics, or success criteria.

## Risks introduced or avoided

- **Avoided by non-adoption:** appearance/identity memorization from pixel models at n=60; domain shift between web-scale pretraining data and participant-recorded webcam video; spurious visual cues (lighting, background, camera angle) re-entering the signal; compute/maintainability cost of an embedding pipeline; overclaiming from an embedding "win" that is really appearance leakage.
- **Already carried (and now documented):** dependence on MediaPipe's pretrained model — licensing is permissive (Apache-2.0), but version drift and subgroup performance are maintenance risks for any live use.

## Checkpoint status

This is a preliminary, evidence-based non-adoption decision, to be revisited in Assignment 8 after fine-tuning across domains is covered. Assignment 8's bounded question, in priority order: (1) does temporal modeling over the existing window-feature sequences beat logistic_full under the identical LOSO protocol (no new data required)? and only if that and the deadline budget allow, (2) do frozen generic embeddings beat the geometric features (requires re-downloading video; currently cost-prohibitive)? The Part A coefficient evidence weakens the case for (2): the current bottleneck looks temporal, not appearance-representational.
