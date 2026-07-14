# Part A — Current portfolio evidence and model status

> Scope boundary: this memo makes the revised modeling decision the assignment asks for.
> The two bounded runs below exist to close the evidence gap A7 named — they are not a model tournament.

## Problem, stakeholder, dataset, target, scope

- **Problem / stakeholder:** Personal early-fatigue aid — given a short live-video window, estimate whether the user appears alert, low-vigilant, or drowsy. Stakeholder is me / a future self-monitoring user; not safety-critical or client-facing.
- **Dataset:** UTA-RLDD, 60 subjects × 3 videos → per-window tabular feature table (18 geometric/temporal model-input features from MediaPipe landmarks; luminance confound columns and `frac_face_missing` reserved for diagnostics per charter). Native portfolio data; no fallback needed.
- **Target:** ordinal `y ∈ {0,1,2}` = {alert, low-vigilant, drowsy}, scored at video level after window→video aggregation.
- **Scope boundary (unchanged from A4):** no raw-pixel / end-to-end CNN modeling; geometric features from the pretrained MediaPipe landmarker are the representation.

## Model candidates tested so far (A5–A8)

| Model | QWK (mean±std) | rank-MAE | Accuracy | macro-F1 | Provenance |
| --- | --- | --- | --- | --- | --- |
| Majority | 0.000 ± 0.000 | 0.944 ± 0.125 | 0.333 ± 0.000 | 0.167 ± 0.000 | A6 floor |
| Luminance-only | 0.048 ± 0.502 | 0.844 ± 0.400 | 0.361 ± 0.248 | 0.258 ± 0.244 | A6 confound check |
| PERCLOS-only | 0.410 ± 0.414 | 0.617 ± 0.357 | 0.517 ± 0.241 | 0.405 ± 0.272 | A6 |
| MLP regularized | 0.363 ± 0.382 | 0.650 ± 0.339 | 0.483 ± 0.233 | 0.370 ± 0.254 | A6 (with A7's luminance-column caveat) |
| Logistic, full features | 0.481 ± 0.378 | 0.567 ± 0.326 | 0.528 ± 0.224 | 0.414 ± 0.255 | A7 champion |
| **Logistic + causal trends (A8)** | **0.484 ± 0.399** | **0.539 ± 0.353** | **0.556 ± 0.251** | **0.449 ± 0.299** | this week |
| GRU over window sequences (A8) | 0.361 ± 0.470 | 0.661 ± 0.386 | 0.478 ± 0.241 | 0.371 ± 0.254 | this week |

## Split / evaluation procedure

Unchanged and still valid: LOSO on `subject_id` from the saved `outputs/folds.json` (60 folds), per-fold subject disjointness asserted, train-fold-only imputation and scaling, window→video probability averaging, video-level QWK / rank-MAE / accuracy / macro-F1 as mean ± std across folds. This week's runs *loaded* the saved folds rather than regenerating them, and a `logistic_full_anchor` run re-executed the A7 champion on those loaded folds: it reproduces the A7 fold-level QWK values exactly (allclose across all 60 folds). Every A8 number is therefore directly comparable to all A5–A7 evidence.

## This week's bounded experiment: is the bottleneck temporal?

A7's coefficient inspection showed the gains over PERCLOS came from eyelid/blink dynamics, and left one prioritized question for A8: does structure *across* windows — how blink behavior evolves over a session — carry signal the per-window tabularization discards? Two complementary probes, both on the identical folds:

1. **`trend_logistic`** — the same multinomial logistic model, fed the same 18 per-window features plus 28 *causal* cross-window trend features: for each of the 9 eyelid-dynamics features, (a) `delta` from the mean of the previous 8 windows (~1 min of past), (b) least-squares `slope` over the trailing 8 windows, (c) `drift` from the expanding mean of all earlier windows in the session; plus `elapsed_min`. Causality is enforced and unit-tested: features at window *t* are computable from windows 0..*t* only, because a live aid never sees the future.
2. **`gru_sequence`** — a small many-to-one GRU (hidden 32, dropout 0.25, weight decay 1e-4, Adam 1e-3, ≤40 epochs, validation-QWK early stopping with patience 8) reading each video's full window sequence and predicting the video label directly. This is the bounded echo of the dataset's published HM-LSTM baseline direction.

## Updated results and comparison to earlier evidence

**Trend features: the middle class finally moves.** `trend_logistic` vs the anchor: QWK is a statistical tie (0.484 vs 0.481 mean; pooled over 180 videos 0.486 vs 0.485), but rank-MAE improves 0.567 → 0.539, accuracy 0.528 → 0.556, macro-F1 0.414 → 0.449, and — the headline — **low-vigilant recall rises 14/60 → 21/60**, the first movement in the stuck middle class across every model tried since A5. Off-by-two corner errors hold at 17/180 (the A7 gain is preserved, not traded away). Alert recall gives up two videos (46 → 44). Paired per fold, trend_logistic beats the anchor in 12 folds, ties in 40, loses in 8.

Summed-over-folds confusion matrices (rows = true, columns = predicted):

```
trend_logistic                    logistic_full anchor (= A7)
         pred a  l  d                      pred a  l  d
true alert   44 10  6             true alert   46 10  4
true low     25 21 14             true low     29 14 17
true drowsy  11 14 35             true drowsy  13 12 35
```

**Where the gain comes from (coefficient evidence, `outputs/temporal_trend_coefficients.csv`):** the new temporal features collectively carry *more* standardized weight than the 18 base features (sum of mean |coef| 10.55 vs 8.88), and the `drift` family dominates: `blink_dur_max_drift` (1.84), `blink_dur_mean_drift` (1.47), `perclos_drift` (1.19). The model learned something interpretable about the middle class: for low-vigilant, blink-duration *level* is high (blink_dur_max +1.56) but blink-duration *drift* is negative (−1.30, −1.11) — long blinks that are **not still worsening** relative to the session's own baseline. Runaway drift marks full drowsiness; elevated-but-stable dynamics mark low vigilance. That distinction is exactly what a per-window representation cannot express.

**The GRU loses on every metric** (QWK 0.361 ± 0.470, corners worsen to 25/180, low-vigilant recall 12/60; per-fold vs anchor: 14 wins / 24 ties / 22 losses). Early stopping fires almost immediately (median best epoch 6 of ≤40), the signature of a sequence model overfitting ~150 training videos long before it learns anything transferable. This reconfirms the A6 lesson at higher stakes: capacity is not the missing ingredient at n=60 subjects — representation is. The temporal signal is real (trend features prove it), but it is simple enough that a linear model on hand-built trends captures it while a GRU drowns trying to rediscover it from 180 examples.

**Confidence behavior:** trend_logistic's confidence-by-correctness gap (0.575 correct vs 0.520 incorrect) is wider than the anchor's (0.489/0.446) but still far too small to support a warning threshold without explicit calibration work — unchanged conclusion since A6.

## Main remaining uncertainty

Low-vigilant recall improved from worst-in-class to 21/60 — still the weakest class and still the class an *early*-fatigue aid most needs. Whether the remaining failures are (a) irreducible label ambiguity at the ordinal midpoint, (b) concentrated in identifiable subject slices, or (c) fixable with ordinal-aware training (CORN) is the question Week 9's failure analysis should answer before any deployment judgment.

---

# Part B — Revised modality and model-strategy decision

## Recommended modeling strategy

**Adopt `trend_logistic` — frozen pretrained landmark features → per-window geometric summaries → causal cross-window trend features → multinomial logistic regression — as the modeling path carried into Week 9 failure analysis and the final package.** The one experiment still planned before the final package is unchanged from A6/A7: a CORN ordinal head on this same representation, judged by the same bar (improve QWK/rank-MAE or corner errors without wider bands).

The charter's simpler-and-equal rule cuts in trend_logistic's favor: it matches the A7 champion on QWK, beats it on rank-MAE, accuracy, and macro-F1, preserves the corner-error gain, and finally moves the primary failure mode (low-vigilant 14→21/60) — while staying a fully interpretable linear model whose 46 standardized coefficients can be read directly. The added cost is 28 derived feature columns, not a new model family.

## Relevance verdicts on course methods

- **Pretrained-model adaptation (frozen feature extraction): directly relevant — and already load-bearing.** The pipeline's representation *is* a frozen pretrained model: MediaPipe's landmarker CNN, specialized to geometric fatigue cues. This was A7's "indirect" relevance; on reflection it is direct — the project is an instance of the Week 7/8 pattern, just with task-specific geometric outputs instead of generic embeddings.
- **Transfer learning via generic embeddings / partial or full fine-tuning: not relevant — now a final rejection**, upgraded from A7's preliminary one (evidence below).
- **Sequence models: tested and rejected at this data scale.** The temporal *signal* is real; the sequence *model* is not the right vehicle for it at n=60 subjects.
- **CNNs on raw pixels: rejected** (unchanged since A4; the luminance-confound and identity-leakage logic still holds).
- **Simpler baselines: retained as floors**, not as the recommendation — PERCLOS-only is now beaten on every headline metric by an equally interpretable model.

## What changed from the Assignment 7 preliminary decision

A7 said: preliminary non-adoption of transfer learning, revisit after Week 8, with the cheapest missing evidence being a temporal-structure test over the existing window sequences. That test has now run, and it resolved the open question in the direction A7's coefficient evidence pointed: the tabularization's real loss was temporal structure, not pixel appearance. The preliminary non-adoption of pixel-level pretrained models is now final; what upgraded is the representation (causal trends added) and the confidence in the decision (anchor-verified, fold-paired evidence rather than inference from coefficients).

## Alternatives considered and rejected

1. **Small GRU over window sequences — rejected on direct evidence.** Worse on all four headline metrics (QWK 0.361 vs 0.484), worse corners (25/180 vs 17/180), worse middle class (12/60 vs 21/60), early stopping at median epoch 6 signaling immediate overfit. 150 training videos cannot feed a sequence model; the trend features capture the same temporal signal with 60 subjects' worth of statistical honesty.
2. **Frozen generic image/face embeddings (the one credible pretrained-vision test from A7) — rejected on cost, risk, and now-weakened motivation.** It requires re-downloading ~111 GB of video for many hours of staged processing; it reopens the appearance/lighting/identity leakage channel the geometric design exists to close (the luminance-only control cannot be extended to embedding space cheaply); and its motivating hypothesis — that the representation was missing something — has now been answered in favor of *temporal* structure, which generic per-window embeddings would not fix (they are still per-window). Expected value no longer justifies the cost before the final package.
3. **Partial or full fine-tuning of a pretrained vision model — rejected.** All of alternative 2's risks, plus memorization of a 60-subject cohort containing identifiable faces, plus compute and maintainability burden, plus the A6/A8 capacity lesson: every added-capacity model tried on this dataset (MLP, GRU) has lost to a linear model on a better representation.
4. **Keep plain `logistic_full` unchanged — rejected, narrowly.** It remains the fallback if Week 9 uncovers a flaw in the trend features (e.g., the cold-start behavior below), but declining an equal-QWK model that improves rank-MAE, macro-F1, and the primary failure mode would privilege inertia over evidence.

## Risks introduced and reduced by the recommended strategy

- **Introduced — cold start / prediction-time availability:** trend features need session history. `delta`/`slope` need ~1 minute; `drift` sharpens as the session baseline accumulates; at window 0 all trend features are 0 by construction, so early-session predictions degrade gracefully toward plain logistic_full behavior — but this must be stated in the deployment context and examined in Week 9.
- **Introduced — session-start assumption:** `drift` measures change from the session's *own* start. A user who begins a session already drowsy has drift ≈ 0 on the very features that flag drowsiness — a structural blind spot the dataset cannot surface (every UTA-RLDD video has a constant label). Named as a Week 9 handoff item.
- **Introduced — mild variance cost:** QWK fold band widens slightly (±0.399 vs ±0.378); 46 features against 60 subjects is more capacity than 18, though LOSO evidence shows it generalizes.
- **Reduced:** middle-class blindness (the stakeholder-critical failure mode) reduced for the first time; no new dependencies (champion path remains sklearn-only; torch stays optional for rejected/diagnostic runs); no new data, licensing, or privacy surface (same table, same upstream extractor).

## What Week 9 failure analysis should examine

Concrete, in priority order: (1) the 39 still-missed low-vigilant videos — are failures concentrated in particular subjects, and do the misses correlate with landmark quality (`frac_face_missing`) or with atypical drift trajectories? (2) the cold-start / already-fatigued-at-start blind spot — slice accuracy by position-in-video and inspect videos whose *first* windows already show elevated blink metrics; (3) the 17 remaining corner errors and whether trend features changed *which* videos sit in the dangerous corner; (4) calibration — whether the widened confidence gap survives per-class inspection, since a warning threshold is the stakeholder-facing output.

---

# Part C — Dataset-audit update and Week 9 handoff

## Audit assumptions that change with the revised strategy

- **Prediction-time input availability (updated):** the model input is no longer "the current 15-second window" but "the current window plus the session's window history." The audit's deployment assumption becomes: sessions are continuous, start near the user's baseline state, and provide ≥1 minute of warm-up before trend features are informative. Cold-start behavior degrades to the A7 champion, not to nonsense — but the assumption is now recorded.
- **Representativeness (sharpened, not new):** drift features assume the session start approximates a personal baseline. Sessions started mid-fatigue violate this silently. The cohort skew already in the audit (51M/9F, ethnicity-skewed) now compounds with this: whether drift trajectories differ by subgroup is unexamined.
- **Reproducibility (extended):** A8 runs load the *saved* `folds.json` rather than regenerating folds, and an anchor run must reproduce the prior champion's fold metrics before any comparison is read. This tightens the audit's reproducibility contract and was verified this week (exact match, all 60 folds).
- **Unchanged:** licensing (UTA-RLDD research license; MediaPipe Apache-2.0 upstream dependency, already documented in A7), privacy/confidentiality (no new data, no raw video retained), memorization (no fine-tuning; linear model on derived features), leakage (subject-wise LOSO with disjointness assertions; causal trend features unit-tested against future peeking).
- **Simpler-model note the assignment asks for:** the recommended strategy *is* the risk-avoiding choice — every pretrained/fine-tuned alternative examined would add licensing surface, memorization risk, leakage channels, or maintainability burden that the linear-model-on-derived-features path avoids.

## Week 9 handoff: the two most important questions

1. **Who are the still-missed low-vigilant videos?** 21/60 recall is progress, not success. Slice the 39 misses by subject, landmark quality, and drift trajectory shape. If they concentrate in identifiable slices (glasses, facial hair, low-quality landmarks, atypical baselines), that is a fairness and deployment-scope finding; if they scatter, that is evidence of irreducible midpoint ambiguity and an argument for widening the middle band in the stakeholder-facing output.
2. **Does the session-baseline assumption fail dangerously?** The drift blind spot (user already drowsy at session start) is the exact inverse of the aid's purpose. The dataset cannot test it directly (constant-label videos), so Week 9 should probe it synthetically: score truncated sessions that *begin* mid-video in drowsy recordings and measure how much the model's drowsy recall depends on seeing an alert-ish start.

---

# Evidence appendix

- **Commands:** `uv run train_temporal` (loads `data/frame_windows.parquet` + saved `outputs/folds.json`; runs `logistic_full_anchor`, `trend_logistic`, `gru_sequence`; writes all `temporal_*` artifacts). `--skip-gru` runs only the logistic-path experiments. Tests: `uv run pytest tests/test_temporal.py` (causality, video-boundary, padding/packing, and known-slope checks).
- **Artifacts (all in `outputs/`, A7 files untouched):** `temporal_manifest.json` (config, seeds, package versions, git SHA), `temporal_metric_summary.csv`, `temporal_fold_metrics.csv`, `temporal_video_predictions.csv`, `temporal_confusion_matrices.csv`, `temporal_diagnostics.csv`, `temporal_learning_curves.csv` (per-epoch GRU traces), `temporal_trend_coefficients.csv`.
- **Protocol:** LOSO, 60 folds loaded from `outputs/folds.json`; seed 42 with per-fold offsets identical to the A6/A7 convention (`random_seed + fold_idx`); train-fold-only preprocessing; GRU used the same 9-subject validation carve-outs recorded in the folds file for early stopping. Anchor reproduction: `logistic_full_anchor` fold-level QWK matches A7's stored `fold_metrics.csv` exactly (numpy allclose, 60/60 folds).
- **Prior evidence referenced:** A6 `metric_summary.csv` (PERCLOS/MLP/luminance/majority rows), A7 `logistic_full_coefficients.csv`, A7 memo (preliminary transfer-learning decision and its missing-evidence list).
- **Dataset citation:** Ghoddoosian, Galib, and Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection," CVPR Workshops 2019.
