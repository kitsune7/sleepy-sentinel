# Assignment 9 — Portfolio Failure Analysis and Responsible AI Risk Review

> Scope note: this memo analyzes the champion adopted in Assignment 8 (`trend_logistic`).
> No new model was built. The one new computation is a bounded *scoring* probe — the
> cold-start truncation test A8's handoff explicitly requested — run with the
> already-chosen model under the identical LOSO protocol.

# Part A — Current model and evidence snapshot

## Problem, stakeholder, dataset, target, scope

- **Problem:** Personal early-fatigue aid — given a short live-video window (**candidate input**), estimate whether the user appears alert, low-vigilant, or drowsy.
- **Stakeholder / use case:** me / a future self-monitoring user during long work sessions; not safety-critical, not client-facing, explicitly not for monitoring others.
- **Dataset:** UTA-RLDD, 60 subjects × 3 videos → per-window tabular feature table (18 geometric/temporal model-input features from MediaPipe landmarks; luminance confound columns and `frac_face_missing` reserved for diagnostics per charter), plus the 28 causal cross-window trend features adopted in A8.
- **Target:** ordinal `y ∈ {0,1,2}` = {alert, low-vigilant, drowsy}, scored at video level after window→video probability averaging.
- **Scope boundary (unchanged since A4):** no raw-pixel / end-to-end CNN modeling; geometric features from the pretrained MediaPipe landmarker are the representation.
- **Current model/strategy (adopted in A8):** `trend_logistic` — frozen pretrained landmark features → per-window geometric summaries → causal cross-window trend features → multinomial logistic regression.
- **Split/evaluation procedure:** leave-one-subject-out (60 folds) from the saved `outputs/folds.json`, subject disjointness asserted per fold, train-fold-only imputation and scaling, video-level QWK / rank-MAE / accuracy / macro-F1 as mean ± std across folds.
- **Strongest current result:** QWK 0.484 ± 0.399, rank-MAE 0.539, accuracy 0.556, macro-F1 0.449 — beats every floor (majority 0.000, luminance-only 0.048, PERCLOS-only 0.410) and is the first model to move low-vigilant recall (14/60 → 21/60 vs. the A7 champion).
- **Known limitations carried in:** low-vigilant remains the weakest class; softmax confidence has never been good enough to gate a warning; the drift features assume the session starts near the user's baseline.

## Model candidates tested so far (A5–A8)

| Model | QWK (mean±std) | rank-MAE | Accuracy | macro-F1 | Provenance |
| --- | --- | --- | --- | --- | --- |
| Majority | 0.000 ± 0.000 | 0.944 ± 0.125 | 0.333 ± 0.000 | 0.167 ± 0.000 | A6 floor |
| Luminance-only | 0.048 ± 0.502 | 0.844 ± 0.400 | 0.361 ± 0.248 | 0.258 ± 0.244 | A6 confound check |
| PERCLOS-only | 0.410 ± 0.414 | 0.617 ± 0.357 | 0.517 ± 0.241 | 0.405 ± 0.272 | A6 |
| MLP regularized | 0.363 ± 0.382 | 0.650 ± 0.339 | 0.483 ± 0.233 | 0.370 ± 0.254 | A6 (with A7's luminance-column caveat) |
| Logistic, full features | 0.481 ± 0.378 | 0.567 ± 0.326 | 0.528 ± 0.224 | 0.414 ± 0.255 | A7 champion |
| **Logistic + causal trends** | **0.484 ± 0.399** | **0.539 ± 0.353** | **0.556 ± 0.251** | **0.449 ± 0.299** | A8 champion |
| GRU over window sequences | 0.361 ± 0.470 | 0.661 ± 0.386 | 0.478 ± 0.241 | 0.371 ± 0.254 | A8, rejected |

**Comparison point:** every number above is directly comparable — A8 verified an anchor
re-run of the A7 champion reproduces its fold-level QWK exactly on the loaded folds, and
this week's probe at truncation 0.0 reproduces the saved A8 champion predictions
(all 180 video labels identical; see the evidence appendix for the version-tolerance note).

---

# Part B — Failure analysis and stakeholder consequences

Assignment 8 handed Week 9 a prioritized list of failure questions. All four were run
(`src/training/failure_analysis.py`, artifacts `outputs/a9_*`). Three meaningful failure
patterns emerged; the fourth analysis (landmark quality) came back *negative* and is
reported inside FP1 because ruling a cause out is also evidence.

## Failure Pattern 1 (FP1) — Low-vigilant misses follow eyelid *level*, and they are overconfident

**Pattern.** The 39 still-missed low-vigilant videos are not random and not
quality-driven: they split by which side of the ordinal midpoint their eyelid *level*
resembles, and the model is *more* confident on the misses than on the hits.

**Evidence** (`a9_low_vigilant_miss_slices.csv`, `a9_low_vigilant_miss_summary.csv`):

| Outcome | n | Mean confidence | Start blink_dur_max | Final blink_dur_max_drift |
| --- | --- | --- | --- | --- |
| Hit | 21 | 0.437 | 0.232 | −0.023 |
| Missed as alert | 25 | 0.512 | 0.132 | −0.020 |
| Missed as drowsy | 14 | 0.614 | 0.314 | +0.098 |

- Missed-as-alert videos look alert from their first minute (start blink_dur_max 0.13 vs 0.23 for hits) and never develop a fatigue trend.
- Missed-as-drowsy videos look fatigued from their first minute (0.31) *and keep deteriorating* (positive final drift) — exactly the "runaway drift" signature A8's coefficients assign to drowsy.
- Confidence is inverted: hits average 0.437 while misses average 0.512–0.614, with individual misses up to 0.92.
- **Ruled out:** landmark quality. Mean `frac_face_missing` is 0.0005–0.0006 for hits and both miss groups alike — the misses are not low-quality videos. Misses also scatter across 39 distinct subjects (every subject has exactly one low-vigilant video; no subject concentration is possible at video level, and subjects who miss low-vigilant are no more likely to also miss drowsy: 15/39 vs 10/21).

**Likely cause (evidence vs. speculation).** The evidence says the errors track the
level features across the ordinal midpoint. The *speculative* part is why: Karolinska Sleepiness Scale (KSS) 6–7
("low-vigilant") is genuinely between the neighboring states, and some sessions
plausibly sit near a boundary for their entire recording. What the evidence cannot
distinguish is irreducible label ambiguity vs. a representation still missing some cue.

**Stakeholder consequence.** In plain language: when the aid is wrong about the exact
state I'm in, it is wrong in a *plausible direction* (one ordinal step) — but it is
also *more sure of itself when wrong than when right* on this class. An "early warning"
that fires confidently in the wrong direction trains the user to ignore it.

**Expected or new?** The midpoint ambiguity was expected (A6 onward). The
overconfidence *direction* — misses more confident than hits — is newly discovered.

**Missing evidence.** Per-video KSS scores (the dataset publishes only the 3-way class),
which would show whether missed-as-alert videos were labeled KSS 6 and missed-as-drowsy
KSS 7; and a calibration method (temperature scaling or CORN cumulative probabilities)
to test whether the inversion is fixable.

## FP2 — The A8 gain lives in the session warm-up (cold-start probe)

**Pattern.** A8 flagged a structural blind spot: `drift` measures change from the
session's *own start*, so a user who opens the aid already fatigued has drift ≈ 0 on
exactly the features that flag fatigue. The dataset cannot show this directly (every
video has one constant label), so the probe simulates it: drop the first 25% / 50% of
each *test* video's windows, restart the clock, recompute trend features from the
truncated start, and score with the same per-fold models.

**Evidence** (`a9_cold_start_summary.csv`):

| Truncation | Accuracy | Alert recall | Low-vig recall | Drowsy recall |
| --- | --- | --- | --- | --- |
| 0% (reproduces A8) | 0.556 | 44/60 | **21/60** | 35/60 |
| 25% | 0.483 | 39/60 | **14/60** | 34/60 |
| 50% | 0.517 | 41/60 | **17/60** | 35/60 |

Two findings, one reassuring and one not:

1. **The feared failure did not materialize.** Drowsy recall barely moves (35 → 34 → 35). The level features (blink durations, PERCLOS) carry drowsy detection on their own; the model does not need to *watch you deteriorate* to call you drowsy.
2. **The damage lands on the middle class.** At 25% truncation, low-vigilant recall falls from 21/60 to 14/60 — *precisely the A7 anchor's number*. The entire A8 improvement is warm-up-dependent: cut the session's first quarter and the trend features stop helping the one class they were adopted to help.

Corroborating evidence from *within* normal sessions (`a9_window_position_accuracy.csv`):
first-decile windows are the least accurate for every class (alert 0.59 vs ~0.70
mid-session; low-vigilant 0.27 vs ~0.33 late; drowsy 0.48 vs ~0.55). Early-session
predictions are measurably weaker even without truncation.

**Likely cause.** Direct: at the cut point all drift/delta features are 0 by
construction and sharpen only as history accumulates (this is enforced, unit-tested
behavior, not speculation). The non-monotonicity (25% worse than 50%) is within noise
at n=60 per class and is not interpreted further.

**Stakeholder consequence.** The aid's stated purpose is *early* fatigue detection. If
I open it mid-episode — the realistic bad case — it degrades to the A7 champion's
behavior on exactly the early-warning class. Deployment framing must say: the first
minutes of a session are reduced-accuracy, and a session should ideally start near
one's alert baseline.

**Expected or new?** Named as a risk in A8 (expected), now quantified — and partially
*weakened*: the drowsy-blindness component did not appear.

**Missing evidence.** Real sessions that begin mid-fatigue with ground truth (the
truncation is a simulation on acted, constant-label recordings); and per-user baselines
persisted across sessions, which would remove the within-session warm-up assumption
entirely.

## FP3 — Corner errors did not shrink; they shuffled — and most point the dangerous way

**Pattern.** A7 and A8 both report 17/180 alert↔drowsy corner errors, which reads as
stability. It isn't: only 13 videos are corners under *both* models.

**Evidence** (`a9_corner_migration.csv`): 13 persistent, 4 fixed by trends
(04/10, 06/10, 09/10, 16/10 — all true-drowsy videos pulled up to low-vigilant or
correct), 4 introduced by trends (20/0 and 45/0: true-alert called drowsy; 34/10 and
53/10: true-drowsy called alert, and 53/10 was *correct* under the anchor). Of
trend_logistic's 17 corners, **11 are drowsy-called-alert** — the worst possible
direction for a fatigue aid — vs. 6 alert-called-drowsy. Related instability at the
fold level: per-fold QWK spans 0.00–1.00 with 22 of 60 folds at ≤ 0 (each LOSO fold
scores only 3 videos, so fold-level metrics are coarse, but the spread is real).

**Likely cause.** Evidence: the corner *count* is at equilibrium while membership
churns, meaning the trend features re-rank borderline videos rather than resolving
them. Speculation: the persistent drowsy→alert corners (9 videos) may be sessions
where fatigue shows in channels the geometric features do not capture (posture,
micro-expressions) or where the subject actively fought sleep — the acted-fatigue
domain gap from the audit.

**Stakeholder consequence.** A drowsy session called *alert* is a silent failure: the
aid says nothing precisely when it matters most. Eleven of 180 videos (6%) fail this
way. Any deployment claim must state that the aid can miss full drowsiness outright,
which is why it can only ever be decision support, never a safety control.

**Expected or new?** Corner errors as the key failure geometry: expected (charter).
That model changes *shuffle* corner membership at constant count: newly discovered.

**Missing evidence.** A qualitative review of the 13 persistent corner videos (what do
these sessions look like?) — feasible but manual, and it must stay local per the
privacy constraints.

## Calibration cross-check (supports FP1/FP3)

Per-class confidence-by-correctness (`a9_calibration_by_class.csv`), trend_logistic:

| Predicted class | Correct (n, mean conf) | Wrong (n, mean conf) | Gap |
| --- | --- | --- | --- |
| Alert | 44, 0.576 | 36, 0.520 | +0.056 |
| Low-vigilant | 21, 0.437 | 24, 0.473 | **−0.036 (inverted)** |
| Drowsy | 35, 0.655 | 20, 0.576 | +0.079 |

The aggregate gap A8 reported (0.575 vs 0.520) survives for alert and drowsy but
*inverts* for the middle class: when the model predicts low-vigilant and is wrong, it
is on average more confident than when it is right. A raw softmax threshold cannot
gate warnings for the class the aid most cares about. This upgrades the
calibration-work item (planned since A6) from nice-to-have to blocking.

---

# Part C — Audit-to-failure trace and responsible AI risk update

Five prior audit/charter items, classified against Week 9 evidence:

### 1. Lighting/luminance confound — **confirmed controlled (risk weakened)**

- **Original concern (A4 charter):** the model might partly be a brightness detector; the luminance-only baseline exists to expose this.
- **Week 9 evidence:** luminance-only sits at QWK 0.048 ± 0.502 (chance); the champion at 0.484 uses geometric features only, and the A9 failure slices show errors tracking eyelid dynamics, not lighting columns.
- **Responsible AI risk:** spurious-correlation / proxy behavior.
- **Stakeholder impact if deployed:** minimal on this axis; the design (geometric features + confound baseline) held.
- **Recommendation effect:** none — this risk no longer argues against use.

### 2. Ordinal midpoint ambiguity (low-vigilant) — **confirmed by observed evidence**

- **Original concern:** KSS 6–7 is genuinely between neighboring states; the charter made ordinal error structure (not accuracy) the headline for this reason.
- **Week 9 evidence:** FP1 — misses split by eyelid level across the midpoint (25 down, 14 up), with overconfidence on misses.
- **Responsible AI risk:** miscommunication of certainty; an aid that overstates its confidence about the very state it exists to catch early.
- **Stakeholder impact:** false reassurance (missed-as-alert) or alarm fatigue (missed-as-drowsy).
- **Recommendation effect:** argues for **limit to decision support** with a widened/hedged middle band in any user-facing output, and blocks any autonomous warning behavior until calibration work lands.

### 3. Session-baseline / cold-start assumption — **revised based on new evidence**

- **Original concern (A8, newly introduced by the trend features):** drift ≈ 0 for a user who starts a session already fatigued; feared consequence was drowsy blindness.
- **Week 9 evidence:** FP2 — drowsy recall is robust to truncation (level features carry it), but the middle-class gain fully evaporates at 25% truncation; early windows are weakest even in full sessions.
- **Responsible AI risk:** silent context-dependence — the aid's advertised improvement exists only under a usage pattern (session starts near baseline) the user was never told about.
- **Stakeholder impact:** a user who opens the aid mid-episode gets last year's model on the early-warning class while believing they have this year's.
- **Recommendation effect:** deployment context must state the warm-up requirement; a future revision should consider persisted per-user baselines. Does not block decision-support use.

### 4. Cohort skew / demographic subgroups — **still untested**

- **Original concern (A4):** 51M/9F, ethnicity-skewed, glasses in 21/180 videos; per-group error rates expected to differ.
- **Week 9 evidence:** none — and now demonstrably none: the feature table carries no demographic columns, so no subgroup slice can be computed from current artifacts. Producing one requires manual annotation of raw videos (kept local per privacy constraints).
- **Responsible AI risk:** unquantified performance disparity; representativeness failure.
- **Stakeholder impact:** for the charter's self-monitoring user, bounded (the aid should be re-validated on its one actual user); for any broader deployment, unbounded and unmeasured.
- **Recommendation effect:** caps the recommendation at **personal decision support**; any multi-user deployment claim would be unsupported by evidence.

### 5. Confidence as a warning gate — **confirmed as a problem (newly sharpened)**

- **Original concern (A5/A6):** softmax confidence barely separates correct from incorrect predictions; a proper calibration step is needed before any threshold drives a warning.
- **Week 9 evidence:** the per-class view shows the gap is not just small but *inverted* for low-vigilant (−0.036).
- **Responsible AI risk:** overconfident errors driving user-facing actions.
- **Stakeholder impact:** warnings would fire most confidently on some of the model's mistakes.
- **Recommendation effect:** blocks any confidence-thresholded warning feature; the planned CORN/calibration experiment is now the highest-value next step (Week 10).

---

# Part D — Sequence, temporal, context, and attention relevance decision

**Decision card:**

- **Does order/time/history/context affect the target outcome?** **Yes.** Fatigue evolves within a session, and causal cross-window trend features produced the only middle-class improvement across five assignments (14→21/60) — direct evidence that history carries signal.
- **Does the current split/evaluation design respect deployment time?** **Yes.** Subject-wise LOSO handles the dominant (identity) leakage axis; within each video, trend features are causal by construction and unit-tested (editing window *t+1* cannot change features at *t*); no feature uses the future. The video-level label is constant per session, so there is no label-time subtlety beyond that simplification.
- **Could future-derived features, duplicates, repeated entities, histories, or source-target overlap leak across splits?** **No.** Folds are disjoint on `subject_id` with a coded assertion; per-subject EAR normalization uses only that subject's own frames; imputation/scaling are fit on train folds only; A8/A9 runs *load* the saved folds and verify an anchor reproduction before comparisons are read.
- **Feature window vs. label window:** available before prediction: the current 15-second window plus the session's own past windows (trend features need ~1 minute before `delta`/`slope` inform, longer for `drift`). Observed after prediction: nothing from the future is used; the "label" is the session's self-reported KSS state, constant across the video — a known dataset simplification rather than a post-prediction outcome.
- **Could behavioral-history use create stakeholder harm?** **Yes, if misused** — the same trend features that flag my fatigue could profile someone else's. Mitigations are charter-level: self-monitoring only, consent of the person on camera, on-device processing, no retention, no raw video or face crops leaving the machine. Cold-start evidence adds a subtler harm: decisions made with *missing context* (no session history) are measurably worse, so the aid must not present early-session output as full-strength.
- **Are attention or sequence methods relevant, irrelevant, or untested?** **Sequence models: tested and rejected on direct evidence** (A8 GRU: worse on every metric, median best-epoch 6 — immediate overfit at 150 training videos). **Attention/transformers: ruled out without testing, on the same capacity logic** — attention layers add parameters on top of the GRU's, and every added-capacity model tried on this dataset (MLP, GRU) has lost to a linear model on a better representation. At n=60 subjects, an attention model would also invite attention-map overinterpretation — reading meaning into weights fit on 180 examples.
- **Decision:** **include the temporal/cold-start risk in the final evidence base** (FP2 rows carry into Week 10); do **not** test sequence or attention models further — the temporal *signal* is already captured by causal trend features, and the data scale cannot honestly feed higher-capacity sequence learners.

Because time *is* relevant here, the practical consequences are already embedded above:
evaluation respects causality (tested), failure analysis includes a temporal probe
(FP2), and deployment risk includes the warm-up requirement. What matters *more* than
further sequence modeling is the failure evidence that has nothing to do with time:
midpoint overconfidence (FP1) and dangerous-direction corners (FP3).

---

# Part E — Week 10 handoff ledger

| Evidence or risk row | What we observed | Stakeholder implication | Confidence | Action for Week 10 |
| --- | --- | --- | --- | --- |
| **Strongest supporting evidence:** champion beats all floors under LOSO | QWK 0.484 ± 0.399 vs majority 0.000 / luminance 0.048 / PERCLOS 0.410; anchor-verified comparability chain A5→A9; causal features unit-tested | The signal is real, geometric (not lighting), and subject-independent in expectation | High (protocol), Medium (effect size — fold band is wide, 22/60 folds at QWK ≤ 0) | Lead evidence row for the deployment judgment; report the fold spread beside the mean, always |
| **Most important failure evidence:** middle class still weak and overconfident | Low-vigilant recall 21/60; misses track eyelid level across the midpoint; misses more confident than hits (0.51–0.61 vs 0.44) | The early-warning class is the weakest, and confidence misleads on exactly that class | High | Pair with calibration experiment result; widen/hedge the middle band in any user-facing output design |
| **Dangerous-direction corners** | 11/180 drowsy-called-alert under the champion; corner membership churns between models (13 persistent, 4 fixed, 4 introduced) | Silent failure when it matters most; caps the aid at decision support | High | Carry as a hard limitation row; qualitative review of the 13 persistent corners if time permits |
| **Highest-priority responsible AI risk:** confidence cannot gate warnings | Per-class calibration gap inverted for low-vigilant (−0.036) | Any confidence-thresholded warning would fire confidently on mistakes | High | Run the planned CORN ordinal head + temperature-scaling check — now the single most valuable experiment left |
| **Cold-start / warm-up dependence** | Drowsy recall robust to truncation; low-vigilant gain fully reverts (21→14) at 25% truncation; early windows weakest in all sessions | Advertised improvement holds only when sessions start near baseline | Medium-High (simulation on acted data) | State warm-up requirement in the deployment context; note persisted per-user baselines as future work |
| **Most important missing evidence** | No demographic subgroup slices possible from current artifacts; no real mid-fatigue session onsets; no per-video KSS scores | Multi-user claims unsupported; midpoint ambiguity not fully attributable | High (that it's missing) | Scope the Week 10 judgment to what the evidence covers: personal, single-user decision support |
| **Preliminary Week 10 implication** | — | — | — | **Limit to decision support**: viable as a personal awareness aid with hedged output and stated warm-up caveat; do not deploy autonomous warnings (calibration blocks); do not use for safety-critical or multi-user monitoring (unmeasured subgroups, corner failures) |

---

# Evidence appendix

- **Commands:** `uv run failure_analysis` (loads `data/frame_windows.parquet` + saved `outputs/folds.json` + saved `outputs/temporal_video_predictions.csv`; writes all `a9_*` artifacts). `--skip-cold-start` runs only the artifact-slicing analyses. Tests: `uv run pytest tests/test_failure_analysis.py` (truncation/renumbering, cut-point-as-session-start causality, corner categorization, miss-direction labeling, calibration math, reproduction grading).
- **Artifacts (all in `outputs/`, A7/A8 files untouched):** `a9_manifest.json` (config, seeds, package versions, git SHA, reproduction check), `a9_low_vigilant_miss_slices.csv`, `a9_low_vigilant_miss_summary.csv`, `a9_corner_migration.csv`, `a9_calibration_by_class.csv`, `a9_cold_start_video_predictions.csv`, `a9_cold_start_summary.csv`, `a9_window_position_accuracy.csv`.
- **Protocol:** LOSO, 60 folds loaded from `outputs/folds.json`; seed 42 with the per-fold offset convention (`random_seed + fold_idx`) identical to A6–A8; training data never truncated (the probe asks how the *deployed* model behaves on truncated sessions); trend features recomputed after truncation so the cut point is the session start.
- **Reproduction check:** the truncation-0.0 probe reproduces the saved A8 `trend_logistic` predictions — all 180 video labels identical. Probabilities matched to 4.8e-3 rather than solver precision because the analysis environment ran a different sklearn/Python version than the A8 run; the graded check (`a9_manifest.json → reproduction_check`) records both facts. Re-running in the pinned `uv` environment is expected to match to ≤1e-6.
- **Code note:** the torch-dependent GRU pieces moved from `training.temporal` to `training.sequence_models` so the champion path and this analysis are importable torch-free — making A8's "champion path remains sklearn-only" claim true at the import level. Behavior unchanged; `train_temporal` updated accordingly.
- **Prior evidence referenced:** A8 memo (handoff questions, trend coefficients, anchor protocol), A7 `logistic_full_coefficients.csv`, A6 `metric_summary.csv` floors, A5 evaluation report (calibration concern), A4 charter/audit (confound design, cohort skew, leakage mitigations).
- **Dataset citation:** Ghoddoosian, Galib, and Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection," CVPR Workshops 2019.
