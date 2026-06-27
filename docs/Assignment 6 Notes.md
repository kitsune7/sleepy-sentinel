# Assignment 6 — Tabular Modeling and Model-Choice Justification

## TL;DR for this assignment (read first)

- The portfolio data is **already tabular** (per-window feature CSVs → ~17 numeric features). So: **no proxy translator needed.** Part A uses the portfolio dataset directly. Say so explicitly — it's a graded distinction.
- The new work is a **fair model-choice comparison**: a **classical tabular baseline** (LightGBM or logistic/ordinal regression) vs. the **MLP we already built in A5**, under the _same_ subject-wise CV, same window→video aggregation, same QWK/rank-MAE/confusion metrics.
- This also closes two gaps the A5 notes left open: no `baselines.py`, and no model-vs-model comparison. Reuse everything possible — splits, metrics, dataset prep all already exist.
- The assignment does **not** require the neural model to win. The deliverable is an honest "MLP earns its complexity / it doesn't" argument.

### What's reusable vs. new

| Already exists (reuse as-is)                              | New for A6                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------------- |
| `src/training/splits.py` (GroupKFold, disjoint asserts)   | A classical baseline model (LightGBM or logistic/ordinal reg)             |
| `src/training/dataset.py` (train-only scaler, fold prep)  | `baselines.py` _(or extend `train.py`'s `model_runs`)_                    |
| `src/evaluation/metrics.py` (QWK, rank-MAE, confusion)    | Results table: baseline vs. MLP, side by side                             |
| `src/training/train.py` (`model_runs`, per-epoch logging) | "Are embeddings appropriate?" note (likely **no** — all features numeric) |
| A5 MLP runs (baseline + regularized)                      | Practical-constraints + responsible-use discussion for _tabular_ choice   |
| `folds.json`, `metric_summary.csv`, etc.                  | Part B checkpoint                                                         |

---

# Part A — Portfolio tabular representation and model comparison

> Scope boundary: one focused comparison, not a model tournament. A small number
> of diagnostic variants is fine; no exhaustive hyperparameter/architecture search.

## A1. Problem, stakeholder, dataset, target, inputs — and data-type declaration

> State the portfolio problem, client/stakeholder scenario, dataset, prediction
> target, candidate inputs, and whether Part A uses native tabular portfolio
> data, a portfolio-derived proxy, or approved fallback case-study materials.

- **Problem / stakeholder:** _(carry forward 1–2 sentences from the charter —
  personal early-fatigue aid; the "client" is me / a future self-monitoring user)_
- **Dataset:** UTA-RLDD, 60 subjects × 3 videos → per-window feature CSVs in
  `data/<subject>/{0,5,10}.csv`. _(fill in window/row counts)_
- **Target:** ordinal `y ∈ {0,1,2}` = {alert, low-vigilant, drowsy}, scored at
  video level.
- **Candidate inputs:** the ~17 model features _(list or point to charter
  "Candidate input features")_.
- **DATA-TYPE DECLARATION (do not skip):** **Native tabular portfolio data.**
  The raw modality is video, but our portfolio representation _is already_ a
  tabular feature table extracted in A-prior work — so we use the portfolio
  dataset directly and **do not** use a Week 6 proxy translator. _(state this
  plainly; it's the cleanest path and graded)_

## A2. Proxy / fallback notes — N/A, but state it

> If using a proxy, describe the transformation and what it preserves/loses.
> If using fallback materials, explain the blocker.

- **Not applicable.** One sentence each:
  - No proxy translator used — data is natively tabular. _(optional: note that
    the video→feature extraction in `src/data_prep/` is *our own* domain
    representation, not the generic Week 6 proxy, and what it deliberately
    discards — raw pixels, identity — which is a feature not a bug here.)_
  - No fallback case study — portfolio data is downloaded, labeled, and legally
    usable for research.

## A3. Feature preparation (leakage + prediction-time availability)

> Prepare numeric/categorical/missing/rare/high-cardinality features without
> leakage; keep prediction-time availability clear.

- **All features numeric** → no categorical encoding / no high-cardinality
  handling needed. _(this is also the reason embeddings are likely N/A — see A6)_
- **Leakage controls (reuse from A5, just cite them):**
  - Train-only `StandardScaler` — `dataset.prepare_fold_datasets`.
  - Subject-wise split, disjointness asserted — `splits.assert_disjoint_subjects`.
  - Per-subject EAR normalization uses only that subject's own frames; alert
    video never used as baseline.
- **Missingness:** `frac_face_missing` gate / interpolation rule _(cite charter)_.
- **Prediction-time availability:** every feature computable from one live
  window — one sentence confirming this still holds for both models.
- ⚠️ **Tree-model note:** if you use LightGBM, the StandardScaler is unnecessary
  for it (trees are scale-invariant) — but keep the _split_ identical. Either
  feed LightGBM unscaled or scaled; just document which. The MLP **must** keep
  the scaler.

## A4. Simple baseline model

> Train at least one simple baseline (logistic/linear, tree-based, or other
> classical model).

- **Chosen baseline:** _(pick ONE primary — recommend **LightGBM**, since the
  charter already names it as a fallback model and it's the strongest classical
  tabular contender; logistic/ordinal regression is the lazier alternative)_
- Key settings: _(fill in — n_estimators, depth, etc. Keep it small.)_
- **Ordinal handling:** note whether you treat it as 3-class or use an
  ordinal/threshold decomposition — must match how QWK is computed.
- Command + seed: _(fill in)_

## A5. Neural tabular model

> Train at least one neural model appropriate for tabular data (MLP / embeddings).

- **Reuse the A5 MLP** (`(64,32)` ReLU, the regularized variant: dropout 0.25 +
  wd 1e-4). _(state whether this is verbatim the A5 run or re-run for A6)_
- **Generalization practices (required for the MLP):**
  - ☐ Training-split normalization — already done (scaler).
  - ☐ **Validation-aware early stopping / checkpoint selection** — A5 used fixed
    100 epochs and flagged this as a gap. **A6 is the place to add early
    stopping on val QWK.** _(if you add it, say so and re-report; if not, justify)_
  - ☐ One regularization control — dropout + weight decay already in place.
- Command + seed: _(fill in)_

## A6. Categorical embeddings — appropriate or not?

> Use categorical embeddings if appropriate; if not, briefly explain why.

- **Likely NOT appropriate** — write 2–3 sentences: all ~17 inputs are
  continuous geometric/temporal summaries; there are no categorical or
  high-cardinality ID features to embed. _(confirm there's truly no categorical
  feature before asserting this.)_

## A7. Fair comparison — same target, split, metrics

> Compare using the same target, split/eval procedure, and task-relevant
> metrics. Validation evidence for selection; test reserved for final reporting.

- Both models: same `folds.json`, same window→video aggregation, same
  `evaluation/metrics.py`.
- **Selection on validation, report on test** — state explicitly which numbers
  drove the choice vs. which are final.

## A8. Results table

> Compact table: each model, key settings, metrics, practical notes.

| Model                          | Key settings                  | QWK (mean±std)  | rank-MAE | Accuracy | macro-F1 | Practical notes |
| ------------------------------ | ----------------------------- | --------------- | -------- | -------- | -------- | --------------- |
| MLP (regularized, A5)          | (64,32), dropout .25, wd 1e-4 | _0.400 ± 0.158_ | _0.611_  | _0.489_  | _0.483_  | _(from A5)_     |
| _Baseline (LightGBM / logreg)_ | _(fill in)_                   | _(fill in)_     | _(fill)_ | _(fill)_ | _(fill)_ | _(fill in)_     |

- Include summed-over-folds **confusion matrices** for both (the alert↔drowsy
  off-by-two corners are the result that matters — see A5).

## A9. Practical constraints

> Interpretability, cost, training/inference complexity, maintainability, data
> size, ease of monitoring.

- _(fill in — honest comparison. Likely angles: LightGBM is more interpretable
  (feature importances), trains/infers trivially, no scaler to maintain at
  serve time; MLP needs the scaler + checkpoint + more monitoring. Data is tiny
  (~60 subjects) which generally favors the simpler model.)_

## A10. Responsible-use concern (tabular-specific)

> At least one: sensitive features, proxy variables, fairness across groups,
> automation bias, human-review needs.

- _(fill in — strongest candidates: **proxy-variable / lighting confound**
  (luminance leaking as a proxy for alertness — still untested per A5);
  **fairness** given the 51M/9F + ethnicity cohort skew; **automation bias** if
  a confidence-gated warning fires on a poorly-calibrated model.)_

## A11. Recommendation

> Is a tabular approach justified? Is the neural model justified over the baseline?

- **Tabular approach justified?** _(almost certainly yes — it's our native rep)_
- **Neural over baseline?** _(answer on evidence: does the MLP beat the
  classical baseline on QWK with non-overlapping bands? If not — and given the
  charter's "simpler-and-equal wins" rule — recommend the simpler model.)_

---

# Part B — Portfolio checkpoint and model-choice note

> Keep concise but specific.

- **Current data readiness:** _(fill — features extracted, folds saved, etc.)_
- **Current baseline / model status:** _(A5 MLP done; classical baseline added
  this week; Stage 0 luminance/PERCLOS floors still not implemented)_
- **One concrete next experiment:** _(recommend: implement the luminance-only
  confound baseline — the single biggest untested risk from A5)_
- **Expected staged improvement before final package:** _(map to charter Stage
  2/3 — LOSO, CORN A/B, feature ablation vs PERCLOS-only)_
- **How Week 6 evidence affects the final model-choice argument:** _(fill —
  e.g. "if LightGBM ≈ MLP, the final pick leans simpler/interpretable")_
- **Charter/audit updates, emphasis, or still-untested:** _(fill — lighting
  confound still untested; confidence calibration still weak)_
- **Relevance of tabular methods / embeddings / simpler baselines:** tabular =
  **directly relevant** (native rep); embeddings = **not relevant** (no
  categoricals); simpler baselines = **directly relevant** (the whole point).

---

# CHECKLIST

### Code / artifacts

- [ ] Add a classical baseline — new `src/training/baselines.py` **or** a new
      entry in `train.py`'s `model_runs`. (LightGBM recommended; logreg = lazy option.)
- [ ] Run baseline over the **existing** `folds.json` (do NOT make a new split).
- [ ] Confirm window→video aggregation + QWK path is shared with the MLP.
- [ ] (Recommended) Add early stopping on val QWK to the MLP and re-run.
- [ ] Save baseline metrics to CSV alongside the A5 outputs.
- [ ] Add/extend a test in `tests/` for the new baseline (project convention +
      global rule: prove it works).
- [ ] Capture: exact command(s), data-split description, seed/nondeterminism note.

### Writeup — Part A

- [ ] A1 problem/stakeholder/dataset/target/inputs **+ explicit "native tabular" declaration**
- [ ] A2 proxy/fallback = N/A, stated
- [ ] A3 preprocessing + leakage + prediction-time availability
- [ ] A4 baseline model + settings
- [ ] A5 neural model + generalization practices (esp. early stopping decision)
- [ ] A6 embeddings = not appropriate, justified
- [ ] A7 same target/split/metrics; val-for-selection / test-for-final stated
- [ ] A8 results table + confusion matrices
- [ ] A9 practical constraints
- [ ] A10 one responsible-use concern
- [ ] A11 recommendation (tabular? neural-over-baseline?)

### Writeup — Part B

- [ ] data readiness · model status · next experiment · staged improvement ·
      Week-6→final-argument link · charter updates · tabular/embedding/baseline relevance

### Package & submit

- [ ] One `.zip`: code/artifacts + run evidence + this writeup (Part A & Part B labeled).
- [ ] Cite UTA-RLDD / CVPRW 2019 paper (charter's only license obligation).
- [ ] Sanity check: no raw video or face crops in the zip (privacy constraint).

---

## Decisions to lock before you start

1. **Baseline model:** LightGBM (stronger, interpretable, charter-sanctioned) vs.
   logistic/ordinal regression (simplest). → _recommend LightGBM._
2. **Early stopping:** add it now (closes an A5 gap, more honest) vs. defer
   (less work, note as still-pending). → _recommend add it._
3. **MLP source:** reuse A5 numbers verbatim vs. re-run for a clean same-session
   A/B. → _recommend re-run if early stopping changes, else reuse._
