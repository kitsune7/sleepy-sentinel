# Assignment 5

> This assignment asks whether a model can be trusted beyond a single aggregate score. You will use training and validation evidence, one generalization intervention, error analysis, subgroup or slice checks, and calibration or confidence evidence when appropriate to make an initial reliability judgment.
> 
> This is not a final portfolio-model assignment. The goal is to practice a trustworthy evaluation process and connect the evidence back to the dataset audit, risks, assumptions, and staged model plan you locked in Assignment 4.

**Note:** Everything in this document in blockquotes is from the assignment description. Everything outside them is a note from me.

## What this assignment entails

> Evaluate a portfolio baseline or initial model candidate if your portfolio project is ready for modeling. Use the dataset audit, evaluation strategy, leakage risks, success criteria, and staged model-improvement plan from Assignment 4 to decide what evidence to collect.
> 
> If your portfolio project is not yet ready for this level of evaluation, use the approved practice model based on the Week 4 NYC TLC trip-duration dataset, or instructor-provided sample results derived from that dataset. In that case, label Part A clearly as practice work, complete the same evaluation process, and explicitly explain how the process will transfer to your portfolio project. Part B must still discuss your portfolio project: name which evidence transfers directly and which portfolio-specific evidence remains missing.
> 
> The assignment should produce early evidence for the final portfolio presentation where possible, but the current model should be treated as a baseline or initial candidate, not as the final recommendation.

## Part A

> Scope boundary: use one main model and one focused generalization intervention. You may include a small number of additional diagnostic runs if they clarify the main result, but identify one primary intervention and do not present the work as a broad hyperparameter search. Do not run an exhaustive hyperparameter search, try many architectures without a specific diagnostic purpose, or attempt to make the model final this week.
> 
> Depth expectation: this section should be evidence-first. A reader should be able to see what the model did, where it failed, what you changed, and whether the change improved trustworthiness or only changed the aggregate score.

---

> State the dataset, task, target, model being evaluated, and whether this is portfolio work or approved practice work.

dataset: UTA Real-Life Drowsiness Dataset (I pre-processed the video files into features)
task: ordinal classification
target: alertness level (0 - alert, 5 - average, 10 - sleepy)
model: _custom_
type: portfolio work

---

> Use a clear train/validation/test strategy or other justified evaluation procedure.

train/validation/test strategy:
- subject-wise cross-validation
- Use `GroupKFold` (groups = `subject_id`)
- **5-fold** (12 held-out subjects per fold), or **Leave-One-Subject-Out (LOSO)** (60 folds) for the gold-standard estimate
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

> Explain why the split strategy is appropriate for the intended use and whether temporal, grouped, repeated-entity, or other leakage concerns affect the split.

---

> Provide evidence for the split choice, not only a rationale. At minimum, report split counts, how the split was created, and one check that the split is plausible for the task, such as target or label distributions by split, time ranges by split, group/entity overlap checks, category coverage checks, or another task-relevant split audit.

---

> Report at least one task-appropriate aggregate metric, using the metric plan from Assignment 4 where feasible.

---

> Compare training and validation behavior using a learning curve, metric table over epochs or settings, or another clear before/after evaluation trace.

---

> When feasible, include both training and validation metrics over epochs. A final metric table alone is not enough to diagnose overfitting or underfitting.

---

> Diagnose whether the evidence suggests overfitting, underfitting, plausible generalization, leakage risk, unstable validation behavior, or insufficient evidence.

---

> Apply at least one generalization technique such as weight decay, dropout, early stopping, data augmentation where appropriate, model simplification, reduced features, or another justified constraint.

---

> Compare behavior before and after the intervention and explain what changed.

---

> Analyze errors by examples, subgroup, slice, condition, time period, source, class, target range, or another task-relevant dimension.

---

> Evaluate calibration or confidence behavior when appropriate for the task. If calibration is not appropriate, briefly explain why.


## Part B

> Your writeup should include a short trace from at least three Assignment 4 audit risks, assumptions, or success criteria to the Week 5 evidence you collected.

Assignment 4 documented portfolio risks:
```md
## Prediction-time availability and leakage risks

Every model feature is computable from a single window of live video at inference time, so there is **no temporal/feature leakage from the future**. The risks that *do* matter here are identity and normalization leakage:

- **Subject leakage (the dominant risk):** windows from one person are highly correlated; if any of a subject's windows land in both train and test, the model recognizes the *face*, not the *state*. **Mitigation:** all splits are **subject-wise** (GroupKFold / LOSO on `subject_id`), with a coded assertion that train/val/test subject sets are disjoint in every fold.
- **Per-subject normalization leakage:** baseline EAR differs by face. Per-subject normalization stats are computed from **that subject's own frames only** (across all 3 of their videos), so nothing crosses between subjects — safe in any fold. Crucially, **the alert video is never used as a per-subject baseline**, because that baseline would not exist at deployment time.
- **Global scaling leakage:** the `StandardScaler` is fit on the **training fold only**, then applied to val/test.
- **Appearance/lighting leakage:** addressed by design (geometric features, not pixels) and *empirically checked* via the luminance-only baseline.
```

---

> Your writeup should include which risks or assumptions were confirmed, reduced, contradicted, newly discovered, or still untested.

---

> Your writeup should include whether the model is reliable enough for the intended use at this stage.

---

> Your writeup should include what additional evidence would be needed before deployment or client-facing recommendation.

---

> Your writeup should include how this evidence should guide the next staged portfolio model improvement.

---

> Your writeup should include if you used a practice model, a short note explaining which parts of the evaluation process transfer directly to your portfolio project and which parts still need portfolio-specific evidence.