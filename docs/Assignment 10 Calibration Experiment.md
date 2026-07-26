# Assignment 10 — Calibration experiment: can the low-vigilant confidence inversion be fixed?

> The A9 handoff (ledger row 4 / Part E) named this the single highest-value
> experiment left: `trend_logistic`'s softmax confidence is *inverted* for the
> low-vigilant class (misses more confident than hits, gap −0.036), so a raw
> confidence threshold cannot gate a warning on the class the aid exists to
> catch early. This experiment tests whether a calibration method fixes it —
> head-to-head against the champion, under the identical LOSO protocol.
>
> **Answer: no.** Both methods roughly halve the inversion but neither flips it
> positive; each carries its own cost. The A10 confidence-gate block stands, now
> on tested evidence rather than a promissory note.

## What was run

Three runs on the A8 trend-feature table, same 60 saved LOSO folds, same
`random_seed + fold_idx` convention, same window→video probability averaging.
No new representation and no deep net — the A8/A9 evidence says added capacity
loses to a linear model at n=60 subjects, so both calibration methods stay
linear.

- **`trend_logistic`** — the A8 champion, raw softmax. Serves as the reproduction anchor.
- **`trend_logistic_temp`** — champion window logits divided by a temperature `T` before softmax. `T` is fit **leak-free**: within each fold the *training* subjects are split into two disjoint halves and cross-fit, so every logit `T` sees was produced by a model that never trained on that subject. The held-out LOSO test subject never informs its own calibration.
- **`trend_corn`** — a **linear CORN ordinal head** (Shi et al. 2022): two chained binary logistic classifiers, P(y>0) and P(y>1 | y>0), whose product is rank-monotonic by construction. Same impute+scale+logistic front end as the champion; enforces the ordinal structure the softmax head ignores.

**Reproduction anchor passed exactly:** the champion run reproduces the saved A8
`trend_logistic` video predictions — all 180 labels identical, max probability
difference 1.1e-16 (`a10_calibration_manifest.json → reproduction_check`). Every
number below is therefore directly comparable to the A5→A9 chain.

## Results

| Run | QWK | rank-MAE | Accuracy | macro-F1 | Low-vig recall | **Low-vig calibration gap** | Video-level ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `trend_logistic` (champion) | **0.486** | **0.539** | **0.556** | 0.546 | 21/60 | **−0.0363** | **0.075** |
| `trend_logistic_temp` (T̄≈2.83) | 0.478 | 0.539 | 0.556 | **0.548** | **23/60** | −0.0179 | 0.123 |
| `trend_corn` | 0.386 | 0.661 | 0.494 | 0.439 | 6/60 | −0.0150 | 0.114 |

*(A tiny QWK shift 0.484→0.486 vs. the A9 memo is `scipy` availability in this
env letting `decision_function` logits reproduce to 1e-16; labels are identical
to A8. Gap is `conf_correct − conf_wrong`; negative = misses more confident than
hits.)*

Full per-class gap table (`a10_calibration_gap.csv`): alert and drowsy gaps stay
positive and healthy under all three heads; **only low-vigilant is inverted, for
all three.** Temperature scaling shrinks the low-vig gap to −0.018 but does not
flip it. CORN reaches −0.015 — the smallest — but by collapsing the class.

## Interpretation (evidence vs. speculation)

1. **Temperature scaling can't fix it because the problem is local, not global.** The mean fitted temperature is **2.83** (the champion is systematically ~3× overconfident), yet its *aggregate* video-level ECE (0.075) is already the best of the three, and softening makes aggregate ECE **worse** (0.123). A single global scalar cannot target a miscalibration that lives only in the low-vigilant region. This is direct evidence for FP1's reading: the low-vig inversion is **genuine label ambiguity at the KSS 6–7 midpoint**, not global overconfidence a temperature can rescale away. *(Observed: T, ECE, gap. Interpretation: the ambiguity mechanism.)*
2. **Temperature scaling is, correctly, argmax-invariant.** `trend_logistic_temp` keeps 100/180 correct and identical QWK/rank-MAE — dividing logits by a positive scalar never moves the decision. The +2 low-vig recall (21→23) comes only from video-level mean-pooling of softened window probabilities, not a changed rule. So temperature scaling is *free* (no accuracy cost) but *insufficient* (does not flip the gate-blocking inversion).
3. **CORN gets the smallest inversion but wrecks the class it was meant to help.** Enforcing ordinal monotonicity drops low-vig recall 21→**6** and QWK 0.486→0.386. The monotone chain pushes borderline mid-class videos to the neighbouring extremes. Rejected — same capacity/scale logic that rejected the GRU: a stricter structural prior does not pay off at n=60 subjects with a genuinely ambiguous middle class.

## Consequence for the deployment judgment

The A10 judgment (**limit to decision support**, no autonomous confidence-gated
warning) **is unchanged and strengthened.** A9 flagged calibration as the
highest-value open item that *might* unblock a hedged warning; the experiment
converts that open item into a **tested negative result**: two standard
calibration methods, run leak-free under the champion's own protocol, cannot
turn the low-vigilant confidence into a trustworthy gate. Temperature scaling is
worth keeping as a free honesty improvement on the alert/drowsy classes (it
softens their overconfidence at no accuracy cost), but it does not license a
warning on the middle class.

**What could still change it** (future work, unchanged from A9's list but now
better motivated): per-video KSS labels to confirm the misses are true KSS 6–7
boundary sessions (would establish the inversion as irreducible), and a
localized/region-specific calibration rather than a single global temperature.

## Evidence appendix

- **Command:** `uv run calibration_experiment` (loads `data/frame_windows.parquet` + saved `outputs/folds.json` + saved `outputs/temporal_video_predictions.csv` for the anchor; writes all `a10_*` artifacts).
- **Tests:** `uv run pytest tests/test_calibration_experiment.py` — CORN rank-monotonicity + rows-sum-to-1, temperature bounds and NLL-minimization softening overconfident logits, temperature argmax-invariance, ECE zero-at-perfect-calibration, calibration-gap inversion sign.
- **Artifacts (`outputs/`, all `a10_` prefix, prior files untouched):** `a10_calibration_manifest.json` (config, seed, package versions, git SHA, reproduction check, temperature summary), `a10_metric_comparison.csv`, `a10_calibration_by_class.csv`, `a10_calibration_gap.csv`, `a10_ece.csv`, `a10_temperatures.csv`, `a10_calibration_video_predictions.csv`.
- **Protocol:** LOSO, 60 folds from `outputs/folds.json`; temperature fit by subject-disjoint 2-way cross-fit on training subjects only; CORN = two chained balanced logistic heads on the 46-feature trend set; ECE binned at 10 on video-level max-probability confidence.
- **Reference:** Shi, Cao, and Raschka, "Deep Neural Networks for Rank-Consistent Ordinal Regression (CORN)," 2022 — the rank-consistent conditional-training scheme, applied here with linear heads.
