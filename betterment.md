# Fix Critical Bugs & Improve Models for IEEE-Worthy Results

> [!NOTE]
> **STATUS: IMPLEMENTED — but three claims in this document turned out to
> be wrong, and two of the fixes below must not be applied as written.**
> Read this box before using anything further down as a spec. Full
> write-up, with the measurements behind each point, is in the
> "Bugs found and fixed" section of
> [`PROJECT_STATUS.md`](PROJECT_STATUS.md). Regression tests pinning every
> fix are in `tests/test_fixes.py` (30 tests, all passing).

### Correction 1 — Fix 2 (LogitNorm) is mathematically wrong as written

This document proposes "fixing" `LogitNormMSPSignal` by centering the
logits, then suggests a "better fix" using the ratio `max_logit / ||z||`.
**Neither works, and no variant can work.** All four candidates were
measured over a sweep of margins and each collapses to at most **two
distinct values**:

| variant | result |
|---|---|
| current, `[0, s]` gauge | constant `0.2689` |
| centered logits (this doc's Fix 2) | constant `0.1956` |
| `max_logit/‖z‖`, `[0, s]` gauge | binary indicator `1[s>0]` — *class identity, strictly worse* |
| `max_logit/‖z‖`, centered | constant `0.7071` |

The reason is structural, not a bug: a softmax reads only logit
*differences*, so at `K = 2` the entire signal is the single scalar gap
`s`. Dividing by `‖z‖_p` removes exactly one degree of freedom — the
overall scale — and at `K = 2` that scale **is** `|s|`. Only `sign(s)`
survives.

So the signal is **excluded from the binary bank and raises on binary
input**, rather than being patched. It is kept unchanged for `K ≥ 3`,
where the normalisation is well defined. This also means Cattelan &
Silva's logit normalisation is *inapplicable to binary classification* —
a limitation of the published method, worth a sentence in the paper.

### Correction 2 — this document's core premise is only half right

The claim at the top ("Fixing these alone should meaningfully improve
aggregator results") overstates what a signal fix can buy. Measured on
20k samples: **on a binary task every Tier-A signal is a strictly
monotone transform of every other one** — MSP, entropy, both margins,
temperature-scaled MSP, and even a *correctly fixed* energy all have
pairwise Spearman |ρ| = **1.0000**.

Tier A therefore supplies exactly **one** distinct ranking. No aggregator
over Tier A alone can order abstentions any differently from MSP, however
it is trained. Fixing energy stops it *actively poisoning* the aggregators
(pre-fix it scored AURC 0.1824 on Adult against random's 0.1284 — worse
than random abstention) but **cannot make it informative**.

This is a structural explanation for the project's central negative
result, independent of any bug: the aggregators never beat MSP
in-distribution because five of their features carried no information the
sixth did not already have, and two of the rest were broken. On binary
problems the only genuinely independent orderings come from **Tier B**
(ensemble disagreement), **Tier C** (geometry/density) and the new
`calib_residual` signal.

### Correction 3 — Fixes 5 and 6 must not both be applied as written

Fix 5 (gate temperature `T: 0.05 -> 0.5`) and Fix 6 (coverage penalty
`lambda: 1.0 -> 10.0`) are each plausible in isolation and **catastrophic
together**. Neither this document nor the first implementation pass checked
whether the target coverage was still *reachable*:

| gate applied to | max reachable `mean(g)` at T=0.5 | target kappa |
|---|---|---|
| squashed score `sigmoid(logit)` (as specified) | **0.7170** | 0.80 — unreachable |
| raw logit (unbounded) | 0.9781 | 0.80 — fine |

With `s` confined to [0, 1] and `tau` to (0, 1), a gate soft enough to have
usable gradients cannot attain coverage 0.8 at all. `relu(kappa - mean(g))^2`
is then permanently active, and at `lambda = 10` it dominates the
selective-risk term. The optimiser's cheapest way to raise `mean(g)` is to
collapse the score toward a constant — which destroys the ranking the loss
is supposed to be learning. Measured on a full 10-seed run:

* `A2_mlp_loss1` on Adult: AURC **0.0317 -> 0.1957**, against random's
  0.1284, i.e. *worse than random abstention*
* `A2_mlp_loss2` on Adult: AURC **0.0303 -> 0.2127**
* the same collapse on German Credit and Electricity

The root cause is the very "double-sigmoid saturation" this document
correctly diagnosed under Fix 5 — but the prescription treated the symptom
(widen `T`) instead of removing the second sigmoid. **The gate now operates
on the raw logit**, with `tau` a free threshold in logit units: coverage
becomes reachable, gradients stay healthy, and `lambda = 10` can bind
legitimately rather than buying a degenerate solution.

General lesson worth keeping: *strengthening a penalty on an unsatisfiable
constraint does not enforce it, it just makes the degenerate escape route
cheaper than the real objective.* Two regression tests now pin this — one
asserts the gate can reach a high coverage, one asserts loss1/loss2 still
produce a usable ranking (AUC > 0.8) rather than a collapsed constant.

### Also fixed, though not listed in this document

* **Tier B had never run once.** `use_ensemble=True` crashed on an
  assertion: the cross-fitted meta path excluded Tier B while the
  deployment path included it, so aggregators would train on `(A+C)`
  columns and be scored on `(A+B+C)`. Its ensemble now trains on
  `D_train` only, making one ensemble valid for scoring `D_meta` *and*
  `D_cal`/`D_test`, with columns grafted onto every split in one fixed
  order. (Answers this document's Open Question 2.)
* **The results schema could silently destroy an experiment.** The
  idempotency key was `(dataset, base_model, seed)` with no `tiers`
  column, so an `(A,B,C)` run overwrote the `(A,C)` rows for the same
  dataset and seed — and §7's signal-family-only ablation could not hold
  both side by side. `tiers` is now part of the key, and
  `make_tables.py`/`make_figures.py` refuse to pool across tier configs.

---

Deep codebase research uncovered **7 critical bugs** and **8 design flaws** that directly explain why aggregators failed to beat MSP. Fixing these should yield genuinely better results.

## User Review Required

> [!CAUTION]
> The research found that 2 of the 12 uncertainty signals are **fundamentally broken** on binary classification (Energy produces class identity instead of uncertainty; LogitNorm outputs a constant for all samples). These corrupted signals are fed into every aggregator, actively degrading their performance. Fixing these alone should meaningfully improve aggregator results.

> [!IMPORTANT]
> The A2 neural aggregators suffer from **vanishing gradients** caused by double-sigmoid saturation (temperature T=0.05 in the soft gate), a **missing bias term** in the AdaptiveGating architecture, and **bounded margins** in the pairwise ranking loss. These are not tuning issues — they are structural defects that prevent the networks from learning properly.

> [!WARNING]
> The A1 LogReg stacking aggregator feeds raw signals **without standardization** into L2-regularized logistic regression. Since Tier C distance signals are orders of magnitude larger than Tier A probability signals, the regularizer crushes the probability signal weights while leaving distance signal weights untouched — exactly backwards from what we want.

## Open Questions — both answered, see above

1. ~~**Scope of this pass:**~~ **Answered: both.** Diabetes-130 (OpenML
   did=4541) was added as the plan's second temporal-shift dataset, and
   all four datasets are being run at 10 seeds post-fix. A pre-fix 10-seed
   baseline was captured first so the before/after comparison is measured
   rather than asserted.
2. ~~**Tier B ensemble signals:**~~ **Answered: fixed in this pass.** The
   `AssertionError` was a meta-vs-deployment column mismatch, not a
   cross-fitting gap — see "Also fixed" above. This turned out to matter
   far more than this document assumed: since Tier A is provably a single
   ranking on binary tasks, Tier B is one of only two real sources of an
   independent ordering.

---

## Proposed Changes

### Signal Fixes (Tier A)

#### [MODIFY] [tier_a.py](src/signals/tier_a.py)

**Fix 1: EnergySignal — broken on binary classification.**
- Current: Binary logits are `[0, s]`, so `logsumexp([0, s]) = ln(1 + e^s)` — this is monotonic with class-1 probability, not uncertainty. Confident class-0 predictions get the *highest* uncertainty score.
- Fix: Center binary logits symmetrically as `[-s/2, +s/2]` before computing energy. Now `energy = -ln(2·cosh(s/2))`, which is symmetric around `s=0` and maximal at the decision boundary.

**Fix 2: LogitNormMSPSignal — degenerate on binary classification.**
- Current: Binary logits `[0, s]` normalized by L2 norm give `[0, sign(s)]`, so softmax always outputs `{0.269, 0.731}` regardless of confidence magnitude. Every sample gets the same score.
- Fix: Center binary logits as `[-s/2, +s/2]` before L2 normalization. Now the unit vector is `[-1/√2, +1/√2]` for all magnitudes, but the softmax of centered logits preserves the relative scale information after normalization. Actually, a better fix: use the **ratio** `max_logit / ||z||` as the confidence score (following Cattelan & Silva 2023 more faithfully), which avoids the softmax degeneracy entirely.

---

### Aggregator Architecture Fixes

#### [MODIFY] [a2_coverage_loss.py](src/aggregators/a2_coverage_loss.py)

**Fix 3: AdaptiveGatingAggregator — missing bias/intercept.**
- Current: `logit(x) = w(x)^T z(x)` with no bias. Since `w` sums to 1 and `z` is zero-mean, the logit is trapped near 0 (σ(0) = 0.5). When the true error rate is 10%, the model mathematically cannot shift its baseline prediction to `log(0.1/0.9) ≈ -2.2`.
- Fix: Add a learnable bias parameter: `logit(x) = w(x)^T z(x) + bias`, initialized to `log(error_rate / (1 - error_rate))` from the meta-set class prior.

**Fix 4: Pairwise ranking loss — margin computed on sigmoid outputs.**
- Current: `margin = s[incorrect] - s[correct]` where `s = sigmoid(logit)`, bounding margin to `(-1, 1)`. The minimum achievable loss is ~0.31 even with perfect separation, and gradients vanish as logits grow.
- Fix: Compute margin directly on raw logits: `margin = logit[incorrect] - logit[correct]`. This gives unbounded margins and healthy gradients.

---

### Loss Function Fixes

#### [MODIFY] [losses.py](src/aggregators/losses.py)

**Fix 5: Soft selective risk — vanishing gradients from T=0.05.**
- Current: Gate `g = σ((τ - s) / 0.05)` creates a near-binary step function. Gradient `σ'(20·(τ-s))` drops to <0.018 when `|τ-s| > 0.2`. Combined with the outer sigmoid on `s`, this creates double-sigmoid saturation that kills learning.
- Fix: Increase temperature to `T=0.5` (or even `T=1.0`) so the gate has a smooth gradient over a meaningful range. The coverage penalty can enforce sharpness at convergence.

**Fix 6: Coverage penalty λ=1.0 is too weak.**
- Current: If target coverage is 0.8 and actual is 0.7, the penalty is `(0.1)^2 × 1.0 = 0.01`, negligible compared to selective risk of ~0.1–0.2.
- Fix: Increase `λ` to `10.0` or `20.0` so the coverage constraint is actually enforced.

---

### A1 Stacking Fixes

#### [MODIFY] [a1_stacking.py](src/aggregators/a1_stacking.py)

**Fix 7: LogRegStacking — no feature scaling.**
- Current: Raw `U_meta` (with Tier A signals in [0,1] and Tier C signals in [0, 1000+]) is fed directly into `LogisticRegression(C=1.0)`. L2 regularization penalizes all weights equally, so the optimizer learns tiny weights for informative small-scale signals (MSP, entropy) and large weights for noisy large-scale signals (Mahalanobis, kNN).
- Fix: Add `StandardScaler` pipeline wrapping the logistic regression. Fit the scaler on `U_meta` during `fit()`, apply during `score()`.

---

### A2 Training Improvements

#### [MODIFY] [a2_coverage_loss.py](src/aggregators/a2_coverage_loss.py)

**Fix 8: Training regime — full-batch, no schedule, no early stopping, only 300 steps.**
- Add **learning rate cosine annealing** (from `lr=1e-2` to `lr=1e-5` over training).
- Increase epochs to `500`.
- Add **dropout (p=0.1)** after each hidden layer to reduce meta-overfitting.
- Add **gradient clipping** (`max_norm=1.0`) to prevent the exploding gradients from the denominator in Loss 1.

---

### Signal Robustness Improvements

#### [MODIFY] [tier_c.py](src/signals/tier_c.py)

**Fix 9: TrustScore — extreme outliers destabilize training.**
- Current: When `d_pred ≈ 0`, trust score can reach `-1e12`, ruining z-score normalization and gradient stability.
- Fix: Clamp the trust score ratio to a reasonable range (e.g., `[-100, 0]`) before returning. This preserves ordering while preventing numerical explosions.

**Fix 10: LocalLabelAgreement — severe quantization (only 11 values).**
- Increase default `k` from 10 to 50 (yielding 51 possible values) for finer granularity. On larger datasets (Adult, Electricity with tens of thousands of samples), this is computationally feasible and produces a much more informative ranking signal.

---

### New Signal: Confidence Disagreement Score

#### [MODIFY] [tier_a.py](src/signals/tier_a.py)

**Addition: `ConfidenceDisagreementSignal` — a signal that IS NOT monotonically equivalent to MSP.**
- All current Tier A signals (MSP, entropy, margin, temp_msp) are monotonic transforms of each other on binary tasks, providing zero additional ranking information.
- New signal: `|p_predicted_class - 0.5|` — measures distance from the decision boundary. Combined with Tier C signals, this gives the aggregator a confidence magnitude that is NOT redundant. Actually, this is also monotonic with MSP for binary. 
- Better approach: Add a **calibration residual signal**: `|predicted_prob - empirical_accuracy_in_bin|` computed via isotonic regression on the meta set. This measures *how miscalibrated* the model is at each confidence level, which is orthogonal to raw confidence and directly relevant to selective prediction.

---

## Verification Plan

### Automated Tests
```powershell
# 1. Run existing unit tests (should still pass after fixes)
python -m pytest tests/ -v

# 2. Run experiments on all 3 datasets with 1 seed to verify fixes work
python scripts/run_all.py dataset=german_credit experiment.n_seeds=1
python scripts/run_all.py dataset=adult experiment.n_seeds=1
python scripts/run_all.py dataset=electricity experiment.n_seeds=1

# 3. Generate updated tables and compare AURC values
python scripts/make_tables.py
```

### Manual Verification
- Compare AURC values before and after fixes for each method across all 3 datasets.
- Specifically verify:
  - `signal_energy` AURC improves dramatically (was worse than random on some datasets).
  - `signal_logitnorm_msp` AURC improves (was near-random due to constant output).
  - A2 aggregators show meaningful improvement over MSP (currently worse).
  - A1_logreg shows improvement from feature scaling.
- Check that the Friedman mean ranks shift: aggregators should rank higher than single signals.
