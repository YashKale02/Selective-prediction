# Project status

Honest accounting against `selective_prediction_project_plan.md`'s
14-week timeline (§9) and engineering checklist (§8), as of this build.
Update this file as the project progresses — it is the map of what's
proven to work vs. what's still a placeholder.

## Environment note (read first)

The plan specifies Python 3.11; **this environment only had Python 3.14
available**. Everything below was built, run, and verified end-to-end on
3.14 (`requirements.txt`/`requirements-freeze.txt` are pinned to versions
that also ship 3.11 wheels, so recreating the venv under 3.11 should work
unchanged). One real compatibility issue surfaced and was patched:
hydra-core 1.3.6's `--shell-completion` argument registration crashes
under Python 3.14's stricter `argparse` validation. `scripts/run_all.py`
carries a small, documented monkeypatch for this — see the comment there.
If you move to Python 3.11, you can leave the shim in (harmless no-op) or
remove it once hydra ships a fix.

## Done and verified end-to-end (real data, not synthetic)

- [x] Repo skeleton matching plan §8 exactly.
- [x] OpenML loaders for Adult and German Credit, with caching.
- [x] Disjoint 4-way split (train/meta/cal/test) + K-fold cross-fitting
      over train∪meta, unit-tested for the no-leakage property.
- [x] LightGBM + logistic-regression base model wrappers.
- [x] Tier A signals (MSP, entropy, margin×2, energy, logit-norm MSP,
      temperature-scaled MSP) + Tier C signals (kNN distance, local label
      agreement, trust score, Mahalanobis).
- [x] Aggregators A0 (rank average), A1 (logreg/LightGBM stacking), A2
      (MLP + adaptive-gating, each trainable with BCE / soft-selective-risk
      / AURC-surrogate / pairwise-ranking loss).
- [x] Conformal risk-control wrapper (Hoeffding + Bonferroni).
- [x] AURC/E-AURC/risk@coverage/coverage@risk/ECE/Brier/NLL metrics, unit
      tested against analytically-known cases (perfect ordering, an evenly-
      interleaved worst-case ordering with an exact closed form at every
      5th point, perfect-calibration ECE).
- [x] Subgroup worst-group-risk / max-min-gap metric.
- [x] Full experiment runner: one call produces the entire §7 baseline
      ladder (random, oracle, every signal, every aggregator) plus the
      naive-vs-cross-fit ablation, for one (dataset, seed).
- [x] Hydra config system + CLI (`scripts/run_all.py`), idempotent writes
      into one long-format `results/results.parquet`.
- [x] `scripts/make_tables.py` (AURC summary, Wilcoxon signed-rank vs. MSP
      with Holm correction, Friedman mean ranks) and
      `scripts/make_figures.py` (risk-coverage curves, the week-2-gate
      MSP-vs-random figure) — both query the parquet, nothing hand-typed.
- [x] **Week-2 gate passed on real data**: `paper/figures/
      week2_gate_msp_vs_random.png` shows MSP thresholding clearly beating
      random abstention on both Adult and German Credit, seed 0, numbers
      backed by `results/results.parquet`.
- [x] IEEEtran paper skeleton (`paper/main.tex`, generic conference class,
      fetched from CTAN) with the structure from §10, `\todo{}` markers
      showing exactly what's unwritten.
- [x] Related-work gap table template (`paper/related_work_gap_table.md`)
      and `paper/references.bib` — **both explicitly marked UNVERIFIED**,
      see below.

## Actual results so far (Adult + German Credit, 5 seeds; Electricity, 10 seeds; Tier A+C, LightGBM)

AURC (mean over 5 seeds; lower is better): on **both** Adult and German
Credit, MSP / temperature-scaled-MSP / entropy / margin are statistically
indistinguishable from each other (they're monotonic transforms of the
same binary-classifier probability, so this is expected, not a bug), and
no aggregator (A0/A1/A2, any loss) beats them by a Holm-corrected-
significant margin yet. Oracle AURC (0.0088 Adult, 0.026 German Credit)
shows real headroom exists, so the ceiling isn't "there's nothing to
gain" — it's that none of the current signals/aggregators reach it
in-distribution.

**This matches the plan's own pre-registered falsification criteria
(§1, §11)**: a well-tuned MSP baseline being very hard to beat
in-distribution is the single most likely outcome the plan tells you to
expect, and it is why RQ2 (shift-regime dependence) is the load-bearing
research question, not RQ1 alone.

**`electricity` (temporal shift, §4) has now been run at 10 seeds**, not
1. Before re-running, `Dataset.is_temporal`'s split was sanity-checked
directly: train/meta/cal/test are contiguous, non-overlapping row ranges
covering all 45,312 rows exactly once (no shuffling, no boundary leakage),
and class balance stays in a reasonable 36–64% range in every partition
(test is 52/48, closest to even). The split itself is sound.

**A real bug was caught and fixed in this dataset's first 10-seed run**:
MSP's AURC came back bit-for-bit identical across all 10 seeds. Cause:
`LightGBMWrapper` inherited LightGBM's own defaults
(`bagging_fraction=feature_fraction=1.0`, i.e. no row/column subsampling),
so with the temporal split fixing train/test row order, there was no
source of randomness left for `random_state` to act on — "10 seeds" was
silently retraining the identical model 10 times. Fixed in
`src/models/lightgbm_model.py` by defaulting to `bagging_fraction=0.8`,
`bagging_freq=1`, `feature_fraction=0.8` (still overridable via
`lgb_kwargs`); verified directly that seed 0 vs. seed 1 now produce
predictions differing by up to 0.50 in probability, and the full test
suite (15 tests) still passes. **Electricity was then re-run from
scratch** with the fix in place — the numbers below are the corrected
ones; discard any earlier Electricity numbers you may have seen quoted
elsewhere in chat history.

With genuine per-seed base-model variance, MSP's AURC is 0.1946 (std
0.0056 — no longer 0.0), still clearly beating random (0.370). Against
aggregators, the picture is now more nuanced than the pre-fix run
suggested: the closest aggregator, `A1_logreg_naive_meta` (AURC 0.1968),
is **no longer significantly different from MSP** (Holm-corrected Wilcoxon
p=0.105, vs. p=0.033 "significantly worse" before the fix) — i.e. once
seed variance is real rather than an artifact, that one aggregator ties
MSP rather than losing to it. Every *other* aggregator (A0_rank, A1_logreg,
A1_lightgbm, all A2 variants) remains significantly worse than MSP
(corrected p=0.033). So the corrected reading is: **one particular naive
aggregator ties MSP under this shift; nothing beats it.** RQ2's hoped-for
effect (aggregation clearly *pulling ahead* under shift) still has **not**
appeared on this dataset. Do not write "our method wins under shift" into
the paper from Electricity; the honest reading is "MSP wins or ties
in-distribution *and* under this one temporal shift, depending on which
aggregator," which is still the plan's second-worst falsification-table
row (§1) rather than the best one, and needs the rest of the shift battery
(Diabetes-130 temporal split, CIFAR-10/100-C) before generalizing beyond
Electricity specifically.

## Post-fix results (10 seeds, Tier A+C, LightGBM) -- CURRENT

These are the numbers in `results/results.parquet` and
`paper/tables/*.csv` as of this commit. Both the parquet and the figures
are now **tracked in git** (see `.gitignore`) so collaborators can see what
was actually run instead of having to reproduce it first.

**The headline changed.** For the first time in this project an aggregator
**beats MSP with statistical significance**:

  * **German Credit: `A0_rank` beats `signal_msp`**, mean AURC delta
    **-0.0082**, Holm-corrected Wilcoxon **p = 0.031** over 10 paired
    seeds. Note this is the *unsupervised* aggregator (rank/z-score
    average, no learning at all) -- not the paper's headline A2.
  * **Electricity (temporal shift): `A2_adaptive_loss3` ties MSP**, delta
    +0.0011, p = 0.055 -- just short of significance, i.e. statistically
    indistinguishable. This is the closest any learned aggregator has come
    to MSP under shift, and it is a direct payoff from two of the fixes
    (the missing gating intercept, and the ranking loss on raw logits).
  * **Adult: nothing beats MSP.** Closest is `A1_logreg` at delta +0.0003
    (p = 0.16), i.e. a tie.

The Friedman ranks make the Tier-A degeneracy visible in the results
themselves: `signal_msp`, `signal_entropy`, `signal_margin_prob`,
`signal_margin_logit`, `signal_temp_msp` and the now-fixed
`signal_energy` all share **exactly** mean rank 4.833, because they are
rank-equivalent by construction on binary tasks (Friedman
chi-sq = 58.11, p < 1e-4). Best aggregator rank is `A0_rank` at 8.0.

Still-honest caveats on these numbers:

  1. **Three datasets, not four.** The `diabetes130` 10-seed run did *not*
     complete -- it was killed when the session ended, roughly 95 minutes
     in (4h42m CPU). The dataset is registered, verified and runs
     end-to-end, but it contributes no rows yet. Everything above is
     Adult + German Credit + Electricity.
  2. **`loss1` and `loss2` remain unreliable** even after the gate fix.
     `A2_mlp_loss1` on Electricity is 0.3455 with std 0.0951 and a worst
     seed of 0.4606 -- above random's 0.3705. Do not put these two in a
     headline table without either fixing them (a quantile-based tau is
     the cleanest option) or reporting the variance honestly.
  3. **A0_rank winning is a mixed message for the paper.** The method that
     beats MSP is the one with no learning in it, which argues for the
     "simple and auditable wins" framing rather than for A2 as the
     contribution.
  4. Significance is still per-seed Wilcoxon (n=10 paired observations),
     not the plan's per-instance paired bootstrap -- see simplification 4.

## Diabetes-130 added (second temporal-shift dataset, plan §4)

`diabetes130` (OpenML did=4541, Strack et al. 2014) is now in the registry
and runs end-to-end. It was verified before use the same way `electricity`
was, and the checks mattered:

  * **Row order is time.** `encounter_id` ascends with row order with only
    4 inversions in 101,765 adjacent pairs, and all four sit inside the
    first 8 rows. `is_temporal=True` is justified. (Both `monotonic=False`
    and `Spearman=1.0` are true at once here, which looks contradictory
    until you locate the inversions -- worth noting because the first
    measurement of this looked like a bug and was not.)
  * **`encounter_id` must be dropped from X, not merely ignored.** It is
    monotone in time, so under a temporal split every test value lies
    outside the training range: a tree model would read it as an explicit
    time index and then degenerate into a single branch at deployment.
    `patient_nbr` is dropped too (a bare identity to memorise, and itself
    ~0.54 Spearman-correlated with row order), as is `weight` (96.9% `?`).
    `loaders.py` gained a `drop_cols` argument that *errors* on a name not
    present, so a typo cannot silently leave a leaking column in place.
  * **Known caveat, deliberately not "fixed": patient recurrence.** 16,773
    patients have more than one encounter (up to 40), so **19.3% of test
    rows belong to a patient also present in D_train∪D_meta**. This is
    patient-level leakage in the strict sense. It is kept because it is
    also the real deployment situation for a readmission model -- you do
    see returning patients -- and removing it would destroy the temporal
    semantics that make this a shift dataset at all. A
    first-encounter-only variant (n = 71,518) is the obvious robustness
    check if a reviewer presses, and should be run before submission.
  * **Target binarised** to the standard "readmitted within 30 days"
    (`<30`) task via a new `positive_class` loader argument, keeping it
    comparable with the other three binary datasets. The native target is
    3-class (`NO` / `>30` / `<30`); the multiclass version is a
    *high-value* future experiment precisely because it would break the
    Tier-A rank degeneracy documented below -- with K >= 3 the Tier-A
    signals stop being monotone transforms of each other, which is the
    single most likely condition under which aggregation could beat MSP.
  * **The shift is weak in label terms.** The `<30` rate drifts only from
    0.109 to 0.106 across row-order deciles, and split positive rates run
    0.1143 (train) to 0.1056 (test). Any covariate shift is not
    accompanied by much label shift, so a null RQ2 result here is weaker
    evidence than it would be on a dataset with a sharper drift. Say so
    rather than counting it as a second independent confirmation.

Split sanity check passed: train/meta/cal/test are contiguous,
non-overlapping row ranges covering all 101,766 rows exactly once, with
adjacent boundaries and no gaps.

## Bugs found and fixed (the "betterment" pass)

A dedicated review pass over the signal and aggregator code found seven
real defects, four of them silently degrading every aggregator result
reported above -- plus one trap discovered only by re-measuring after the
first fix attempt (items 5-6 below), which is why the numbers here come
from a *second* full battery rather than the first. All are fixed, each with a regression test in
`tests/test_fixes.py` written against the *symptom* rather than the
implementation. The pre-fix results are archived at
`results/results_prefix_archive.parquet` so the before/after comparison
can be re-derived rather than taken on trust.

**1. `EnergySignal` measured class identity, not uncertainty.** The base
wrappers emit binary logits in the `[0, s]` gauge, so
`-logsumexp([0, s]) = -ln(1 + e^s)` is *monotonically decreasing in the
margin* -- a rank-equivalent copy of `P(class 0)`. A confident class-0
prediction (`s = -8`) scored `-0.0003` while the maximally uncertain
boundary (`s = 0`) scored `-0.6931`, i.e. the confident prediction was
ranked as *more* uncertain. This is why `signal_energy` came out **worse
than random abstention** on Adult (AURC 0.1824 vs. random's 0.1284). Fixed
by mean-centering the logits before the logsumexp, making energy symmetric
in the margin and maximal at the boundary. Note the gauge, not the
formula, was at fault: energy is not shift-invariant, so `[0, s]` was an
arbitrary and wrong choice of coordinates for it.

**2. `LogitNormMSPSignal` is mathematically degenerate for K = 2.** A
softmax reads only logit *differences*, so at K = 2 the whole signal is
the single scalar gap `s`; dividing by `||z||_p` removes exactly the
overall scale, which at K = 2 *is* `|s|`, leaving only `sign(s)`. The
pre-fix implementation returned the constant `0.2689` for every `s != 0`,
which induces an arbitrary ordering -- hence its Friedman mean rank of
20.0, *below random abstention's* 19.7. Every candidate repair collapses
identically (measured: centering -> constant 0.1956;
`max_logit/||z||` on the raw gauge -> the binary indicator `1[s > 0]`;
the same ratio on centered logits -> constant 0.7071), so the signal is
now **excluded from the binary bank** and raises on binary input, while
being kept unchanged for K >= 3 where the normalisation is well-defined.
*This is worth a sentence in the paper*: Cattelan & Silva's logit
normalisation is inapplicable to binary classification, which is a
limitation of the published method rather than of this codebase.

**3. `AdaptiveGatingAggregator` had no intercept.** The score was
`logit(x) = w(x)^T z(x)` where `w` is a softmax (so it sums to 1) over a
z-scored `z` (so zero-mean per column), which pins the mean logit near 0
-- a predicted error probability of 0.5. With a base error rate of 10% the
correct mean logit is `log(0.1/0.9) ~= -2.2`, and no parameter in the
architecture could express it; the convex-combination constraint also caps
the logit's range at `max_j z_j(x)`, so scaling could not compensate.
Fixed with a learnable bias initialised at the meta-set log-odds.

**4. The pairwise ranking loss (Loss 3) had bounded margins.** The caller
passed `s = sigmoid(logit)`, confining the margin to (-1, 1), so
`softplus(-margin)` could not fall below `softplus(-1) = 0.313` even for a
*perfectly* separated ranking, and the gradient vanished exactly where the
ranking was becoming confident. Now computed on the raw logit: unbounded
margins, loss -> 0 for a correct ranking, healthy gradients. The optimised
ranking is unchanged (sigmoid is monotone, so every pairwise sign is the
same); only the loss surface differs.

**5 and 6. The coverage-targeted losses (Loss 1 / Loss 2) were untrainable
-- and the obvious fix made it far worse.** Two defects and a trap:

*The original defects.* The gate temperature was `T = 0.05`, making
`sigmoid((tau - s)/T)` a near-step function whose derivative falls below
0.018 once `|tau - s| > 0.2`; composed with the already-saturating sigmoid
producing `s`, almost no training row retained a usable gradient. And the
coverage penalty weight was `lambda = 1.0`, so missing a target coverage of
0.8 by a full 10 points cost `(0.1)^2 * 1.0 = 0.01` against a
selective-risk term of order 0.1-0.2 -- a *coverage-targeted* objective
could ignore its own coverage target essentially for free.

*The trap.* Applying the two obvious fixes (`T -> 0.5`, `lambda -> 10.0`)
while still gating on the **squashed** score is catastrophic. With `s` in
[0, 1] and `tau` in (0, 1), the maximum attainable `mean(g)` at `T = 0.5`
is only **0.7170**, below the 0.80 target -- measured directly. The
coverage penalty is therefore permanently active, and at `lambda = 10` it
dominates; the optimiser's cheapest way to raise `mean(g)` is to collapse
the score toward a constant, destroying the ranking. This was caught only
because the first post-fix 10-seed battery showed `A2_mlp_loss1` on Adult
going from AURC 0.0317 to **0.1957** -- *worse than random abstention*
(0.1284) -- and `A2_mlp_loss2` from 0.0303 to 0.2127, with the same
collapse on German Credit and Electricity.

*The actual fix.* Gate on the **raw logit**, with `tau` a free threshold in
logit units. The logit is unbounded, so any coverage is reachable
(max `mean(g)` = 0.978 at `T = 0.5`) while gradients stay healthy, and this
removes the double-saturation at its source instead of widening `T` to
compensate for it. `lambda = 10` then binds legitimately. Two regression
tests pin the property: one asserts the gate can reach a high coverage,
one asserts loss1/loss2 still yield a usable ranking (AUC > 0.8) rather
than a collapsed constant.

*Lesson worth carrying:* strengthening a penalty on an **unsatisfiable**
constraint does not enforce the constraint -- it makes the degenerate
escape route cheaper than the real objective. Check reachability before
raising a penalty weight.

**7. `LogRegStackingAggregator` fed unstandardised features to an
L2-regularised model.** Tier-A signals are probabilities in [0, 1] while
Tier-C signals are raw distances reaching 1e3+, and L2 penalises
coefficient magnitude -- so a small-scale feature needs a larger
coefficient for the same influence and is penalised far more for it. The
least-cost solution shrinks the *informative* probability signals towards
zero and leans on the large-scale distance signals, exactly backwards
(MSP is the strongest single signal in every dataset here;
`knn_distance`/`mahalanobis` are among the weakest). Fixed with a
`StandardScaler` pipeline. This also made the headline A1-vs-A2 comparison
unfair to A1, since A2 already standardised its inputs.

Additionally fixed, as robustness rather than correctness: `TrustScore`
could return ~1e12 when a test point coincided with a training point of
its predicted class, which alone dominated the z-score standardisation
every aggregator applies (now clamped); `LocalLabelAgreement` could take
only k+1 = 11 distinct values, forcing large blocks of ties that a
risk-coverage curve must then traverse in arbitrary order (k raised to
50); and the A2 training regime (300 full-batch steps, fixed lr, no
schedule, no regularisation, no clipping) gained cosine annealing, 500
epochs, dropout 0.1 and gradient clipping.

### Tier B now runs at all

`use_ensemble=True` previously crashed on an assertion and had therefore
**never been exercised**. The cause was a column-set mismatch: the
cross-fitted meta path deliberately excluded Tier B while the deployment
path included it, so the aggregators would have trained on (A+C) columns
and been scored on (A+B+C). Tier B's ensemble is now trained on
**D_train only**, which makes one ensemble valid for scoring D_meta *and*
D_cal/D_test (both disjoint from D_train), and its columns are grafted
onto every split in one fixed order -- exactly like temperature scaling --
so the meta and deployment feature spaces agree by construction. Verified
end-to-end on German Credit with all three ensemble signals present.
This does not cross-fit the ensemble (an ensemble-of-ensembles over
K folds x M members remains out of scope); the cost is that the ensemble
sees 60% of the data rather than 75%.

### The finding that reframes the negative result

Measured directly (`tests/test_fixes.py` pins it): **on a binary task,
every Tier-A signal is a strictly monotone transform of every other one**
-- MSP, entropy, both margins, temperature-scaled MSP, and even the
*fixed* energy all have pairwise Spearman |rho| = 1.0000. Tier A therefore
supplies exactly **one** distinct ranking, so no aggregator over Tier A
alone can order abstentions any differently from MSP, no matter how it is
trained. That is a structural explanation for the central negative result
above, independent of any bug: the aggregators were never able to beat MSP
in-distribution because five of their features carried no information the
sixth did not already have, and two of the remaining ones were broken.

Two consequences worth carrying into the paper:

  * On binary problems the only genuinely different orderings come from
    Tier B (ensemble disagreement), Tier C (geometry/density) and the new
    `calib_residual` signal. Tier A is a single bit dressed as six
    features. Any future "aggregate many uncertainty signals" claim needs
    to state which of its signals are actually rank-independent.
  * A new signal was added to break the degeneracy deliberately:
    `CalibrationResidualSignal` fits an isotonic map
    `confidence -> empirical accuracy` on a held-out split and scores
    `|conf - iso(conf)|`, i.e. local miscalibration. Because `iso` is
    monotone but crosses the identity line, the residual is V-shaped
    rather than monotone in confidence -- measured Spearman against MSP is
    0.25, the first Tier-A signal that is not rank-equivalent to it. It
    adds no new *raw* information (it is a function of confidence alone),
    but it adds *expressivity*: a monotone aggregator over `{conf}` can
    only reproduce MSP's ordering, whereas given `conf` and the residual it
    can express "in the confidence band where this model is systematically
    overconfident, treat it as riskier" -- and the ideal ordering, by
    `P(error | x)`, is genuinely non-monotone in reported confidence for a
    miscalibrated model.

### Results schema change

`results/results.parquet` rows now carry a `tiers` column, and the
idempotency key in `scripts/run_all.py` is
`(dataset, base_model, tiers, seed)`. Without this an (A,B,C) run silently
overwrote the (A,C) rows for the same dataset and seed -- they are
different experiments -- and §7's signal-family-only ablation could not
hold both side by side. `scripts/make_tables.py` now refuses to pool
across tier configurations and takes a `--tiers` argument to pick one.
The pre-schema parquet is archived at
`results/results_prefix_archive.parquet`; `run_all.py` fails loudly rather
than merging a file that predates the column.

## Deliberate simplifications (be honest about these, per the plan's own ethos)

1. **Temperature scaling is not K-fold cross-fitted.** It's fit once, on
   D_meta, using a model trained only on D_train (`model_train` in
   `runner.py`) — already non-leaky since D_train and D_meta are disjoint,
   but it does *not* go through the same 5-fold cross-fitting loop as the
   other Tier A/C signals, so its distribution on D_meta may differ
   slightly from model_final's (trained on D_train∪D_meta) behavior at
   deployment. Documented in `runner.py`'s module docstring.
2. **Tier B (ensemble) is not cross-fitted.** `use_ensemble=True` now
   trains one ensemble on **D_train only** and uses it to score D_meta and
   D_cal/D_test alike. Both are disjoint from D_train, so nothing is scored
   in-sample and the meta/deployment feature spaces match by construction
   -- but the ensemble sees 60% of the data rather than the 75% the other
   deployment signals get. Cross-fitting an ensemble-of-ensembles (K folds
   × M members) is real additional engineering, still out of scope. (This
   replaced an arrangement that trained on D_train∪D_meta and included Tier
   B only in the deployment path, which crashed -- see "Tier B now runs at
   all" above.)
3. **Conformal wrapper is intentionally simple, not tight.** Hoeffding's
   inequality + Bonferroni over a bounded quantile grid is auditable in
   ~130 lines but is provably loose at small target risk α (its variance-
   obliviousness means very small α can require more calibration data than
   is available — this was measured directly: even 20,000 well-behaved
   synthetic calibration points could not satisfy α=0.01/0.02/0.05 under
   this bound, only α=0.10). This is expected behavior of Hoeffding, not a
   bug — but it means small-α rows in a coverage-at-guaranteed-risk table
   may show 0 coverage even when a tighter bound (Hoeffding-Bentkus, or
   `mapie`/`crepes` per plan §8) would find a usable threshold. Swap in a
   tighter bound before trusting small-α numbers for the paper.
4. **No raw per-instance caching.** `results/results.parquet` stores
   aggregated summary rows (risk/accuracy at each coverage checkpoint,
   AURC, E-AURC, ...), not the raw per-test-instance (uncertainty,
   correct) arrays. This is sufficient for everything currently computed,
   but the plan's exact "paired bootstrap on the test set for AURC
   differences" (§7) needs the raw arrays. `scripts/make_tables.py`
   currently substitutes a per-seed-paired Wilcoxon test instead (valid,
   but coarser — 5 paired observations per comparison instead of a
   bootstrap over ~thousands of test points). Add raw-array caching to
   `runner.py` if you need the finer test before submission.
5. **Tier D (tree-specific) signals are not implemented.** Out-of-bag /
   boosting-round disagreement and leaf co-occurrence distance are listed
   in the plan (§3.1) but not built here. Note this is now a more
   interesting gap than it looked: since Tier A is provably a single
   ranking on binary tasks, Tier D would be one of the few remaining
   sources of a genuinely independent ordering.
6. **No neural base model / MC-dropout / image datasets yet.** Everything
   above runs on tabular data with LightGBM/logreg. CIFAR-10/100 +
   ResNet-18/WideResNet, and the MC-dropout Tier-B source that requires a
   dropout-bearing network, are unstarted (plan explicitly allows dropping
   image experiments first under schedule pressure — §11).
7. **4 of the planned 12 tabular datasets so far** (`adult`,
   `german_credit`, `electricity` and `diabetes130`, all at 10 seeds). Everything is wired to scale to the full
   12-dataset registry in `src/data/loaders.py:REGISTRY` — just add
   entries and run `scripts/run_all.py dataset=<name>
   experiment.n_seeds=10`. Note that for a temporal dataset (`is_temporal=
   True`), the train/meta/cal/test split itself is deterministic (split by
   row order, not by `seed`) — running multiple seeds on `electricity`
   only varies model-training and cross-fitting randomness, not the split
   itself. (This used to also mean *zero* base-model training randomness
   on `electricity` specifically, since LightGBM's un-tuned defaults have
   no row/column subsampling for `random_state` to act on — fixed, see
   above; seeds now genuinely retrain a different base model.)

## Not started (still exactly as scoped in the plan)

- Datasets: Bank Marketing, Give Me Some Credit, Covertype, MiniBooNE,
  Higgs, Telco churn, ann-thyroid, Jannis/Road-Safety.
- Shift experiments: Electricity and Diabetes-130 both have real 10-seed
  temporal-split runs (see above). CIFAR-10-C/100-C and synthetic
  covariate shift are still unrun.
- Published baselines requiring retraining: SelectiveNet, Deep Gamblers,
  ConfidNet.
- Full ablation suite (§7): leave-one-signal-out, signal-family-only,
  meta-set-size sweep, signal correlation/PCA — the building blocks (the
  runner's `tiers` argument, the naive-vs-cross-fit comparison) exist;
  the sweep scripts that iterate over them do not yet.
- Cost-model curves (sweeping `c_error/c_review`).
- CD (critical-difference) diagram rendering (mean ranks are computed in
  `make_tables.py`; the actual diagram plot is not drawn yet).
- Literature verification (see next section) and the resulting related-
  work prose in `main.tex` §2.
- Everything in the paper past the `\todo{}` markers: full Results,
  Ablations, Conclusion sections, and the submission checklist in plan
  §10 (venue-specific IEEEtran template, page limit, anonymization).

## Citations: verified

Literature verification (Week 1's gate) is done. Every entry in
`paper/references.bib` and `paper/related_work_gap_table.md` has been
checked against a live source (arXiv / ACM DL / IEEE Xplore / the venue's
own proceedings page) — no longer "recalled from memory." A handful of
corrections surfaced during verification (an author name, a venue year,
one renamed bib key) — see the "Corrections made during verification"
section of `related_work_gap_table.md` for the full list. The related-
work prose in `main.tex` §2 itself is still unwritten (`\todo{}` marker),
but it can now be built on trustworthy citations.

## Suggested next steps, in priority order

1. **Scale seeds to 10 (done for adult/german_credit/electricity) and add
   a handful more tabular datasets from the registry (§4)** — the runner
   and CLI already support this; it's compute-bound, not code-bound.
   Re-run `adult`/`german_credit` at 10 seeds too now that the LightGBM
   subsampling fix is in, since their base-model seed variance was
   presumably also thinner than intended before the fix (they're not
   temporal, so it wasn't *zero* variance like Electricity, but it was
   likely too small — worth confirming, not assuming).
2. Add raw-array caching to `runner.py` if the paired-bootstrap test
   matters for the final paper (currently substituted with a coarser
   per-seed Wilcoxon test).
3. Only then: SelectiveNet/Deep Gamblers/ConfidNet reproduction, image
   experiments, and the full ablation sweep scripts.
