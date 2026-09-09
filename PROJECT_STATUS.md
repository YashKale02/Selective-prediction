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
(a second temporal-shift dataset -- Diabetes-130 was tried and dropped,
see below -- and CIFAR-10/100-C) before generalizing beyond Electricity
specifically.

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

  1. **Three datasets.** `diabetes130` was tried as a fourth (temporal
     shift) but was dropped entirely -- too many incomplete fields -- before
     ever producing a result row; see "Diabetes-130: added, then dropped"
     below. Everything above is Adult + German Credit + Electricity, and
     the plan's second-shift-dataset slot is open for a different choice.
  2. **`loss1` and `loss2` remained unreliable** even after the gate fix --
     `A2_mlp_loss1` on Electricity was 0.3455 with std 0.0951 and a worst
     seed of 0.4606, above random's 0.3705. **Now fixed** (quantile-based
     tau, see the "Bugs found and fixed" section, item 8) but **not yet
     re-run** on Adult/German Credit/Electricity, so the numbers in this
     section still reflect the old, unreliable version of these two
     methods. Re-run before using them in a headline table.
  3. **A0_rank winning is a mixed message for the paper.** The method that
     beats MSP is the one with no learning in it, which argues for the
     "simple and auditable wins" framing rather than for A2 as the
     contribution.
  4. Significance is still per-seed Wilcoxon (n=10 paired observations),
     not the plan's per-instance paired bootstrap -- see simplification 4.

## THE CENTRAL RESULT (10 seeds, Tier A+C, LightGBM)

**Read the comparison against the best *single signal*, not against MSP.**
Comparing everything to MSP is what the project did for months, and on one
dataset it produces a headline that is largely an artefact. Both columns
are given below; the second is the one that supports a claim about
*aggregation*.

| dataset | K | base acc | MSP AURC | best single signal | best aggregator | agg vs MSP | **agg vs best single** |
|---|---|---|---|---|---|---|---|
| adult | 2 | 0.87 | 0.03016 | energy *(= MSP)* | 0.03043 | +0.9% | **+0.9% (hurts)** |
| electricity | 2 | 0.80 | 0.19459 | energy *(= MSP)* | 0.19566 | +0.5% | **+0.5% (hurts)** |
| german_credit | 2 | 0.79 | 0.13298 | energy *(= MSP)* | 0.12477 | -6.2% | **-6.2%** |
| satimage | 6 | 0.93 | 0.00998 | entropy (0.00996) | 0.00933 | -6.5% | **-6.4%** |
| letter | 26 | 0.97 | 0.00131 | entropy (0.00131) | 0.00123 | -6.3% | **-6.3%** |
| wine_quality_white | 7 | 0.68 | 0.18459 | **trust_score (0.14125)** | 0.13942 | **-24.5%** | **-1.3%** |

Two *separate* findings live in that table, and conflating them would
overclaim:

**Finding 1 -- aggregation genuinely synthesises, by a strikingly stable
margin.** On german_credit, satimage and letter the best aggregator beats
the best single signal by **-6.2%, -6.4%, -6.3%**. Three datasets, three
near-identical numbers, spanning K=2/6/26 and AURCs three orders of
magnitude apart. `A1_logreg` -- plain logistic-regression stacking -- is
the winner on satimage and letter (9/10 seeds each; seed-averaged
bootstrap CIs [-0.001004, -0.000319] and [-0.000127, -0.000040], both
excluding zero).

**Finding 2 -- MSP is not always the best single signal, and when it is
not, that dominates everything else.** On wine_quality_white, plain
`trust_score` (Tier C geometry) beats MSP by **-23.5%** on its own. The
aggregator's headline -24.5% vs MSP is therefore *almost entirely
trust_score*, with aggregation contributing only -1.3% on top. Ten methods
clear Holm correction against MSP on this dataset -- the project's first
unambiguous Holm-corrected wins -- but the honest reading is **"MSP is the
wrong signal here"**, not "aggregation is powerful". Practically this may
matter more than Finding 1, and it is cheaper to act on.

**Where aggregation actively hurts:** adult and electricity (+0.9%,
+0.5%). Both are binary datasets where Tier C is uninformative, so the
bank collapses to a single effective ranking and the aggregator can only
add estimation variance.

**No mechanism currently explains which datasets benefit -- two candidate
explanations were proposed and both were falsified by direct check.** This
is recorded rather than quietly dropped, because the temptation to ship a
tidy mechanism is exactly what a reviewer will probe.

*Candidate 1: "K >= 3 breaks the Tier-A degeneracy, so multiclass wins."*
Falsified two ways. german_credit is **binary** and aggregation still helps
-6.2%. And on satimage, `entropy` is still **1.0000** rank-identical to MSP
(measured), so Tier-A differentiation cannot be what produced satimage's
-6.4%. Measured |Spearman| against MSP, seed 0:

| dataset | K | entropy | margin_prob | margin_logit | energy | temp_msp |
|---|---|---|---|---|---|---|
| adult | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| german_credit | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| electricity | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| satimage | 6 | 1.0000 | 0.9997 | 0.9985 | 0.9608 | 0.9943 |
| wine_quality_white | 7 | 0.9935 | 0.9983 | 0.9947 | **0.7089** | 0.9739 |

The binary row is exact and analytically proven. Multiclass breaks it only
*partially and unevenly* -- `energy` decorrelates most, `entropy` barely
at all.

*Candidate 2: "aggregation helps when Tier-C geometry is informative."*
Also falsified. Ratio of the best Tier-C signal's AURC to MSP's, against
whether aggregation beat the best single signal:

| dataset | bestTierC / MSP | aggregation | consistent? |
|---|---|---|---|
| adult | 1.64 | hurts | yes |
| **electricity** | **1.16** | **hurts** | -- |
| **german_credit** | **1.16** | **helps -6.2%** | **no: same ratio, opposite outcome** |
| **letter** | **1.60** | **helps -6.3%** | **no: weak Tier C, still helps** |
| satimage | 1.14 | helps -6.4% | yes |
| wine_quality_white | 0.77 | helps | yes |

**What is therefore established vs. speculative.** Established: the binary
Tier-A degeneracy (exact, proven, tested); that aggregation beats the best
single signal by a strikingly stable 6.2-6.4% on three datasets; that it
*hurts* on two; that Tier B never helps; and that the simplest aggregator
wins. Speculative and currently **unexplained**: which datasets fall in
which group. Neither class count nor Tier-C quality predicts it. Write the
paper around the proven negative result plus the observation, and label
the selection question as open -- do not ship a mechanism that the
project's own data contradicts.

**Caveat on german_credit specifically:** its test split is ~150 rows with
~31 errors, so its -6.2% may be noise. Check whether its bootstrap CI
excludes zero before counting it as one of the three.

**Tier B (ensemble disagreement) does not help, on any dataset tested.**
On wine it *dilutes* the aggregator's edge over the best single signal
(-1.3% -> -0.2%); on satimage it shrank the win slightly (-0.000648 ->
-0.000595). Its own signals are weak: on wine, `ensemble_vote_entropy`
scores 0.292 against random's 0.320. Genuinely independent signals are not
automatically *useful* signals, and each one costs the aggregator another
dimension to fit from a small meta-set.

**The elaborate aggregators lose to the simple one.** `A1_logreg` (plain
logistic regression) is the best aggregator on satimage, letter and wine.
The A2 neural family with the custom coverage-targeted losses -- the
project's original headline contribution -- never wins. This should be
stated plainly in the paper rather than buried.

## Multiclass datasets added (the mechanism experiment)

Three K >= 3 datasets are now in the registry: **wine_quality_white**
(id 40498, K=7), **letter** (id 6, K=26) and **satimage** (id 182, K=6).
Ids and shapes were verified directly against OpenML, not recalled.

**Why these exist.** The project's central negative result has a single
proven cause: on a *binary* task every Tier-A signal is a strictly monotone
transform of every other one, so Tier A supplies exactly one ranking and no
aggregator over it can order abstentions differently from MSP. At K >= 3
that degeneracy provably breaks -- top1-minus-top2 in probability space and
in logit space stop being monotone in each other, entropy stops being a
function of the top probability alone, and `logitnorm_msp` becomes
well-defined (it is excluded on binary input as provably constant). These
datasets are therefore the direct test of *"aggregation helps exactly when
the signals are not redundant"*, which is what would convert a null result
into a demonstrated mechanism. A first single-seed run on a multiclass
dataset already shows the predicted split: `signal_entropy` (0.000430),
`signal_msp` (0.000455), `signal_margin_prob` (0.000463),
`signal_margin_logit` (0.000472) and `signal_temp_msp` (0.000484) land on
**different** AURCs where on binary they were bit-identical.

**They were screened on test-set error count, which is the binding
constraint and is easy to miss.** A risk-coverage curve is built entirely
out of the *errors* in D_test, so an "easy" dataset produces a curve made
of noise regardless of how many rows it has. Measured with the standard
split and base model before admitting anything:

| candidate | K | rows | test errors | verdict |
|---|---|---|---|---|
| wine_quality_white | 7 | 4,898 | **232** | kept (base acc 0.684) |
| letter | 26 | 20,000 | **102** | kept (base acc 0.966) |
| satimage | 6 | 6,430 | **70** | kept (base acc 0.928) |
| yeast | 10 | 1,484 | 89 | rejected -- only 223 test rows |
| pendigits | 10 | 10,992 | **11** | rejected -- unusable |
| texture | 11 | 5,500 | **9** | rejected -- unusable |
| optdigits | 10 | 5,620 | **8** | rejected -- unusable |
| segment | 7 | 2,310 | **5** | rejected -- unusable |

`segment` and `pendigits` were briefly registered before this screen was
run, and `segment`'s smoke-test rows were deleted from the parquet when it
was rejected. **Carry the lesson: screen a selective-prediction dataset on
its test-error count, not its row count.** A 2,310-row dataset at 98.6%
accuracy yields five errors, and nothing statistically meaningful can be
built on five errors.

Scope limits, stated so they are not overclaimed: none of the three has a
natural demographic subgroup column, so they contribute to RQ1 and the
mechanism question but produce **no RQ4 rows**; and none is temporal, so
they say **nothing about RQ2's shift question** -- that slot is still open
(see below).

## Diabetes-130: added, then dropped (plan §4's second shift dataset)

`diabetes130` (OpenML did=4541, Strack et al. 2014) was registered as the
plan's second temporal-shift dataset and verified the same way
`electricity` was (temporal row order confirmed via `encounter_id`
ordering; `encounter_id`/`patient_nbr`/`weight` dropped as leaking or
near-useless columns; a known, deliberately-not-fixed patient-recurrence
caveat: 19.3% of test rows belong to a patient also seen in
D_train∪D_meta). Two 10-seed run attempts followed: the first died
silently around 95 minutes in with no visibility into how far it got (its
stdout was fully buffered); the second, run with unbuffered output to fix
that blind spot, showed a single seed taking ~27 minutes on ~100k rows
with 5-fold cross-fitting and Tier-C's nearest-neighbor signals — on pace
for ~4.5 hours for all 10.

**Explicit decision: dropped from the project**, before either attempt
produced a single result row, on data-quality grounds — `weight` alone was
96.9% missing (a `?` sentinel) on top of the patient-recurrence caveat
above. Removed from `src/data/loaders.py`'s `REGISTRY`,
`configs/dataset/diabetes130.yaml` deleted, and its cached raw OpenML pull
(`data_cache/openml_4541.parquet`) deleted. No result of any kind exists
for this dataset anywhere in `results/results.parquet` or the paper
tables, so nothing needed to be purged from either. If reviving it later,
the verification notes above (and the full loader arguments) are
preserved in git history rather than repeated here.

The plan's second-shift-dataset slot (§4) is open again as a result — see
"Suggested next steps" below.

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

**8. `loss1`/`loss2`'s freely learned `tau` could still miss the target
coverage, even after fixing the gate (items 5-6 above).** The earlier fix
made the target coverage *reachable* by gating on the raw logit, but `tau`
remained a free `nn.Parameter` trained jointly with the network against a
quadratic coverage *penalty* -- an indirect way to hit a target that has a
direct answer. Measured symptom: `A2_mlp_loss1` on Electricity had AURC
0.3455 (std 0.0951, worst seed 0.4606) -- worse than random abstention
(0.3705) -- because a learned threshold can wander into a bad
risk/coverage tradeoff, especially on small meta-sets (German Credit's
D_meta is ~150 rows). Fixed by replacing the learned `tau` with
`losses.quantile_tau`: the `kappa`-quantile of the current batch's logits,
recomputed fresh every training step and detached from the gradient. This
makes `kappa`-coverage hold **by construction** for any network weights,
so the coverage penalty term becomes a redundant backstop rather than a
live constraint competing with the risk term for the gradient. Two
regression tests pin this (`tests/test_fixes.py`): one confirms the
quantile hits the target coverage exactly, one confirms `MLPAggregator` no
longer carries a `tau`/`taus` parameter at all. Caught during review: the
first implementation had the quantile direction backwards
(`torch.quantile(s_logit, 1 - kappa)` instead of `kappa`), which the
coverage-exactness test failed on immediately -- worth noting as a
reminder that a test written *for* a fix can still catch a bug *in* the
fix, if it checks the actual property rather than just "does it run."
**Not yet re-run**: this fix landed after Adult/German Credit/Electricity's
current 10-seed numbers were already in `results/results.parquet`, so
those rows still reflect the old, unreliable `loss1`/`loss2`. Re-run all
datasets to get corrected `A2_mlp_loss1`/`A2_mlp_loss2` numbers before
trusting them in the paper.

**9. `loss1`/`loss2`'s threshold is now a quantile, not a learned
parameter** -- see item 8 above (kept as a separate numbered entry because
it is a distinct defect from 5-6, found only after those were fixed).

**10. Model wrappers emitted a class-space that shifted when a training
split missed a rare class -- a silent-correctness bug, not just a crash.**
sklearn and LightGBM return one probability/logit column per class *present
in their training data*, in sorted order. `wine_quality_white` has 7 quality
levels but only ~5 rows of the rarest in 4,898, so a 60% train split can
easily contain 6 of the 7. Everything downstream indexes these matrices by
label -- temperature scaling does `logp[np.arange(n), y]` -- which then goes
wrong in one of two ways:

  * the missing class is the **last** one -> `IndexError`. This is how the
    bug was caught (the `wine_quality_white` sweep config died with
    "index 6 is out of bounds for axis 1 with size 6").
  * the missing class is in the **middle** -> every column above it shifts
    left, so `logp[i, y_i]` silently reads a *different class's*
    probability. **No error, just wrong numbers**, propagating into
    temperature scaling, the calibration residual, and every signal that
    reads a per-label probability.

The second case is far more dangerous and is why the fix belongs in the
model wrappers rather than in each consumer. `BaseModelWrapper` now takes
`n_classes` (the *whole dataset's* class count) and scatters the model's
local columns into global label order, padding never-seen classes with
probability 0 and logits at `row_max - 32` (finite, softmax weight ~1e-14,
so it survives the energy signal's `logsumexp` and can never enter a top-2
margin). `run_experiment` passes `n_classes` to every fit -- `model_train`,
each cross-fit fold model, the Tier B ensemble members and `model_final`.
`predict` was corrected in the same pass to return global labels in both
the aligned and unaligned cases.

`tests/test_class_alignment.py` (13 tests, both wrappers) pins this,
including the quiet case explicitly: it asserts that class *c*'s
probability sits at column *c* after alignment, by comparing against an
unaligned model and mapping through its `classes_`. Measured
demonstration of what was at stake: with class 2 held out of training, the
unaligned model's **column 3 contains class 4's probability**.

*Note this only ever affected multiclass datasets, so it was latent for as
long as the project was binary-only -- and it surfaced within minutes of
adding the first multiclass dataset with a rare class. Worth remembering
before trusting any "it has always worked" argument about a code path that
has never actually been exercised.*

### The R3/R5/R6/R7 pass (issues raised in `full_research_analysis.md`)

Four issues that document listed as open were worked through; two were real
and fixed, one was tightened substantially, and **one turned out not to be
a defect at all** -- recorded here because "we looked and it was fine" is
worth as much as a fix.

**R3 -- conformal bound was provably too loose (FIXED).** The simple
Hoeffding UCB is variance-oblivious, which bites hardest exactly where a
selective classifier lives: at small risk. Measured on 20,000 well-behaved
synthetic calibration points it admitted **zero coverage** at
alpha in {0.01, 0.02, 0.05} -- it abstained on everything, which makes RQ3
unanswerable in practice rather than merely conservative. Replaced with the
**Hoeffding-Bentkus** bound of Bates et al. (RCPS) / Angelopoulos et al.
("Learn Then Test") -- the minimum of a Chernoff-Hoeffding (KL-form) tail
bound and Bentkus's binomial tail bound, where the `e` factor on the
Bentkus term is what makes that minimum valid. On the same synthetic case
HB recovers **0.117 coverage at alpha=0.02 and 0.417 at alpha=0.05** (from
0.0 and 0.0), and improves alpha=0.10 from 0.864 to 0.930. The old bound
is kept selectable as `bound="hoeffding"` so the before/after stays
*measurable* rather than asserted. `tests/test_conformal.py` (12 tests)
pins both halves of the claim: that HB never exceeds the old bound, and --
critically -- that the guarantee still **holds empirically**, via a
calibrate-then-evaluate-on-fresh-data loop counting violations against
delta. That validity test doubles as the "conformal coverage empirical
verification" experiment `full_research_analysis.md` lists as a P1 gap
(its claim C6).

**R7 -- no per-instance caching, so §7's paired bootstrap was impossible
(FIXED).** `run_experiment` now takes `raw_out_dir` and writes one
compressed `.npz` per `(dataset, model, tiers, seed)` holding each method's
test scores plus the *shared* `incorrect` vector -- shared because it
depends only on the frozen base model, so storing it per method would
inflate the files ~20x for no information. With float32 scores and
compression a full sweep costs a few MB rather than the ~200MB a
per-method long-format parquet would, and `results/raw/` is gitignored
(regenerable intermediate; the summary parquet and the bootstrap table
remain the tracked outputs). `scripts/make_bootstrap_table.py` then runs
the real test: resample test *instances*, apply the same resampled indices
to both methods, recompute AURC, and report a percentile CI on the
difference. First run already shows the mechanism finding landing in the
prescribed test -- every binary Tier-A signal comes out at **exactly
0.000000 delta with a zero-width CI** (rank-identical, so every resample
agrees), while oracle and random fall correctly on either side.
`tests/test_raw_cache_and_bootstrap.py` pins the round-trip, the
zero-delta-under-monotone-transform property, that the test has power
(oracle beats random with a CI clear of 0), and that the resampling is
genuinely *paired* (paired spread must be tighter than unpaired).

**R6 -- cross-fit folds all shared one seed (FIXED).** Every fold model was
fit with the outer `seed`, so the K models differed only by which rows they
saw; their subsampling draws followed an identical pattern, leaving them
less independent than the cross-fitting design intends. `fit_model_fn` may
now take the fold index and derives `seed * 1000 + fold` (the same pattern
`_train_ensemble` already used). `cross_fitted_signals` detects the
two-argument form by signature inspection, so single-argument callers keep
working -- inspected once up front rather than via try/except per fold, so
a genuine `TypeError` from inside a fit function can't be silently
swallowed and retried.

**R5 -- `np.random.seed` "should be `default_rng`" (NOT A DEFECT).**
Audited rather than assumed: **nothing in `src/` or `scripts/` makes a bare
`np.random.<dist>()` call**, and the only two random consumers construct
their own explicitly-seeded `RandomState`; every sklearn/LightGBM/torch
object is passed an explicit `random_state`/`seed`. Swapping the API would
therefore change no behaviour anywhere. The global call is kept as a
deliberate net for third-party paths that *do* read global NumPy state when
`random_state=None`, a `new_generator(seed)` helper is provided for new
code, and the reasoning is recorded in `seeding.py` so this is not
re-litigated. Not every flagged item is a bug.

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
7. **3 of the planned 12 tabular datasets so far** (`adult`,
   `german_credit`, `electricity`, all at 10 seeds; `diabetes130` was
   tried as a 4th and dropped, see above). Everything is wired to scale
   to the full 12-dataset registry in `src/data/loaders.py:REGISTRY` —
   just add entries and run `scripts/run_all.py dataset=<name>
   experiment.n_seeds=10`. Note that for a temporal dataset (`is_temporal=
   True`), the train/meta/cal/test split itself is deterministic (split by
   row order, not by `seed`) — running multiple seeds on `electricity`
   only varies model-training and cross-fitting randomness, not the split
   itself. (This used to also mean *zero* base-model training randomness
   on `electricity` specifically, since LightGBM's un-tuned defaults have
   no row/column subsampling for `random_state` to act on — fixed, see
   above; seeds now genuinely retrain a different base model.)

## RQ4 (subgroup disparity): measured for the first time this pass

`results/results.parquet` has carried `max_min_gap@0.8`/`max_min_gap@0.9`
rows (worst-case-minus-best-case subgroup selective risk, at a fixed
overall coverage threshold — `src/metrics/subgroup.py`) since the metric
was implemented, but nothing in `paper/tables/` ever summarized them: RQ4
was *computed* but never *reported*. `scripts/make_subgroup_table.py`
(new) closes that gap — same per-seed-paired Wilcoxon/Holm-correction
pattern as the AURC table, applied to the subgroup gap instead, for every
dataset with a `subgroup_col` (Adult, German Credit; Electricity has none
by construction, and Diabetes-130 -- which would have been a third -- was
dropped from the project before contributing any results, see above).

**The finding is the Jones et al. (2021) failure mode, replicated
empirically, not avoided.** On Adult, at both coverage thresholds, **8
different non-oracle methods significantly *widen* the sex/race subgroup
gap relative to MSP** (Holm-corrected p<0.05: `A0_rank`, `A1_lightgbm`,
`A2_adaptive_loss3`, `random`, `signal_calib_residual`,
`signal_knn_distance`, `signal_mahalanobis`, `signal_trust_score`), and
**zero** methods narrow it significantly (the only significant *narrowing*
in the whole table is `oracle`, which trivially has zero error and is not
a real method). On German Credit, no non-oracle method reaches
significance in either direction. So the honest RQ4 answer so far is: no
evidence multi-criteria abstention *helps* subgroup disparity, and
suggestive (Adult-specific, not yet replicated) evidence it can make it
*worse* — the opposite of the hoped-for direction, and worth a section of
its own rather than a footnote. See `paper/tables/subgroup_gap_summary.csv`
and `subgroup_gap_wilcoxon_vs_msp.csv`.

## Signal correlation heatmap: built (item 5, "cheapest strong figure")

`scripts/make_signal_correlation_heatmap.py` (new) computes the full
signal-bank Spearman correlation matrix on D_test without training any
aggregator (two base-model fits only, so it's cheap to run alongside
heavier jobs), and renders it as a heatmap. Run for Adult, German Credit,
and Electricity (`paper/figures/signal_correlation_<dataset>_AC.png`); all
three confirm the degeneracy finding above the fold. `calib_residual`'s
Spearman-vs-MSP is dataset-dependent (0.75 on Adult, 0.62 on Electricity,
0.33 on German Credit) — all comfortably non-degenerate, but higher than
the 0.25 figure quoted earlier in this document, which came from the unit
test's synthetic toy dataset, not real data. Use the real per-dataset
numbers in the paper, not the toy-dataset one.

## Not started (still exactly as scoped in the plan)

- Datasets: Bank Marketing, Give Me Some Credit, Covertype, MiniBooNE,
  Higgs, Telco churn, ann-thyroid, Jannis/Road-Safety.
- Shift experiments: Electricity has a real 10-seed temporal-split run
  (see above). Diabetes-130 was tried as a second shift dataset and
  dropped (see "Diabetes-130: added, then dropped") — the plan's
  second-shift-dataset slot (§4) is open again; pick a different
  candidate rather than retrying this one. CIFAR-10-C/100-C and synthetic
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

## Paper: drafted end-to-end (was a skeleton with 11 TODOs)

`paper/main.tex` now has real prose in every section: abstract,
introduction (contributions rewritten to match what was actually found),
related work, experimental setup, results, ablations, limitations and
conclusion. **Two `\todo{}` markers remain and both are the author's
calls**: the department/institution line, and whether to strip the
acknowledgment for double-blind submission.

**Every number is generated, none typed.** `scripts/make_latex_tables.py`
emits `paper/tables/tab_*.tex` from `results/results.parquet`, and
`main.tex` `\input`s them, per plan §8's rule that no number may exist in
the paper except as a query against the results file. Re-running the sweep
and re-running that script updates the paper without anyone editing a
figure by hand -- and removes the classic failure where a table and its
surrounding prose drift apart.

Structural check (there is no LaTeX toolchain in this environment, so it
is **not compiled** -- do that before trusting it): environments balanced,
braces balanced, every `\cite` key resolves against `references.bib`, and
no bib entry is left uncited. ~3,600 words.

**How the paper is framed, and why.** Not as "our method wins" -- the data
does not support that. The spine is the proven binary rank-degeneracy
result, then the conditional and modest aggregation benefit, then three
negative engineering findings, with the selection question stated as open.
The introduction explicitly reports that the pre-registered
regime-dependence prediction was **not** confirmed, rather than quietly
reframing it, and the related-work section states that logit
normalisation's binary degeneracy is a limitation of the published method
rather than of this implementation.

## Suggested next steps, in priority order

1. **Re-run Adult, German Credit, and Electricity** now that `loss1`/
   `loss2` use the quantile-based tau fix (item 8 above) — every existing
   `A2_mlp_loss1`/`A2_mlp_loss2` number in `results/results.parquet` still
   reflects the old, unreliable version. Cheap (minutes per dataset).
2. **Extend Tier B (ensemble disagreement) to German Credit and
   Electricity.** It ran successfully on Adult for the first time this
   pass (10 seeds, `tiers=ABC`) — one dataset can't say whether Tier B
   changes the RQ1/RQ2 picture, only suggest it.
3. **Pick a replacement second shift dataset** for the plan's §4 slot that
   Diabetes-130 vacated (see above) — Electricity alone is a single data
   point for RQ2's shift claim.
4. **Diabetes-130-style experiment, but native 3-class, on whatever
   replaces it (or a similarly multiclass dataset).** This is the
   highest-value *new* experiment available: the entire negative result so
   far traces to one proven fact — on binary tasks, MSP/entropy/margin/
   temp-scaling are forced into identical rankings (Spearman = 1.0000,
   pinned by `tests/test_fixes.py`). At K>=3 classes that degeneracy
   provably breaks. Confirming aggregation helps specifically where
   signals stop being redundant would turn a null result into a
   demonstrated mechanism, not just an honest shrug.
5. **A tighter conformal bound (Hoeffding-Bentkus)** for RQ3 — the current
   Hoeffding+Bonferroni wrapper is documented as unable to satisfy
   α=0.01/0.02/0.05 even on 20,000 well-behaved synthetic points (only
   α=0.10 works), so RQ3 is currently "mechanically valid" but not
   practically demonstrated.
6. Add raw-array caching to `runner.py` if the paired-bootstrap test
   matters for the final paper (currently substituted with a coarser
   per-seed Wilcoxon test).
7. More datasets from the registry, the CD diagram, then
   SelectiveNet/Deep Gamblers/ConfidNet reproduction and image
   experiments — valuable, but only once the above settles what the
   paper's central claim actually is.

**On framing**: the strongest, most defensible contributions right now are
the rank-degeneracy proof, `calib_residual` (the signal that breaks it),
and the RQ4 disparity-widening finding (above) — not "our aggregator
wins." Item 4 is what would upgrade the framing from "when does
aggregation help, and why usually not" (honest but modest) to a
demonstrated mechanism (stronger).
