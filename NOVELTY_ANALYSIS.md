# Novelty analysis and recommended changes

> Produced by running `exhaustive_research_novelty_discovery_prompt.md`
> against this repository's committed results
> (`results/results.parquet`, `results/raw/*.npz`,
> `results/shift_results.parquet`) as of commit `0a36f78`.
>
> **Every number below is recomputed from those files, not quoted from
> another document.** Where a claim in `PROJECT_STATUS.md` is contradicted,
> the recomputation is shown so it can be checked or refuted.

---

## Executive summary — three findings, in order of consequence

1. **The project's headline positive result is partly a winner's curse.**
   `PROJECT_STATUS.md` reports "aggregation beats the best single signal by
   a strikingly stable −6.2% / −6.4% / −6.3% on german_credit, satimage and
   letter". Two of those three survive when the aggregator is fixed in
   advance instead of being chosen as the best of nine. **german_credit's
   does not: it flips to +0.85% (Tier A+C) / +4.44% (Tier A+B+C), with
   `A1_logreg` beating the best single signal in only 4 of 10 seeds.** The
   −6.17% comes from selecting `A0_rank` post hoc on a 150-row test split
   with ~34 errors. See §2.

2. **The mechanism `PROJECT_STATUS.md` records as "currently unexplained"
   appears to be resolved, and the reason it looked unexplained is that the
   diagnostic was measured in the wrong place.** Rank-redundancy among
   Tier-A signals was measured *globally* over the whole test set. AURC does
   not integrate globally — it integrates over the ranking near each
   abstention threshold. Measured **locally, on the top-10% riskiest points
   by MSP**, the minimum |Spearman| between MSP and the rest of the Tier-A
   bank separates the six datasets perfectly and with zero seed overlap:
   1.0000 on all three datasets where aggregation hurts, ≤0.53 on all three
   where it helps. Both of the candidate mechanisms the project falsified
   used global statistics. See §3 — this is the strongest novelty in the
   repository.

3. **RQ2 — the load-bearing "aggregation helps under shift" claim — has
   only ever been tested on the one dataset where finding 2 predicts it
   cannot work.** The entire shift sweep in `results/shift_results.parquet`
   is german_credit × gaussian_noise × 3 intensities: a binary dataset with
   local redundancy exactly 1.0000. On it, the gap over the best single
   signal *widens* with shift intensity (+1.3% → +7.9% → +9.8%), i.e. the
   pre-registered prediction is currently falsified on the only evidence
   available — on a dataset where a null was predictable in advance. See §5.

---

## 1. Phase 0 — the problem, restated at first principles

| | |
|---|---|
| Input | `x ∈ ℝ^d` (tabular), a frozen base classifier `f`, and a signal bank `u(x) ∈ ℝ^m` derived from `f` and from the training geometry |
| Output | a scalar risk score `s(x)`, and a threshold `τ̂` inducing accept/abstain |
| Latent variable | `Pr[f(x) ≠ y | x]`, the pointwise error probability |
| Objective | minimise `AURC = ∫ risk(coverage) d(coverage)`, i.e. a *ranking* objective |
| Hidden information | the label `y`; the true error probability |
| Constraints | post-hoc (no base retraining); the meta-set is small (150–7 500 rows) |

**The single most important structural fact, and it is underexploited in the
current framing:** AURC is a function of the *ranking* `s` induces, and
nothing else. Two scores with the same ranking have identical AURC. This has
three consequences that the architecture does not currently act on:

- **Only ordinal information in `u(x)` can ever matter.** Any monotone
  reparameterisation of a signal is free. The project already proved half of
  this (the binary Tier-A degeneracy). It has not drawn the constructive
  half: the correct object to study is the bank's *discordance structure*,
  not its correlation matrix.
- **A ranking objective is not uniformly weighted.** AURC weights a
  swap between ranks 3 and 4 far more than one between ranks 3 000 and
  3 001, because the low-coverage end of the curve has the fewest accepted
  points and the largest risk swings. A *global* redundancy statistic
  therefore answers a question AURC never asks. This is exactly the error
  in §3.
- **Calibration is irrelevant to AURC and the project measures both.**
  `signal_calib_residual` is a calibration-magnitude signal being judged by
  a pure ranking metric. Its poor standalone AURC (0.0536 on adult vs MSP's
  0.0302) is close to guaranteed by construction and should not be read as
  the signal being uninformative.

### The true bottleneck is not the aggregator

The project's bottleneck was assumed to be *aggregator capacity* (hence the
A2 neural family and the custom coverage-targeted losses). The committed
results say otherwise: **`A1_logreg`, plain logistic-regression stacking, is
the best aggregator on satimage, letter and wine, and the A2 family never
wins on any dataset.** The bottleneck is **the information content of the
bank**, and specifically *where in the ranking* that information lives.
Every remaining unit of effort spent on aggregator architecture is spent on
the wrong component.

---

## 2. Phase 18 (falsification), applied to this project's own headline

`PROJECT_STATUS.md`'s central table selects, per dataset, the best of ~9
aggregators and the best of ~14 signals, then reports the difference. On
binary datasets the signal side has one effective degree of freedom (proven
Tier-A degeneracy) while the aggregator side has nine largely independent
ones. **The comparison is therefore biased in the aggregator's favour, and
the bias grows as test-set error count falls.** That is measurable:

Recomputed from `results/results.parquet` (10 seeds, LightGBM):

| dataset | test errors | best-of-9 agg vs best single | **`A1_logreg` fixed in advance** | `A1_logreg` seed win-rate |
|---|---:|---:|---:|---:|
| german_credit | 34 | **−6.17%** (`A0_rank`) | **+0.85%** | **4/10** |
| satimage | 67 | −6.38% (`A1_logreg`) | −6.38% | 9/10 |
| letter | 82 | −6.29% (`A1_logreg`) | −6.29% | 9/10 |
| wine_quality_white | 237 | −1.29% (`A1_logreg`) | −1.29% | 7/10 |
| adult | 944 | +0.88% (`A1_logreg`) | +0.88% | 4/10 |
| electricity | 2 533 | +0.37% (`A2_adaptive`) | +2.60% | 0/10 |

*(Tier A+C, matching the project's own headline table. Tier A+B+C gives the
same qualitative picture; german_credit moves to +4.44%.)*

Spearman(test errors, best-of-9 margin) = **0.943, p = 0.005** — monotone
across all six datasets. Fix the aggregator in advance and it collapses to
**0.371, p = 0.47**. The correlation was selection, not science.

**Consequence for the paper.** Finding 1 as written ("three datasets, three
near-identical numbers") must become **two** datasets, and the near-identity
of the three numbers loses its rhetorical force — it was two real ~6.3%
effects plus one artefact that happened to land nearby. The existing caveat
("german_credit's test split is ~150 rows, its −6.2% may be noise") pointed
at the right dataset for the wrong reason: the problem is not variance, it
is *post-hoc selection over nine candidates* on a high-variance metric. A
reviewer who computes the fixed-aggregator column will find this
immediately.

**This is also a contribution, not just a correction.** "Selective-
prediction benchmarks report best-of-N method selection on a metric whose
variance is set by the test *error* count, not the test row count, and the
resulting optimistic bias is large enough to manufacture a headline effect"
is a methodological finding with a clean, cheap demonstration and immediate
relevance to every benchmark in this literature, including Pugnana et al.
(2024)'s 18 baselines. See idea **N2** in §6.

---

## 3. Phase 13 — root cause of the "unexplained" mechanism

`PROJECT_STATUS.md` states: *"No mechanism currently explains which datasets
benefit — two candidate explanations were proposed and both were falsified
by direct check."* Both candidates were measured with **global** statistics:

- Candidate 1 ("K ≥ 3 breaks Tier-A degeneracy") was rejected because
  global |Spearman| between entropy and MSP on satimage is 1.0000, and
  because german_credit is binary yet appeared to help.
- Candidate 2 ("aggregation helps when Tier-C geometry is informative") was
  rejected on a global best-Tier-C/MSP AURC ratio.

Candidate 1's second leg is void: german_credit does **not** help (§2). Its
first leg is measuring the wrong thing. AURC integrates over the ranking
near each threshold, so the statistic that matters is redundancy **restricted
to the points that are candidates for abstention**.

### The diagnostic

Let `M` be the MSP risk ranking on an evaluation set, `Q_q` the top-`q`
fraction of points by `M`, and define

```
    rho_local(q) = min           |Spearman( u_msp , u_j )|  restricted to Q_q
                 j ∈ TierA\{msp}
```

Computed from `results/raw/*ABC*.npz`, all 10 seeds, `q = 0.10`:

| dataset | K | ρ_local(0.10) mean | [min, max] over seeds | ρ_global | aggregation (fixed `A1_logreg`) |
|---|---:|---:|---|---:|---|
| adult | 2 | **1.0000** | [1.000, 1.000] | 1.000 | **hurts** +0.88% |
| electricity | 2 | **1.0000** | [1.000, 1.000] | 1.000 | **hurts** +12.73% |
| german_credit | 2 | **1.0000** | [1.000, 1.000] | 1.000 | **hurts** +4.44% |
| letter | 26 | **0.463** | [0.388, 0.528] | 0.856 | **helps** −6.07% |
| satimage | 6 | **0.335** | [0.250, 0.440] | 0.891 | **helps** −5.85% |
| wine_quality_white | 7 | **0.245** | [0.150, 0.393] | 0.668 | helps −0.20% |

**Perfect separation, 6/6 datasets, with a gap of 0.47 and zero overlap
across 60 dataset-seed pairs.** Both previously-tested candidates had two
counterexamples each.

Note the `ρ_global` column: on satimage it is 0.891 and on letter 0.856 —
high enough to read as "the bank is redundant", which is precisely the
reading that killed Candidate 1. Locally the same banks decorrelate to 0.335
and 0.463. **The degeneracy is global on binary tasks and only global on
multiclass tasks; multiclass breaks it exactly where AURC is sensitive.**

An entropy-based effective rank of the full rank-transformed signal bank on
`Q_0.10` separates the datasets too (4.62/4.78/4.94 hurts vs 5.51/5.81/7.37
helps), but with a much thinner margin, so `ρ_local` is the better statistic.

### Why this is the strongest result in the repository

- It **resolves a question the project has recorded as open**, without
  requiring a single new experiment — the evidence is already committed.
- It is a **pre-deployment decision procedure**, not just an explanation.
  `ρ_local` needs only a frozen base model and unlabelled data; it can be
  computed on `D_meta` before an aggregator is ever trained, and it says
  whether aggregation can possibly help. That converts a null result into an
  actionable rule.
- It **generalises the project's own theorem in the right direction.** The
  binary Tier-A degeneracy theorem is the `q → 1`, `K = 2` corner of a
  general statement about where in the ranking a signal bank is
  rank-degenerate. That is a strictly stronger and more interesting theorem.
- It **explains a body of prior negative results, not just this project's.**
  Feng et al. (ICLR 2023)'s "a tuned MSP is enough" and the flatness of
  Pugnana et al. (2024)'s single-signal comparisons are predicted by
  `ρ_local ≈ 1` on the mostly-binary, mostly-easy benchmarks in use. That is
  the difference between a paper about this system and a paper about the
  field.

### What could still kill it

Six datasets, and the three "helps" datasets are all `K ≥ 6` while all three
"hurts" datasets are `K = 2`, so `ρ_local` and `K ≥ 3` are not yet separated
as explanations. **The decisive experiment is a multiclass dataset with high
`ρ_local`, or a binary dataset with low `ρ_local`.** The second is reachable:
`ρ_local` on a binary task is 1.0000 *by the proven theorem* for Tier A, so
extend the min over the full bank including Tier B/C, where a binary dataset
can have low local redundancy. If a low-`ρ_local` binary dataset still fails
to benefit, the mechanism is really just `K ≥ 3` and the contribution
shrinks to a reframing. This must be run before the claim is written up.

---

## 4. Phase 5 — assumption attack on the current architecture

| # | Assumption | Where | If false | Recommended action |
|---|---|---|---|---|
| A1 | Aggregation needs a flexible function class | A2 MLP + gating | A2 never wins on any dataset; `A1_logreg` wins on 3 | **Demote A2 to an ablation.** Keep it to answer "does capacity help?" — the answer is no, and that is a result |
| A2 | A coverage-targeted loss beats BCE | `losses.py` | `A2_mlp_bce` ≥ the loss1/loss2/loss3 variants on adult, german, wine | Report the null plainly; it is one of the paper's cleaner negatives |
| A3 | Signals should be combined by value | all aggregators | Only the ranking matters (§1) | **Rank-transform the bank before aggregation.** `A0_rank` already does and it is the best aggregator on german_credit and electricity — the two datasets where every learned aggregator hurts. That is not a coincidence and is not currently investigated |
| A4 | Redundancy is a global property | the falsification of Candidate 1 | It is local (§3) | Make `ρ_local` a first-class measured quantity |
| A5 | One score, one threshold, all inputs | `runner.py` → conformal | Signal informativeness varies across the input space | Per-region aggregation — the honest version of what `A2_adaptive` was reaching for; see **N4** |
| A6 | More independent signals help | Tier B rationale | Tier B is independent and never helps, and *dilutes* wine's edge from −1.3% to −0.2% | Already correctly documented. Sharpen it: independence is necessary, not sufficient; the binding constraint is meta-set size |
| A7 | Exchangeability holds at calibration time | `risk_control.py` | Stated to fail under shift | Measure the violation rate. Currently claimed as a limitation but never quantified |
| A8 | AURC is the right target | everywhere | AURC integrates over coverages nobody deploys at | Add risk@coverage at deployment-relevant operating points as a *primary* metric, not a secondary one |

**A3 deserves emphasis.** `A0_rank` — a rank/z-score average with no
learning at all — is the best aggregator on both datasets where learning
actively hurts, and it is the mechanism-consistent choice: it discards
exactly the non-ordinal information AURC cannot use, and it has no
parameters to overfit a 150-row meta-set with. The natural family to explore
is not "bigger networks" but **rank-space aggregation with increasing
amounts of learning** (rank average → learned rank weights → learned
per-region rank weights), which is a monotone ladder in capacity where the
project's own data predicts the low end wins.

---

## 5. RQ2 is untested where it could work

`results/shift_results.parquet` contains **126 rows: german_credit ×
gaussian_noise × intensity {0.0, 0.5, 2.0}**. Recomputed:

| intensity | MSP AURC | best single | `A1_logreg` | `A1_logreg` vs best single |
|---:|---:|---:|---:|---:|
| 0.0 | 0.1294 | 0.1294 | 0.1312 | +1.3% |
| 0.5 | 0.1482 | 0.1465 | 0.1580 | **+7.9%** |
| 2.0 | 0.1840 | 0.1840 | 0.2021 | **+9.8%** |

The pre-registered RQ2 prediction is that the aggregation gap *narrows and
reverses* as shift intensity rises. **It widens monotonically.** On the
current evidence RQ2 is falsified.

But this is the single dataset in the registry with `ρ_local = 1.0000`
across every seed — the mechanism in §3 predicts a null there *before any
shift is applied*, and shift can only add estimation noise to an aggregator
that has no extra information to exploit. **The sweep was run on the one
dataset where it could not succeed.** Until it runs on satimage / letter /
wine, RQ2 has no evidence either way. This is the highest-value experiment
in the repository and it needs no new code — `scripts/run_shift_sweep.py`
already takes a dataset argument.

---

## 6. Phase 20 — research idea frontier

Scored /10. Only ideas that survive Phase 18 are listed; ideas discarded as
falsified are in §8.

### The three strongest — in priority order

**N1. Where a confidence signal is redundant: boundary-local rank degeneracy
in selective prediction.** (§3)
Novelty: **new theory + new diagnostic + explains prior negatives.** The
theorem generalises the project's proven binary result from a global corner
case to a statement about redundancy as a function of position in the
ranking; the diagnostic is computable pre-deployment on unlabelled data;
6/6 perfect separation with zero seed overlap is unusually clean evidence
for a mechanism claim. Closest prior work: the project's own binary
degeneracy proof (a special case), and Cattelan & Silva (UAI 2024) (a
different repair to a related problem). Nothing in the literature measures
redundancy locally in the ranking.
Novelty 8 · Significance 8 · Feasibility 9 · Risk 4 (the `K` confound, §3)
· **Priority 1**

**N2. The winner's curse in selective-prediction benchmarks.** (§2)
Novelty: **new evaluation methodology + a measured negative.** AURC's
variance is set by test *error* count, not row count; benchmarks report
best-of-N; the resulting bias is large enough to manufacture a −6% headline
(demonstrated on this project's own prior claim, which is the most credible
possible demonstration). Deliverables: the bias as a function of error count
and N; a corrected protocol (pre-registered method, or a selection-adjusted
CI); a re-analysis of one published benchmark. Closest prior work: general
model-selection-bias literature; nothing specific to selective prediction,
where the error-count dependence is the distinctive part.
Novelty 7 · Significance 8 · Feasibility 10 · Risk 2 · **Priority 2**

**N3. Rank-space aggregation, and why capacity is the wrong axis.** (§4, A3)
Novelty: **new formulation of a known component + a strong null.** AURC is
ordinal, therefore aggregation should be too; `A0_rank` beating every
learned aggregator on the two hardest datasets is the seed of a principled
family (rank average → learned rank weights → per-region rank weights) with
a hypothesis that the low-capacity end wins, against the field's prior. Pairs
naturally with N1: rank-space aggregation on `Q_q` is where `ρ_local` says
the information is.
Novelty 6 · Significance 7 · Feasibility 9 · Risk 3 · **Priority 3**

### Worth building, second tier

**N4. Region-conditional abstention.** Partition the input space by
`ρ_local` computed locally, and use the rank-average aggregator where the
bank is degenerate and a learned one where it is not. This is what
`A2_adaptive` was reaching for, but conditioned on a quantity that
*provably* determines whether learning can help rather than on a learned
gate. Novelty 6 · Feasibility 7 · Risk 5.

**N5. Quantify the conformal violation rate under shift.** The code and the
paper both state that exchangeability fails under shift; neither measures
by how much. A table of empirical `P(risk > α)` against the nominal `1 − δ`
across the shift sweep is cheap, is currently missing, and is the kind of
honest negative that strengthens a paper. Novelty 4 · Feasibility 10 ·
Risk 1. **Do this regardless of which direction you pick** — it closes a
claim the paper currently makes without evidence.

**N6. Screen datasets by error count, and say so.** The project already
learned this ("screen on test-error count, not row count") and the lesson
generalises: a large fraction of tabular selective-prediction benchmarks
cannot resolve a 6% AURC effect at all. A short "how many errors do you need"
power analysis is a genuine service contribution and directly supports N2.
Novelty 4 · Feasibility 10 · Risk 1.

### Higher-risk

**N7. Ordinal-aware abstention for ordinal labels.** wine_quality_white has
ordinal classes and is the dataset where geometric `trust_score` beats every
logit signal by 23.5%. That is suggestive: for ordinal labels, "which class"
errors are not exchangeable in cost, and a distance-aware risk score may be
the right object. Under-explored, but `n = 1` dataset. Novelty 7 ·
Feasibility 6 · Risk 7.

**N8. A calibration-aware ranking metric.** `signal_calib_residual` is a
calibration signal judged by a ranking metric that is invariant to
calibration (§1) — it is being evaluated by a metric that structurally
cannot reward it. Either evaluate it against a risk-calibration metric, or
drop it. Novelty 5 · Feasibility 8 · Risk 5.

---

## 7. Recommended reframing of the paper

Current framing: *"we propose a multi-signal aggregator with
coverage-targeted losses and conformal risk control."* Every component of
that sentence is contradicted by the repository's own results — the
aggregator wins on 2 of 6 datasets, the coverage-targeted losses never win,
and the conformal guarantee is unverified under the conditions that matter.

Proposed framing:

> **When does combining uncertainty signals help a selective classifier?**
> We show that whether multi-signal aggregation can beat the best single
> confidence signal is determined by the signal bank's rank-redundancy
> *restricted to the abstention region*, not by its global redundancy, class
> count, or the aggregator's capacity. We prove the degenerate case exactly
> (on binary tasks every logit-derived signal induces one ranking), give a
> local redundancy statistic that predicts the outcome on 6/6 datasets
> before any aggregator is trained, and show that the apparent gains
> reported for aggregation are inflated by best-of-N selection on a metric
> whose variance is governed by test-set error count.

Contributions in that framing, in descending strength: (i) the local
degeneracy theory and diagnostic (N1); (ii) the benchmarking bias (N2);
(iii) the exact binary theorem, already proven and tested; (iv) rank-space
aggregation and the capacity null (N3); (v) the honest negatives — Tier B,
the A2 family, the coverage-targeted losses, the conformal violation rate.

This is a stronger paper than the original plan, it is supported by evidence
already in the repository, and it does not require the aggregator to win.

---

## 8. Ideas checked and discarded

Recorded so they are not re-proposed.

| Idea | Why discarded |
|---|---|
| "Aggregation helps when base error rate is high" | Non-monotone: adult 0.129 hurts, letter 0.028 helps |
| "Aggregation helps when a non-Tier-A signal is individually competitive" | Falsified by letter: best non-Tier-A signal is 25% *worse* than MSP, aggregation still helps −6.1% |
| "Aggregation helps when Tier-C geometry is informative" | Already falsified in `PROJECT_STATUS.md`; reconfirmed |
| "K ≥ 3 breaks the degeneracy" (as stated) | Not falsified after §2 removes german_credit, but **confounded** with `ρ_local` on the current six datasets — it is the null hypothesis N1 must beat, not a discarded idea |
| Bigger / deeper aggregators | A2 never wins; capacity is not the bottleneck (§4, A1) |
| More independent signals (Tier B) | Independent and useless; dilutes the wine effect |
| Repairing `logitnorm_msp` for binary | Proven impossible; correctly closed already |

---

## 9. Concrete change list

Ordered by value per unit of effort. No item requires new infrastructure
except where noted.

### Must do before the paper is written

1. **Add the fixed-aggregator column to the central table** and correct
   Finding 1 from three datasets to two. german_credit's −6.2% is a
   selection artefact (§2). *Analysis only, no runs.*
2. **Implement `ρ_local` as a measured quantity** — a function in
   `src/metrics/`, emitted per `(dataset, seed)` into
   `results/results.parquet`, computed on `D_meta` as well as `D_test` so
   the pre-deployment claim is supported. *~80 lines; the analysis in §3
   already exists as a script.*
3. **Run the shift sweep on satimage, letter and wine_quality_white.** RQ2
   currently rests on the one dataset where a null was predictable (§5).
   `scripts/run_shift_sweep.py` already takes a dataset argument. *Compute
   only.*
4. **Report the conformal violation rate under shift** (N5). The paper makes
   a claim it does not measure. *Small addition to the sweep's reporting.*

### Should do

5. **Find the decisive dataset for N1** — a binary task with low `ρ_local`
   over the full bank, or a multiclass task with high `ρ_local` — to separate
   `ρ_local` from `K ≥ 3` (§3, "what could still kill it"). Screen candidates
   on test-error count first, per the project's own lesson. *This is the
   experiment the flagship claim stands or falls on.*
6. **Add the rank-space aggregation ladder** (N3): `A0_rank` (exists) →
   learned rank weights → per-region rank weights. Predicts the low end
   wins, which is a testable claim against the field's prior. *~150 lines.*
7. **Demote the A2 family and the coverage-targeted losses to ablations**
   and state the null plainly (§4, A1/A2). This is a subtraction, and it
   makes the paper more honest and shorter.
8. **Re-cut every method comparison on risk@coverage at deployment-relevant
   operating points**, alongside AURC (§4, A8).

### Nice to have

9. Either evaluate `signal_calib_residual` against a calibration-sensitive
   metric or drop it (N8) — currently it is judged by a metric that cannot
   reward it.
10. The error-count power analysis (N6): how many test errors are needed to
    resolve a 5% AURC difference at 80% power. Supports N2 and the dataset
    screen, and is ~30 lines of bootstrap.
11. Bump letter / satimage / wine to the same seed budget as the binary
    datasets if any of them ran short, so the two surviving wins in §2 rest
    on the same footing as the nulls.

### Do not do

- Do not add aggregator capacity, new loss functions, or new neural
  architectures. Six datasets of committed evidence say the aggregator is
  not the bottleneck.
- Do not add more Tier-B-style independent-but-weak signals.
- Do not add datasets without first screening test-error count.
