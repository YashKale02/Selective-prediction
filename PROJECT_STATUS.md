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

## Deliberate simplifications (be honest about these, per the plan's own ethos)

1. **Temperature scaling is not K-fold cross-fitted.** It's fit once, on
   D_meta, using a model trained only on D_train (`model_train` in
   `runner.py`) — already non-leaky since D_train and D_meta are disjoint,
   but it does *not* go through the same 5-fold cross-fitting loop as the
   other Tier A/C signals, so its distribution on D_meta may differ
   slightly from model_final's (trained on D_train∪D_meta) behavior at
   deployment. Documented in `runner.py`'s module docstring.
2. **Tier B (ensemble) is not cross-fitted at all.** `use_ensemble=True`
   trains one ensemble on the full D_train∪D_meta pool for both meta-signal
   computation and deployment scoring. Cross-fitting an ensemble-of-
   ensembles (K folds × M members) is real additional engineering, out of
   scope for this pass.
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
   in the plan (§3.1) but not built here.
6. **No neural base model / MC-dropout / image datasets yet.** Everything
   above runs on tabular data with LightGBM/logreg. CIFAR-10/100 +
   ResNet-18/WideResNet, and the MC-dropout Tier-B source that requires a
   dropout-bearing network, are unstarted (plan explicitly allows dropping
   image experiments first under schedule pressure — §11).
7. **3 of the planned 12 tabular datasets so far** (`adult`,
   `german_credit` at 5 seeds each; `electricity`, the temporal-shift
   dataset, now at 10 seeds). Everything is wired to scale to the full
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
  Higgs, Diabetes-130, Telco churn, ann-thyroid, Jannis/Road-Safety.
- Shift experiments: Electricity now has a real 10-seed run (see above);
  Diabetes-130 temporal split (the `Dataset.is_temporal` plumbing exists
  in `src/data/splits.py` and would need a registry entry), CIFAR-10-C/
  100-C, synthetic covariate shift are all still unrun.
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
