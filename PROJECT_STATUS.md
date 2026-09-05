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

## Actual results so far (Adult + German Credit, 5 seeds; Electricity, 1 seed; Tier A+C, LightGBM)

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

A first single-seed run on `electricity` (temporal shift, §4) shows the
same pattern: MSP still clearly beats random (AURC 0.197 vs. 0.378), and
still beats every aggregator tried here (best aggregator AURC 0.207-0.213
vs. MSP's 0.197) — so on this one shift dataset, at one seed, RQ2's hoped
-for effect (aggregation pulling ahead under shift) has **not** appeared
yet either. With only one seed and one shift dataset this is not a
statistically meaningful test of RQ2 — it is a single data point, reported
here so the next person doesn't have to re-derive it. Do not write "our
method wins" into the paper from this data; do not write "the method
fails under shift" either — that needs the full shift battery (Electricity
at 10 seeds, Diabetes-130 temporal split, CIFAR-10/100-C) that hasn't run
yet. The honest reading right now is "ties in both regimes tested so far,
at low sample sizes" — exactly the third row of the plan's own
falsification table (§1) if it holds up at scale, not the bottom row.

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
   dataset, at 1 seed). Everything is wired to scale to the full
   12-dataset registry in `src/data/loaders.py:REGISTRY` — just add
   entries and run `scripts/run_all.py dataset=<name>
   experiment.n_seeds=10`. Note that for a temporal dataset (`is_temporal=
   True`), the train/meta/cal/test split itself is deterministic (split by
   row order, not by `seed`) — running multiple seeds on `electricity`
   only varies model-training and cross-fitting randomness, not the split,
   which is weaker variance evidence than the random-split datasets get
   from the same seed count.

## Not started (still exactly as scoped in the plan)

- Datasets: Bank Marketing, Give Me Some Credit, Covertype, MiniBooNE,
  Higgs, Diabetes-130, Telco churn, ann-thyroid, Jannis/Road-Safety.
- Shift experiments: Electricity/Diabetes-130 temporal split (the
  `Dataset.is_temporal` plumbing exists in `src/data/splits.py` and is
  wired for `electricity`, but no run has been done), CIFAR-10-C/100-C,
  synthetic covariate shift.
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

## Citations: unverified, by design of this build

No literature search was run this session. `paper/references.bib` and
`paper/related_work_gap_table.md` transcribe the project plan's own §2
citations, which the plan itself flags as "recalled from memory... some
may be misattributed." Every entry is marked `UNVERIFIED` in both files.
**Verify all of them against IEEE Xplore/arXiv/Semantic Scholar before
they appear in a real submission** — this is Week 1's gate in the plan's
own timeline and has not been done here.

## Suggested next steps, in priority order

1. Verify the literature (plan §2) and fill in the gap table — it's the
   one piece of this that requires a human/web-search step this session
   didn't have.
2. Run the temporal-shift dataset (`electricity`) end-to-end and confirm
   `Dataset.is_temporal` produces a sane split; this is the cheapest real
   test of RQ2.
3. Scale seeds to 10 and add a handful more tabular datasets from the
   registry (§4) — the runner and CLI already support this; it's
   compute-bound, not code-bound.
4. Add raw-array caching to `runner.py` if the paired-bootstrap test
   matters for the final paper.
5. Only then: SelectiveNet/Deep Gamblers/ConfidNet reproduction, image
   experiments, and the full ablation sweep scripts.
