# Risk-Aware Multi-Criteria Selective Prediction

Research codebase for the IEEE-conference project described in
[`selective_prediction_project_plan.md`](selective_prediction_project_plan.md).
One paragraph version: train a fixed base classifier, extract a vector of
uncertainty signals per input, learn an aggregator that maps that vector to
an abstention score with a coverage-targeted ranking loss, wrap the score
in a conformal risk-control layer, and evaluate the accuracy-coverage
trade-off against single-signal baselines in- and out-of-distribution.

**Read [`PROJECT_STATUS.md`](PROJECT_STATUS.md) first.** It is the honest,
current account of what in the 14-week plan is implemented and verified
end-to-end vs. still scaffolded/TODO — this README describes the system
that exists; that file describes how far along it is.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or requirements-freeze.txt for an exact pin

pytest tests/ -v                                   # unit tests (metrics, splits, signals)
python scripts/run_all.py dataset=adult             # one dataset, default config
python scripts/run_all.py dataset=german_credit experiment.n_seeds=10
python scripts/make_tables.py                       # AURC table + Wilcoxon/Friedman
python scripts/make_figures.py                      # risk-coverage + week-2-gate figures
```

Every run appends to `results/results.parquet` (long format: one row per
`(dataset, base_model, method, seed, coverage-or-summary-subgroup)`) —
per the plan's engineering rule (§8), no number in the paper should exist
anywhere except as a query against that file. `scripts/make_tables.py` and
`scripts/make_figures.py` are that query layer.

## What's implemented

- **Data** (`src/data/`): OpenML loaders for a curated dataset registry;
  the disjoint 4-way split (train/meta/cal/test); K-fold cross-fitting
  over train∪meta to produce honest, non-leaky signal vectors for
  aggregator training (project plan §6 — the leakage trap the plan calls
  the most common silent failure mode in this kind of project).
- **Base models** (`src/models/`): LightGBM and logistic-regression
  wrappers behind one interface (`fit` / `predict_proba` / `logits` /
  `features`), so every signal is model-agnostic.
- **Signals** (`src/signals/`): Tier A (MSP, entropy, margin, energy,
  logit-norm MSP, temperature-scaled MSP), Tier B (ensemble vote entropy,
  mean pairwise KL, top-class variance), Tier C (kNN distance, local label
  agreement, trust score, Mahalanobis distance). Tier D (tree-specific) is
  not implemented — see PROJECT_STATUS.md.
- **Aggregators** (`src/aggregators/`): A0 (rank/z-score average, no
  learning), A1 (logistic regression / LightGBM stacking on correctness),
  A2 (a small MLP, and an instance-adaptive gating variant, both trained
  with the coverage-targeted losses in project plan §3.3 — soft selective
  risk, an AURC surrogate, and a pairwise ranking loss — plus plain BCE for
  the A1-vs-A2 ablation).
- **Conformal wrapper** (`src/conformal/`): a Hoeffding-bound,
  Bonferroni-corrected threshold search giving `P(selective risk <= alpha)
  >= 1 - delta` in-distribution. Documented as deliberately simple and
  auditable rather than tight — see the conservativeness note in
  PROJECT_STATUS.md before reading too much into small-alpha numbers.
- **Metrics** (`src/metrics/`): risk-coverage curve, AURC, E-AURC,
  risk@coverage, coverage@risk, failure-prediction AUROC, ECE (equal-mass
  bins), Brier, NLL, and subgroup worst-group-risk / max-min gap — all
  unit-tested against analytically-known cases (`tests/test_metrics.py`).
- **Experiment runner** (`src/experiment/runner.py`): wires all of the
  above into one call producing the full §7 baseline ladder (random, every
  single signal, every aggregator, oracle) plus the naive-vs-cross-fit
  ablation, for one `(dataset, seed)`.
- **Config** (`configs/`, Hydra) and **CLI** (`scripts/run_all.py`).
- **Paper skeleton** (`paper/main.tex`, IEEEtran, generic conference
  class) with the structure from project plan §10, and
  `paper/related_work_gap_table.md` / `paper/references.bib` — both
  explicitly marked **UNVERIFIED**, see below.

## What's real vs. scaffolded — the short version

Two real OpenML datasets (Adult, German Credit) have been run end-to-end,
5 seeds each, through the entire pipeline including cross-fitting,
conformal calibration, and the statistical tests — this is not a demo on
synthetic data. The result on both matches the paper plan's own
pre-registered prediction (§1, RQ2 and §11's risk register): **a properly
tuned MSP/temperature-scaled-MSP baseline is very hard to beat
in-distribution**; no aggregator here beats it by a statistically
significant margin yet on these two datasets. That is not a failure of the
implementation — it is the expected in-distribution result the plan
explicitly tells you to anticipate, and precisely why RQ2 (shift-regime
dependence) and the conformal-coverage numbers, not raw in-distribution
AURC, are the load-bearing claims. See PROJECT_STATUS.md for exactly what
would need to run next (shift datasets, more seeds, more datasets) to
actually test that.

## Citations are unverified — do not submit as-is

`paper/references.bib` and `paper/related_work_gap_table.md` are
transcribed directly from the project plan's own §2, which explicitly
flagged its citations as "recalled from memory... verify each one... some
may be misattributed." Nothing in this repository has checked them against
a live source. Verify every entry before writing the related-work section
for real.

## Repository layout

See `configs/`, `src/`, `scripts/`, `results/`, `paper/`, `tests/` — the
layout matches project plan §8 exactly.
