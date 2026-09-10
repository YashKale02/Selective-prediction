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

### Two findings worth knowing before you read any result here

Both came out of the bug-fix pass described in
[`betterment.md`](betterment.md) (which carries the same summary at the
top) and are written up in full, with the measurements, under
"Bugs found and fixed" in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

1. **On a binary task, every Tier-A signal is a monotone transform of
   every other one.** MSP, entropy, both margins, temperature-scaled MSP
   and even a correctly-fixed energy all have pairwise Spearman
   |ρ| = 1.0000 — so Tier A supplies exactly **one** distinct ranking, and
   *no aggregator over Tier A alone can order abstentions differently from
   MSP*, however it is trained. This is a structural explanation for the
   project's central negative result that is independent of any bug. On
   binary problems the only genuinely independent orderings come from Tier
   B (ensemble disagreement), Tier C (geometry/density) and the new
   `calib_residual` signal.

2. **Logit-norm MSP is mathematically degenerate for K = 2 and cannot be
   repaired.** A softmax reads only logit *differences*, so at K = 2 the
   whole signal is one scalar gap `s`; dividing by `‖z‖_p` removes exactly
   the overall scale, which at K = 2 *is* `|s|`, leaving only `sign(s)`.
   Every candidate repair was measured and each collapses to ≤2 distinct
   values. It is now excluded from the binary bank rather than "fixed" —
   which also means Cattelan & Silva's logit normalisation is
   **inapplicable to binary classification**, a limitation of the
   published method rather than of this code.

Two of the twelve uncertainty signals were also outright broken before
this pass, in ways that made them *worse than random abstention* while
being fed as features into every aggregator. If you are comparing against
numbers from an earlier commit, they are not trustworthy.

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
- **Signals** (`src/signals/`): Tier A (MSP, entropy, both margins,
  energy, temperature-scaled MSP, and `calib_residual` — an isotonic
  miscalibration residual, the only Tier-A signal that is *not*
  rank-equivalent to MSP on a binary task; logit-norm MSP is included only
  for K ≥ 3, see finding 2 above), Tier B (ensemble vote entropy, mean
  pairwise KL, top-class variance — enabled with `signals=tier_a_b_c`),
  Tier C (kNN distance, local label agreement, trust score, Mahalanobis
  distance). Tier D (tree-specific) is not implemented — see
  PROJECT_STATUS.md.
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
- **Paper** — `final_ieee/main.tex` is the authoritative submission: a
  single-focus 6-page IEEE conference paper on the boundary-local
  redundancy diagnostic (`rho_local`) and why it is needed, self-contained
  with its own `IEEEtran.cls`, `references.bib` (all entries verified
  against live sources) and only the figures it actually uses.
  `paper/tables/` and `paper/figures/` remain the pipeline's general
  output directory — every analysis script writes there — but `paper/`
  no longer holds a competing paper draft.

## What's real vs. scaffolded — the short version

Three real OpenML datasets (Adult, German Credit, and Electricity, the
last one temporal-shift) have been run end-to-end at 10 seeds each,
through the entire pipeline including cross-fitting, conformal
calibration, and the statistical tests — this is not a demo on synthetic
data. (Diabetes-130 was registered and attempted as a second shift
dataset, but was dropped — too many incomplete fields, see
PROJECT_STATUS.md — before a run ever completed.) The result across all
three matches the paper plan's own pre-registered prediction (§1, RQ2 and
§11's risk register): **a properly tuned MSP/temperature-scaled-MSP
baseline is very hard to beat in-distribution**; no aggregator here beats
it by a statistically significant margin yet. That is not a failure of the
implementation — it is the expected in-distribution result the plan
explicitly tells you to anticipate, and precisely why RQ2 (shift-regime
dependence) and the conformal-coverage numbers, not raw in-distribution
AURC, are the load-bearing claims. See PROJECT_STATUS.md for exactly what
would need to run next (shift datasets, more seeds, more datasets) to
actually test that.

## Citations

`final_ieee/references.bib` (the paper's actual bibliography) carries its
own header note recording that every entry was checked against a live
source (arXiv / ACM DL / IEEE Xplore / NeurIPS-ICML-ICLR proceedings) as
of the date given there. `paper/related_work_gap_table.md` is the older,
project-plan-era gap analysis this was checked against; it predates the
current paper's single-focus scope and is kept for that provenance, not
as a citation list to trust on its own.

## Repository layout

`src/`, `scripts/`, `configs/`, `results/`, `tests/` are the reproducible
pipeline. `paper/tables/` and `paper/figures/` are that pipeline's output
directory (every `scripts/make_*`/`analyze_*` script writes there).
`final_ieee/` is the paper itself — self-contained, compiles on its own,
and is the only LaTeX project in this repository that should be treated
as a submission draft.
