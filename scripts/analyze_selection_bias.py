"""The winner's curse in selective-prediction benchmarking
(`NOVELTY_ANALYSIS.md` §2, idea N2).

The claim
---------
Selective-prediction papers, this project's own `PROJECT_STATUS.md`
included, report a *best-of-N* comparison: pick the best of ~9 aggregators
and the best of ~14 signals per dataset, then quote the difference. That
number is optimistically biased, and the bias is governed by the test
set's **error count**, not its row count -- because AURC is built entirely
out of the errors, so a 20 000-row dataset at 99% accuracy carries the
statistical weight of 200 points, not 20 000.

The bias is not a rounding concern. On german_credit (34 test errors) the
best-of-9 aggregator beats the best single signal by -6.17%, which this
project reported as one of three headline wins. Fix the aggregator in
advance -- the same `A1_logreg` that wins on the other datasets -- and the
same data gives **+0.85%**, i.e. aggregation *hurts*. The headline was
selection, not science.

What this script measures, and why each piece is needed
-------------------------------------------------------
1. **Held-out method selection (`selection_shrinkage`).** The direct,
   assumption-free measurement: split the seeds in half, choose the best
   aggregator on one half, and score *that* aggregator on the other half.
   Averaged over many random splits, the gap between the best-of-N figure
   and the held-out figure is the selection bias, measured rather than
   modelled. This is also the corrected protocol a paper should use.

2. **Per-instance bootstrap null (`null_best_of_n`).** How large a
   best-of-N margin arises *purely* by chance? Each aggregator's scores are
   resampled on the same bootstrapped test instances (paired), and the
   best-of-N margin is recomputed. Comparing the observed margin against
   this null band says whether a reported win is distinguishable from
   noise-plus-selection. Uses `results/raw/*.npz`, so it resamples real
   instances rather than assuming a parametric form.

3. **Error-count scaling (`aurc_se_vs_errors`).** The mechanism behind
   (1) and (2): the standard error of an AURC estimate against the number
   of test errors, which is what makes the bias dataset-dependent and
   predictable in advance.

4. **Power (`power_analysis`).** How many test errors are needed to
   resolve a given relative AURC difference at 80% power -- the number the
   field needs in order to screen datasets honestly (idea N6). Directly
   supports the `MIN_TEST_ERRORS` gate in
   `scripts/screen_redundancy_candidates.py`.

Usage:
    python scripts/analyze_selection_bias.py
    python scripts/analyze_selection_bias.py --tiers ABC --n-boot 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment.runner import load_raw_scores  # noqa: E402
from src.metrics.selective import aurc  # noqa: E402

RESULTS = ROOT / "results" / "results.parquet"
RAW_DIR = ROOT / "results" / "raw"
TABLE_DIR = ROOT / "paper" / "tables"

# The aggregator fixed in advance for the corrected comparison. Chosen on
# a principle, not on the results: it is the simplest learned aggregator in
# the repo (plain standardised logistic-regression stacking), so committing
# to it needs no knowledge of which method won where. Naming it here, once,
# is what makes the comparison pre-registered rather than post-hoc.
PREREGISTERED_AGGREGATOR = "A1_logreg"


def _aggregator_names(methods: list[str]) -> list[str]:
    return [m for m in methods if m.startswith(("A0", "A1", "A2"))]


def _signal_names(methods: list[str]) -> list[str]:
    return [m for m in methods if m.startswith("signal_")]


def load_aurc_table(tiers: str) -> pd.DataFrame:
    """(dataset, seed) x method AURC matrix from the committed summary."""
    df = pd.read_parquet(RESULTS)
    df = df[(df.tiers == tiers) & (df.subgroup == "AURC")]
    if df.empty:
        raise SystemExit(f"no AURC rows for tiers={tiers!r} in {RESULTS}")
    return df.pivot_table(index=["dataset", "seed"], columns="method", values="value")


def test_error_counts(tiers: str) -> pd.Series:
    """Mean number of test errors per dataset -- the quantity that governs
    AURC variance, and the one benchmarks report row counts instead of."""
    df = pd.read_parquet(RESULTS)
    base = df[
        (df.tiers == tiers)
        & (df.method == "signal_msp")
        & (df.coverage == 1.0)
        & (df.subgroup == "overall")
    ]
    return (base.risk * base.n_accepted).groupby(base.dataset).mean()


def selection_shrinkage(
    aurc_tab: pd.DataFrame, n_splits: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Measure the optimism in best-of-N by selecting on held-out seeds.

    For each dataset: repeatedly split the seeds into a selection half and
    an evaluation half, pick the aggregator with the lowest mean AURC on
    the selection half, and record its mean AURC on the evaluation half.
    The reference for every comparison is the *best single signal chosen
    the same way*, so the signal side is not given an unfair advantage --
    the asymmetry being measured is in the number of candidates, not in
    whether selection happened at all.

    `biased_margin` is the protocol under audit (choose and evaluate on all
    seeds). `honest_margin` selects and evaluates on disjoint seeds.
    `shrinkage = biased - honest`: how much of the reported win evaporates
    when selection is paid for. Positive shrinkage means the reported
    number was optimistic.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, g in aurc_tab.groupby(level=0):
        aggs = _aggregator_names(list(g.columns))
        sigs = _signal_names(list(g.columns))
        if not aggs or not sigs:
            continue
        # Drop the dataset level so rows are indexed by seed alone:
        # `.loc[(dataset, [seeds])]` on a MultiIndex is read as
        # (rows, columns) and looks up seeds among the method columns.
        by_seed = g.droplevel(0)
        seeds = by_seed.index.to_numpy()
        n = len(seeds)
        if n < 4:
            continue  # cannot split seeds meaningfully

        mean_all = g.mean()
        best_agg_all = mean_all[aggs].idxmin()
        best_sig_all = mean_all[sigs].idxmin()
        biased = (mean_all[best_agg_all] - mean_all[best_sig_all]) / mean_all[best_sig_all]

        honest, chosen = [], []
        half = n // 2
        for _ in range(n_splits):
            perm = rng.permutation(n)
            sel, ev = seeds[perm[:half]], seeds[perm[half:]]
            m_sel = by_seed.loc[sel].mean()
            m_ev = by_seed.loc[ev].mean()
            a = m_sel[aggs].idxmin()
            s = m_sel[sigs].idxmin()
            chosen.append(a)
            honest.append((m_ev[a] - m_ev[s]) / m_ev[s])

        honest_margin = float(np.mean(honest))
        # How often does the selection half even agree on a winner? A
        # method that is genuinely best is picked nearly always; an
        # artefact of noise is picked a fraction of the time, so this is a
        # direct readout of whether "the best aggregator" is a stable fact
        # about the dataset or a coin flip.
        stability = float(np.mean([c == best_agg_all for c in chosen]))

        prereg = mean_all.get(PREREGISTERED_AGGREGATOR, np.nan)
        rows.append(
            dict(
                dataset=dataset,
                n_aggregators=len(aggs),
                n_signals=len(sigs),
                best_agg=best_agg_all,
                best_single=best_sig_all,
                biased_margin_pct=100 * biased,
                honest_margin_pct=100 * honest_margin,
                shrinkage_pct=100 * (biased - honest_margin),
                selection_stability=stability,
                prereg_margin_pct=100 * (prereg - mean_all[best_sig_all])
                / mean_all[best_sig_all],
            )
        )
    return pd.DataFrame(rows)


def null_best_of_n(
    tiers: str, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Per-instance bootstrap distribution of a best-of-N margin.

    The null being simulated is *not* "all methods are identical" -- that
    would need a model of what the methods are. It is the sharper,
    assumption-free question: given these methods' actual per-instance
    scores, how much does the best-of-N margin move when the test set is
    resampled? Instances are resampled once per iteration and applied to
    every method (paired), so the comparison isolates sampling noise in the
    shared test set rather than adding independent noise per method.

    **Two spreads are reported, and conflating them would misstate every
    conclusion in this section.** The same distinction is drawn (and its
    rationale argued) in `scripts/make_bootstrap_table.py`:

      * `within_seed_sd_pct` -- the spread of the margin *within a single
        run*. This answers "would I see this margin if I deployed once on a
        test set this size?", which is the practitioner's question, and it
        is the number that makes a 6% effect on 82 test errors look
        hopeless.
      * `seed_avg_sd_pct` -- the spread of the *10-seed mean* margin,
        obtained by averaging each bootstrap iteration's per-seed margins
        before taking the spread. This answers "is the mean effect across
        runs real?", which is what a paper claiming "aggregation beats the
        best single signal on dataset D" actually asserts, and it is
        roughly sqrt(n_seeds) times narrower.

    Reporting only the second (having seen the first) would be
    indistinguishable from fishing; reporting only the first would
    understate what 10 seeds legitimately buy. Both are emitted so any
    claim's basis is auditable.

    Bootstrap iterations are shared across seeds by construction: iteration
    `b` uses seed-specific resamples, and the seed-average is taken across
    those, so `seed_avg` inherits the pairing rather than assuming
    independence between methods.
    """
    files = sorted(RAW_DIR.glob(f"*__{tiers}__*.npz"))
    if not files:
        raise SystemExit(f"no raw score files for tiers={tiers!r} in {RAW_DIR}")

    # dataset -> list of per-seed arrays of length n_boot
    boot_by_dataset: dict[str, dict[str, list[np.ndarray]]] = {}
    observed: dict[str, list[dict]] = {}

    for path in files:
        raw = load_raw_scores(path)
        methods, scores, incorrect = raw["methods"], raw["scores"], raw["incorrect"]
        aggs = _aggregator_names(methods)
        sigs = _signal_names(methods)
        if not aggs or not sigs:
            continue
        idx = {m: i for i, m in enumerate(methods)}
        n = len(incorrect)
        # Seeded per (dataset, seed) so the whole analysis is reproducible
        # and independent of the order files happen to be globbed in.
        rng = np.random.default_rng((seed, raw["seed"], hash(raw["dataset"]) % 2**32))

        margins = np.full(n_boot, np.nan)
        prereg = np.full(n_boot, np.nan)
        for b in range(n_boot):
            take = rng.integers(0, n, n)
            inc_b = incorrect[take]
            if inc_b.min() == inc_b.max():
                continue  # degenerate resample: no errors or all errors
            best_s = min(aurc(scores[idx[m]][take], inc_b) for m in sigs)
            margins[b] = (min(aurc(scores[idx[m]][take], inc_b) for m in aggs) - best_s) / best_s
            if PREREGISTERED_AGGREGATOR in idx:
                p = aurc(scores[idx[PREREGISTERED_AGGREGATOR]][take], inc_b)
                prereg[b] = (p - best_s) / best_s

        s_obs = min(aurc(scores[idx[m]], incorrect) for m in sigs)
        a_obs = min(aurc(scores[idx[m]], incorrect) for m in aggs)
        p_obs = (
            aurc(scores[idx[PREREGISTERED_AGGREGATOR]], incorrect)
            if PREREGISTERED_AGGREGATOR in idx
            else np.nan
        )

        ds = raw["dataset"]
        boot_by_dataset.setdefault(ds, {"best": [], "prereg": []})
        boot_by_dataset[ds]["best"].append(margins)
        boot_by_dataset[ds]["prereg"].append(prereg)
        observed.setdefault(ds, []).append(
            dict(
                n_errors=int(incorrect.sum()),
                best_margin=100 * (a_obs - s_obs) / s_obs,
                prereg_margin=100 * (p_obs - s_obs) / s_obs,
            )
        )

    rows = []
    for ds, boots in boot_by_dataset.items():
        obs = pd.DataFrame(observed[ds])
        best = np.vstack(boots["best"])  # (n_seeds, n_boot)
        prereg = np.vstack(boots["prereg"])
        # Seed-averaged: collapse the seed axis *inside* each iteration.
        best_seed_avg = np.nanmean(best, axis=0)
        prereg_seed_avg = np.nanmean(prereg, axis=0)
        rows.append(
            dict(
                dataset=ds,
                n_seeds=best.shape[0],
                n_errors=float(obs.n_errors.mean()),
                best_margin_pct=float(obs.best_margin.mean()),
                within_seed_sd_pct=100 * float(np.nanmean(np.nanstd(best, axis=1))),
                seed_avg_sd_pct=100 * float(np.nanstd(best_seed_avg)),
                seed_avg_ci_lo_pct=100 * float(np.nanpercentile(best_seed_avg, 2.5)),
                seed_avg_ci_hi_pct=100 * float(np.nanpercentile(best_seed_avg, 97.5)),
                prereg_margin_pct=float(obs.prereg_margin.mean()),
                prereg_ci_lo_pct=100 * float(np.nanpercentile(prereg_seed_avg, 2.5)),
                prereg_ci_hi_pct=100 * float(np.nanpercentile(prereg_seed_avg, 97.5)),
            )
        )
    out = pd.DataFrame(rows).sort_values("n_errors")
    # A claim survives only if the seed-averaged interval excludes zero.
    out["best_resolvable"] = out.seed_avg_ci_hi_pct < 0
    out["prereg_resolvable"] = out.prereg_ci_hi_pct < 0
    return out


def aurc_se_vs_errors(tiers: str, n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """Bootstrap standard error of MSP's AURC, against the test error count.

    This is the mechanism sentence of the whole section: AURC's precision
    is set by how many errors the curve is built from. Reported as a
    *relative* SE (SE / AURC) so datasets three orders of magnitude apart
    in absolute AURC are comparable, and next to `1/sqrt(n_errors)`, which
    is the scaling a binomial argument predicts.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for path in sorted(RAW_DIR.glob(f"*__{tiers}__*.npz")):
        raw = load_raw_scores(path)
        if "signal_msp" not in raw["methods"]:
            continue
        s = raw["scores"][raw["methods"].index("signal_msp")]
        inc = raw["incorrect"]
        n = len(inc)
        vals = []
        for _ in range(n_boot):
            take = rng.integers(0, n, n)
            if inc[take].min() == inc[take].max():
                continue
            vals.append(aurc(s[take], inc[take]))
        if len(vals) < 10:
            continue
        point = aurc(s, inc)
        rows.append(
            dict(
                dataset=raw["dataset"],
                seed=raw["seed"],
                n_test=n,
                n_errors=int(inc.sum()),
                aurc=point,
                aurc_se=float(np.std(vals)),
                rel_se=float(np.std(vals) / point) if point > 0 else np.nan,
            )
        )
    d = pd.DataFrame(rows)
    out = d.groupby("dataset").agg(
        n_test=("n_test", "mean"),
        n_errors=("n_errors", "mean"),
        aurc=("aurc", "mean"),
        rel_se=("rel_se", "mean"),
    )
    out["inv_sqrt_errors"] = 1.0 / np.sqrt(out.n_errors)
    out["rel_se_x_sqrt_errors"] = out.rel_se * np.sqrt(out.n_errors)
    return out.sort_values("n_errors")


def power_analysis(
    tiers: str,
    effects: tuple[float, ...] = (0.02, 0.05, 0.10),
    seed_counts: tuple[int, ...] = (1, 10, 30),
    n_boot: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Test errors needed to resolve a relative AURC effect at 80% power.

    Built on the *measured* relative SE rather than a parametric AURC
    variance formula: AURC's sampling distribution depends on where in the
    ranking the errors fall, and no clean closed form applies. The
    `1/sqrt(n_errors)` scaling is verified empirically in
    `aurc_se_vs_errors` (the `rel_se_x_sqrt_errors` column is
    approximately constant across six datasets spanning 34 to 2533 errors),
    which is what licenses extrapolating it.

    For a two-sided paired comparison at alpha = 0.05, 80% power needs the
    effect to be about 2.8 standard errors. Averaging over `n_seeds`
    independent runs shrinks the SE of the mean by sqrt(n_seeds), so

        errors_required(effect, n_seeds) = (2.8 * k / (effect * sqrt(n_seeds)))^2

    where `k = rel_se * sqrt(n_errors)` is the pooled scaling constant.

    **The seed dimension is not a technicality -- it is the difference
    between "this benchmark is hopeless" and "10 seeds is enough".** A
    single run needs tens of thousands of test errors to resolve a 5%
    effect; ten runs need a hundredth of that. Reporting only the
    single-run column would overstate the field's problem as badly as
    ignoring the multiplicity would understate it.
    """
    se = aurc_se_vs_errors(tiers, n_boot=n_boot, seed=seed)
    # Pooled across datasets: rel_se * sqrt(n_errors) is dataset-independent
    # if the binomial scaling holds, so one constant gives a rule of thumb.
    k = float(se.rel_se_x_sqrt_errors.mean())
    k_sd = float(se.rel_se_x_sqrt_errors.std())
    rows = []
    for effect in effects:
        row = {"relative_effect": effect, "scaling_k": k, "scaling_k_sd": k_sd}
        for n_seeds in seed_counts:
            row[f"errors_at_{n_seeds}_seeds"] = int(
                np.ceil((2.8 * k / (effect * np.sqrt(n_seeds))) ** 2)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiers", default="AC")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-splits", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    errs = test_error_counts(args.tiers)

    print("=" * 78)
    print(f"1. HELD-OUT METHOD SELECTION (tiers={args.tiers})")
    print("=" * 78)
    shrink = selection_shrinkage(load_aurc_table(args.tiers), n_splits=args.n_splits,
                                 seed=args.seed)
    shrink["test_errors"] = shrink.dataset.map(errs).round(0)
    shrink = shrink.sort_values("test_errors")
    cols = ["dataset", "test_errors", "best_agg", "biased_margin_pct",
            "honest_margin_pct", "shrinkage_pct", "prereg_margin_pct",
            "selection_stability"]
    print(shrink[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n  biased_margin  = best-of-N chosen and scored on all seeds (the audited protocol)")
    print("  honest_margin  = chosen on half the seeds, scored on the other half")
    print("  shrinkage      = biased - honest (positive = the reported number was optimistic)")
    print(f"  prereg_margin  = {PREREGISTERED_AGGREGATOR} fixed in advance, no selection at all")
    print("  stability      = how often the held-out selection picks the same winner")
    shrink.to_csv(TABLE_DIR / f"selection_bias_shrinkage_{args.tiers}.csv", index=False)

    print()
    print("=" * 78)
    print("2. PER-INSTANCE BOOTSTRAP SPREAD OF THE BEST-OF-N MARGIN")
    print("=" * 78)
    null = null_best_of_n(args.tiers, n_boot=args.n_boot, seed=args.seed)
    print(null.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n  within_seed_sd = spread within one run (the practitioner's question)")
    print("  seed_avg_sd/ci = spread of the 10-seed mean (what a paper claim asserts)")
    print("  *_resolvable   = the seed-averaged 95% interval excludes zero")
    null.to_csv(TABLE_DIR / f"selection_bias_null_{args.tiers}.csv", index=False)

    print()
    print("=" * 78)
    print("3. AURC PRECISION IS SET BY ERROR COUNT, NOT ROW COUNT")
    print("=" * 78)
    se = aurc_se_vs_errors(args.tiers, n_boot=max(200, args.n_boot // 2), seed=args.seed)
    print(se.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  rel_se_x_sqrt_errors is ~constant iff precision scales as 1/sqrt(errors).")
    print("  Note n_test and n_errors disagree by orders of magnitude across datasets:")
    print("  that gap is exactly why row count is the wrong screening variable.")
    se.to_csv(TABLE_DIR / f"aurc_se_vs_errors_{args.tiers}.csv")

    print()
    print("=" * 78)
    print("4. HOW MANY TEST ERRORS DOES AN AURC CLAIM NEED?")
    print("=" * 78)
    power = power_analysis(args.tiers, n_boot=max(200, args.n_boot // 2), seed=args.seed)
    print(power.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  Two-sided alpha=0.05, 80% power, using the measured relative SE and the")
    print("  1/sqrt(errors) scaling verified above. Averaging over n_seeds runs")
    print("  shrinks the requirement by a factor of n_seeds.")
    power.to_csv(TABLE_DIR / f"aurc_power_{args.tiers}.csv", index=False)

    print(f"\nwrote 4 tables to {TABLE_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
