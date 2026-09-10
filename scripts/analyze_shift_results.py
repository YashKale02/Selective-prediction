"""Analyse `results/shift_results.parquet`: RQ2, and the conformal
guarantee's decay under covariate shift.

Reads the parquet rather than re-running, so the same sweep can be
re-analysed without recomputation.

Three things are reported, and the first exists because of a trap worth
naming
-----------------------------------------------------------------------
**1. The aggregation gap, in absolute *and* relative terms.** Under shift
the base model degrades, so AURC itself grows -- on Satimage from
$0.009$ to $0.326$ between intensity 0 and 2. An absolute AURC gap that
"grows monotonically with shift" can therefore be entirely an artefact of
the denominator moving, and reporting only the absolute gap would let a
null masquerade as RQ2's confirmation. Both are printed side by side; the
relative column is the one that tests the hypothesis.

**2. `rho_local` against intensity.** The mechanism's own covariate. A
dataset pinned at `rho_local = 1.0000` (any binary task, by theorem)
cannot show an aggregation effect at any intensity, so a null there is
predictable in advance rather than evidence about shift.

**3. The conformal violation rate.** The threshold is calibrated on the
clean D_cal and applied to shifted test data, so a violation is
`realised selective risk > alpha`. Reported against the nominal `delta`,
and always alongside coverage: abstaining on everything cannot violate a
risk bound, so a violation rate read without coverage would score a
degenerate all-abstain solution as a success.

Usage:
    python scripts/analyze_shift_results.py
    python scripts/analyze_shift_results.py --dataset satimage
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHIFT_PATH = ROOT / "results" / "shift_results.parquet"
TABLE_DIR = ROOT / "paper" / "tables"

PREREGISTERED_AGGREGATOR = "A1_logreg"
NOMINAL_DELTA = 0.1


def _aggs(methods) -> list[str]:
    return [m for m in methods
            if m.startswith(("A0", "A1", "A2")) and not m.endswith("__conformal")]


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregation gap vs the best single signal, per (dataset, intensity)."""
    rows = []
    for (dataset, intensity), g in df.groupby(["dataset", "intensity"]):
        per_method = g.groupby("method")["aurc"].mean()
        sigs = [m for m in per_method.index if m.startswith("signal_")]
        aggs = _aggs(per_method.index)
        if not sigs or not aggs:
            continue
        best_sig = per_method[sigs].min()
        best_agg_name = per_method[aggs].idxmin()
        prereg = per_method.get(PREREGISTERED_AGGREGATOR, np.nan)
        rows.append(
            dict(
                dataset=dataset,
                intensity=intensity,
                base_error=g.base_error_rate.mean(),
                rho_local=g.rho_local.mean() if "rho_local" in g else np.nan,
                msp_aurc=per_method.get("signal_msp", np.nan),
                best_single_aurc=best_sig,
                best_agg=best_agg_name,
                # Absolute gaps move with AURC itself under shift, so the
                # relative columns are the ones that test RQ2.
                abs_gap_best=per_method[best_agg_name] - best_sig,
                rel_gap_best_pct=100 * (per_method[best_agg_name] - best_sig) / best_sig,
                abs_gap_prereg=prereg - best_sig,
                rel_gap_prereg_pct=100 * (prereg - best_sig) / best_sig,
                n_seeds=g.seed.nunique(),
            )
        )
    return pd.DataFrame(rows).sort_values(["dataset", "intensity"])


def conformal_table(df: pd.DataFrame) -> pd.DataFrame:
    if "conformal_violated" not in df.columns:
        return pd.DataFrame()
    conf = df[df.conformal_violated.notna()]
    if conf.empty:
        return pd.DataFrame()
    return (
        conf.groupby(["dataset", "conformal_alpha", "intensity"])
        .agg(violation_rate=("conformal_violated", "mean"),
             mean_coverage=("conformal_coverage", "mean"),
             mean_risk=("conformal_risk", "mean"),
             n=("conformal_violated", "size"))
        .reset_index()
    )


def report_rq2(gaps: pd.DataFrame) -> None:
    from scipy.stats import spearmanr

    print("=" * 84)
    print("RQ2: does the aggregation advantage grow with shift intensity?")
    print("=" * 84)
    for dataset, g in gaps.groupby("dataset"):
        g = g.sort_values("intensity")
        print(f"\n--- {dataset} ---")
        show = g[["intensity", "base_error", "rho_local", "msp_aurc",
                  "best_single_aurc", "abs_gap_best", "rel_gap_best_pct",
                  "rel_gap_prereg_pct"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        rho_pinned = (
            "rho_local" in g and np.allclose(g.rho_local.dropna(), 1.0, atol=1e-9)
        )
        if rho_pinned:
            print("  rho_local == 1.0000 at every intensity (binary task, by theorem):")
            print("  the Tier-A bank supplies ONE ordering in the abstention region, so")
            print("  no aggregator over it can help at any intensity. A null here is")
            print("  predicted in advance and is not evidence about shift.")

        if len(g) >= 4:
            abs_r = spearmanr(g.intensity, g.abs_gap_best)
            rel_r = spearmanr(g.intensity, g.rel_gap_best_pct)
            print(f"  Spearman(intensity, ABSOLUTE gap) = {abs_r.statistic:+.3f} "
                  f"(p={abs_r.pvalue:.4f})")
            print(f"  Spearman(intensity, RELATIVE gap) = {rel_r.statistic:+.3f} "
                  f"(p={rel_r.pvalue:.4f})")
            # RQ2 predicts the advantage grows, i.e. the (signed) gap becomes
            # more negative, i.e. a NEGATIVE correlation with intensity.
            if abs_r.statistic < -0.5 and rel_r.statistic > -0.5:
                print("  CAUTION: the absolute gap grows but the relative gap does not.")
                print("  Under shift AURC itself grows, so the absolute trend is largely")
                print("  a moving denominator. Report the relative column as the test.")
            elif rel_r.statistic < -0.5:
                print("  --> RQ2 supported: the gap grows in relative terms too.")
            else:
                print("  --> RQ2 not supported on this dataset.")


def report_signal_takeover(df: pd.DataFrame) -> None:
    """Which single signal is best at each intensity, and how the
    aggregators fare against it.

    This is the diagnostic that explains RQ2's failure rather than merely
    recording it. A post-hoc aggregator is fit on the *clean* meta split,
    so it learns the signal-informativeness profile of the clean
    distribution. If shift changes which signals are informative -- and it
    does, dramatically, with Tier-C geometry overtaking MSP -- then the
    learned weights are fit to a profile that no longer holds, and the
    aggregator can end up worse than either the best single signal or the
    parameter-free rank average. That is a failure of *learned
    aggregation under shift*, not of the signal bank, and the two have
    opposite fixes.
    """
    print("\n" + "=" * 84)
    print("Which signal wins under shift, and what the aggregators do about it")
    print("=" * 84)
    for dataset, g in df.groupby("dataset"):
        piv = g.groupby(["intensity", "method"])["aurc"].mean().unstack()
        sigs = [c for c in piv.columns if c.startswith("signal_")]
        if not sigs:
            continue
        report = pd.DataFrame(
            {
                "best_single": piv[sigs].idxmin(axis=1).str.replace("signal_", "", regex=False),
                "best_single_aurc": piv[sigs].min(axis=1),
                "msp_aurc": piv.get("signal_msp"),
                "A0_rank": piv.get("A0_rank"),
                PREREGISTERED_AGGREGATOR: piv.get(PREREGISTERED_AGGREGATOR),
            }
        )
        print(f"\n--- {dataset} ---")
        print(report.to_string(float_format=lambda v: f"{v:.4f}"))
        changed = report.best_single.nunique() > 1
        if changed:
            print("  The identity of the best single signal CHANGES with intensity:")
            print(f"    {' -> '.join(report.best_single.tolist())}")
            print("  The aggregators were fit on the clean meta split, where the")
            print("  eventual winner was not the informative signal. Compare the")
            print("  learned aggregator against the parameter-free rank average:")
            worst_i = report.index.max()
            r = report.loc[worst_i]
            if np.isfinite(r.get("A0_rank", np.nan)) and np.isfinite(
                r.get(PREREGISTERED_AGGREGATOR, np.nan)
            ):
                print(f"    at intensity {worst_i}: best single {r.best_single_aurc:.4f}, "
                      f"A0_rank {r.A0_rank:.4f}, "
                      f"{PREREGISTERED_AGGREGATOR} {r[PREREGISTERED_AGGREGATOR]:.4f}")


def report_conformal(conf: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print(f"Conformal risk control under shift (nominal delta = {NOMINAL_DELTA})")
    print("=" * 84)
    if conf.empty:
        print("No conformal rows. Re-run scripts/run_shift_sweep.py to populate them.")
        return
    for dataset, g in conf.groupby("dataset"):
        print(f"\n--- {dataset} ---")
        piv = g.pivot_table(index="conformal_alpha", columns="intensity",
                            values="violation_rate")
        cov = g.pivot_table(index="conformal_alpha", columns="intensity",
                            values="mean_coverage")
        print("  violation rate P(risk > alpha):")
        print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
        print("  mean coverage at the same thresholds:")
        print(cov.to_string(float_format=lambda v: f"{v:.3f}"))

        clean = g[g.intensity == 0]
        if not clean.empty:
            ok = clean.violation_rate.max() <= NOMINAL_DELTA + 1e-9
            print(f"  in-distribution (intensity 0) worst violation rate: "
                  f"{clean.violation_rate.max():.3f} "
                  f"{'-- within nominal delta, as claimed' if ok else '-- EXCEEDS delta'}")
        # A row with coverage 0 cannot violate a risk bound; flag it so a
        # trivially-satisfied guarantee is never read as a real one.
        degenerate = g[(g.mean_coverage < 1e-9)]
        if not degenerate.empty:
            alphas = sorted(degenerate.conformal_alpha.unique())
            print(f"  note: alpha in {alphas} abstains on everything (coverage 0), so its")
            print("  zero violation rate is vacuous, not a success.")
        shifted = g[g.intensity > 0]
        if not shifted.empty:
            worst = shifted.loc[shifted.violation_rate.idxmax()]
            print(f"  worst under shift: alpha={worst.conformal_alpha}, "
                  f"intensity={worst.intensity}, violation rate "
                  f"{worst.violation_rate:.3f} at coverage {worst.mean_coverage:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="", help="restrict to one dataset")
    p.add_argument("--kind", default="gaussian_noise")
    args = p.parse_args()

    if not SHIFT_PATH.exists():
        raise SystemExit(f"{SHIFT_PATH} not found; run scripts/run_shift_sweep.py")
    df = pd.read_parquet(SHIFT_PATH)
    df = df[df.shift_kind == args.kind]
    if args.dataset:
        df = df[df.dataset == args.dataset]
    if df.empty:
        raise SystemExit("no rows after filtering")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gaps = gap_table(df)
    conf = conformal_table(df)

    report_rq2(gaps)
    report_signal_takeover(df)
    report_conformal(conf)

    gaps.to_csv(TABLE_DIR / f"shift_gaps_{args.kind}.csv", index=False)
    print(f"\nwrote {(TABLE_DIR / f'shift_gaps_{args.kind}.csv').relative_to(ROOT)}")
    if not conf.empty:
        conf.to_csv(TABLE_DIR / f"shift_conformal_{args.kind}.csv", index=False)
        print(f"wrote {(TABLE_DIR / f'shift_conformal_{args.kind}.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
