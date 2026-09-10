"""Boundary-local rank redundancy of the signal bank (NOVELTY_ANALYSIS.md §3).

The project's falsification of "K >= 3 breaks the Tier-A degeneracy" rested
on a *global* rank correlation: on satimage, |Spearman(entropy, MSP)| over
the whole test set is 1.0000, so the bank looked redundant. But AURC is a
ranking functional -- it integrates risk over coverage, so a swap between
ranks 3 and 4 moves it far more than a swap between ranks 3000 and 3001.
The redundancy that matters is therefore redundancy *restricted to the
points that are candidates for abstention*, and a global statistic answers
a question AURC never asks.

Define, for an evaluation set and the MSP risk ranking M:

    Q_q            = the top-q fraction of points by M (the abstention region)
    rho_local(q)   = min over j in TierA (excluding msp) of |Spearman(u_msp, u_j)| on Q_q

Measured at q = 0.10 on the committed runs, this separates the six datasets
perfectly and with no seed overlap: exactly 1.0000 on all three where
aggregation hurts, <= 0.53 on all three where it helps -- while the global
statistic reads 0.86-0.89 on two of the "helps" datasets and so hides the
effect entirely.

Usage:
    python scripts/analyze_local_redundancy.py [--q 0.10] [--tiers ABC]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Tier A = the logit-derived bank. On binary input these are provably a
# single ranking (see src/signals/tier_a.py); the point of rho_local is
# that on multiclass input they stay near-degenerate *globally* while
# decorrelating sharply inside the abstention region.
TIER_A = {
    "signal_msp",
    "signal_entropy",
    "signal_margin_prob",
    "signal_margin_logit",
    "signal_energy",
    "signal_temp_msp",
    "signal_logitnorm_msp",
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def effective_rank(ranks: np.ndarray) -> float:
    """Entropy-based effective rank of a rank-transformed signal matrix.

    `ranks` is (m signals, n points). Standardising rows and taking the
    Gram matrix gives the rank correlation matrix; the exponentiated
    Shannon entropy of its normalised eigenvalues counts how many
    *independent* orderings the bank really supplies -- a continuous
    stand-in for "how many distinct ways can this bank sort the data".
    Reported alongside rho_local because it uses the whole bank rather
    than Tier A alone, but it separates the datasets by a much thinner
    margin, so rho_local is the statistic to prefer.
    """
    R = (ranks - ranks.mean(axis=1, keepdims=True)) / (ranks.std(axis=1, keepdims=True) + 1e-12)
    corr = R @ R.T / R.shape[1]
    ev = np.clip(np.linalg.eigvalsh(corr), 1e-12, None)
    ev = ev / ev.sum()
    return float(np.exp(-(ev * np.log(ev)).sum()))


def analyse_one(path: str, q: float) -> dict | None:
    z = np.load(path, allow_pickle=True)
    methods = list(z["methods"])
    scores = z["scores"]
    index = {name: i for i, name in enumerate(methods)}
    if "signal_msp" not in index:
        return None

    msp = scores[index["signal_msp"]]
    # The abstention region: the top-q riskiest points by MSP. Floored at
    # 30 points so a small test split (german_credit's ~150 rows) still
    # yields a rank correlation that means anything.
    k = max(30, int(q * len(msp)))
    sel = np.argsort(-msp)[:k]

    signals = [m for m in methods if m.startswith("signal_")]
    ranks = np.vstack([pd.Series(scores[index[s]][sel]).rank().to_numpy() for s in signals])

    others = [s for s in signals if s in TIER_A and s != "signal_msp"]
    local = [abs(spearmanr(msp[sel], scores[index[s]][sel]).statistic) for s in others]
    glob_ = [abs(spearmanr(msp, scores[index[s]]).statistic) for s in others]

    return {
        "dataset": str(z["dataset"]),
        "seed": int(z["seed"]),
        "n_test": len(msp),
        "n_region": k,
        "rho_local": float(np.nanmin(local)),
        "rho_global": float(np.nanmin(glob_)),
        "eff_rank_local": effective_rank(ranks),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.10, help="abstention-region fraction")
    ap.add_argument("--tiers", default="ABC", help="tier config to read from results/raw")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(REPO, "results", "raw", f"*{args.tiers}*.npz")))
    if not paths:
        raise SystemExit(f"no raw score files for tiers={args.tiers!r} in results/raw/")

    rows = [r for r in (analyse_one(p, args.q) for p in paths) if r is not None]
    df = pd.DataFrame(rows)

    summary = df.groupby("dataset").agg(
        seeds=("seed", "count"),
        rho_local=("rho_local", "mean"),
        rho_local_min=("rho_local", "min"),
        rho_local_max=("rho_local", "max"),
        rho_global=("rho_global", "mean"),
        eff_rank_local=("eff_rank_local", "mean"),
    )
    print(f"Boundary-local rank redundancy, q={args.q}, tiers={args.tiers}\n")
    print(summary.round(4).sort_values("rho_local").to_string())
    print(
        "\nrho_local == 1.0000 means the Tier-A bank supplies one ranking inside the\n"
        "abstention region, so no aggregator over it can reorder abstentions. Compare\n"
        "the rho_global column: it is what made this effect invisible."
    )

    out = os.path.join(REPO, "paper", "tables", f"local_redundancy_{args.tiers}.csv")
    summary.to_csv(out)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
