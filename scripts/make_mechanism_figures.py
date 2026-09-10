"""Figures for the mechanism and benchmarking-power results
(`NOVELTY_ANALYSIS.md` §2, §3, §5).

Five figures, each carrying one claim:

  1. `redundancy_global_vs_local` -- global vs boundary-local Tier-A
     redundancy per dataset. The claim: measuring redundancy globally
     hides the structure AURC responds to. This is the paper's key
     diagnostic figure.
  2. `rho_local_vs_gain` -- rho_local against the realised aggregation
     gain, across datasets, with the K=2 group marked. Shows the
     separation *and* the confound honestly in one panel.
  3. `capacity_dose_response` -- the within-dataset decoupling
     experiment: rho_local vs gain at fixed K, across base-model
     capacity settings.
  4. `aurc_precision_vs_errors` -- relative AURC standard error against
     test error count, log-log, with the 1/sqrt(errors) reference. The
     claim: precision is set by error count, not row count.
  5. `conformal_violation_vs_shift` -- the conformal guarantee's
     empirical violation rate as shift intensity rises, against nominal
     delta.

Each figure is skipped with a message (not a crash) if its inputs are
absent, so this can be run at any point during a partial sweep.

Usage:
    python scripts/make_mechanism_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment.runner import load_raw_scores  # noqa: E402
from src.metrics.redundancy import local_rank_redundancy  # noqa: E402
from src.metrics.selective import aurc  # noqa: E402

RAW_DIR = ROOT / "results" / "raw"
FIG_DIR = ROOT / "paper" / "figures"
TABLE_DIR = ROOT / "paper" / "tables"
CAPACITY_PATH = ROOT / "results" / "capacity_sweep.parquet"
SHIFT_PATH = ROOT / "results" / "shift_results.parquet"

# The Tier-A sub-bank whose binary rank-degeneracy is proven. Named here
# rather than derived from the bank so the figure and the claim cannot
# drift apart if the bank grows a signal.
TIER_A = [
    "signal_msp", "signal_entropy", "signal_margin_prob", "signal_margin_logit",
    "signal_energy", "signal_temp_msp", "signal_logitnorm_msp",
]
PREREG = "A1_logreg"
Q = 0.10


def _tier_a_present(methods: list[str]) -> list[str]:
    return [m for m in TIER_A if m in methods]


def collect_per_dataset(tiers: str) -> pd.DataFrame:
    """rho_local (global and local) plus the realised gain, per dataset."""
    from scipy.stats import spearmanr

    rows = []
    for path in sorted(RAW_DIR.glob(f"*__{tiers}__*.npz")):
        raw = load_raw_scores(path)
        methods, scores, incorrect = raw["methods"], raw["scores"], raw["incorrect"]
        if "signal_msp" not in methods or PREREG not in methods:
            continue
        idx = {m: i for i, m in enumerate(methods)}
        bank = {m: scores[idx[m]] for m in _tier_a_present(methods)}
        if len(bank) < 2:
            continue

        local = local_rank_redundancy(bank, reference="signal_msp", q=Q,
                                      sub_bank=list(bank))
        msp = bank["signal_msp"]
        glob = min(
            abs(spearmanr(msp, v).statistic)
            for k, v in bank.items() if k != "signal_msp"
        )
        sigs = [m for m in methods if m.startswith("signal_")]
        best_single = min(aurc(scores[idx[m]], incorrect) for m in sigs)
        prereg = aurc(scores[idx[PREREG]], incorrect)
        rows.append(
            dict(
                dataset=raw["dataset"], seed=raw["seed"],
                rho_local=local.rho_local, rho_global=glob,
                n_errors=int(incorrect.sum()),
                gain_pct=100 * (prereg - best_single) / best_single,
            )
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby("dataset")
        .agg(rho_local=("rho_local", "mean"), rho_global=("rho_global", "mean"),
             n_errors=("n_errors", "mean"), gain_pct=("gain_pct", "mean"),
             gain_sd=("gain_pct", "std"), seeds=("seed", "count"))
        .reset_index()
    )


# Class counts, needed only to mark the confound in figure 2. Hard-coded
# rather than reloaded from OpenML: these are facts about the registry, and
# a figure should not need a network call.
K_BY_DATASET = {
    "adult": 2, "german_credit": 2, "electricity": 2,
    "satimage": 6, "wine_quality_white": 7, "letter": 26,
}


def fig_global_vs_local(df: pd.DataFrame) -> Path | None:
    if df.empty:
        print("[skip] fig 1: no raw scores")
        return None
    d = df.sort_values("rho_local")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.2, d.rho_global, width=0.4, label="global (whole test set)",
           color="#b3b3b3", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, d.rho_local, width=0.4,
           label=f"boundary-local (top {int(Q * 100)}% by MSP risk)",
           color="#7570b3", edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n(K={K_BY_DATASET.get(n, '?')})" for n in d.dataset], fontsize=8
    )
    ax.set_ylabel(r"min$_j$ |Spearman(MSP, $u_j$)| over Tier A")
    ax.set_ylim(0, 1.08)
    ax.set_title("Tier-A redundancy is global on binary tasks, and only global on multiclass ones")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    # The point of the figure: on the two datasets where aggregation works,
    # the global bar says "redundant" and the local bar says otherwise.
    for xi, (_, r) in zip(x, d.iterrows()):
        if r.rho_global - r.rho_local > 0.3:
            ax.annotate("", xy=(xi + 0.2, r.rho_local + 0.03),
                        xytext=(xi - 0.2, r.rho_global + 0.03),
                        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.4))
    fig.tight_layout()
    out = FIG_DIR / "redundancy_global_vs_local.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_rho_vs_gain(df: pd.DataFrame) -> Path | None:
    if df.empty:
        print("[skip] fig 2: no raw scores")
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        k = K_BY_DATASET.get(r.dataset, 0)
        binary = k == 2
        ax.errorbar(
            r.rho_local, r.gain_pct,
            yerr=(r.gain_sd / np.sqrt(r.seeds)) if np.isfinite(r.gain_sd) else None,
            marker="s" if binary else "o", markersize=9,
            color="#d95f02" if binary else "#1b9e77",
            capsize=3, linestyle="none",
        )
        ax.annotate(f"{r.dataset} (K={k})", (r.rho_local, r.gain_pct),
                    textcoords="offset points", xytext=(7, 5), fontsize=7.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\rho_{local}$ (Tier A, top 10% by MSP risk)")
    ax.set_ylabel(f"AURC gain of {PREREG} vs best single signal (%)\n(negative = aggregation helps)")
    ax.set_title("Where the signal bank is locally redundant, aggregation cannot help")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="s", color="#d95f02", linestyle="none",
               label=r"K = 2 ($\rho_{local}$ = 1 by theorem)"),
        Line2D([], [], marker="o", color="#1b9e77", linestyle="none", label="K >= 3"),
    ], fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    # Stated on the figure itself so it cannot be quoted without its caveat.
    ax.text(0.02, 0.02,
            "Confounded: every K=2 dataset is at 1.0 by theorem.\n"
            "See the capacity sweep for the fixed-K test.",
            transform=ax.transAxes, fontsize=7, style="italic",
            va="bottom", color="#444444")
    fig.tight_layout()
    out = FIG_DIR / "rho_local_vs_gain.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_capacity(path: Path = CAPACITY_PATH) -> Path | None:
    if not path.exists():
        print("[skip] fig 3: no capacity sweep yet")
        return None
    d = pd.read_parquet(path)
    datasets = sorted(d.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.3 * len(datasets), 4.4),
                             squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        g = d[d.dataset == dataset]
        k = int(g.K.iloc[0])
        m = g.groupby("capacity", sort=False).agg(
            rho=("rho_local_test", "mean"), gain=("gain_vs_best_single_pct", "mean"),
            gain_sd=("gain_vs_best_single_pct", "std"), n=("seed", "count"),
            acc=("base_acc", "mean"),
        )
        ax.errorbar(m.rho, m.gain, yerr=m.gain_sd / np.sqrt(m.n.clip(lower=1)),
                    marker="o", capsize=3, color="#1b9e77" if k > 2 else "#d95f02",
                    linewidth=1.2)
        for cap, r in m.iterrows():
            ax.annotate(f"{cap}\nacc={r.acc:.2f}", (r.rho, r.gain),
                        textcoords="offset points", xytext=(6, -4), fontsize=6.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(r"$\rho_{local}$")
        ax.set_ylabel("gain vs best single signal (%)")
        title = f"{dataset} (K={k}, fixed)"
        if k == 2:
            title += "\nCONTROL: rho pinned at 1.0"
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(r"Decoupling $\rho_{local}$ from K: base-model capacity swept at fixed K",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "capacity_dose_response.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_precision(tiers: str) -> Path | None:
    src = TABLE_DIR / f"aurc_se_vs_errors_{tiers}.csv"
    if not src.exists():
        print(f"[skip] fig 4: run analyze_selection_bias.py --tiers {tiers} first")
        return None
    d = pd.read_csv(src, index_col=0)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(d.n_errors, d.rel_se, s=70, color="#7570b3", zorder=3,
               edgecolor="black", linewidth=0.5)
    for name, r in d.iterrows():
        ax.annotate(f"{name}\n(n_test={int(r.n_test)})", (r.n_errors, r.rel_se),
                    textcoords="offset points", xytext=(8, -2), fontsize=7)
    grid = np.geomspace(d.n_errors.min() * 0.7, d.n_errors.max() * 1.4, 50)
    k = float((d.rel_se * np.sqrt(d.n_errors)).mean())
    ax.plot(grid, k / np.sqrt(grid), color="#d95f02", linestyle="--", linewidth=1.3,
            label=rf"$k/\sqrt{{n_{{errors}}}}$, $k$={k:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("test errors (not test rows)")
    ax.set_ylabel("relative bootstrap SE of AURC")
    ax.set_title("AURC precision is governed by error count, not sample size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG_DIR / "aurc_precision_vs_errors.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_conformal(path: Path = SHIFT_PATH) -> Path | None:
    if not path.exists():
        print("[skip] fig 5: no shift results")
        return None
    d = pd.read_parquet(path)
    if "conformal_violated" not in d.columns or d.conformal_violated.isna().all():
        print("[skip] fig 5: shift results predate conformal tracking; re-run the sweep")
        return None
    conf = d[d.conformal_violated.notna()]
    datasets = sorted(conf.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 4.4),
                             squeeze=False, sharey=True)
    # A threshold that accepts nothing cannot violate a risk bound, so its
    # 0.0 violation rate is vacuous rather than a success. Without this
    # distinction german_credit -- where every alpha abstains on 100% of
    # inputs -- reads as "the guarantee held under shift", which is the
    # opposite of what happened. Vacuous points are drawn hollow and the
    # curve dashed.
    VACUOUS_COVERAGE = 1e-6
    for ax, dataset in zip(axes[0], datasets):
        g = conf[conf.dataset == dataset]
        for alpha in sorted(g.conformal_alpha.unique()):
            ga = g[g.conformal_alpha == alpha]
            viol = ga.groupby("intensity")["conformal_violated"].mean()
            cov = ga.groupby("intensity")["conformal_coverage"].mean()
            vacuous = cov <= VACUOUS_COVERAGE
            line, = ax.plot(viol.index, viol.values,
                            linestyle=":" if vacuous.all() else "-",
                            linewidth=1.2, label=rf"$\alpha$={alpha}", zorder=2)
            colour = line.get_color()
            ax.scatter(viol.index[~vacuous], viol.values[~vacuous.to_numpy()],
                       s=42, color=colour, zorder=3)
            ax.scatter(viol.index[vacuous], viol.values[vacuous.to_numpy()],
                       s=42, facecolors="none", edgecolors=colour, zorder=3)
        ax.axhline(0.1, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("shift intensity")
        n_vac = int((g.groupby("conformal_alpha")["conformal_coverage"].mean()
                     <= VACUOUS_COVERAGE).sum())
        ax.set_title(f"{dataset}\n({n_vac} of "
                     f"{g.conformal_alpha.nunique()} "
                     r"$\alpha$ abstain on everything)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    axes[0][0].set_ylabel(r"empirical $P(\mathrm{risk} > \alpha)$")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="black", linestyle="--", label=r"nominal $\delta$=0.1"),
               Line2D([], [], marker="o", color="grey", linestyle="none",
                      label="coverage > 0 (real)"),
               Line2D([], [], marker="o", color="grey", linestyle="none",
                      markerfacecolor="none", label="coverage = 0 (vacuous)")]
    axes[0][-1].legend(handles=handles, fontsize=7, loc="center right")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Conformal risk control under covariate shift: measured violation rate",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "conformal_violation_vs_shift.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiers", default="AC")
    args = p.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    per_dataset = collect_per_dataset(args.tiers)
    if not per_dataset.empty:
        out = TABLE_DIR / f"mechanism_per_dataset_{args.tiers}.csv"
        per_dataset.to_csv(out, index=False)
        print(f"\n=== rho_local vs realised gain (tiers={args.tiers}) ===")
        print(per_dataset.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"wrote {out.relative_to(ROOT)}")

    written = [
        fig_global_vs_local(per_dataset),
        fig_rho_vs_gain(per_dataset),
        fig_capacity(),
        fig_precision(args.tiers),
        fig_conformal(),
    ]
    print()
    for path in written:
        if path is not None:
            print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
