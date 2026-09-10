"""RQ2 shift-intensity sweep: does the aggregation advantage over MSP grow
as covariate shift intensifies?

This is the sharpest available form of RQ2. "Does aggregation beat MSP on a
second shifted dataset?" is one noisy comparison; "does the gap over MSP
grow monotonically with a shift we control" is a *trend* with a
pre-specified direction, and a null there is far more informative than a
null on one more dataset.

Protocol (matching the temporal setting's logic): fit everything -- base
model, cross-fitted signals, aggregators, conformal threshold -- on the
**clean** train/meta/cal splits exactly once, then evaluate every method on
D_test perturbed at increasing intensity. Nothing is refit per intensity,
because that is the deployment situation being modelled: the system was
built before the shift arrived.

Writes long-format rows to `results/shift_results.parquet` (kept separate
from `results.parquet`, whose rows are all clean-test-set numbers -- mixing
them would silently pool two different experiments) and a figure of
AURC-delta-vs-MSP against intensity.

Two things this sweep now also measures, both previously missing
(`NOVELTY_ANALYSIS.md` §5, N5)
-----------------------------------------------------------------------
**1. The conformal violation rate.** `src/conformal/risk_control.py` and
the paper both state that exchangeability fails under shift, so the
distribution-free guarantee `P(selective risk <= alpha) >= 1 - delta` is
valid in-distribution only. Neither has ever measured *by how much* it
fails. The threshold `tau_hat` is chosen on the **clean** D_cal (as it
would be in deployment, before the shift arrives) and then applied to the
shifted D_test; a violation is `realised risk > alpha`. Averaged over
seeds this is an empirical violation rate, directly comparable to the
nominal `delta`. A guarantee claimed but never checked is the kind of thing
a reviewer tests first.

**2. `rho_local` under shift.** The mechanism in §3 predicts *where*
aggregation can help. If shift changes the signal bank's boundary-local
redundancy, that is the mechanism's own account of why the aggregation gap
moves with intensity -- and if `rho_local` stays pinned at 1.0000 (as it
must on any binary dataset, by theorem), then a null result under shift was
predictable before the sweep was run, which is exactly what happened on
german_credit.

Usage:
    python scripts/run_shift_sweep.py --dataset adult --n-seeds 5
    python scripts/run_shift_sweep.py --dataset adult --kind subpopulation
    python scripts/run_shift_sweep.py --dataset satimage --n-seeds 10
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

from src.aggregators.a0_rank import RankAverageAggregator  # noqa: E402
from src.aggregators.a1_stacking import (  # noqa: E402
    LightGBMStackingAggregator,
    LogRegStackingAggregator,
)
from src.aggregators.a2_coverage_loss import (  # noqa: E402
    AdaptiveGatingAggregator,
    MLPAggregator,
)
from src.data import splits as split_utils  # noqa: E402
from src.data.loaders import load as load_dataset  # noqa: E402
from src.conformal.risk_control import coverage_at_guaranteed_risk_table  # noqa: E402
from src.data.shift import apply_covariate_shift  # noqa: E402
from src.experiment.runner import (  # noqa: E402
    HELD_OUT_FIT_SIGNALS,
    _build_signal_bank,
    _fit_model,
)
from src.experiment.seeding import set_seed  # noqa: E402
from src.metrics.redundancy import local_rank_redundancy  # noqa: E402
from src.metrics.selective import aurc  # noqa: E402
from src.signals.tier_a import (  # noqa: E402
    CalibrationResidualSignal,
    TemperatureScaledMSPSignal,
    default_tier_a_bank,
)

RESULTS_PATH = ROOT / "results" / "shift_results.parquet"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_INTENSITIES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)

# Risk levels for the conformal guarantee, matching the main runner so the
# in-distribution and under-shift numbers are directly comparable.
CONFORMAL_ALPHAS = (0.01, 0.02, 0.05, 0.10)
CONFORMAL_DELTA = 0.1


def run_one_seed(
    dataset_name: str,
    seed: int,
    kind: str,
    intensities: tuple[float, ...],
    model_name: str = "lightgbm",
    tiers: tuple[str, ...] = ("A", "C"),
    n_cv_folds: int = 5,
) -> list[dict]:
    set_seed(seed)
    ds = load_dataset(dataset_name)
    sp = split_utils.four_way_split(len(ds.y), y=ds.y, seed=seed, temporal=ds.is_temporal)

    X_train, y_train = ds.X.iloc[sp.train_idx], ds.y[sp.train_idx]
    X_meta, y_meta = ds.X.iloc[sp.meta_idx], ds.y[sp.meta_idx]
    # D_cal stays CLEAN: the conformal threshold is calibrated before the
    # shift arrives, which is the deployment situation whose guarantee is
    # under test. Calibrating on shifted data would be a different (and
    # unavailable) experiment.
    X_cal, y_cal = ds.X.iloc[sp.cal_idx], ds.y[sp.cal_idx]
    X_test, y_test = ds.X.iloc[sp.test_idx], ds.y[sp.test_idx]
    pool_idx = np.concatenate([sp.train_idx, sp.meta_idx])
    X_pool, y_pool = ds.X.iloc[pool_idx], ds.y[pool_idx]
    n_classes = int(len(ds.classes))

    # --- fit on CLEAN data only (see module docstring)
    model_train = _fit_model(model_name, X_train, y_train, seed=seed)
    held_out = [
        TemperatureScaledMSPSignal().fit(X_meta, y_meta, model_train),
        CalibrationResidualSignal().fit(X_meta, y_meta, model_train),
    ]
    held_out_names = [s.name for s in held_out]

    def fit_model_fn(idx_abs: np.ndarray, fold: int = 0):
        Xi, yi = ds.X.iloc[idx_abs], ds.y[idx_abs]
        m = _fit_model(model_name, Xi, yi, seed=seed * 1000 + fold)
        bank = _build_signal_bank(tiers, n_classes=n_classes)
        bank.fit(Xi, yi, m)
        return {"model": m, "bank": bank}

    def compute_signals_fn(bundle: dict, idx_abs: np.ndarray) -> np.ndarray:
        Xi = ds.X.iloc[idx_abs]
        u = bundle["bank"].transform(Xi, bundle["model"])
        pred = bundle["model"].predict_proba(Xi).argmax(axis=1)
        correct = (pred == ds.y[idx_abs]).astype(float).reshape(-1, 1)
        return np.hstack([u, correct])

    oof = split_utils.cross_fitted_signals(
        fit_model_fn, compute_signals_fn, pool_idx,
        y_pool=y_pool, n_folds=n_cv_folds, seed=seed,
    )
    U_pool, correct_pool = oof[:, :-1], oof[:, -1].astype(bool)
    is_meta = np.isin(pool_idx, sp.meta_idx)
    U_meta_cf, correct_meta = U_pool[is_meta], correct_pool[is_meta]

    model_final = _fit_model(model_name, X_pool, y_pool, seed=seed)
    final_bank = _build_signal_bank(tiers, n_classes=n_classes)
    final_bank.fit(X_pool, y_pool, model_final)
    signal_names = list(final_bank.names) + held_out_names

    def score_all_signals(X: pd.DataFrame) -> np.ndarray:
        cols = [final_bank.transform(X, model_final)]
        cols += [s.score(X, model_final).reshape(-1, 1) for s in held_out]
        return np.hstack(cols)

    U_meta_full = np.hstack(
        [U_meta_cf] + [s.score(X_meta, model_train).reshape(-1, 1) for s in held_out]
    )

    aggregators = {
        "A0_rank": RankAverageAggregator(),
        "A1_logreg": LogRegStackingAggregator(seed=seed),
        "A1_lightgbm": LightGBMStackingAggregator(seed=seed),
        "A2_mlp_bce": MLPAggregator(loss="bce", seed=seed),
        "A2_mlp_loss1": MLPAggregator(loss="loss1", seed=seed),
        "A2_mlp_loss2": MLPAggregator(loss="loss2", seed=seed),
        "A2_mlp_loss3": MLPAggregator(loss="loss3", seed=seed),
        "A2_adaptive_loss3": AdaptiveGatingAggregator(loss="loss3", seed=seed),
    }
    for agg in aggregators.values():
        agg.fit(U_meta_full, correct_meta)

    # --- conformal thresholds, calibrated on the clean D_cal
    U_cal = score_all_signals(X_cal)
    pred_cal = model_final.predict_proba(X_cal).argmax(axis=1)
    incorrect_cal = (pred_cal != y_cal).astype(int)
    conformal_tau: dict[str, dict[float, float]] = {}
    for nm, agg in aggregators.items():
        table = coverage_at_guaranteed_risk_table(
            agg.score(U_cal), incorrect_cal,
            alphas=CONFORMAL_ALPHAS, delta=CONFORMAL_DELTA,
        )
        conformal_tau[nm] = {a: ct.tau for a, ct in table.items()}

    # Tier-A names for rho_local: the sub-bank whose binary rank-degeneracy
    # is proven, matching src/metrics/redundancy.py's headline statistic.
    tier_a_names = [
        sig.name for sig in default_tier_a_bank(n_classes=n_classes)
        if not isinstance(sig, HELD_OUT_FIT_SIGNALS)
    ]

    rows = []
    for intensity in intensities:
        # One draw per (seed, intensity); the returned index (subpopulation
        # only) must be applied to y as well -- apply_covariate_shift
        # returns both so they cannot desynchronise.
        rng = np.random.default_rng(seed * 7919 + int(intensity * 1000))
        X_sh, idx = apply_covariate_shift(
            X_test, kind=kind, intensity=intensity, rng=rng, reference=X_train
        )
        y_sh = y_test if idx is None else y_test[idx]

        U_sh = score_all_signals(X_sh)
        pred_sh = model_final.predict_proba(X_sh).argmax(axis=1)
        incorrect_sh = (pred_sh != y_sh).astype(int)
        base_err = float(incorrect_sh.mean())

        shifted_signals = {nm: U_sh[:, j] for j, nm in enumerate(signal_names)}
        red = local_rank_redundancy(
            shifted_signals, reference="msp", q=0.10, sub_bank=tier_a_names
        )

        def _row(method: str, score: np.ndarray, **extra) -> dict:
            row = dict(
                dataset=dataset_name, base_model=model_name,
                tiers="".join(sorted(tiers)), shift_kind=kind,
                intensity=intensity, seed=seed, method=method,
                aurc=float(aurc(score, incorrect_sh)),
                base_error_rate=base_err, n_test=int(len(incorrect_sh)),
                # The mechanism's own covariate, tracked alongside the
                # outcome so a null can be attributed rather than guessed
                # at. Constant 1.0000 on any binary dataset, by theorem.
                rho_local=red.rho_local,
                conformal_alpha=np.nan, conformal_tau=np.nan,
                conformal_coverage=np.nan, conformal_risk=np.nan,
                conformal_violated=np.nan,
            )
            row.update(extra)
            return row

        for j, nm in enumerate(signal_names):
            rows.append(_row(f"signal_{nm}", U_sh[:, j]))
        for nm, agg in aggregators.items():
            s_sh = agg.score(U_sh)
            rows.append(_row(nm, s_sh))
            # Apply the clean-calibrated threshold to the shifted test set.
            # A violation is `realised selective risk > alpha`; with
            # coverage 0 (tau = -inf, no threshold satisfied the bound on
            # D_cal) the system abstains on everything, which cannot
            # violate a risk bound and is recorded as such rather than as
            # a NaN that would quietly drop out of the mean.
            for alpha, tau in conformal_tau[nm].items():
                accepted = s_sh <= tau
                n_acc = int(accepted.sum())
                realised = float(incorrect_sh[accepted].mean()) if n_acc else 0.0
                rows.append(
                    _row(
                        f"{nm}__conformal", s_sh,
                        conformal_alpha=alpha, conformal_tau=float(tau),
                        conformal_coverage=n_acc / len(s_sh),
                        conformal_risk=realised,
                        conformal_violated=float(realised > alpha),
                    )
                )
        rows.append(_row("oracle", incorrect_sh.astype(float)))
        rows.append(_row("random", np.random.default_rng(seed).random(len(incorrect_sh))))
    return rows


def plot_gap_vs_intensity(df: pd.DataFrame, dataset: str, kind: str) -> Path:
    """AURC(method) - AURC(MSP) against shift intensity. RQ2 predicts this
    trends *downward* (the aggregator's advantage grows) for at least some
    aggregator; a flat or upward line is the null."""
    base = (
        df[df.method == "signal_msp"]
        .groupby("intensity")["aurc"].mean()
        .rename("msp_aurc")
    )
    methods = [m for m in df.method.unique() if m.startswith(("A0", "A1", "A2"))]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in sorted(methods):
        g = df[df.method == m].groupby("intensity")["aurc"].mean()
        gap = (g - base).reindex(sorted(df.intensity.unique()))
        ax.plot(gap.index, gap.values, marker="o", label=m, linewidth=1.4)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"{kind} shift intensity")
    ax.set_ylabel("AURC(method) - AURC(MSP)   (< 0 = beats MSP)")
    ax.set_title(f"RQ2: aggregation gap vs. shift intensity -- {dataset} ({kind})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"shift_gap_{dataset}_{kind}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def report_conformal_violations(df: pd.DataFrame) -> None:
    """Empirical violation rate of the conformal risk guarantee, by shift
    intensity, against the nominal `delta`.

    The guarantee is `P(selective risk <= alpha) >= 1 - delta`, so at
    delta = 0.1 a violation rate above ~0.10 means exchangeability has
    broken badly enough to void the bound in practice -- which the code
    predicts under shift but has never measured. At intensity 0 the shift
    is a no-op, so that row doubles as an in-distribution sanity check:
    the rate there should sit at or below delta.

    `coverage` is reported alongside because a bound can also be "kept"
    trivially by abstaining on everything (coverage 0 cannot violate a
    risk bound), and a violation rate read without coverage would make
    that degenerate case look like a success.
    """
    conf = df[df.method.str.endswith("__conformal") & df.conformal_alpha.notna()]
    if conf.empty:
        print("\n=== conformal violations: no rows (re-run this sweep to populate) ===")
        return
    print("\n=== conformal violation rate under shift (nominal delta = "
          f"{CONFORMAL_DELTA}) ===")
    tab = (
        conf.groupby(["conformal_alpha", "intensity"])
        .agg(violation_rate=("conformal_violated", "mean"),
             mean_coverage=("conformal_coverage", "mean"),
             mean_risk=("conformal_risk", "mean"),
             n=("conformal_violated", "size"))
        .reset_index()
    )
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    worst = tab.loc[tab.violation_rate.idxmax()]
    print(f"  worst: alpha={worst.conformal_alpha}, intensity={worst.intensity}, "
          f"violation rate {worst.violation_rate:.3f} vs nominal {CONFORMAL_DELTA}")


def report_rho_local(df: pd.DataFrame) -> None:
    """Boundary-local Tier-A redundancy against shift intensity.

    On a binary dataset this is 1.0000 at every intensity by theorem, and
    printing it makes the "this dataset could not have shown an effect"
    reading explicit rather than an argument made after the fact."""
    if "rho_local" not in df.columns or df.rho_local.isna().all():
        return
    r = df.groupby("intensity")["rho_local"].mean()
    print("\n=== rho_local (Tier A, q=0.10) vs shift intensity ===")
    for i, v in r.items():
        print(f"  intensity={i:<6} rho_local={v:.4f}")
    if np.allclose(r.to_numpy(), 1.0, atol=1e-9):
        print("  Pinned at 1.0000: the Tier-A bank supplies ONE ordering in the")
        print("  abstention region at every intensity, so no aggregator over it can")
        print("  reorder abstentions. A null result here was predictable in advance.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="adult")
    p.add_argument("--kind", default="gaussian_noise",
                   help="gaussian_noise | feature_scale | subpopulation")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--intensities", default=",".join(str(i) for i in DEFAULT_INTENSITIES))
    args = p.parse_args()

    intensities = tuple(float(x) for x in args.intensities.split(","))
    all_rows: list[dict] = []
    for seed in range(args.n_seeds):
        print(f"[shift] {args.dataset} kind={args.kind} seed={seed}", flush=True)
        all_rows += run_one_seed(args.dataset, seed, args.kind, intensities)

    new = pd.DataFrame(all_rows)
    key = ["dataset", "base_model", "tiers", "shift_kind", "seed"]
    if RESULTS_PATH.exists():
        old = pd.read_parquet(RESULTS_PATH)
        mask = old.set_index(key).index.isin(new.set_index(key).index)
        new = pd.concat([old[~mask], new], ignore_index=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    new.to_parquet(RESULTS_PATH)
    print(f"[shift] wrote {len(new)} rows -> {RESULTS_PATH}")

    sub = new[(new.dataset == args.dataset) & (new.shift_kind == args.kind)]
    fig_path = plot_gap_vs_intensity(sub, args.dataset, args.kind)
    print(f"[shift] wrote {fig_path}")

    report_conformal_violations(sub)
    report_rho_local(sub)

    print("\n=== base error rate and best aggregator gap vs. MSP, by intensity ===")
    msp = sub[sub.method == "signal_msp"].groupby("intensity")["aurc"].mean()
    aggs = sub[sub.method.str.startswith(("A0", "A1", "A2"))]
    for i in sorted(sub.intensity.unique()):
        err = sub[sub.intensity == i]["base_error_rate"].mean()
        gaps = aggs[aggs.intensity == i].groupby("method")["aurc"].mean() - msp.loc[i]
        best = gaps.idxmin()
        print(f"  intensity={i:<5} base_err={err:.4f}  msp_aurc={msp.loc[i]:.4f}  "
              f"best={best} gap={gaps.min():+.5f}")


if __name__ == "__main__":
    main()
