"""The rho_local / K decoupling experiment, run *within* a dataset.

Why this experiment exists
--------------------------
`NOVELTY_ANALYSIS.md` §3 shows `rho_local` separates all six registered
datasets perfectly, but the three where aggregation helps are all K >= 6
and the three where it hurts are all K = 2. So `rho_local` is confounded
with class count, and the mechanism claim is not yet earned.

The obvious fix -- find a K >= 3 dataset that is nevertheless locally
redundant -- was attempted first and **failed for a structural reason**,
which is itself worth reporting. `scripts/screen_redundancy_candidates.py`
screened 16 multiclass OpenML candidates and found the two requirements to
be in direct tension:

    candidates with >= 50 test errors : 5, max rho_local  = 0.458
    candidates with rho_local >= 0.6  : 4, max test errors =     9

That is not bad luck. Local redundancy tracks base *accuracy*
(Spearman 0.50 across the 16, against only 0.25 for class count): an
accurate model's abstention region contains a handful of near-binary
confusions, so the Tier-A bank stays redundant there -- and an accurate
model also produces almost no test errors, so its risk-coverage curve
cannot resolve anything. The very property that makes a dataset locally
redundant is the property that makes it unmeasurable.

So the confound cannot be broken by choosing a different dataset. It can be
broken by holding the dataset -- and therefore K -- **fixed**, and moving
`rho_local` with something else.

The design
----------
Base-model capacity is that something else. Weakening the base model (fewer
trees, shallower, less data) raises its error rate, which widens the
abstention region and pulls more classes into contention there, which
should *lower* `rho_local`. K never changes. Neither does the dataset, the
split protocol, the signal bank, or the aggregator.

That converts the confounded between-dataset comparison into a within-
dataset dose-response, and the mechanism makes a directional prediction
that class count cannot make at all:

    H1: within one K >= 3 dataset, the aggregation gain over the best
        single signal tracks rho_local across capacity settings.

    H0 (the confound): rho_local is a proxy for K, so at fixed K the gain
        should be flat in rho_local.

`K >= 3` matters here because at `K = 2` the theorem forces
`rho_local == 1.0000` at every capacity, so the sweep has no dose to
respond to. That makes a binary dataset the ideal **control**, and
`--dataset` accepts one: if `rho_local` moves at all on a binary task, the
implementation is wrong, not the theory.

Protocol notes
--------------
Everything except base-model capacity is held to the main runner's
protocol: the same four-way split, the same cross-fitted honest meta
signals, the same aggregators. Capacity is varied through
`LightGBMWrapper`'s kwargs and, for the weakest settings, by subsampling
D_train -- the runner's `_fit_model` is bypassed for exactly this reason
and no other.

Reported per (capacity, seed): base accuracy, test error count,
`rho_local`, `delta_local` (the labelled Part-2 diagnostic, computed on
D_meta so it stays a pre-deployment quantity), and the realised AURC gain
of the pre-registered aggregator against the best single signal.

Usage:
    python scripts/run_capacity_sweep.py --dataset satimage --n-seeds 5
    python scripts/run_capacity_sweep.py --dataset adult --n-seeds 3   # control
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aggregators.a0_rank import RankAverageAggregator  # noqa: E402
from src.aggregators.a1_stacking import (  # noqa: E402
    LightGBMStackingAggregator,
    LogRegStackingAggregator,
)
from src.data import splits as split_utils  # noqa: E402
from src.data.loaders import load as load_dataset  # noqa: E402
from src.experiment.runner import HELD_OUT_FIT_SIGNALS, _build_signal_bank  # noqa: E402
from src.experiment.seeding import set_seed  # noqa: E402
from src.metrics.redundancy import (  # noqa: E402
    effective_binarity,
    incremental_boundary_auroc,
    local_rank_redundancy,
)
from src.metrics.selective import aurc  # noqa: E402
from src.models.lightgbm_model import LightGBMWrapper  # noqa: E402
from src.signals.tier_a import (  # noqa: E402
    CalibrationResidualSignal,
    TemperatureScaledMSPSignal,
    default_tier_a_bank,
)

RESULTS_PATH = ROOT / "results" / "capacity_sweep.parquet"
TABLE_DIR = ROOT / "paper" / "tables"
FIG_DIR = ROOT / "paper" / "figures"

# Capacity ladder, weakest first. `train_frac` subsamples D_train, which is
# the only lever strong enough to make an accurate multiclass model
# genuinely error-prone -- tree count and depth alone bottom out well
# before base accuracy moves far on an easy dataset.
#
# Named settings rather than a numeric grid so a row in the output table
# says what was actually done to the model.
CAPACITY_LADDER: list[tuple[str, dict]] = [
    ("tiny", dict(n_estimators=3, num_leaves=2, max_depth=1, train_frac=0.02)),
    ("very_weak", dict(n_estimators=5, num_leaves=3, max_depth=2, train_frac=0.05)),
    ("weak", dict(n_estimators=15, num_leaves=4, max_depth=2, train_frac=0.15)),
    ("medium", dict(n_estimators=50, num_leaves=8, max_depth=3, train_frac=0.40)),
    ("strong", dict(n_estimators=150, num_leaves=31, max_depth=-1, train_frac=1.00)),
    ("full", dict(n_estimators=300, num_leaves=31, max_depth=-1, train_frac=1.00)),
]

# Fixed in advance, matching scripts/analyze_selection_bias.py. The whole
# point of this sweep is a dose-response in one pre-committed quantity, so
# taking a best-of-N per capacity setting would reintroduce exactly the
# selection artefact §2 is about.
PREREGISTERED_AGGREGATOR = "A1_logreg"


def _fit_capacity_model(X, y, seed: int, n_classes: int, params: dict):
    """Fit a LightGBM base model at a given point on the capacity ladder.

    `train_frac` subsamples the rows *stratified by label*, so a weak
    setting reduces the amount of evidence without also changing the class
    balance the model is asked to learn -- an unstratified draw at
    train_frac=0.02 can drop a rare class entirely and turn a K-class
    problem into a smaller one, which would confound the very quantity
    being measured.
    """
    params = dict(params)
    train_frac = params.pop("train_frac", 1.0)
    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        keep: list[np.ndarray] = []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            # At least 2 rows per class so every class survives and the
            # objective stays K-class.
            k = max(2, int(round(train_frac * len(idx))))
            keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        sel = np.sort(np.concatenate(keep))
        X, y = X.iloc[sel], y[sel]
    return LightGBMWrapper(seed=seed, n_classes=n_classes, **params).fit(X, y), len(y)


def run_one(dataset_name: str, seed: int, capacity: str, params: dict,
            tiers: tuple[str, ...] = ("A", "C"), n_cv_folds: int = 5) -> dict:
    set_seed(seed)
    t0 = time.time()
    ds = load_dataset(dataset_name)
    n_classes = int(len(ds.classes))
    sp = split_utils.four_way_split(len(ds.y), y=ds.y, seed=seed, temporal=ds.is_temporal)

    X_train, y_train = ds.X.iloc[sp.train_idx], ds.y[sp.train_idx]
    X_meta, y_meta = ds.X.iloc[sp.meta_idx], ds.y[sp.meta_idx]
    X_test, y_test = ds.X.iloc[sp.test_idx], ds.y[sp.test_idx]
    pool_idx = np.concatenate([sp.train_idx, sp.meta_idx])
    y_pool = ds.y[pool_idx]

    model_train, n_train_used = _fit_capacity_model(X_train, y_train, seed, n_classes, params)
    held_out = [
        TemperatureScaledMSPSignal().fit(X_meta, y_meta, model_train),
        CalibrationResidualSignal().fit(X_meta, y_meta, model_train),
    ]
    held_out_names = [s.name for s in held_out]

    # Honest cross-fitted meta signals, same as the main runner -- the
    # aggregator must not be trained on in-sample signals, and that is
    # doubly important here because a weak model's in-sample overconfidence
    # is much larger than a strong one's.
    def fit_model_fn(idx_abs: np.ndarray, fold: int = 0):
        Xi, yi = ds.X.iloc[idx_abs], ds.y[idx_abs]
        m, _ = _fit_capacity_model(Xi, yi, seed * 1000 + fold, n_classes, params)
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
        fit_model_fn, compute_signals_fn, pool_idx, y_pool=y_pool,
        n_folds=n_cv_folds, seed=seed,
    )
    U_pool, correct_pool = oof[:, :-1], oof[:, -1].astype(bool)
    is_meta = np.isin(pool_idx, sp.meta_idx)
    U_meta_cf, correct_meta = U_pool[is_meta], correct_pool[is_meta]

    model_final, _ = _fit_capacity_model(
        ds.X.iloc[pool_idx], y_pool, seed, n_classes, params
    )
    final_bank = _build_signal_bank(tiers, n_classes=n_classes)
    final_bank.fit(ds.X.iloc[pool_idx], y_pool, model_final)
    signal_names = list(final_bank.names) + held_out_names

    def score_signals(X, model) -> np.ndarray:
        cols = [final_bank.transform(X, model)]
        cols += [s.score(X, model).reshape(-1, 1) for s in held_out]
        return np.hstack(cols)

    U_test = score_signals(X_test, model_final)
    U_meta_full = np.hstack(
        [U_meta_cf] + [s.score(X_meta, model_train).reshape(-1, 1) for s in held_out]
    )
    probs_test = model_final.predict_proba(X_test)
    incorrect_test = (probs_test.argmax(axis=1) != y_test).astype(int)

    # --- diagnostics -----------------------------------------------------
    # Tier A only, and only the signals whose binary rank-degeneracy is
    # proven: rho_local's claim is about that sub-bank (see
    # src/metrics/redundancy.py on why the full-bank variant is degenerate).
    tier_a_names = [
        s.name for s in default_tier_a_bank(n_classes=n_classes)
        if not isinstance(s, HELD_OUT_FIT_SIGNALS)
    ]
    test_signals = {nm: U_test[:, j] for j, nm in enumerate(signal_names)}
    meta_signals = {nm: U_meta_full[:, j] for j, nm in enumerate(signal_names)}

    red_test = local_rank_redundancy(test_signals, reference="msp", q=0.10,
                                     sub_bank=tier_a_names)
    red_meta = local_rank_redundancy(meta_signals, reference="msp", q=0.10,
                                     sub_bank=tier_a_names)
    # Part 2 on D_meta: labelled, but uses no test data, so it stays a
    # pre-deployment screening quantity.
    #
    # Measured at two region widths on purpose. D_meta is small (15% of the
    # dataset -- 965 rows on satimage), so its top-10% region holds ~96
    # points and only ~10 errors, and an AUROC over 10 errors is mostly
    # noise. q=0.25 trades some locality for an estimate that is actually
    # resolvable. Reporting both makes the trade visible instead of
    # hiding it behind one chosen width, and lets the paper say which
    # width the screening rule should use.
    incorrect_meta = (~correct_meta).astype(int)
    info_meta = incremental_boundary_auroc(
        meta_signals, incorrect_meta, reference="msp", q=0.10, seed=seed
    )
    info_meta_wide = incremental_boundary_auroc(
        meta_signals, incorrect_meta, reference="msp", q=0.25, seed=seed
    )

    # --- realised outcome ------------------------------------------------
    aggregators = {
        "A0_rank": RankAverageAggregator(),
        "A1_logreg": LogRegStackingAggregator(seed=seed),
        "A1_lightgbm": LightGBMStackingAggregator(seed=seed),
    }
    agg_aurc = {}
    for nm, agg in aggregators.items():
        agg.fit(U_meta_full, correct_meta)
        agg_aurc[nm] = float(aurc(agg.score(U_test), incorrect_test))
    sig_aurc = {nm: float(aurc(test_signals[nm], incorrect_test)) for nm in signal_names}
    best_sig = min(sig_aurc, key=sig_aurc.get)

    prereg = agg_aurc[PREREGISTERED_AGGREGATOR]
    return dict(
        dataset=dataset_name,
        K=n_classes,
        capacity=capacity,
        seed=seed,
        n_train_used=n_train_used,
        base_acc=float(1.0 - incorrect_test.mean()),
        test_errors=int(incorrect_test.sum()),
        n_test=int(len(y_test)),
        rho_local_test=red_test.rho_local,
        rho_local_meta=red_meta.rho_local,
        eff_rank_test=red_test.effective_rank,
        eff_binarity_test=effective_binarity(probs_test, test_signals["msp"], q=0.10),
        delta_local_meta=info_meta.delta_local,
        delta_local_meta_q25=info_meta_wide.delta_local,
        n_errors_meta_region=info_meta.n_errors_region,
        n_errors_meta_region_q25=info_meta_wide.n_errors_region,
        msp_aurc=sig_aurc["signal_msp"] if "signal_msp" in sig_aurc else sig_aurc["msp"],
        best_single_aurc=sig_aurc[best_sig],
        best_single=best_sig,
        prereg_aurc=prereg,
        # The outcome: negative = the pre-registered aggregator beats the
        # best single signal. This is the quantity H1 says tracks rho_local.
        gain_vs_best_single_pct=100 * (prereg - sig_aurc[best_sig]) / sig_aurc[best_sig],
        runtime=time.time() - t0,
    )


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "K", "capacity"], sort=False)
        .agg(
            seeds=("seed", "count"),
            base_acc=("base_acc", "mean"),
            test_errors=("test_errors", "mean"),
            rho_local_test=("rho_local_test", "mean"),
            rho_local_meta=("rho_local_meta", "mean"),
            eff_binarity=("eff_binarity_test", "mean"),
            delta_local_meta=("delta_local_meta", "mean"),
            delta_local_meta_q25=("delta_local_meta_q25", "mean"),
            n_err_meta_q25=("n_errors_meta_region_q25", "mean"),
            gain_pct=("gain_vs_best_single_pct", "mean"),
            gain_sd=("gain_vs_best_single_pct", "std"),
        )
        .reset_index()
    )


def report_hypothesis_test(summary: pd.DataFrame, per_seed: pd.DataFrame) -> None:
    """H1 vs H0 at fixed K, per dataset."""
    from scipy.stats import spearmanr

    print("\n" + "=" * 78)
    print("H1: at fixed K, the aggregation gain tracks rho_local")
    print("=" * 78)
    for dataset, g in summary.groupby("dataset"):
        k = int(g.K.iloc[0])
        rho, gain = g.rho_local_test.to_numpy(), g.gain_pct.to_numpy()
        spread = float(np.nanmax(rho) - np.nanmin(rho))
        print(f"\n{dataset} (K={k}), {len(g)} capacity settings")
        print(f"  rho_local range: {np.nanmin(rho):.4f} .. {np.nanmax(rho):.4f} "
              f"(spread {spread:.4f})")
        if k == 2:
            # The control. The theorem says every Tier-A signal is a
            # monotone function of the single logit gap, so rho_local is
            # 1.0000 regardless of how the model is trained.
            exact = bool(np.allclose(rho, 1.0, atol=1e-9))
            print(f"  CONTROL (binary): rho_local == 1.0000 at every capacity? "
                  f"{'YES -- as the theorem requires' if exact else 'NO -- INVESTIGATE'}")
            print(f"  gain range: {np.nanmin(gain):+.2f}% .. {np.nanmax(gain):+.2f}% "
                  f"(should never be usefully negative)")
            continue
        if spread < 0.05:
            print("  rho_local barely moved: the capacity ladder failed to create a "
                  "dose here, so this dataset cannot test H1.")
            continue
        # Per-seed correlation, so the test uses all the data rather than
        # 6 capacity means.
        ps = per_seed[per_seed.dataset == dataset]
        res_seed = spearmanr(ps.rho_local_test, ps.gain_vs_best_single_pct)
        res_mean = spearmanr(rho, gain)
        print(f"  Spearman(rho_local, gain), capacity means: "
              f"{res_mean.statistic:+.3f} (p={res_mean.pvalue:.4f}, n={len(g)})")
        print(f"  Spearman(rho_local, gain), per seed:       "
              f"{res_seed.statistic:+.3f} (p={res_seed.pvalue:.4g}, n={len(ps)})")
        print("  H1 predicts a POSITIVE correlation (higher redundancy -> less gain,")
        print("  and `gain` is signed so that less gain is a larger number).")
        verdict = (
            "supports H1" if res_seed.statistic > 0 and res_seed.pvalue < 0.05
            else "does NOT support H1 at fixed K"
        )
        print(f"  --> {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="satimage")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--capacities", default="",
                   help="comma-separated subset of the capacity ladder")
    args = p.parse_args()

    wanted = {c.strip() for c in args.capacities.split(",") if c.strip()}
    ladder = [(nm, pr) for nm, pr in CAPACITY_LADDER if not wanted or nm in wanted]

    rows = []
    for capacity, params in ladder:
        for seed in range(args.n_seeds):
            print(f"[capacity] {args.dataset} {capacity} seed={seed}", flush=True)
            try:
                rows.append(run_one(args.dataset, seed, capacity, params))
            except Exception as exc:  # noqa: BLE001
                # A capacity setting so weak that a fold loses a class can
                # legitimately fail; recorded, not silently skipped.
                print(f"    FAILED ({capacity}, seed={seed}): "
                      f"{type(exc).__name__}: {exc}", flush=True)

    if not rows:
        raise SystemExit("no capacity setting completed")

    new = pd.DataFrame(rows)
    key = ["dataset", "capacity", "seed"]
    if RESULTS_PATH.exists():
        old = pd.read_parquet(RESULTS_PATH)
        mask = old.set_index(key).index.isin(new.set_index(key).index)
        new = pd.concat([old[~mask], new], ignore_index=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(RESULTS_PATH)
    print(f"\n[capacity] wrote {len(new)} rows -> {RESULTS_PATH.relative_to(ROOT)}")

    summary = summarise(new)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "capacity_sweep_summary.csv", index=False)

    print("\n=== Capacity sweep (weakest base model first) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    report_hypothesis_test(summary, new)
    print(f"\nwrote {(TABLE_DIR / 'capacity_sweep_summary.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
