"""End-to-end experiment runner.

Wires together: dataset load -> four-way split -> honest (cross-fitted)
meta signals -> final deployment model -> aggregators -> conformal wrapper
-> metrics -> one long-format DataFrame row per (method, coverage) per
§8's results-schema rule ("no number in the paper should exist anywhere
except as a query against [the results parquet]").

Design choice, stated explicitly (this is the kind of simplification the
project plan asks you to be honest about, not hide, §6/§7):

  - `model_train`  : fit on D_train only. Used *only* to fit temperature
                     scaling on D_meta (already honest by construction,
                     since model_train never saw D_meta -- no cross-fitting
                     needed for this one signal).
  - fold models    : K-fold cross-fit over D_train u D_meta (§6's fix).
                     Used to compute honest out-of-fold Tier A (non-temperature)
                     and Tier C signals, plus correctness labels, for
                     aggregator (meta) training data.
  - `model_final`  : fit on D_train u D_meta. This is the deployed model:
                     it produces every prediction and signal used at
                     evaluation time on D_cal and D_test.

Tier B (ensemble disagreement) is optional (`use_ensemble=True`) and, when
enabled, trains its own small ensemble on D_train u D_meta for both the
meta-signal computation and deployment scoring -- cross-fitting an
ensemble-of-ensembles is out of scope for this runner; see
PROJECT_STATUS.md for the honest accounting of this simplification.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from ..aggregators.a0_rank import RankAverageAggregator
from ..aggregators.a1_stacking import LightGBMStackingAggregator, LogRegStackingAggregator
from ..aggregators.a2_coverage_loss import AdaptiveGatingAggregator, MLPAggregator
from ..conformal.risk_control import coverage_at_guaranteed_risk_table
from ..data import splits as split_utils
from ..data.loaders import Dataset, load as load_dataset
from ..metrics.selective import (
    aurc,
    e_aurc,
    ece,
    failure_prediction_auroc,
    risk_at_coverage,
)
from ..metrics.subgroup import worst_group_selective_risk
from ..models.lightgbm_model import LightGBMWrapper
from ..models.logreg_model import LogRegWrapper
from ..signals.base import Signal, SignalBank
from ..signals.ensemble import default_tier_b_bank
from ..signals.tier_a import TemperatureScaledMSPSignal, default_tier_a_bank
from ..signals.tier_c import default_tier_c_bank
from .seeding import set_seed

MODEL_REGISTRY = {"lightgbm": LightGBMWrapper, "logreg": LogRegWrapper}

REPORT_COVERAGES = (1.0, 0.95, 0.90, 0.80, 0.70, 0.50)
CONFORMAL_ALPHAS = (0.01, 0.02, 0.05, 0.10)


def _build_signal_bank(tiers: tuple[str, ...], ensemble: Optional[list] = None) -> SignalBank:
    signals: list[Signal] = []
    if "A" in tiers:
        signals += [s for s in default_tier_a_bank() if not isinstance(s, TemperatureScaledMSPSignal)]
    if "C" in tiers:
        signals += default_tier_c_bank()
    if "B" in tiers:
        assert ensemble is not None, "Tier B requires an ensemble"
        signals += default_tier_b_bank(ensemble)
    return SignalBank(signals)


def _fit_model(model_name: str, X, y, seed: int):
    return MODEL_REGISTRY[model_name](seed=seed).fit(X, y)


def _train_ensemble(model_name: str, X, y, n_members: int, base_seed: int) -> list:
    return [_fit_model(model_name, X, y, seed=base_seed * 1000 + i) for i in range(n_members)]


def run_experiment(
    dataset_name: str,
    model_name: str = "lightgbm",
    tiers: tuple[str, ...] = ("A", "C"),
    seed: int = 0,
    n_cv_folds: int = 5,
    use_ensemble: bool = False,
    n_ensemble_members: int = 5,
    conformal_delta: float = 0.1,
    subgroup_col: Optional[str] = None,
) -> pd.DataFrame:
    set_seed(seed)
    t_start = time.time()

    ds: Dataset = load_dataset(dataset_name)
    n = len(ds.y)
    sp = split_utils.four_way_split(n, y=ds.y, seed=seed, temporal=ds.is_temporal)

    X_train, y_train = ds.X.iloc[sp.train_idx], ds.y[sp.train_idx]
    X_meta, y_meta = ds.X.iloc[sp.meta_idx], ds.y[sp.meta_idx]
    X_cal, y_cal = ds.X.iloc[sp.cal_idx], ds.y[sp.cal_idx]
    X_test, y_test = ds.X.iloc[sp.test_idx], ds.y[sp.test_idx]

    pool_idx = np.concatenate([sp.train_idx, sp.meta_idx])
    y_pool = ds.y[pool_idx]

    # --- model_train: D_train only, used solely to fit temperature on D_meta
    model_train = _fit_model(model_name, X_train, y_train, seed=seed)
    temp_signal = TemperatureScaledMSPSignal().fit(X_meta, y_meta, model_train)

    # --- honest cross-fitted Tier A(-temp)/C signals + correctness labels
    # over the D_train u D_meta pool (§6's fix for the leakage trap).
    def fit_model_fn(idx_abs: np.ndarray):
        Xi, yi = ds.X.iloc[idx_abs], ds.y[idx_abs]
        model = _fit_model(model_name, Xi, yi, seed=seed)
        bank = _build_signal_bank(tuple(t for t in tiers if t != "B"))
        bank.fit(Xi, yi, model)
        return {"model": model, "bank": bank}

    def compute_signals_fn(bundle: dict, idx_abs: np.ndarray) -> np.ndarray:
        Xi = ds.X.iloc[idx_abs]
        u = bundle["bank"].transform(Xi, bundle["model"])
        pred = bundle["model"].predict_proba(Xi).argmax(axis=1)
        correct = (pred == ds.y[idx_abs]).astype(float).reshape(-1, 1)
        return np.hstack([u, correct])

    oof = split_utils.cross_fitted_signals(
        fit_model_fn, compute_signals_fn, pool_idx, y_pool=y_pool, n_folds=n_cv_folds, seed=seed
    )
    U_pool, correct_pool = oof[:, :-1], oof[:, -1].astype(bool)
    is_meta = np.isin(pool_idx, sp.meta_idx)
    U_meta_cf, correct_meta_cf = U_pool[is_meta], correct_pool[is_meta]

    signal_names = _build_signal_bank(tuple(t for t in tiers if t != "B")).names

    # --- naive (leaky) meta signals, for the naive-vs-cross-fit ablation
    naive_bank = _build_signal_bank(tuple(t for t in tiers if t != "B"))
    naive_bank.fit(X_train, y_train, model_train)
    U_meta_naive = naive_bank.transform(X_meta, model_train)
    pred_naive = model_train.predict_proba(X_meta).argmax(axis=1)
    correct_meta_naive = pred_naive == y_meta

    # --- model_final: deployed model, D_train u D_meta
    X_pool, y_pool_full = ds.X.iloc[pool_idx], y_pool
    model_final = _fit_model(model_name, X_pool, y_pool_full, seed=seed)

    ensemble_final = None
    if use_ensemble:
        ensemble_final = _train_ensemble(model_name, X_pool, y_pool_full, n_ensemble_members, seed)

    final_bank = _build_signal_bank(tiers, ensemble=ensemble_final)
    # Non-temperature parts of the bank are fit on the full pool (their own
    # geometry-index / no-op fits, as documented in tier_a.py/tier_c.py);
    # temperature is grafted in from `temp_signal` (fit honestly on D_meta
    # by model_train above -- see module docstring).
    final_bank.fit(X_pool, y_pool_full, model_final)
    final_signal_names = final_bank.names + ["temp_msp"]

    def score_final(X) -> np.ndarray:
        u = final_bank.transform(X, model_final)
        t = temp_signal.score(X, model_final).reshape(-1, 1)
        return np.hstack([u, t])

    U_cal = score_final(X_cal)
    U_test = score_final(X_test)
    pred_cal = model_final.predict_proba(X_cal).argmax(axis=1)
    pred_test = model_final.predict_proba(X_test).argmax(axis=1)
    incorrect_cal = (pred_cal != y_cal).astype(int)
    incorrect_test = (pred_test != y_test).astype(int)

    # Align meta-signal columns with final-signal columns (final adds
    # temperature at the end; cross-fit meta path didn't compute it, so we
    # append it here using model_train -- honest, since model_train never
    # saw D_meta).
    U_meta_cf_full = np.hstack([U_meta_cf, temp_signal.score(X_meta, model_train).reshape(-1, 1)])
    U_meta_naive_full = np.hstack(
        [U_meta_naive, temp_signal.score(X_meta, model_train).reshape(-1, 1)]
    )
    assert final_signal_names == signal_names + ["temp_msp"]

    aggregator_factories = {
        "A0_rank": lambda: RankAverageAggregator(),
        "A1_logreg": lambda: LogRegStackingAggregator(seed=seed),
        "A1_lightgbm": lambda: LightGBMStackingAggregator(seed=seed),
        "A2_mlp_bce": lambda: MLPAggregator(loss="bce", seed=seed),
        "A2_mlp_loss1": lambda: MLPAggregator(loss="loss1", seed=seed),
        "A2_mlp_loss2": lambda: MLPAggregator(loss="loss2", seed=seed),
        "A2_mlp_loss3": lambda: MLPAggregator(loss="loss3", seed=seed),
        "A2_adaptive_loss3": lambda: AdaptiveGatingAggregator(loss="loss3", seed=seed),
    }

    rows = []
    runtime_so_far = time.time() - t_start

    def _row(cov=np.nan, risk=np.nan, accuracy=np.nan, n_accepted=np.nan, subgroup="overall",
              value=np.nan, method=""):
        return dict(
            dataset=dataset_name,
            base_model=model_name,
            method=method,
            seed=seed,
            coverage=cov,
            risk=risk,
            accuracy=accuracy,
            n_accepted=n_accepted,
            subgroup=subgroup,
            value=value,
            runtime=runtime_so_far,
        )

    def add_rows(method: str, uncertainty_test: np.ndarray, extra_conformal: Optional[dict] = None):
        for cov in REPORT_COVERAGES:
            risk = risk_at_coverage(uncertainty_test, incorrect_test, cov)
            k = max(1, min(len(uncertainty_test), int(round(cov * len(uncertainty_test)))))
            rows.append(
                _row(cov=cov, risk=risk, accuracy=1.0 - risk, n_accepted=k,
                     subgroup="overall", method=method)
            )
        rows.append(_row(subgroup="AURC", value=aurc(uncertainty_test, incorrect_test), method=method))
        rows.append(_row(subgroup="E_AURC", value=e_aurc(uncertainty_test, incorrect_test), method=method))
        rows.append(
            _row(
                subgroup="failure_auroc",
                value=failure_prediction_auroc(uncertainty_test, incorrect_test),
                method=method,
            )
        )
        if extra_conformal:
            for alpha, ct in extra_conformal.items():
                rows.append(
                    _row(
                        cov=ct.coverage_at_tau,
                        risk=ct.empirical_risk_at_tau,
                        accuracy=1.0 - ct.empirical_risk_at_tau,
                        subgroup=f"conformal_alpha_{alpha}",
                        method=method,
                    )
                )
        if subgroup_col and subgroup_col in ds.X.columns:
            group_test = ds.X.iloc[sp.test_idx][subgroup_col].values
            for cov in (0.9, 0.8):
                res = worst_group_selective_risk(uncertainty_test, incorrect_test, group_test, cov)
                rows.append(
                    _row(
                        cov=cov,
                        subgroup=f"max_min_gap@{cov}",
                        value=res["max_min_gap"],
                        method=method,
                    )
                )

    # --- Random abstention baseline (lower bound, §7)
    rng = np.random.RandomState(seed)
    add_rows("random", rng.rand(len(y_test)))

    # --- Oracle ordering (upper bound, §7)
    add_rows("oracle", incorrect_test.astype(float))

    # --- Single-signal baselines, evaluated directly off the final bank
    for j, name in enumerate(final_signal_names):
        add_rows(f"signal_{name}", U_test[:, j])

    # --- Aggregators, trained on the honest cross-fitted meta signals
    for agg_name, factory in aggregator_factories.items():
        agg = factory()
        agg.fit(U_meta_cf_full, correct_meta_cf)
        s_test = agg.score(U_test)
        s_cal = agg.score(U_cal)
        conformal = coverage_at_guaranteed_risk_table(
            s_cal, incorrect_cal, alphas=CONFORMAL_ALPHAS, delta=conformal_delta
        )
        add_rows(agg_name, s_test, extra_conformal=conformal)

    # --- Naive-vs-cross-fit ablation (§6, §9 week-3 gate): same aggregator
    # class (A1 logreg) trained on naive vs. honest meta signals.
    agg_naive = LogRegStackingAggregator(seed=seed)
    agg_naive.fit(U_meta_naive_full, correct_meta_naive)
    add_rows("A1_logreg_naive_meta", agg_naive.score(U_test))

    return pd.DataFrame(rows)
