"""Tier A signals (§3.1): logit-derived, free, always available. Each one
only reads `model.predict_proba` / `model.logits`, so they work identically
for LightGBM and logistic-regression wrappers.

Two of these — TemperatureScaledMSP and LogitNormMSP — are also listed in
§7's baseline ladder as "the cheap fix[es] that often close the gap"; they
live here and not just in aggregators because the whole point of §7 is to
compare them *as standalone baselines* against the learned aggregator.

Important leakage note (§6): `TemperatureScaledMSP.fit` must be called with
D_meta, never with D_train or D_test — the experiment runner is responsible
for passing the right split. Fitting T on D_train (which the base model has
already memorized) or D_test (which must be touched once, at the end) is
exactly the kind of quiet leakage bug this project is designed to avoid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .base import Signal


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class MSPSignal(Signal):
    """Max softmax probability, negated so higher = more uncertain. This is
    Chow's rule (§2) — the baseline everything else in the paper has to
    beat."""

    name = "msp"

    def fit(self, X_train, y_train, model) -> "MSPSignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        p = model.predict_proba(X)
        return 1.0 - p.max(axis=1)


class EntropySignal(Signal):
    name = "entropy"

    def fit(self, X_train, y_train, model) -> "EntropySignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        p = np.clip(model.predict_proba(X), 1e-12, 1.0)
        return -(p * np.log(p)).sum(axis=1)


class MarginSignal(Signal):
    """Top-1 minus top-2, in probability or logit space (§3.1 item 3).
    Negated so higher = more uncertain (a small margin is uncertain)."""

    name = "margin"

    def __init__(self, space: str = "prob"):
        assert space in ("prob", "logit")
        self.space = space
        self.name = f"margin_{space}"

    def fit(self, X_train, y_train, model) -> "MarginSignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        arr = model.predict_proba(X) if self.space == "prob" else model.logits(X)
        sorted_arr = np.sort(arr, axis=1)
        margin = sorted_arr[:, -1] - sorted_arr[:, -2]
        return -margin


class EnergySignal(Signal):
    """Negative energy score (§2 Liu et al. 2020; §3.1 item 4):
    -logsumexp(z). Higher raw energy score is more OOD-like in the original
    paper's convention; we keep that sign directly since it already means
    "more uncertain" for larger values here (energy = -logsumexp(z), and
    logsumexp(z) shrinks — energy grows — as the model's max logit shrinks)."""

    name = "energy"

    def fit(self, X_train, y_train, model) -> "EnergySignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        z = model.logits(X)
        return -logsumexp(z, axis=1)


class LogitNormMSPSignal(Signal):
    """Logit-norm-normalised MSP (§2 Cattelan & Silva 2023; §3.1 item 5):
    softmax(z / ||z||_p), then negated max prob."""

    name = "logitnorm_msp"

    def __init__(self, p: int = 2):
        self.p = p

    def fit(self, X_train, y_train, model) -> "LogitNormMSPSignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        z = model.logits(X)
        norm = np.linalg.norm(z, ord=self.p, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-12, None)
        p = _softmax(z / norm, axis=1)
        return 1.0 - p.max(axis=1)


class TemperatureScaledMSPSignal(Signal):
    """Temperature-scaled MSP (§2 Guo et al. 2017; §3.1 item 6). T is fit by
    minimizing NLL on whatever split is passed to `.fit` — the caller MUST
    pass D_meta, never D_train or D_test (see module docstring)."""

    name = "temp_msp"

    def __init__(self):
        self.T: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, model) -> "TemperatureScaledMSPSignal":
        z = model.logits(X_train)
        y = np.asarray(y_train)

        def nll(log_T: float) -> float:
            T = np.exp(log_T)
            logp = z / T - logsumexp(z / T, axis=1, keepdims=True)
            return -logp[np.arange(len(y)), y].mean()

        res = minimize_scalar(nll, bounds=(np.log(1e-2), np.log(1e2)), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        z = model.logits(X)
        p = _softmax(z / self.T, axis=1)
        return 1.0 - p.max(axis=1)


def default_tier_a_bank() -> list[Signal]:
    return [
        MSPSignal(),
        EntropySignal(),
        MarginSignal(space="prob"),
        MarginSignal(space="logit"),
        EnergySignal(),
        LogitNormMSPSignal(),
        TemperatureScaledMSPSignal(),
    ]
