"""Conformal risk-control threshold selection (§3.5, §2 Angelopoulos et al.,
"Conformal Risk Control" 2022 and "Learn Then Test" 2021).

Given any heuristic risk score s(x) (from any aggregator in this repo) and
a calibration set D_cal, choose the largest threshold `tau_hat` (i.e. the
highest-coverage threshold) such that a finite-sample upper confidence
bound on selective risk stays below the user's target `alpha`, with
confidence `1 - delta`. Under exchangeability between D_cal and the
deployment distribution, this gives

    P( selective_risk(tau_hat) <= alpha ) >= 1 - delta

distribution-free, with no assumption about how the aggregator was trained
— it wraps A0/A1/A2 identically.

Implementation note: we use a Hoeffding-style upper confidence bound (UCB)
on the empirical selective risk at each candidate threshold, searched over
a grid of thresholds from `s(x)` on D_cal, and Bonferroni-correct across
the grid so the "search over thresholds" step doesn't itself invalidate the
guarantee. This is a direct, auditable instantiation of the same idea as
Learn-Then-Test's Hoeffding-Bentkus bound; swap in `mapie`/`crepes` (§8)
for a more powerful (tighter) bound if empirical coverage in the
in-distribution ablation (§9 week-7 gate) shows this one is too
conservative.

Caveat, stated in the paper too (§3.5): exchangeability fails under
distribution shift, so the guarantee is valid in-distribution only and is
*evaluated empirically, not claimed*, under shift.
"""
from __future__ import annotations

import dataclasses

import numpy as np


def _hoeffding_ucb(risk_hat: float, n: int, delta: float) -> float:
    """Upper confidence bound on the true mean of a [0,1]-bounded random
    variable given its empirical mean over n samples, at confidence 1-delta."""
    if n == 0:
        return 1.0
    return float(risk_hat + np.sqrt(np.log(1.0 / delta) / (2.0 * n)))


@dataclasses.dataclass
class ConformalThreshold:
    tau: float
    alpha: float
    delta: float
    empirical_risk_at_tau: float
    ucb_at_tau: float
    coverage_at_tau: float
    n_cal: int


def select_threshold(
    s_cal: np.ndarray,
    incorrect_cal: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    n_grid: int = 200,
) -> ConformalThreshold:
    """Select tau_hat = the largest threshold (highest coverage) such that
    the Bonferroni-corrected Hoeffding UCB on selective risk is <= alpha.

    `s_cal`: (n,) risk scores on D_cal (higher = more uncertain; abstain
             when s(x) > tau).
    `incorrect_cal`: (n,) in {0,1}, 1 if the frozen base model was wrong.
    """
    n = len(s_cal)
    assert n > 0, "calibration set must be non-empty"

    # Candidate thresholds: `n_grid` quantiles of the observed scores. This
    # must stay bounded (not grow with n) -- the Bonferroni correction below
    # divides delta by the grid size, so a grid that silently grows to
    # size ~n (e.g. by including every unique observed score) makes the
    # bound needlessly, severely conservative on large calibration sets. A
    # coarser grid can only miss the exact optimal threshold by one
    # quantile step; it never invalidates the guarantee itself.
    grid = np.unique(np.quantile(s_cal, np.linspace(0, 1, n_grid)))
    delta_per_test = delta / len(grid)  # Bonferroni over the threshold search

    best = None
    for tau in grid:
        accepted = s_cal <= tau
        n_acc = int(accepted.sum())
        if n_acc == 0:
            continue
        risk_hat = float(incorrect_cal[accepted].mean())
        ucb = _hoeffding_ucb(risk_hat, n_acc, delta_per_test)
        coverage = n_acc / n
        if ucb <= alpha:
            if best is None or coverage > best.coverage_at_tau:
                best = ConformalThreshold(
                    tau=float(tau),
                    alpha=alpha,
                    delta=delta,
                    empirical_risk_at_tau=risk_hat,
                    ucb_at_tau=ucb,
                    coverage_at_tau=coverage,
                    n_cal=n,
                )

    if best is None:
        # No threshold satisfies the risk bound at this alpha/delta/n_cal:
        # abstain on everything (coverage 0) rather than silently violate
        # the guarantee. This is itself an informative result (report it,
        # don't hide it) -- it means either D_cal is too small for this
        # alpha, or the aggregator isn't good enough at any coverage.
        best = ConformalThreshold(
            tau=float(-np.inf),
            alpha=alpha,
            delta=delta,
            empirical_risk_at_tau=0.0,
            ucb_at_tau=0.0,
            coverage_at_tau=0.0,
            n_cal=n,
        )
    return best


def coverage_at_guaranteed_risk_table(
    s_cal: np.ndarray,
    incorrect_cal: np.ndarray,
    alphas: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
    delta: float = 0.1,
) -> dict[float, ConformalThreshold]:
    """The primary conformal table from §3.5/§7: coverage at guaranteed
    risk <= alpha, for alpha in {1%, 2%, 5%, 10%}."""
    return {a: select_threshold(s_cal, incorrect_cal, alpha=a, delta=delta) for a in alphas}
