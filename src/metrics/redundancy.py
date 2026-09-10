"""Boundary-local redundancy and informativeness diagnostics.

This module formalises the two-part diagnostic that `NOVELTY_ANALYSIS.md`
§3 identified as the mechanism governing when multi-signal aggregation can
beat the best single confidence signal. It is the paper's central
methodological contribution, so the reasoning is spelled out here rather
than left in a notebook.

Why a *local* statistic at all
------------------------------
AURC is a pure ranking functional: sort by uncertainty, walk the prefix,
average the cumulative error rate. Two scores that induce the same total
order have *identical* AURC, so only the ordinal content of a signal can
ever matter, and any monotone reparameterisation of a signal is free.

Crucially the ranking is not weighted uniformly. At prefix length `k` the
cumulative error rate has denominator `k`, so a swap between ranks 3 and 4
moves the curve by O(1/3) while a swap between ranks 3000 and 3001 moves it
by O(1/3000). AURC is therefore dominated by the ordering of the points
most likely to be abstained on -- the tail of the risk ranking -- and a
statistic computed over the *whole* evaluation set answers a question AURC
never asks.

This is not a technicality. Measured on this project's own runs, the
minimum |Spearman| between MSP and the rest of the Tier-A bank reads
0.86-0.89 globally on satimage and letter -- high enough to conclude "the
bank is redundant, aggregation cannot help" -- while the same banks
decorrelate to 0.33-0.46 inside the top-10% risk region, which is where
aggregation demonstrably *does* help by ~6% AURC. Measuring redundancy
globally is what made the mechanism look unexplained.

The two parts, and why one is not enough
----------------------------------------
**Part 1 -- necessity (`local_rank_redundancy`, needs no labels).**
`rho_local(q)` is the minimum absolute Spearman correlation between the
reference signal (MSP) and every other signal in a nominated sub-bank,
restricted to the top-`q` fraction of points by the reference. If
`rho_local = 1` then every signal in the sub-bank induces the same order as
MSP on that region, so the sub-bank carries exactly **one** ordinal degree
of freedom there. Any aggregator over it is then a function `h(t)` of that
single latent quantity, and the only way `h` can reorder points is by being
non-monotone -- for which the bank supplies no evidence, since a
non-monotone `h` is exactly what a monotone-link estimate would have to
contradict. AURC being invariant to monotone transforms, aggregation cannot
help. On a binary task this is not an empirical measurement but a
mathematical certainty: every Tier-A signal is a strictly monotone function
of the single logit gap (see `src/signals/tier_a.py`), so
`rho_local == 1.0000` exactly, at every `q`, for every binary dataset.

That exactness is also the limitation. Because `rho_local(TierA) = 1` is
*forced* for `K = 2`, "binary" and "locally redundant" cannot be
distinguished by any binary dataset, and the honest test of the mechanism
has to come from multiclass datasets that are *also* locally redundant --
see `effective_binarity` below, which measures the property that drives it
and so turns that search from a blind sweep into a targeted one.

**Part 2 -- sufficiency (`incremental_boundary_auroc`, needs labels).**
Non-redundancy is necessary but demonstrably not sufficient: this project's
Tier B (ensemble disagreement) signals are strongly decorrelated from MSP
and never help on any dataset, because being *independent* is not the same
as being *informative*. Worse, taking `rho_local` over the full A+B+C bank
is degenerate in the opposite direction -- Tier C distance signals are
weakly correlated with MSP on every dataset, binary or not (measured: 0.011
to 0.072 across all six), so the minimum collapses to ~0 everywhere and
separates nothing.

So the second half of the diagnostic asks the labelled question directly:
inside the abstention region, how much better can the *bank* rank errors
than the best *single* signal can? `incremental_boundary_auroc` fits an
honest out-of-fold linear stack on the meta-split signals and reports

    Delta_local(q) = AUROC_local(stacked bank) - max_j AUROC_local(u_j)

The comparison is against the best single signal, not against MSP, because
that is the quantity the paper reports as its outcome -- a diagnostic whose
baseline differs from the outcome's baseline would mispredict exactly the
dataset where some non-MSP signal is individually strong (which is
wine_quality_white, where `trust_score` alone beats MSP by 23.5%).

Both parts are computable on `D_meta` before any aggregator is trained and
before `D_test` is touched, which is what makes them a *screening* rule
rather than a post-hoc rationalisation.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

# Minimum number of points to keep in the abstention region. A Spearman
# correlation over a handful of points is noise, and german_credit's test
# split is only ~150 rows, so the top-10% region would otherwise be 15
# points. Kept as a floor rather than a hard requirement so small datasets
# degrade to "a wider region" instead of to a NaN.
MIN_REGION = 30


def abstention_region(reference: np.ndarray, q: float, min_size: int = MIN_REGION) -> np.ndarray:
    """Indices of the top-`q` fraction of points by `reference` uncertainty.

    `reference` follows the repo-wide convention that higher = abstain
    sooner, so the abstention region is the *largest* values. Returned in
    descending-uncertainty order, which makes the region nest as `q` grows
    (`region(0.1)` is a prefix of `region(0.25)`) -- useful when sweeping
    `q` and wanting the sequence to be interpretable.
    """
    reference = np.asarray(reference, dtype=float)
    n = len(reference)
    if n == 0:
        return np.empty(0, dtype=int)
    k = int(np.ceil(q * n))
    k = max(min(min_size, n), min(k, n))
    # Descending by uncertainty: `-reference` ascending. Stable so ties
    # resolve by original position rather than unpredictably, which keeps
    # the statistic reproducible across numpy versions.
    return np.argsort(-reference, kind="stable")[:k]


@dataclasses.dataclass
class LocalRedundancy:
    """Result of `local_rank_redundancy`."""

    rho_local: float
    """min_j |Spearman(reference, u_j)| on the abstention region."""

    argmin_signal: str
    """Which signal achieved that minimum -- i.e. the most independent
    ordering the sub-bank supplies where it matters."""

    per_signal: dict[str, float]
    """|Spearman| against the reference for every signal, so a reader can
    see whether the minimum is an outlier or the whole bank decorrelates."""

    effective_rank: float
    """Entropy-based effective rank of the rank-transformed sub-bank on the
    same region: `exp(-sum p_i log p_i)` over the normalised eigenvalues of
    the rank-correlation matrix. Counts how many independent orderings the
    bank really supplies. Reported alongside `rho_local` because it uses
    the whole bank rather than pairwise-against-reference, but it separates
    the datasets by a much thinner margin, so `rho_local` is the headline."""

    n_region: int
    q: float


def effective_rank(values: np.ndarray) -> float:
    """Entropy-based effective rank of a (m signals, n points) matrix.

    Rows are **rank-transformed here**, so the Gram matrix is the *Spearman*
    correlation matrix and the result depends only on the orderings the
    signals induce -- which is the only thing AURC can see. The
    exponentiated Shannon entropy of its normalised eigenvalues is then a
    smooth count of independent orderings: 1.0 for a perfectly redundant
    bank, m for m orthogonal ones.

    Ranking internally rather than trusting the caller is deliberate. Fed
    raw values, a Pearson version reports 1.35 for the bank
    `[t, 2t + 1, exp(t)]` -- three *rank-identical* signals -- because
    `exp` is monotone but not linear. That would understate redundancy
    exactly where this project's signals live: Tier A is full of monotone
    non-linear transforms of each other. Re-ranking an already-ranked
    matrix is a no-op, so callers may pass either.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        return float("nan")
    ranked = np.vstack([rankdata(row) for row in values])
    centred = ranked - ranked.mean(axis=1, keepdims=True)
    scale = centred.std(axis=1, keepdims=True)
    # A constant signal (e.g. a degenerate score on a tiny region) has zero
    # spread; leaving it as 0/0 would poison the whole eigendecomposition,
    # so it is mapped to a zero row, which contributes no direction.
    safe = np.divide(centred, scale, out=np.zeros_like(centred), where=scale > 1e-12)
    corr = safe @ safe.T / safe.shape[1]
    ev = np.clip(np.linalg.eigvalsh(corr), 1e-12, None)
    ev = ev / ev.sum()
    return float(np.exp(-(ev * np.log(ev)).sum()))


def local_rank_redundancy(
    signals: dict[str, np.ndarray],
    reference: str = "msp",
    q: float = 0.10,
    sub_bank: Optional[Sequence[str]] = None,
    min_size: int = MIN_REGION,
) -> LocalRedundancy:
    """Part 1 of the diagnostic: `rho_local(q)` over a nominated sub-bank.

    `signals` maps signal name -> uncertainty vector (higher = abstain
    sooner). `sub_bank` restricts which signals enter the minimum; pass the
    Tier-A names to get the headline statistic. Defaults to every signal
    present, which is the *degenerate* choice discussed in the module
    docstring -- Tier C decorrelates from MSP everywhere, so the full-bank
    minimum is ~0 on every dataset and separates nothing. The default is
    permissive rather than correct on purpose: callers should be explicit
    about which sub-bank they are making a claim about.

    Raises if `reference` is absent, rather than silently picking another
    signal -- a `rho_local` measured against the wrong reference is not a
    weaker version of this statistic, it is a different one.
    """
    if reference not in signals:
        raise KeyError(f"reference signal {reference!r} not in bank {sorted(signals)}")

    names = list(signals) if sub_bank is None else [s for s in sub_bank if s in signals]
    others = [nm for nm in names if nm != reference]
    ref = np.asarray(signals[reference], dtype=float)
    region = abstention_region(ref, q=q, min_size=min_size)

    if len(others) == 0 or len(region) < 3:
        return LocalRedundancy(
            rho_local=float("nan"),
            argmin_signal="",
            per_signal={},
            effective_rank=float("nan"),
            n_region=len(region),
            q=q,
        )

    ref_region = ref[region]
    per_signal: dict[str, float] = {}
    for nm in others:
        u = np.asarray(signals[nm], dtype=float)[region]
        if np.ptp(u) == 0 or np.ptp(ref_region) == 0:
            # A signal that is constant on the region supplies no ordering
            # at all. That is maximal redundancy (it can never reorder
            # anything), not maximal independence -- treating a constant as
            # rho = 0 would make a useless signal look like the most
            # valuable one in the bank.
            per_signal[nm] = 1.0
            continue
        rho = spearmanr(ref_region, u).statistic
        per_signal[nm] = float(abs(rho)) if np.isfinite(rho) else 1.0

    argmin = min(per_signal, key=per_signal.get)
    # `effective_rank` rank-transforms internally, so raw values are fine.
    sub_bank_values = np.vstack([np.asarray(signals[nm], dtype=float)[region] for nm in names])

    return LocalRedundancy(
        rho_local=per_signal[argmin],
        argmin_signal=argmin,
        per_signal=per_signal,
        effective_rank=effective_rank(sub_bank_values),
        n_region=len(region),
        q=q,
    )


def _auroc(score: np.ndarray, incorrect: np.ndarray) -> float:
    """Failure-prediction AUROC, NaN when undefined (no errors or all
    errors on the region -- which happens routinely on a narrow abstention
    region of an accurate model, so it must not raise)."""
    from sklearn.metrics import roc_auc_score

    incorrect = np.asarray(incorrect)
    if len(incorrect) < 2 or incorrect.min() == incorrect.max():
        return float("nan")
    return float(roc_auc_score(incorrect, score))


@dataclasses.dataclass
class BoundaryInformativeness:
    """Result of `incremental_boundary_auroc`."""

    delta_local: float
    """AUROC_local(stacked bank) - max_j AUROC_local(u_j). Positive means
    the bank can rank errors inside the abstention region better than any
    single signal can, which is the headroom an aggregator could realise."""

    auroc_stack: float
    auroc_best_single: float
    best_single_signal: str
    auroc_reference: float
    """AUROC of the reference signal alone, so `delta` against MSP (what a
    naive diagnostic would use) can be recovered and contrasted."""

    n_region: int
    n_errors_region: int
    q: float


def incremental_boundary_auroc(
    signals: dict[str, np.ndarray],
    incorrect: np.ndarray,
    reference: str = "msp",
    q: float = 0.10,
    n_folds: int = 5,
    seed: int = 0,
    min_size: int = MIN_REGION,
) -> BoundaryInformativeness:
    """Part 2 of the diagnostic: labelled headroom inside the abstention
    region, measured out-of-fold.

    The stack is a standardised L2 logistic regression on the full signal
    bank -- deliberately the *simplest* aggregator in the repo (`A1_logreg`),
    because the point is to measure the bank's information content, not an
    aggregator's capacity. Anything richer would confound "the bank carries
    extra signal" with "this particular model class can exploit it", and
    this project's own results already show capacity is not the binding
    constraint.

    Scored **out-of-fold**: the stack is fit on `n_folds - 1` folds and
    scores the held-out one. An in-sample AUROC here would be optimistic by
    exactly the amount that makes a useless bank look informative, which
    would defeat the purpose of a screening rule.

    The region is chosen by the reference signal on the *full* input, then
    intersected with each fold, so the region definition does not depend on
    the fold split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if reference not in signals:
        raise KeyError(f"reference signal {reference!r} not in bank {sorted(signals)}")

    names = list(signals)
    incorrect = np.asarray(incorrect).astype(int)
    ref = np.asarray(signals[reference], dtype=float)
    region = abstention_region(ref, q=q, min_size=min_size)
    U = np.column_stack([np.asarray(signals[nm], dtype=float) for nm in names])

    nan = BoundaryInformativeness(
        delta_local=float("nan"), auroc_stack=float("nan"),
        auroc_best_single=float("nan"), best_single_signal="",
        auroc_reference=float("nan"), n_region=len(region),
        n_errors_region=int(incorrect[region].sum()) if len(region) else 0, q=q,
    )
    # The stack needs both classes present in every training fold, so it
    # needs at least `n_folds` of each class overall. Below that the
    # diagnostic is genuinely unavailable and must say so rather than
    # returning a number built from one fold.
    if len(region) < 3 or incorrect.min() == incorrect.max():
        return nan
    if min(np.bincount(incorrect, minlength=2)) < n_folds:
        return nan

    oof = np.full(len(incorrect), np.nan)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(U, incorrect):
        if len(np.unique(incorrect[tr])) < 2:
            continue
        model = Pipeline(
            [("scale", StandardScaler()),
             ("clf", LogisticRegression(max_iter=2000, random_state=seed))]
        )
        model.fit(U[tr], incorrect[tr])
        oof[te] = model.predict_proba(U[te])[:, 1]

    scored = region[np.isfinite(oof[region])]
    if len(scored) < 3:
        return nan

    y_region = incorrect[scored]
    auroc_stack = _auroc(oof[scored], y_region)
    per_signal = {
        nm: _auroc(np.asarray(signals[nm], dtype=float)[scored], y_region) for nm in names
    }
    finite = {nm: v for nm, v in per_signal.items() if np.isfinite(v)}
    if not finite or not np.isfinite(auroc_stack):
        return dataclasses.replace(nan, n_region=len(scored),
                                   n_errors_region=int(y_region.sum()))

    best_nm = max(finite, key=finite.get)
    return BoundaryInformativeness(
        delta_local=float(auroc_stack - finite[best_nm]),
        auroc_stack=float(auroc_stack),
        auroc_best_single=float(finite[best_nm]),
        best_single_signal=best_nm,
        auroc_reference=float(per_signal.get(reference, float("nan"))),
        n_region=len(scored),
        n_errors_region=int(y_region.sum()),
        q=q,
    )


def effective_binarity(probs: np.ndarray, reference: np.ndarray, q: float = 0.10,
                       min_size: int = MIN_REGION) -> float:
    """Mean top-2 probability mass `p_(1) + p_(2)` on the abstention region.

    This is the *mechanistic driver* of `rho_local`, and the reason the
    K-decoupling experiment is targetable rather than a blind dataset sweep.

    Every Tier-A signal is a function of the whole probability vector, but
    when the top two classes carry almost all the mass the vector is
    effectively two-dimensional, and each Tier-A signal collapses to a
    function of the single top-2 gap -- exactly the `K = 2` situation in
    which they are provably rank-identical. So a `K >= 3` dataset whose
    abstention region has top-2 mass near 1 should behave like a binary one
    and show `rho_local` near 1, while one with mass spread over three or
    more competing classes should decorrelate.

    That prediction is what separates the mechanism from the confound: if
    `rho_local` merely tracked `K`, effective binarity would be irrelevant.
    Returns a value in [0, 1]; 1.0 for K = 2 by construction.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 2:
        return float("nan")
    region = abstention_region(reference, q=q, min_size=min_size)
    if len(region) == 0:
        return float("nan")
    top2 = np.sort(probs[region], axis=1)[:, -2:].sum(axis=1)
    return float(np.mean(top2))
