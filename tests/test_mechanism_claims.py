"""Integration tests for the paper's mechanism claims, run through the
*real* signal implementations rather than synthetic score vectors.

`tests/test_redundancy.py` pins the diagnostics' mathematical properties in
isolation. This file pins the claims the paper actually makes about the
pipeline, so that a change to a Tier-A signal cannot quietly invalidate a
sentence in the paper while every unit test still passes.

Deliberately self-contained: no OpenML fetch and no dependence on
`results/raw/`, which is gitignored (bulky, regenerable), so these run on a
fresh clone with no network.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.experiment.runner import HELD_OUT_FIT_SIGNALS
from src.metrics.redundancy import local_rank_redundancy
from src.models.logreg_model import LogRegWrapper
from src.signals.tier_a import default_tier_a_bank


def _binary_dataset(n=600, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
    y = (X["x1"] + 0.5 * X["x2"] + 0.4 * rng.randn(n) > 0).astype(int).values
    return X, y


def _multiclass_dataset(n=900, k=4, seed=0):
    """A K-class problem with genuinely three-way-ambiguous regions.

    Class means are placed on a circle so that near the centre several
    classes compete at once -- which is the condition under which the
    Tier-A bank stops being a function of one scalar. A well-separated
    blob layout would leave only pairwise confusions and reproduce the
    binary degeneracy, which is the very thing being distinguished here.
    """
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centres = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.2
    y = rng.randint(0, k, size=n)
    pts = centres[y] + rng.randn(n, 2) * 1.1
    X = pd.DataFrame({"x1": pts[:, 0], "x2": pts[:, 1]})
    return X, y


def _tier_a_scores(X, y, n_classes, seed=0):
    """The Tier-A sub-bank whose rank-degeneracy the theorem is about.

    Excludes the two signals that must be fit on a split the base model has
    not seen (temperature scaling, isotonic calibration residual): fitting
    them in-sample here would measure the model's own overconfidence rather
    than the geometric claim, and the runner holds them out for the same
    reason.
    """
    model = LogRegWrapper(seed=seed, n_classes=n_classes).fit(X, y)
    bank = [
        s for s in default_tier_a_bank(n_classes=n_classes)
        if not isinstance(s, HELD_OUT_FIT_SIGNALS)
    ]
    scores = {}
    for sig in bank:
        sig.fit(X, y, model)
        scores[sig.name] = sig.score(X, model)
    return scores


@pytest.mark.parametrize("q", [0.05, 0.10, 0.25, 0.50, 1.00])
def test_binary_tier_a_is_locally_degenerate_at_every_region_width(q):
    """The theorem, measured through the real signals.

    On a binary task every Tier-A signal is a strictly monotone function of
    the single logit gap, so the bank supplies exactly one ordering -- and
    that holds on *any* subset of points, hence at every `q`. This is what
    makes `rho_local == 1.0000` a mathematical certainty rather than an
    observation for all three binary datasets in the registry, and it is
    why no binary dataset can serve as the K-decoupling experiment.
    """
    X, y = _binary_dataset()
    scores = _tier_a_scores(X, y, n_classes=2)
    res = local_rank_redundancy(scores, reference="msp", q=q, sub_bank=list(scores))
    assert res.rho_local == pytest.approx(1.0, abs=1e-9), (
        f"binary Tier-A degeneracy broken at q={q}: "
        f"{res.argmin_signal} has |rho|={res.rho_local:.6f}"
    )
    # One ordinal degree of freedom, whatever the column count.
    assert res.effective_rank == pytest.approx(1.0, abs=1e-4)


def test_logitnorm_is_absent_from_the_binary_bank():
    # Not decoration: `logitnorm_msp` is provably constant at K=2, and a
    # constant scores as maximally redundant, so if it ever leaked into the
    # binary bank the test above would still pass while measuring the wrong
    # bank. Pinned so that stays true.
    scores = _tier_a_scores(*_binary_dataset(), n_classes=2)
    assert "logitnorm_msp" not in scores


def test_multiclass_breaks_the_degeneracy_locally():
    """The complement of the theorem: at K >= 3 the bank gains ordinal
    degrees of freedom, so `rho_local` must fall strictly below 1.

    Only the direction is asserted, not a magnitude: how far it falls is a
    property of the dataset (measured 0.23-0.46 on the registry's three
    multiclass datasets) and pinning a number here would make the test a
    restatement of one synthetic layout rather than of the claim.
    """
    X, y = _multiclass_dataset(k=4)
    scores = _tier_a_scores(X, y, n_classes=4)
    res = local_rank_redundancy(scores, reference="msp", q=0.10, sub_bank=list(scores))
    assert res.rho_local < 0.999, (
        "multiclass Tier-A bank is still locally rank-degenerate; the "
        "mechanism claim depends on this being false"
    )
    assert res.effective_rank > 1.05


def test_global_redundancy_understates_local_structure_on_real_signals():
    """The reason the mechanism looked unexplained for months.

    On a multiclass task the global minimum |Spearman| over Tier A can sit
    high enough to read as "redundant" while the abstention region carries
    a materially more independent ordering. Measured on the registry:
    satimage 0.89 global vs 0.33 local, letter 0.86 vs 0.46. Asserting the
    *inequality* (not the gap size, which is dataset-specific) pins the
    claim that the two statistics can disagree, and in which direction.
    """
    X, y = _multiclass_dataset(k=4)
    scores = _tier_a_scores(X, y, n_classes=4)
    msp = scores["msp"]
    global_rho = min(
        abs(spearmanr(msp, v).statistic) for k, v in scores.items() if k != "msp"
    )
    local = local_rank_redundancy(scores, reference="msp", q=0.10, sub_bank=list(scores))
    assert local.rho_local <= global_rho + 1e-9, (
        f"local redundancy ({local.rho_local:.4f}) exceeded global "
        f"({global_rho:.4f}); the diagnostic's premise is that restricting to "
        f"the abstention region can only reveal disagreement, not hide it"
    )


def test_rho_local_is_unchanged_by_a_monotone_reparameterisation():
    """AURC is invariant to monotone transforms of a score, so a diagnostic
    that claims to predict AURC outcomes must be too. A statistic that
    moved when MSP was replaced by, say, `-log(1 - msp)` would be measuring
    the parameterisation rather than the ranking."""
    X, y = _multiclass_dataset(k=4)
    scores = _tier_a_scores(X, y, n_classes=4)
    base = local_rank_redundancy(scores, reference="msp", q=0.10, sub_bank=list(scores))

    warped = dict(scores)
    warped["msp"] = np.expm1(3.0 * scores["msp"])  # strictly increasing
    warped["entropy"] = np.sqrt(scores["entropy"] - scores["entropy"].min() + 1e-12)
    after = local_rank_redundancy(warped, reference="msp", q=0.10, sub_bank=list(warped))

    assert after.rho_local == pytest.approx(base.rho_local, abs=1e-9)
    assert after.effective_rank == pytest.approx(base.effective_rank, abs=1e-6)
