"""Unit tests for the boundary-local redundancy diagnostics
(`src/metrics/redundancy.py`), which are the paper's central mechanism
claim -- so each analytically-known property is pinned here rather than
verified only by the fact that a downstream table looked plausible.

The load-bearing test is `test_global_redundancy_can_hide_local_structure`:
it constructs a bank that a *global* correlation reads as redundant while
the abstention region carries a genuinely independent ordering. That is
precisely the failure mode that made this project's own mechanism look
unexplained, and it is the reason the statistic is local.
"""
import numpy as np
import pytest

from src.metrics.redundancy import (
    abstention_region,
    effective_binarity,
    effective_rank,
    incremental_boundary_auroc,
    local_rank_redundancy,
)

TIER_A_LIKE = ["msp", "entropy", "margin"]


def test_monotone_transforms_give_rho_exactly_one():
    # The binary Tier-A situation: every signal is a strictly monotone
    # function of one latent scalar, so all induce the same ranking and
    # rho_local must be exactly 1 -- not merely close to it. This is the
    # analytic case the project already proved for K = 2.
    rng = np.random.default_rng(0)
    t = rng.normal(size=500)
    signals = {
        "msp": t,
        "entropy": np.tanh(3 * t),  # strictly increasing
        "margin": np.exp(t),  # strictly increasing
        "flipped": -2.0 * t - 7.0,  # strictly decreasing -> |rho| = 1 too
    }
    res = local_rank_redundancy(signals, reference="msp", q=0.10)
    assert res.rho_local == pytest.approx(1.0, abs=1e-12)
    # One ordinal degree of freedom, however many columns there are.
    assert res.effective_rank == pytest.approx(1.0, abs=1e-6)


def test_independent_signal_drives_rho_down():
    rng = np.random.default_rng(1)
    t = rng.normal(size=800)
    signals = {"msp": t, "entropy": t**3, "noise": rng.normal(size=800)}
    res = local_rank_redundancy(signals, reference="msp", q=0.10)
    assert res.argmin_signal == "noise"
    assert res.rho_local < 0.4
    # entropy is still a monotone transform of the reference, so the bank
    # supplies two orderings, not three.
    assert 1.5 < res.effective_rank < 2.6


def test_global_redundancy_can_hide_local_structure():
    """The core claim: a bank can be globally redundant and locally not.

    `partner` tracks the reference almost perfectly over the bulk of the
    range but is scrambled inside the top decile. A global Spearman is
    dominated by the 90% that agrees and reads ~1; the local statistic
    sees the disagreement where AURC actually integrates.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(2)
    n = 2000
    ref = np.sort(rng.normal(size=n))  # ascending uncertainty
    partner = ref.copy()
    tail = slice(int(0.9 * n), n)  # the abstention region
    partner[tail] = rng.permutation(partner[tail])

    signals = {"msp": ref, "partner": partner}

    global_rho = abs(spearmanr(ref, partner).statistic)
    local = local_rank_redundancy(signals, reference="msp", q=0.10)

    assert global_rho > 0.97, "the construction should look redundant globally"
    assert local.rho_local < 0.3, "and independent locally"
    # The whole point: the two statistics disagree by a wide margin.
    assert global_rho - local.rho_local > 0.6


def test_constant_signal_counts_as_redundant_not_independent():
    # A signal that is constant on the region supplies no ordering, so it
    # can never reorder anything. Scoring it as rho = 0 (maximally
    # independent) would make a useless column look like the bank's most
    # valuable one -- the exact inversion this guards against.
    rng = np.random.default_rng(3)
    t = rng.normal(size=300)
    signals = {"msp": t, "dead": np.full(300, 0.5)}
    res = local_rank_redundancy(signals, reference="msp", q=0.10)
    assert res.per_signal["dead"] == pytest.approx(1.0)


def test_region_is_the_most_uncertain_points_and_nests():
    ref = np.arange(100, dtype=float)
    small = abstention_region(ref, q=0.10, min_size=1)
    big = abstention_region(ref, q=0.25, min_size=1)
    # Higher = abstain sooner, so the region is the top of the range.
    assert small.min() >= 90
    assert len(small) == 10 and len(big) == 25
    # Nesting makes a q-sweep interpretable as a widening window.
    assert set(small).issubset(set(big))


def test_region_respects_min_size_floor_on_small_inputs():
    ref = np.arange(50, dtype=float)
    region = abstention_region(ref, q=0.10, min_size=30)
    assert len(region) == 30  # 10% would be 5 points, which is noise
    assert len(abstention_region(np.arange(10, dtype=float), q=0.1, min_size=30)) == 10


def test_effective_rank_counts_independent_orderings():
    rng = np.random.default_rng(4)
    n = 4000
    base = rng.normal(size=n)
    # Rank-identical rows, including a monotone *non-linear* one: the
    # statistic must read one ordering, not three. A Pearson version on
    # raw values reports ~1.35 here, which is why ranking happens inside.
    redundant = np.vstack([base, 2 * base + 1, np.exp(base)])
    assert effective_rank(redundant) == pytest.approx(1.0, abs=0.05)
    independent = rng.normal(size=(4, n))
    assert effective_rank(independent) == pytest.approx(4.0, abs=0.3)


def test_missing_reference_raises_rather_than_substituting():
    # Silently measuring against a different reference would return a
    # plausible number answering a different question.
    with pytest.raises(KeyError):
        local_rank_redundancy({"entropy": np.arange(10.0)}, reference="msp")
    with pytest.raises(KeyError):
        incremental_boundary_auroc(
            {"entropy": np.arange(10.0)}, np.zeros(10, dtype=int), reference="msp"
        )


def test_sub_bank_restriction_changes_the_claim():
    # rho_local over Tier A is the headline statistic; adding a
    # geometry-style column that decorrelates everywhere collapses the
    # minimum to ~0 and separates nothing. Pinned because this is the
    # documented reason the full-bank variant is degenerate.
    rng = np.random.default_rng(5)
    t = rng.normal(size=600)
    signals = {"msp": t, "entropy": t**3, "knn": rng.normal(size=600)}
    tier_a_only = local_rank_redundancy(signals, reference="msp", q=0.10,
                                        sub_bank=TIER_A_LIKE[:2])
    full_bank = local_rank_redundancy(signals, reference="msp", q=0.10)
    assert tier_a_only.rho_local == pytest.approx(1.0, abs=1e-12)
    assert full_bank.rho_local < 0.4


def test_incremental_auroc_is_positive_when_bank_adds_information():
    # Errors driven by four weakly-informative signals, none of which is
    # close to sufficient alone. A stack over the bank must rank errors
    # inside the abstention region substantially better than the best
    # single signal can -- that gap is exactly the headroom an aggregator
    # could realise, so it must be detected clearly, not marginally.
    rng = np.random.default_rng(6)
    n = 3000
    parts = {f"s{i}": rng.normal(size=n) for i in range(4)}
    logit = 1.2 * sum(parts.values())
    incorrect = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    signals = {"msp": parts["s0"], **{k: v for k, v in parts.items() if k != "s0"}}

    res = incremental_boundary_auroc(signals, incorrect, reference="msp", q=0.25, seed=0)
    # Measured 0.170 for this construction, against -0.006 for the
    # redundant bank in the next test: the two regimes are far apart, so
    # the threshold does not need to sit on a knife edge.
    assert res.delta_local > 0.08
    assert res.auroc_stack > res.auroc_best_single


def test_incremental_auroc_is_not_positive_on_a_redundant_bank():
    # A bank of monotone transforms of the reference carries no extra
    # information, so out-of-fold stacking must not beat the best single
    # signal by any meaningful margin. (Small negative values are expected
    # and fine -- that is the honest cost of estimating a stack.)
    rng = np.random.default_rng(7)
    n = 3000
    ref = rng.normal(size=n)
    incorrect = (rng.random(n) < 1 / (1 + np.exp(-ref))).astype(int)
    res = incremental_boundary_auroc(
        {"msp": ref, "mono1": np.tanh(ref), "mono2": ref**3},
        incorrect, reference="msp", q=0.25, seed=0,
    )
    assert res.delta_local < 0.02


def test_incremental_auroc_baseline_is_best_single_not_reference():
    # wine_quality_white in miniature: a non-reference signal is the strong
    # one. The diagnostic must measure headroom over *that*, otherwise it
    # reports large headroom on exactly the dataset where aggregation adds
    # almost nothing over the best single signal.
    rng = np.random.default_rng(8)
    n = 3000
    ref = rng.normal(size=n)
    strong = rng.normal(size=n)
    incorrect = (rng.random(n) < 1 / (1 + np.exp(-3.0 * strong))).astype(int)
    res = incremental_boundary_auroc(
        {"msp": ref, "strong": strong}, incorrect, reference="msp", q=0.25, seed=0
    )
    assert res.best_single_signal == "strong"
    assert res.auroc_best_single > res.auroc_reference
    # Headroom over the best single signal is small; over the reference it
    # would look large. Both are reported so the contrast is visible.
    assert res.delta_local < 0.05
    assert res.auroc_stack - res.auroc_reference > 0.1


def test_incremental_auroc_returns_nan_when_region_has_no_errors():
    signals = {"msp": np.arange(200.0)}
    incorrect = np.zeros(200, dtype=int)
    res = incremental_boundary_auroc(signals, incorrect, reference="msp", q=0.1)
    assert np.isnan(res.delta_local)


def test_effective_binarity_is_one_for_two_classes():
    rng = np.random.default_rng(9)
    p = rng.random((200, 2))
    p /= p.sum(axis=1, keepdims=True)
    ref = rng.normal(size=200)
    assert effective_binarity(p, ref, q=0.1) == pytest.approx(1.0, abs=1e-12)


def test_effective_binarity_detects_a_third_competing_class():
    n = 400
    ref = np.arange(n, dtype=float)
    near_binary = np.tile([0.5, 0.49, 0.01], (n, 1))
    three_way = np.tile([0.34, 0.33, 0.33], (n, 1))
    assert effective_binarity(near_binary, ref, q=0.1) == pytest.approx(0.99, abs=1e-9)
    assert effective_binarity(three_way, ref, q=0.1) == pytest.approx(0.67, abs=1e-9)
