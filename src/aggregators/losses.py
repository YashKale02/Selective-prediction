"""The coverage-targeted losses from §3.3 — the thing that is supposed to
make A2 more than stacking (§11: "this is just stacking" is the expected
review objection; the A1-vs-A2 ablation on these losses is the direct
rebuttal, so it must be run early per §3.3/§9 week-5 gate).

All losses take:
    s          : (n,) tensor, the aggregator's continuous risk score
    incorrect  : (n,) tensor of {0,1}, 1 if the frozen base model was wrong

and are expressed directly in terms of the formulas in §3.3, with a
learnable threshold `tau` standing in for the operating point rather than a
fixed constant, so gradient descent can find the threshold that realizes
the target coverage kappa instead of it being a hand-picked constant.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_selective_risk_loss(
    s: torch.Tensor,
    incorrect: torch.Tensor,
    tau: torch.Tensor,
    kappa: float,
    T: float = 0.05,
    lam: float = 1.0,
) -> torch.Tensor:
    """Loss 1 (§3.3): soft selective risk at target coverage kappa.

    g(x) = sigmoid((tau - s(x)) / T)   -- soft "accept" gate (accept when
                                           s(x) is comfortably below tau)
    L = sum g*err / sum g  +  lam * relu(kappa - mean(g))^2
    """
    g = torch.sigmoid((tau - s) / T)
    num = (g * incorrect).sum()
    den = g.sum().clamp_min(1e-6)
    selective_risk = num / den
    coverage_penalty = F.relu(kappa - g.mean()) ** 2
    return selective_risk + lam * coverage_penalty


def aurc_surrogate_loss(
    s: torch.Tensor,
    incorrect: torch.Tensor,
    taus: torch.Tensor,
    kappa_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    T: float = 0.05,
    lam: float = 1.0,
) -> torch.Tensor:
    """Loss 2 (§3.3): sum Loss-1 over a grid of target coverages, so a
    single score is trained to be good across the whole risk-coverage
    curve rather than at one operating point. `taus` is a per-kappa
    learnable threshold vector, same length as `kappa_grid`."""
    total = s.new_zeros(())
    for tau_k, kappa in zip(taus, kappa_grid):
        total = total + soft_selective_risk_loss(s, incorrect, tau_k, kappa, T=T, lam=lam)
    return total / len(kappa_grid)


def pairwise_ranking_loss(
    s: torch.Tensor, incorrect: torch.Tensor, n_pairs: int = 2048, generator=None
) -> torch.Tensor:
    """Loss 3 (§3.3): for (correct, incorrect) pairs, penalise
    sigmoid(s(correct) - s(incorrect)) via a logistic pairwise loss —
    directly optimises the ranking AURC measures, and per §3.3 is "better
    behaved than Loss 1 on small tabular datasets"."""
    correct_idx = torch.nonzero(incorrect == 0, as_tuple=True)[0]
    incorrect_idx = torch.nonzero(incorrect == 1, as_tuple=True)[0]
    if len(correct_idx) == 0 or len(incorrect_idx) == 0:
        # Degenerate batch (e.g. a tiny meta-set or a perfect/failing base
        # model) -- no valid pairs, contribute zero loss rather than NaN.
        return s.new_zeros(())
    kwargs = {"generator": generator} if generator is not None else {}
    ci = correct_idx[torch.randint(0, len(correct_idx), (n_pairs,), **kwargs)]
    ii = incorrect_idx[torch.randint(0, len(incorrect_idx), (n_pairs,), **kwargs)]
    margin = s[ii] - s[ci]  # want this > 0 (incorrect scored higher-risk)
    return F.softplus(-margin).mean()


def bce_loss(s_logit: torch.Tensor, incorrect: torch.Tensor) -> torch.Tensor:
    """Plain correctness-classification loss -- what A1 optimises. Kept
    here so A2 can be run in "BCE mode" for the direct A1-vs-A2 ablation
    on loss choice alone, holding the function class fixed (§3.3, §7
    ablation 4)."""
    return F.binary_cross_entropy_with_logits(s_logit, incorrect)
