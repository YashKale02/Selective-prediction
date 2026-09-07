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


# Gate temperature for the soft accept/reject decision.
#
# **Bug fix (vanishing gradients from an over-sharp gate).** This was 0.05,
# which makes `sigmoid((tau - s)/T)` a near-step function: its derivative
# carries a factor 1/T = 20 at the threshold but decays as
# `sigmoid'(20*(tau-s))`, falling below 0.018 once |tau - s| > 0.2. Since
# the caller passes `s = sigmoid(logit)`, which is itself already
# saturating, the two sigmoids compose into a double-saturation that leaves
# almost every training point with no usable gradient -- only the handful of
# rows sitting within ~0.2 of tau could learn anything, so the network
# barely moved from its initialisation. T=0.5 keeps the gate smooth across
# the whole [0, 1] range that `s` actually occupies; the coverage penalty
# below is what drives the gate towards a decisive split at convergence, so
# sharpness does not need to be baked into T.
DEFAULT_GATE_T = 0.5

# Weight on the coverage constraint.
#
# **Bug fix (constraint too weak to bind).** This was 1.0, which left the
# penalty numerically irrelevant: missing a target coverage of 0.8 by a full
# 10 points costs `(0.1)^2 * 1.0 = 0.01`, against a selective risk term of
# order 0.1-0.2. The optimiser could therefore ignore the requested coverage
# almost entirely and still report a good loss, which defeats the point of a
# *coverage-targeted* objective (§3.3). At 10.0 the same 10-point miss costs
# 0.1, i.e. comparable to the risk term, so the constraint actually binds.
DEFAULT_COVERAGE_LAMBDA = 10.0


def soft_selective_risk_loss(
    s: torch.Tensor,
    incorrect: torch.Tensor,
    tau: torch.Tensor,
    kappa: float,
    T: float = DEFAULT_GATE_T,
    lam: float = DEFAULT_COVERAGE_LAMBDA,
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
    T: float = DEFAULT_GATE_T,
    lam: float = DEFAULT_COVERAGE_LAMBDA,
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
    s_logit: torch.Tensor, incorrect: torch.Tensor, n_pairs: int = 2048, generator=None
) -> torch.Tensor:
    """Loss 3 (§3.3): for (correct, incorrect) pairs, penalise a logistic
    pairwise loss on the score margin — directly optimises the ranking that
    AURC measures, and per §3.3 is "better behaved than Loss 1 on small
    tabular datasets".

    **Bug fix (margin was computed on squashed scores).** The caller used to
    pass `s = sigmoid(logit)`, so the margin `s[incorrect] - s[correct]` was
    confined to (-1, 1). That capped how well the objective could ever be
    satisfied -- `softplus(-margin)` cannot fall below `softplus(-1) = 0.313`
    even for a *perfectly* separated ranking -- so a converged model still
    reported a large loss, and worse, the gradient through the sigmoid
    vanished exactly where the ranking was becoming confident. The parameter
    is now the raw **logit**, giving unbounded margins, a loss that tends to
    0 for a correct ranking, and healthy gradients throughout.

    Note that only the *scale* of the margin changes, not the ranking being
    optimised: sigmoid is strictly monotone, so the sign of every pairwise
    comparison is identical either way.
    """
    correct_idx = torch.nonzero(incorrect == 0, as_tuple=True)[0]
    incorrect_idx = torch.nonzero(incorrect == 1, as_tuple=True)[0]
    if len(correct_idx) == 0 or len(incorrect_idx) == 0:
        # Degenerate batch (e.g. a tiny meta-set or a perfect/failing base
        # model) -- no valid pairs, contribute zero loss rather than NaN.
        return s_logit.new_zeros(())
    kwargs = {"generator": generator} if generator is not None else {}
    ci = correct_idx[torch.randint(0, len(correct_idx), (n_pairs,), **kwargs)]
    ii = incorrect_idx[torch.randint(0, len(incorrect_idx), (n_pairs,), **kwargs)]
    margin = s_logit[ii] - s_logit[ci]  # want > 0 (incorrect = higher risk)
    return F.softplus(-margin).mean()


def bce_loss(s_logit: torch.Tensor, incorrect: torch.Tensor) -> torch.Tensor:
    """Plain correctness-classification loss -- what A1 optimises. Kept
    here so A2 can be run in "BCE mode" for the direct A1-vs-A2 ablation
    on loss choice alone, holding the function class fixed (§3.3, §7
    ablation 4)."""
    return F.binary_cross_entropy_with_logits(s_logit, incorrect)
