"""A2 — the coverage-targeted aggregator (§3.2-§3.3), the paper's headline
contribution, plus the instance-adaptive signal-weighting variant (§3.4,
"second novelty axis").

Both classes share the same function-class-vs-A1 comparison: same input
u(x), same general capacity (a small MLP), different training loss. That is
deliberate — §3.3 says the A1-vs-A2 ablation must hold the function class
fixed and vary only the loss, or the comparison proves nothing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .losses import aurc_surrogate_loss, bce_loss, pairwise_ranking_loss, soft_selective_risk_loss


class _Standardizer:
    def fit(self, U: np.ndarray) -> "_Standardizer":
        self.mean_ = U.mean(axis=0)
        self.std_ = U.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, U: np.ndarray) -> np.ndarray:
        return (U - self.mean_) / self.std_


def _make_mlp(m_in: int, hidden: int, out: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = m_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


class _BaseTorchAggregator:
    """Shared training loop for MLPAggregator and AdaptiveGatingAggregator.
    Subclasses implement `_build_net` and `_score_logit`."""

    KAPPA_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    def __init__(
        self,
        loss: str = "loss1",
        target_coverage: float = 0.8,
        hidden: int = 32,
        depth: int = 2,
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ):
        assert loss in ("bce", "loss1", "loss2", "loss3")
        self.loss_name = loss
        self.target_coverage = target_coverage
        self.hidden = hidden
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.scaler = _Standardizer()
        self.net: nn.Module | None = None
        self.tau: nn.Parameter | None = None  # loss1: scalar
        self.taus: nn.Parameter | None = None  # loss2: vector over KAPPA_GRID

    def _build_net(self, m_in: int) -> nn.Module:
        raise NotImplementedError

    def _score_logit(self, net_out: torch.Tensor, U_t: torch.Tensor) -> torch.Tensor:
        """Map the network's raw output (and, for the adaptive-gating
        subclass, the standardized input it was computed from) to a scalar
        risk logit per row."""
        raise NotImplementedError

    def fit(self, U_meta: np.ndarray, correct_meta: np.ndarray) -> "_BaseTorchAggregator":
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)

        Un = self.scaler.fit(U_meta).transform(U_meta)
        U_t = torch.tensor(Un, dtype=torch.float32)
        incorrect = torch.tensor(1 - correct_meta.astype(int), dtype=torch.float32)

        self.net = self._build_net(Un.shape[1])
        params = list(self.net.parameters())

        if self.loss_name == "loss1":
            self.tau = nn.Parameter(torch.zeros(()))
            params.append(self.tau)
        elif self.loss_name == "loss2":
            self.taus = nn.Parameter(torch.zeros(len(self.KAPPA_GRID)))
            params.append(self.taus)

        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.epochs):
            opt.zero_grad()
            net_out = self.net(U_t)
            logit = self._score_logit(net_out, U_t)
            s = torch.sigmoid(logit)

            if self.loss_name == "bce":
                loss = bce_loss(logit, incorrect)
            elif self.loss_name == "loss1":
                loss = soft_selective_risk_loss(
                    s, incorrect, torch.sigmoid(self.tau), kappa=self.target_coverage
                )
            elif self.loss_name == "loss2":
                loss = aurc_surrogate_loss(
                    s, incorrect, torch.sigmoid(self.taus), kappa_grid=self.KAPPA_GRID
                )
            else:  # loss3
                loss = pairwise_ranking_loss(s, incorrect, generator=gen)

            loss.backward()
            opt.step()

        return self

    def score(self, U: np.ndarray) -> np.ndarray:
        Un = self.scaler.transform(U)
        U_t = torch.tensor(Un, dtype=torch.float32)
        with torch.no_grad():
            logit = self._score_logit(self.net(U_t), U_t)
            s = torch.sigmoid(logit)
        return s.numpy()


class MLPAggregator(_BaseTorchAggregator):
    """A2 core: a small MLP over u(x) -> scalar risk score, trained with
    one of the losses in §3.3."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"A2_mlp_{self.loss_name}"

    def _build_net(self, m_in: int) -> nn.Module:
        return _make_mlp(m_in, self.hidden, out=1, depth=self.depth)

    def _score_logit(self, net_out: torch.Tensor, U_t: torch.Tensor) -> torch.Tensor:
        return net_out.squeeze(-1)


class AdaptiveGatingAggregator(_BaseTorchAggregator):
    """§3.4: instance-adaptive signal weighting. A gating network
    w(x) = softmax(h(u(x))) over the m signals, and
    s(x) = sum_j w_j(x) * z_j(x) where z_j is the per-signal z-score.
    Exposes `weights(U)` for the interpretability figure in §3.4 (plotting
    the learned weight distribution for in-distribution vs corrupted
    inputs)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"A2_adaptive_{self.loss_name}"

    def _build_net(self, m_in: int) -> nn.Module:
        return _make_mlp(m_in, self.hidden, out=m_in, depth=self.depth)

    def _score_logit(self, net_out: torch.Tensor, U_t: torch.Tensor) -> torch.Tensor:
        # net_out: (n, m) gating logits over the m standardized signals in
        # U_t; s(x) = sum_j w_j(x) * z_j(x), per §3.4.
        w = torch.softmax(net_out, dim=1)
        return (w * U_t).sum(dim=1)

    def weights(self, U: np.ndarray) -> np.ndarray:
        """Learned per-signal gating weights w(x) for interpretability
        (§3.4's headline figure)."""
        Un = self.scaler.transform(U)
        U_t = torch.tensor(Un, dtype=torch.float32)
        with torch.no_grad():
            logits = self.net(U_t)
            w = torch.softmax(logits, dim=1)
        return w.numpy()
