"""Tier C signals (§3.1): representation / data-geometry signals that need
the training set available at inference time. All of them call
`model.features(X)` for the representation (see the caveat in
`models/base.py` about tabular models using the preprocessed input space
rather than a learned embedding).

Leakage note (§6): "Tier C signals index the training set only. Never let a
test point be its own neighbour." Every signal here is fit once on
D_train's features and never re-fit or re-indexed on the split being
scored, so a query point drawn from D_meta/D_cal/D_test can never appear in
its own neighbour set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import Signal


class _TrainIndexedSignal(Signal):
    """Shared machinery: fit a k-NN index on D_train's feature
    representation and labels, once."""

    def __init__(self, k: int = 10):
        self.k = k
        self.nn: NearestNeighbors | None = None
        self.y_train: np.ndarray | None = None
        self.n_classes: int = 0

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, model) -> "_TrainIndexedSignal":
        feats = model.features(X_train)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(feats)
        self.y_train = np.asarray(y_train)
        self.n_classes = int(self.y_train.max()) + 1
        return self

    def _neighbors(self, X: pd.DataFrame, model):
        feats = model.features(X)
        dist, idx = self.nn.kneighbors(feats, n_neighbors=self.k)
        return dist, idx


class KNNDistanceSignal(_TrainIndexedSignal):
    """Mean distance to the k nearest training points (§3.1 item 13).
    Larger distance -> further from the training manifold -> more
    uncertain."""

    name = "knn_distance"

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        dist, _ = self._neighbors(X, model)
        return dist.mean(axis=1)


class LocalLabelAgreementSignal(_TrainIndexedSignal):
    """Fraction of the k nearest training neighbours sharing the *predicted*
    label (§3.1 item 14) — the one Tier-C signal that targets aleatoric
    rather than epistemic uncertainty (genuinely ambiguous regions of
    input space, not just far-from-training ones). Negated so higher score
    = more uncertain (low local agreement with the model's own prediction)."""

    name = "local_label_agreement"

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        pred = model.predict_proba(X).argmax(axis=1)
        _, idx = self._neighbors(X, model)
        neighbor_labels = self.y_train[idx]  # (n, k)
        agreement = (neighbor_labels == pred[:, None]).mean(axis=1)
        return 1.0 - agreement


class TrustScoreSignal(_TrainIndexedSignal):
    """Trust score (§2 Jiang et al. 2018, §3.1 item 15): ratio of the
    distance to the nearest training point of a class *other than* the
    predicted one, over the distance to the nearest training point *of*
    the predicted class. High ratio = trustworthy; we negate to the
    project's "higher = more uncertain" convention."""

    name = "trust_score"

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, model) -> "TrustScoreSignal":
        feats = model.features(X_train)
        self.y_train = np.asarray(y_train)
        self.n_classes = int(self.y_train.max()) + 1
        self.per_class_nn = {
            c: NearestNeighbors(n_neighbors=1).fit(feats[self.y_train == c])
            for c in range(self.n_classes)
            if (self.y_train == c).sum() > 0
        }
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        feats = model.features(X)
        pred = model.predict_proba(X).argmax(axis=1)
        n = feats.shape[0]
        dist_to_class = np.full((n, self.n_classes), np.inf)
        for c, nn in self.per_class_nn.items():
            d, _ = nn.kneighbors(feats, n_neighbors=1)
            dist_to_class[:, c] = d[:, 0]

        d_pred = dist_to_class[np.arange(n), pred]
        masked = dist_to_class.copy()
        masked[np.arange(n), pred] = np.inf
        d_other = masked.min(axis=1)
        trust = d_other / np.clip(d_pred, 1e-12, None)
        return -trust


class MahalanobisSignal(Signal):
    """Mahalanobis distance to the predicted class's centroid in feature
    space (§2 Lee et al. 2018, §3.1 item 12), with a shared (tied) covariance
    across classes, shrinkage-regularised for numerical stability on
    small/high-dimensional splits."""

    name = "mahalanobis"

    def __init__(self, shrinkage: float = 1e-3):
        self.shrinkage = shrinkage
        self.centroids: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, model) -> "MahalanobisSignal":
        feats = model.features(X_train)
        y = np.asarray(y_train)
        n_classes = int(y.max()) + 1
        d = feats.shape[1]
        centroids = np.zeros((n_classes, d))
        cov = np.zeros((d, d))
        for c in range(n_classes):
            mask = y == c
            if mask.sum() == 0:
                continue
            centroids[c] = feats[mask].mean(axis=0)
            diff = feats[mask] - centroids[c]
            cov += diff.T @ diff
        cov /= max(len(y) - n_classes, 1)
        cov += self.shrinkage * np.eye(d)
        self.centroids = centroids
        self.precision = np.linalg.pinv(cov)
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        feats = model.features(X)
        pred = model.predict_proba(X).argmax(axis=1)
        diff = feats - self.centroids[pred]
        # Mahalanobis^2 = diff @ precision @ diff, per row
        return np.einsum("ij,jk,ik->i", diff, self.precision, diff)


def default_tier_c_bank(k: int = 10) -> list[Signal]:
    return [
        KNNDistanceSignal(k=k),
        LocalLabelAgreementSignal(k=k),
        TrustScoreSignal(),
        MahalanobisSignal(),
    ]
