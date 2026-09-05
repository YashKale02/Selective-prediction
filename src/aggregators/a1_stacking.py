"""A1 — linear + GBM stacking on correctness (§3.2): fit a classifier to
predict `1[f(x) != y]` from `u(x)` on a disjoint meta-split. This is
ordinary stacking; it is the internal baseline A2 has to beat (§3.3,
§11 risk register: "this is just stacking" is the expected review
objection, and the A1-vs-A2 ablation is the direct answer to it).
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


class LogRegStackingAggregator:
    name = "A1_logreg"

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C = C
        self.seed = seed
        self.model: LogisticRegression | None = None

    def fit(self, U_meta: np.ndarray, correct_meta: np.ndarray) -> "LogRegStackingAggregator":
        incorrect = 1 - correct_meta.astype(int)
        self.model = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        self.model.fit(U_meta, incorrect)
        return self

    def score(self, U: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(U)[:, 1]


class LightGBMStackingAggregator:
    name = "A1_lightgbm"

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed
        self.model: LGBMClassifier | None = None

    def fit(self, U_meta: np.ndarray, correct_meta: np.ndarray) -> "LightGBMStackingAggregator":
        incorrect = 1 - correct_meta.astype(int)
        self.model = LGBMClassifier(
            n_estimators=self.n_estimators, random_state=self.seed, verbosity=-1
        )
        self.model.fit(U_meta, incorrect)
        return self

    def score(self, U: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(U)[:, 1]
