"""LightGBM wrapper — the primary tabular base model (§5: "it's the actual
state of the art here").

`logits`: LightGBM's `predict(..., raw_score=True)` returns the true
pre-sigmoid/pre-softmax margins, so this is an exact logit, not a surrogate.
For binary tasks LightGBM returns a single raw score per row (the positive
class margin); we expand it to a 2-column [0, z] logit matrix so every
downstream signal can assume a (n, K) logits shape uniformly.

`features`: LightGBM has no learned embedding, so we use the preprocessed
input space as the representation for Tier-C distance-based signals. This
is a standard, documented approximation for GBM base models (see §3.1,
Tier C docstring) — not a hidden shortcut.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .base import BaseModelWrapper
from .preprocessing import build_preprocessor


class LightGBMWrapper(BaseModelWrapper):
    def __init__(self, seed: int = 0, n_estimators: int = 300, **lgb_kwargs):
        self.seed = seed
        self.n_estimators = n_estimators
        self.lgb_kwargs = lgb_kwargs
        self.preprocessor = None
        self.model: LGBMClassifier | None = None
        self.n_classes_: int = 0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LightGBMWrapper":
        self.preprocessor = build_preprocessor(X)
        Xt = self.preprocessor.fit_transform(X)
        self.n_classes_ = int(len(np.unique(y)))
        objective = "binary" if self.n_classes_ == 2 else "multiclass"
        self.model = LGBMClassifier(
            random_state=self.seed,
            n_estimators=self.n_estimators,
            objective=objective,
            num_class=self.n_classes_ if objective == "multiclass" else None,
            verbosity=-1,
            **self.lgb_kwargs,
        )
        self.model.fit(Xt, y)
        self.classes_ = self.model.classes_
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._transform(X))

    def logits(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._transform(X)
        raw = self.model.predict(Xt, raw_score=True)
        raw = np.asarray(raw)
        if raw.ndim == 1:
            # binary: raw is the positive-class margin -> expand to (n, 2)
            z = np.zeros((raw.shape[0], 2), dtype=float)
            z[:, 1] = raw
            return z
        return raw

    def features(self, X: pd.DataFrame) -> np.ndarray:
        return self._transform(X)
