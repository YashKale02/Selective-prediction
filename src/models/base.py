"""Uniform base-model interface (§8: "every signal implements the same
interface... one file per signal"; models get the same treatment so signals
never need to know which base model produced them).

Every wrapper exposes:
    fit(X, y) -> self
    predict_proba(X) -> (n, K) array
    logits(X) -> (n, K) array of pre-softmax scores (real margins where the
                 model provides them; a monotone surrogate otherwise, noted
                 per-wrapper below)
    features(X) -> (n, d) array, the representation Tier-C signals compute
                   distances in. For tabular models without a learned
                   embedding, this is the preprocessed input space itself —
                   a documented approximation, not a hidden one.
    classes_ : array of class labels seen during fit (0..K-1)

Preprocessing (imputation, scaling, categorical encoding) is fit *inside*
each wrapper's `.fit`, never outside it, so that plugging a wrapper into the
cross-fitting loop in `src/data/splits.py` can never leak fold information
through a globally-fit preprocessor (§6, "fit preprocessing inside the
training fold only").
"""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd


class BaseModelWrapper(abc.ABC):
    classes_: np.ndarray

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModelWrapper": ...

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abc.abstractmethod
    def logits(self, X: pd.DataFrame) -> np.ndarray: ...

    @abc.abstractmethod
    def features(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
