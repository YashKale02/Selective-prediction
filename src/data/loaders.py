"""Dataset loaders.

Every loader returns a `Dataset` (features `X`, labels `y`, optional subgroup
columns for fairness analysis, and metadata). Pulled from OpenML per the
project plan §4, which asks that datasets come from curated suites
(OpenML-CC18 / Grinsztajn et al.) rather than being hand-assembled, so that
dataset choice can't be attacked as cherry-picked.

Raw OpenML pulls are cached to disk under `data_cache/` (gitignored) because
repeated fetches are slow and this module is called dozens of times per
experiment sweep.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclasses.dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray  # integer-encoded class labels, 0..K-1
    classes: list
    subgroup_cols: list[str]  # column names in X usable for RQ4 fairness analysis
    task: str  # "binary" or "multiclass"
    is_temporal: bool = False  # True if row order encodes time (for shift splits, §4)


def _openml_frame_cache_path(dataset_id: int) -> Path:
    return CACHE_DIR / f"openml_{dataset_id}.parquet"


def fetch_openml_dataset(
    dataset_id: int,
    name: str,
    target_column: Optional[str] = None,
    subgroup_cols: Optional[list[str]] = None,
    is_temporal: bool = False,
) -> Dataset:
    """Fetch and cache a dataset from OpenML by numeric dataset id.

    Uses sklearn's fetch_openml under the hood (it talks to the same OpenML
    API and is already a project dependency), which is simpler and more
    robust across OpenML API versions than round-tripping through raw ARFF.
    """
    cache_path = _openml_frame_cache_path(dataset_id)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        df = bunch.frame
        if target_column is None:
            target_column = bunch.target.name if bunch.target is not None else df.columns[-1]
        # Persist target column name in an attrs-preserving way: parquet drops
        # DataFrame.attrs, so we just require the caller to pass the same
        # target_column each time (documented below) or infer it below.
        df.to_parquet(cache_path)
        df.attrs["target_column"] = target_column

    if target_column is None:
        raise ValueError(
            f"target_column must be specified for dataset {dataset_id} on a cache hit "
            f"(parquet caching does not preserve DataFrame.attrs)."
        )

    y_raw = df[target_column]
    X = df.drop(columns=[target_column])

    # Encode target to 0..K-1 ints, preserving a stable class ordering.
    y_cat = pd.Categorical(y_raw)
    y = y_cat.codes.astype(np.int64)
    classes = list(y_cat.categories)
    task = "binary" if len(classes) == 2 else "multiclass"

    subgroup_cols = [c for c in (subgroup_cols or []) if c in X.columns]

    return Dataset(
        name=name,
        X=X,
        y=y,
        classes=classes,
        subgroup_cols=subgroup_cols,
        task=task,
        is_temporal=is_temporal,
    )


# --- Curated registry, per project plan §4 -------------------------------
# OpenML dataset ids verified against openml.org at the time this file was
# written. Re-check ids if OpenML re-numbers a dataset.
REGISTRY = {
    "adult": dict(
        dataset_id=1590,
        target_column="class",
        subgroup_cols=["sex", "race"],
    ),
    "german_credit": dict(
        dataset_id=31,
        target_column="class",
        subgroup_cols=["personal_status", "age", "foreign_worker"],
    ),
    "electricity": dict(
        dataset_id=151,
        target_column="class",
        subgroup_cols=[],
        is_temporal=True,
    ),
}


def load(name: str) -> Dataset:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(REGISTRY)}")
    spec = dict(REGISTRY[name])
    dataset_id = spec.pop("dataset_id")
    ds = fetch_openml_dataset(dataset_id=dataset_id, name=name, **spec)
    return ds
