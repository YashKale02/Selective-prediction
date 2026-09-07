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
    drop_cols: Optional[list[str]] = None,
    positive_class: Optional[str] = None,
) -> Dataset:
    """Fetch and cache a dataset from OpenML by numeric dataset id.

    Uses sklearn's fetch_openml under the hood (it talks to the same OpenML
    API and is already a project dependency), which is simpler and more
    robust across OpenML API versions than round-tripping through raw ARFF.

    `drop_cols` removes columns from X *after* caching (so the cache stays a
    faithful copy of the OpenML original). This is not cosmetic: on a
    temporal dataset, any column that encodes row order -- e.g.
    Diabetes-130's `encounter_id`, which is monotone in time -- lets a tree
    model read the time index directly, and because a temporal split puts
    every test value outside the training range, the model degenerates into
    a single branch. Such columns must be dropped, not merely ignored.

    `positive_class` binarises a multiclass target to
    `(y_raw == positive_class)`. Used for Diabetes-130, whose native target
    has three levels (`NO` / `>30` / `<30`) but whose standard task in the
    literature is the binary "readmitted within 30 days" (`<30`) question --
    which also keeps it comparable to the other (binary) datasets here.
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

    if drop_cols:
        present = [c for c in drop_cols if c in X.columns]
        missing = sorted(set(drop_cols) - set(present))
        if missing:
            raise ValueError(
                f"drop_cols for dataset {name} names column(s) not present: "
                f"{missing}. Fix the registry rather than silently ignoring "
                f"them -- a typo here would leave a leaking column in X."
            )
        X = X.drop(columns=present)

    if positive_class is not None:
        # Binarise: 1 == positive_class, 0 == everything else.
        as_str = y_raw.astype(str)
        levels = set(as_str.unique())
        if positive_class not in levels:
            raise ValueError(
                f"positive_class={positive_class!r} not found in target "
                f"{target_column!r} of dataset {name}; levels are "
                f"{sorted(levels)}."
            )
        y = (as_str == positive_class).to_numpy().astype(np.int64)
        classes = [f"not_{positive_class}", positive_class]
        task = "binary"
    else:
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
    # Diabetes-130 (Strack et al. 2014), the plan's second temporal-shift
    # dataset (§4). Verified before use, the same way `electricity` was:
    #   * Row order is time. `encounter_id` is ascending in row order with
    #     only 4 inversions in 101,765 adjacent pairs (all of them inside
    #     the first 8 rows), so `is_temporal=True` is justified.
    #   * `encounter_id` / `patient_nbr` are dropped from X. `encounter_id`
    #     is monotone in time, which under a temporal split would hand the
    #     model an explicit time index whose test values all lie outside the
    #     training range; `patient_nbr` is a bare identity the model could
    #     memorise (and is itself ~0.54 Spearman-correlated with row order).
    #   * `weight` is dropped: 96.9% of rows carry the '?' sentinel.
    #   * KNOWN CAVEAT, not fixable by column choice: patients recur across
    #     encounters (16,773 patients have >1 encounter, up to 40), so 19.3%
    #     of test rows belong to a patient also present in D_train∪D_meta.
    #     This is patient-level leakage in the strict sense, but it is also
    #     exactly the real deployment situation for a readmission model
    #     (you *do* see returning patients), and removing it would destroy
    #     the temporal semantics. Reported, not hidden -- see
    #     PROJECT_STATUS.md. A first-encounter-only variant is the obvious
    #     robustness check if a reviewer presses on it.
    #   * Target binarised to the standard "readmitted <30 days" task,
    #     keeping it comparable to the other binary datasets.
    "diabetes130": dict(
        dataset_id=4541,
        target_column="readmitted",
        subgroup_cols=["race", "gender", "age"],
        is_temporal=True,
        drop_cols=["encounter_id", "patient_nbr", "weight"],
        positive_class="<30",
    ),
}


def load(name: str) -> Dataset:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(REGISTRY)}")
    spec = dict(REGISTRY[name])
    dataset_id = spec.pop("dataset_id")
    ds = fetch_openml_dataset(dataset_id=dataset_id, name=name, **spec)
    return ds
