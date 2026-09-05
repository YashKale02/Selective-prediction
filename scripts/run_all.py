"""Hydra-driven experiment entry point (§8: "hydra-core -- do not
hand-roll argparse").

Examples:
    python scripts/run_all.py dataset=adult
    python scripts/run_all.py dataset=german_credit signals=tier_a
    python scripts/run_all.py dataset=adult experiment.n_seeds=10

Appends results to `results/results.parquet` (creating it if absent) --
the single long-format file every table/figure in the paper queries
(§8's non-negotiable engineering rule).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Python 3.14 / hydra-core 1.3.6 compatibility shim ---------------------
# Python 3.14's argparse added an eager `_check_help` validation at
# `add_argument` time that assumes `help=` is always a real string.
# hydra-core 1.3.6 (the latest release as of writing) registers its
# `--shell-completion` flag with a lazily-stringified `LazyCompletionHelp`
# object, which trips that new check and crashes before any of this
# project's own code runs -- entirely unrelated to anything in this repo.
# There is no newer hydra-core release that fixes this yet (checked via
# `pip index versions hydra-core`). Disabling the eager check is safe: it
# only restores the pre-3.14 behavior of resolving `help=` lazily when
# `--help` is actually invoked, which nothing here relies on anyway.
argparse.ArgumentParser._check_help = lambda self, action: None  # type: ignore[method-assign]

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.runner import run_experiment  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "results.parquet"


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    all_dfs = []
    for seed in range(cfg.experiment.n_seeds):
        print(f"[run_all] dataset={cfg.dataset.name} model={cfg.model.name} "
              f"tiers={cfg.signals.tiers} seed={seed}")
        df = run_experiment(
            dataset_name=cfg.dataset.name,
            model_name=cfg.model.name,
            tiers=tuple(cfg.signals.tiers),
            seed=seed,
            n_cv_folds=cfg.experiment.n_cv_folds,
            use_ensemble=cfg.signals.get("use_ensemble", False),
            n_ensemble_members=cfg.signals.get("n_ensemble_members", 5),
            conformal_delta=cfg.experiment.conformal_delta,
            subgroup_col=cfg.dataset.get("subgroup_col"),
        )
        all_dfs.append(df)

    new_results = pd.concat(all_dfs, ignore_index=True)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    if RESULTS_PATH.exists():
        existing = pd.read_parquet(RESULTS_PATH)
        # Replace any prior rows for this exact (dataset, model, seed) combo
        # so re-running a config is idempotent instead of duplicating rows.
        key_cols = ["dataset", "base_model", "seed"]
        mask = existing.set_index(key_cols).index.isin(
            new_results.set_index(key_cols).index
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_results], ignore_index=True)
    else:
        combined = new_results

    combined.to_parquet(RESULTS_PATH)
    print(f"[run_all] wrote {len(combined)} rows -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
