"""Wine-quality-white trust_score-exclusion ablation.

Tests whether `trust_score` and the aggregator's advantage over the best
*remaining* single signal are substitutes -- i.e. whether trust_score's
strength is masking a Tier-A-driven local-redundancy-breaking effect that
would otherwise show up as aggregator gain (see NOVELTY_ANALYSIS.md §3's
`rho_local` diagnostic, and the "wine anomaly": low rho_local but only a
-1.3% aggregator-vs-best-single-signal edge, the smallest of the three
"helps" datasets).

Deliberately does **not** touch `results/results.parquet`: this ablation
changes what's *in* the signal bank for a tier label that already means
something else there ("AC" = the full Tier A + Tier C bank), so writing
into the same file under the same key would either collide with the real
AC-tier wine rows or need a new key convention project-wide. Writes to a
separate file instead.

No new dataset, no new base-model architecture, no change to n_estimators
or splits -- only the signal bank composition changes, so this is cheap:
same cross-fitting cost as any other tier_a_c run.

Usage:
    python scripts/run_trust_score_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment.runner import run_experiment  # noqa: E402

OUT_PATH = ROOT / "results" / "ablation_wine_no_trust_score.parquet"


def main() -> None:
    rows = []
    for seed in range(10):
        print(f"[ablation] wine_quality_white, trust_score excluded, seed={seed}")
        df = run_experiment(
            "wine_quality_white",
            tiers=("A", "C"),
            seed=seed,
            exclude_signals=("trust_score",),
        )
        rows.append(df)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(OUT_PATH)
    print(f"[ablation] wrote {len(combined)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
