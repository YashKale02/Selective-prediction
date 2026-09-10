"""Screen OpenML candidates for the rho_local / K decoupling experiment.

The problem this solves
-----------------------
`NOVELTY_ANALYSIS.md` §3 shows `rho_local` separates all six current
datasets perfectly, but all three "aggregation helps" datasets are K >= 6
and all three "hurts" datasets are K = 2. So the mechanism is confounded
with class count, and a reviewer's first question is "isn't rho_local just
a proxy for multiclass?"

One direction of that test is provably unreachable. On a binary task every
Tier-A signal is a strictly monotone function of the single logit gap, so
`rho_local(TierA) == 1.0000` *by theorem*, at every q, for every binary
dataset -- there is no binary dataset with low Tier-A `rho_local` to find.
(Extending the minimum over the full A+B+C bank does not rescue it: Tier C
decorrelates from MSP on every dataset, binary or not, so the full-bank
minimum collapses to ~0 everywhere and separates nothing. Measured, and
recorded in `src/metrics/redundancy.py`.)

That leaves exactly one decisive experiment: **a K >= 3 dataset that is
nevertheless locally redundant** (`rho_local` near 1). If aggregation fails
there, `rho_local` predicts the outcome where `K >= 3` does not, and the
confound is broken. If aggregation still helps, the mechanism really is
just class count and the contribution shrinks to a reframing -- which is a
result worth having either way.

Why the search is targeted rather than blind
--------------------------------------------
`effective_binarity` -- the mean top-2 probability mass in the abstention
region -- is the mechanistic driver. When the top two classes carry nearly
all the mass, the probability vector is effectively 2-dimensional and every
Tier-A signal collapses to a function of the single top-2 gap, reproducing
the binary degeneracy at K >= 3. So the candidate we want is a multiclass
dataset whose *confusions are pairwise*: many classes overall, but only two
in contention at any given uncertain point.

Screening protocol
------------------
Deliberately cheap -- one split, one base model, Tier-A signals only. It
must be possible to screen dozens of candidates for the price of one real
run, so nothing here cross-fits, trains an ensemble, or builds Tier C.

Two gates, both hard-won lessons already recorded in this repo:

  * **test errors >= MIN_TEST_ERRORS.** A risk-coverage curve is built
    entirely out of the errors in D_test, so an accurate dataset yields a
    curve made of noise however many rows it has. `segment` (2310 rows,
    98.6% accurate) gives *five* test errors. Screen on error count, never
    row count.
  * **K >= 3**, since the binary case is settled analytically.

Usage:
    python scripts/screen_redundancy_candidates.py                # default list
    python scripts/screen_redundancy_candidates.py --ids 23,60,1475
    python scripts/screen_redundancy_candidates.py --q 0.25
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import splits as split_utils  # noqa: E402
from src.data.loaders import fetch_openml_dataset  # noqa: E402
from src.metrics.redundancy import (  # noqa: E402
    effective_binarity,
    local_rank_redundancy,
)
from src.experiment.runner import HELD_OUT_FIT_SIGNALS  # noqa: E402
from src.metrics.selective import aurc  # noqa: E402
from src.models.lightgbm_model import LightGBMWrapper  # noqa: E402
from src.signals.tier_a import default_tier_a_bank  # noqa: E402

# A curve built on fewer errors than this cannot resolve the ~6% AURC
# effect the paper is about; see `scripts/analyze_selection_bias.py` for
# the power calculation that pins the number down.
MIN_TEST_ERRORS = 50

# `rho_local` at or above this counts as "locally redundant", i.e. a
# candidate for the decoupling experiment. Chosen from the observed gap:
# the three datasets where aggregation helps sit at 0.25-0.46, so anything
# above ~0.8 is unambiguously on the redundant side of that gap.
REDUNDANT_THRESHOLD = 0.80

# Candidate multiclass datasets, with the target column each needs. Drawn
# from OpenML-CC18 and neighbours, spanning a wide range of class counts
# and base accuracies so the screen sees both regimes rather than only the
# one it hopes to find. Ids are OpenML data ids.
CANDIDATES: list[tuple[int, str, str]] = [
    (23, "cmc", "Contraceptive_method_used"),
    (60, "waveform_5000", "class"),
    (54, "vehicle", "Class"),
    (188, "eucalyptus", "Utility"),
    (1475, "first_order_theorem_proving", "Class"),
    (1478, "har", "Class"),
    (375, "japanese_vowels", "speaker"),
    (1466, "cardiotocography", "Class"),
    (40670, "dna", "class"),
    (469, "analcatdata_dmft", "Prevention"),
    (307, "vowel", "Class"),
    (40691, "wine_quality_red", "class"),
    (1497, "wall_robot_navigation", "Class"),
    (26, "nursery", "class"),
    (12, "mfeat_factors", "class"),
    (1476, "gas_drift", "Class"),
]


def screen_one(
    dataset_id: int, name: str, target: str, q: float, seed: int = 0
) -> dict:
    ds = fetch_openml_dataset(dataset_id, name=name, target_column=target)
    n_classes = int(len(ds.classes))
    sp = split_utils.four_way_split(len(ds.y), y=ds.y, seed=seed, temporal=False)

    # Screening fits on D_train only and evaluates on D_test. That is not
    # the deployment model the real pipeline uses (which also sees D_meta),
    # so the accuracy here is a slight underestimate -- fine for a screen,
    # and it keeps the cost to one model fit.
    X_train, y_train = ds.X.iloc[sp.train_idx], ds.y[sp.train_idx]
    X_test, y_test = ds.X.iloc[sp.test_idx], ds.y[sp.test_idx]

    model = LightGBMWrapper(seed=seed, n_classes=n_classes).fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    incorrect = (probs.argmax(axis=1) != y_test).astype(int)

    # Temperature scaling and the calibration residual are excluded: both
    # must be fit on a split the base model has not seen (D_meta), which
    # this one-fit screen does not set up. The remaining Tier-A signals
    # are exactly the ones whose binary rank-degeneracy is proven, so
    # rho_local here measures the quantity the theorem is about.
    bank = [
        s for s in default_tier_a_bank(n_classes=n_classes)
        if not isinstance(s, HELD_OUT_FIT_SIGNALS)
    ]
    signals = {}
    for sig in bank:
        sig.fit(X_train, y_train, model)
        signals[sig.name] = sig.score(X_test, model)

    red = local_rank_redundancy(signals, reference="msp", q=q, sub_bank=list(signals))
    return dict(
        dataset_id=dataset_id,
        name=name,
        K=n_classes,
        n_rows=len(ds.y),
        n_test=len(y_test),
        test_errors=int(incorrect.sum()),
        base_acc=float(1.0 - incorrect.mean()),
        rho_local=red.rho_local,
        argmin_signal=red.argmin_signal,
        eff_rank=red.effective_rank,
        eff_binarity=effective_binarity(probs, signals["msp"], q=q),
        msp_aurc=float(aurc(signals["msp"], incorrect)),
        n_tier_a=len(signals),
    )


def verdict(row: pd.Series) -> str:
    if row.K < 3:
        return "skip: binary (settled analytically)"
    if row.test_errors < MIN_TEST_ERRORS:
        return f"reject: only {row.test_errors} test errors"
    if row.rho_local >= REDUNDANT_THRESHOLD:
        return "*** DECISIVE CANDIDATE: K>=3 and locally redundant ***"
    return "usable, but locally non-redundant (adds a data point, not a decoupling)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", default="", help="comma-separated OpenML ids to restrict to")
    p.add_argument("--q", type=float, default=0.10, help="abstention-region fraction")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    wanted = {int(x) for x in args.ids.split(",") if x.strip()} or None
    candidates = [c for c in CANDIDATES if wanted is None or c[0] in wanted]

    rows, failures = [], []
    for dataset_id, name, target in candidates:
        print(f"[screen] {name} (id={dataset_id}) ...", flush=True)
        try:
            rows.append(screen_one(dataset_id, name, target, q=args.q, seed=args.seed))
        except Exception as exc:  # noqa: BLE001 -- a screen must survive one bad candidate
            # Recorded rather than swallowed: a candidate that fails to
            # load is not evidence about the mechanism, and silently
            # dropping it would make the screen look cleaner than it was.
            failures.append((name, dataset_id, f"{type(exc).__name__}: {exc}"))
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=1)

    if not rows:
        raise SystemExit("no candidate screened successfully")

    df = pd.DataFrame(rows).sort_values("rho_local", ascending=False)
    df["verdict"] = df.apply(verdict, axis=1)

    out = ROOT / "paper" / "tables" / f"redundancy_screen_q{args.q}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    show = ["name", "K", "n_test", "test_errors", "base_acc", "rho_local",
            "eff_binarity", "eff_rank", "argmin_signal"]
    print(f"\n=== Redundancy screen, q={args.q} (sorted by rho_local) ===")
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Verdicts ===")
    for _, r in df.iterrows():
        print(f"  {r['name']:<28} K={r['K']:<3} errors={r['test_errors']:<5} "
              f"rho_local={r['rho_local']:.4f}  {r['verdict']}")

    decisive = df[df.verdict.str.startswith("***")]
    print(f"\n{len(decisive)} decisive candidate(s) for the K-decoupling experiment.")
    if len(decisive):
        print("Register these and run the full pipeline on them:")
        for _, r in decisive.iterrows():
            print(f"  - {r['name']} (id={r['dataset_id']}, K={r['K']}, "
                  f"rho_local={r['rho_local']:.4f}, "
                  f"eff_binarity={r['eff_binarity']:.4f})")
    else:
        print("None found in this candidate list. The mechanism claim stays "
              "confounded with K until one is; widen --ids before writing §3 "
              "as a mechanism rather than an observation.")

    if failures:
        print(f"\n{len(failures)} candidate(s) failed to screen:")
        for name, dataset_id, err in failures:
            print(f"  - {name} (id={dataset_id}): {err}")

    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
