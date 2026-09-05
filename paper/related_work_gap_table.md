# Related-work gap table (project plan §2, phase-1 deliverable)

**Status: template only, citations UNVERIFIED.** Every row below is
transcribed from `selective_prediction_project_plan.md` §2, which was
itself explicit that these are titles recalled from memory, not verified
citations. This session had no literature-search capability wired in, so
none of these have been checked against IEEE Xplore / arXiv / Semantic
Scholar. **Do this before writing the paper's related-work section** —
per the plan, "the empty cells in that table are your paper's
justification. Build the table before you commit to the contribution
wording," and an unverified or misattributed citation is worse than a
missing one.

For each row: confirm the exact title, authors, venue, and year; fix the
corresponding entry in `paper/references.bib` (which carries the same
`UNVERIFIED` note); then check the box.

| Verified? | Method | Signals used | Needs retraining? | Guarantee? | Evaluated under shift? | Subgroup analysis? |
|---|---|---|---|---|---|---|
| [ ] | Chow's rule (1970) | MSP | No | No (assumes correct posteriors) | No | No |
| [ ] | SelectiveNet (Geifman & El-Yaniv, ICML 2019) | learned selection head | **Yes** | No | Not by default | No |
| [ ] | Deep Gamblers (Liu et al., NeurIPS 2019) | abstention-as-extra-class | **Yes** | No | Not by default | No |
| [ ] | ConfidNet (Corbière et al., NeurIPS 2019) | auxiliary confidence head | **Yes** | No | Not by default | No |
| [ ] | Self-Adaptive Training (Huang et al., 2020) | training dynamics | **Yes** | No | Not by default | No |
| [ ] | Learning to Defer (Mozannar & Sontag, ICML 2020) | human-in-the-loop | **Yes** | No | Not by default | No |
| [ ] | CRL (Moon et al., ICML 2020) | confidence ranking | **Yes** | No | Not by default | No |
| [ ] | Feng et al., "Towards better selective classification" (ICLR 2023) | tuned MSP | No | No | Partially | No |
| [ ] | Cattelan & Silva, logit-norm repair (2023) | logit normalization | No (post-hoc) | No | Partially | No |
| [ ] | Jones et al., "Selective classification can magnify disparities" (ICLR 2021) | MSP | No | No | No | **Yes** |
| [ ] | 2024 benchmark (arXiv:2401.12708? — **confirm ID**) | many, compared | Mixed | Mixed | Mixed | Mixed |
| [ ] | Conformal risk control (Angelopoulos et al. 2022) / Learn-Then-Test (2021) | any (wraps a score) | No | **Yes** | Assumption-breaking under shift | No |
| — | **This work** | Tiers A+B+C (+D optional), aggregated | No | **Yes** (conformal wrapper) | Yes (temporal + CIFAR-C) | Yes (RQ4) |

## Why this table is the paper's justification

Reading across the columns: every learned-abstention baseline needs
retraining the base model; none of them ship a distribution-free
guarantee; shift and subgroup evaluation are inconsistent or absent across
the field. The empty/No cells are exactly the gap the four contributions
in `main.tex`'s introduction claim to fill. If verification changes any of
these cells (e.g. a method turns out to already report shift results),
update the contribution wording accordingly — the honesty of that mapping
matters more than how favorable the table looks.

## Verification checklist per entry

1. Search the exact title on arXiv / Semantic Scholar / IEEE Xplore.
2. Confirm authors, venue, and year match what's in `references.bib`.
3. Skim the abstract to confirm the "signals used" / "needs retraining"
   / "guarantee" / "shift" / "subgroup" columns above are accurate — the
   plan's one-line descriptions are a starting point, not verified facts.
4. Update `references.bib`'s entry (remove the `UNVERIFIED` note) and tick
   the box in this table.
