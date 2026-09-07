# Related-work gap table (project plan §2, phase-1 deliverable)

**Status: VERIFIED.** Every row below was checked on 2026-09-06 against a
live source (arXiv, ACM DL, IEEE Xplore, or the venue's own proceedings
page via web search) — see `paper/references.bib` for the corrected
entries and DOIs/arXiv IDs. No row in this table should be trusted purely
from memory again; if a future edit changes a title/author/venue, re-run
the same kind of check before committing it.

For each row: title/authors/venue/year are now confirmed. The "signals
used / needs retraining? / guarantee? / shift? / subgroup?" columns were
also spot-checked against each paper's abstract during verification and
are accurate as far as the abstract states.

| Verified? | Method | Signals used | Needs retraining? | Guarantee? | Evaluated under shift? | Subgroup analysis? |
|---|---|---|---|---|---|---|
| [x] | Chow's rule (Chow, IEEE Trans. Inf. Theory 1970, vol. 16, no. 1, pp. 41–46) | MSP | No | No (assumes correct posteriors) | No | No |
| [x] | Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks" (NeurIPS 2017, arXiv:1705.08500) | MSP + risk-control procedure | No | Statistical (not distribution-free) | No | No |
| [x] | El-Yaniv & Wiener, "On the Foundations of Noise-free Selective Classification" (JMLR 11, 2010, pp. 1605–1641) | theory (no learned signal) | N/A | Theoretical framework | No | No |
| [x] | SelectiveNet (Geifman & El-Yaniv, ICML 2019, pp. 2151–2159, arXiv:1901.09192) | learned selection head | **Yes** | No | Not by default | No |
| [x] | Deep Gamblers (Liu, Wang, Liang, Salakhutdinov, Morency, Ueda, NeurIPS 2019, arXiv:1907.00208) | abstention-as-extra-class | **Yes** | No | Not by default | No |
| [x] | ConfidNet (Corbière, Thome, Bar-Hen, Cord, Pérez, NeurIPS 2019, arXiv:1910.04851) | auxiliary confidence head (TCP) | **Yes** | No | Not by default | No |
| [x] | Self-Adaptive Training (Huang, Zhang, Zhang, NeurIPS 2020, arXiv:2002.10319) | training dynamics | **Yes** | No | Not by default | No |
| [x] | Learning to Defer (Mozannar & Sontag, ICML 2020, arXiv:2006.01862) | human-in-the-loop | **Yes** | No | Not by default | No |
| [x] | CRL (Moon, Kim, Shin, Hwang, ICML 2020, pp. 7034–7044) | confidence ranking | **Yes** | No | Not by default | No |
| [x] | Feng, Ahmed, Hajimirsadeghi, Abdi, "Towards Better Selective Classification" (ICLR 2023) | tuned MSP | No | No | Partially | No |
| [x] | Cattelan & Silva, "How to Fix a Broken Confidence Estimator" (arXiv:2305.15508, 2023; UAI 2024) | logit normalization (post-hoc) | No (post-hoc) | No | Partially | No |
| [x] | Jones, Sagawa, Koh, Kumar, Liang, "Selective Classification Can Magnify Disparities Across Groups" (ICLR 2021, arXiv:2010.14134) | MSP | No | No | No | **Yes** |
| [x] | Pugnana, Perini, Davis, Ruggieri, "Deep Neural Network Benchmarks for Selective Classification" (JDMLR 2024, arXiv:2401.12708) | 18 baselines, many signals | Mixed | Mixed | Mixed | Mixed |
| [x] | Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster, arXiv:2208.02814, 2022) | any (wraps a score) | No | **Yes** | Assumption-breaking under shift | No |
| [x] | Learn Then Test (Angelopoulos, Bates, Candès, Jordan, Lei, arXiv:2110.01052, 2021; also ICML 2022) | any (wraps a score) | No | **Yes** | Assumption-breaking under shift | No |
| — | **This work** | Tiers A+B+C (+D optional), aggregated | No | **Yes** (conformal wrapper) | Yes (temporal + CIFAR-C) | Yes (RQ4) |

Supporting references (uncertainty/OOD signal sources and calibration
background, not selective-classification methods themselves, so not given
their own row): Guo et al. calibration (ICML 2017, arXiv:1706.04599),
Lakshminarayanan et al. deep ensembles (NeurIPS 2017), Gal & Ghahramani
MC-dropout (ICML 2016, arXiv:1506.02142), Liu et al. energy-based OOD
(NeurIPS 2020, arXiv:2010.03759), Lee et al. Mahalanobis OOD (NeurIPS
2018), Sun et al. KNN OOD (ICML 2022, arXiv:2204.06507), Jiang et al.
trust score (NeurIPS 2018, arXiv:1805.11783), Huang et al. GradNorm
(NeurIPS 2021, arXiv:2110.00218) — all confirmed, see `references.bib`.
Vovk, Gammerman & Shafer's *Algorithmic Learning in a Random World*
(Springer, 2005) and Angelopoulos & Bates's "A Gentle Introduction to
Conformal Prediction" (arXiv:2107.07511, 2021) are foundational/tutorial
references, also confirmed.

## Corrections made during verification

- `benchmark2024` was a guess at arXiv:2401.12708 — **confirmed correct**.
  Title and arXiv ID matched; bib key renamed to `pugnana2024benchmark`
  (author list: Pugnana, Perini, Davis, Ruggieri) and `main.tex`'s
  `\cite{benchmark2024}` was updated to the new key.
- Deep Gamblers' fourth author is **Ruslan** Salakhutdinov, not "Russ R.
  Salakhutdinov" as the plan draft had it.
- Cattelan & Silva's paper title was exactly right, but its published
  venue is **UAI 2024**, not 2023 as the plan assumed — 2023 is only the
  arXiv preprint date (arXiv:2305.15508). Pick one convention (preprint
  year vs. venue year) and apply it consistently across the bibliography.
- Sun et al.'s KNN-OOD paper lists the author as "Jerry Zhu" on the
  ICML/arXiv record, not "Xiaojin Zhu" as guessed (same person, different
  name form used on this paper).
- Self-Adaptive Training is a **NeurIPS 2020** paper (arXiv:2002.10319),
  not a venue-less `@misc` — don't confuse it with the different, later
  arXiv:2101.08732 paper by an overlapping author set with a similar
  title.
- Everything else (Chow, Geifman & El-Yaniv x2, ConfidNet, Mozannar &
  Sontag, Moon et al., Feng et al., Jones et al., the OOD-detection set,
  Guo et al., Lakshminarayanan et al., Gal & Ghahramani, Vovk et al., and
  the two Angelopoulos et al. conformal papers) matched the plan's
  recollection closely; only minor formatting (page numbers, arXiv IDs)
  was added.

## Why this table is the paper's justification

Reading across the columns: every learned-abstention baseline needs
retraining the base model; none of them ship a distribution-free
guarantee; shift and subgroup evaluation are inconsistent or absent across
the field. The empty/No cells are exactly the gap the four contributions
in `main.tex`'s introduction claim to fill. This mapping is now built on
verified citations rather than recalled ones, so the contribution wording
in `main.tex` can be trusted as resting on real sources — though the
one-line "signals used" et al. descriptions above are still abstract-level
summaries, not a full read of each paper; deepen them before the
related-work prose is finalized if any claim needs to bear real weight.
