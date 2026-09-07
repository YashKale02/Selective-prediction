# Action Plan: Transforming the Codebase into a High-Impact IEEE Paper

> [!WARNING]
> Written before the bug-fix pass in commit `bf964fc`. Seven
> signal/aggregator defects were found afterwards (two signals outright
> broken, Tier B never executing, the A2 aggregators unable to train), so
> any results, scores or rankings quoted here are not trustworthy. See the
> top of [`betterment.md`](betterment.md) and "Bugs found and fixed" in
> [`PROJECT_STATUS.md`](PROJECT_STATUS.md).


> **Target Venue:** IEEE Conference (e.g., IEEE BigData, ICMLA, or IEEE TNNLS / CVPR-W)  
> **Current Status:** Complete working pipeline, 3 real datasets evaluated, verified citations, working test suite.  
> **Goal:** Step-by-step roadmap to elevate the paper from a scaffolded project into a compelling, rigorous, and highly citable publication.

---

## 1. The Core Strategy: Reframe the Narrative

### The Fatal Trap to Avoid
Many machine learning papers fail at review because the authors promise an *"unprecedented state-of-the-art neural architecture"* only for the empirical tables to show that a simple baseline (like standard confidence) beats or matches their method. Reviewers immediately reject such papers for lack of evidentiary support.

### The Winning Narrative
Frame the paper as an **authoritative empirical reality check and benchmark** on selective prediction:
- **Working Title Options:**
  - *Option A (Recommended):* *"When Does Uncertainty Aggregation Help? An Empirical Study of Multi-Criteria Selective Prediction Under Distribution Shift"*
  - *Option B:* *"The Limits of Complexity: A Critical Benchmark of Learned Abstention on Tabular Benchmarks"*
  - *Option C:* *"Risk-Aware Selective Prediction: Evaluating Multi-Signal Aggregation and Conformal Guarantees Under Shift"*
- **The Core Argument:**
  1. Under well-calibrated posteriors, **Chow’s rule (MSP) is Bayes-optimal**—meaning beating it in-distribution is mathematically improbable.
  2. While complex neural aggregators and surrogate ranking losses are widely celebrated in computer vision, on **tabular data they suffer from severe meta-overfitting and optimization instability**, especially under distribution shift.
  3. Simple, auditable **linear stacking (`A1_logreg`) provides superior stability and minimax risk control**, matching or nudging MSP without the failure modes of deep ranking networks.
  4. Feature-space geometric metrics (Mahalanobis distance, kNN distance), staple OOD detectors in deep vision, **fail completely on heterogeneous tabular spaces**.

---

## 2. Six Actionable Pillars for Improvement

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PAPER IMPROVEMENT ROADMAP                         │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ 1. Empirical      │ 2. Diagnostic     │ 3. Mathematical                │
│    Scaling        │    Heatmap        │    Upgrades                    │
│ • 6+ datasets     │ • Signal-error    │ • Hoeffding-Bentkus            │
│ • 5-10 seeds      │   correlation     │ • Tighter coverage             │
│ • Real Wilcoxons  │ • Redundancy proof│ • Conformal audit              │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ 4. Subgroup       │ 5. Related Work   │ 6. Complete Draft              │
│    Fairness (RQ4) │    Prose          │    in main.tex                 │
│ • Disparity audit │ • Gap table prose │ • Embed real tables            │
│ • Worst-group risk│ • Full citations  │ • Clear all \todo{}            │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

---

### Pillar 1: Expand Empirical Breadth (Scale Datasets & Seeds)

**Current state:** 3 datasets evaluated at 1 seed.  
**Target:** $\ge 6$ datasets evaluated across 5–10 random seeds.

1. **Scale Seeds to 5:**
   Running 5 seeds generates standard deviations (`AURC_std`) and unlocks valid Wilcoxon signed-rank significance testing with Holm-Bonferroni corrections in `scripts/make_tables.py`:
   ```powershell
   python scripts/run_all.py dataset=adult experiment.n_seeds=5
   python scripts/run_all.py dataset=german_credit experiment.n_seeds=5
   python scripts/run_all.py dataset=electricity experiment.n_seeds=5
   ```
2. **Add 3 More Tabular Datasets from the Registry:**
   In [`src/data/loaders.py`](file:///c:/Users/Admin/Desktop/Selective_predict/src/data/loaders.py), register and run:
   - `bank_marketing` (imbalanced tabular financial data)
   - `diabetes_130` (high-stakes medical prediction with hospital shift)
   - `covertype` or `churn` (multi-feature classification)
3. **Generate the Critical Difference (CD) Diagram:**
   Once $\ge 5$ datasets are evaluated, use the Friedman test output to render a publication-quality Critical Difference diagram (using Orange or matplotlib) showing cliques of statistically equivalent methods.

---

### Pillar 2: The "Killer" Diagnostic — Signal Correlation Heatmap

Reviewers need to see **why** adding 12 signals did not dramatically beat 1 signal. A visual diagnostic of signal redundancy provides instant clarity.

- **Action:** Create a script `scripts/make_correlation_heatmap.py` that computes the **Spearman rank correlation matrix** between all candidate uncertainty signals and true classification correctness $\mathbf{1}[f(x) = y]$.
- **What this demonstrates:**
  - **Tier A collinearity:** Proves empirically that $\text{corr}(\text{MSP}, \text{Margin}) = 1.0$ and $\text{corr}(\text{MSP}, \text{Entropy}) \approx -0.99$, demonstrating zero additive entropy in binary tasks.
  - **Tier C disconnect:** Proves that $\text{corr}(\text{Mahalanobis}, \text{Correctness}) \approx 0.0$ on tabular data, giving empirical grounding to why distance metrics fail.
- **Placement:** Include this figure in Section 6 (*Ablations & Diagnostics*) of the paper.

---

### Pillar 3: Strengthen the Conformal Guarantee Layer (RQ3)

**Current state:** [`src/conformal/risk_control.py`](file:///c:/Users/Admin/Desktop/Selective_predict/src/conformal/risk_control.py) uses a standard Hoeffding bound with Bonferroni correction.  
**Issue:** Vanilla Hoeffding is variance-oblivious and loose, meaning for small target risks (e.g., $\alpha = 0.02$ or $0.05$), the conservative threshold results in $0\%$ coverage unless the calibration set is massive.

- **Upgrade Options:**
  1. **Hoeffding-Bentkus Bound:** Significantly tighter for small sample sizes and small risk levels.
  2. **Empirical Bernstein Bound:** Takes the empirical variance of the loss into account, allowing much higher coverage at the same certified risk level.
- **Reporting:** Include a table of **Coverage at Certified Risk $\alpha \in \{0.05, 0.10, 0.15\}$** across methods.

---

### Pillar 4: Audit Subgroup Fairness & Disparities (RQ4)

Selective classification often magnifies societal bias: a model may achieve an overall $5\%$ error rate by rejecting $90\%$ of minority applicants or concentrating all residual errors on them.

- **Implementation:** The project already contains [`src/metrics/subgroup.py`](file:///c:/Users/Admin/Desktop/Selective_predict/src/metrics/subgroup.py) which computes:
  - `worst_group_risk`: Maximum error rate among demographic subgroups.
  - `max_min_risk_gap`: Disparity between best-served and worst-served groups.
  - `rejection_rate_by_group`: Disparity in abstention rates across groups.
- **Benchmark Groups:**
  - Adult dataset: `sex` and `race`.
  - German Credit dataset: `age` and `foreign_worker`.
- **Contribution:** Demonstrate whether multi-signal aggregation reduces this unfairness compared to single-signal confidence thresholding.

---

### Pillar 5: Complete the Paper Draft ([`paper/main.tex`](file:///c:/Users/Admin/Desktop/Selective_predict/paper/main.tex))

Replace every remaining `\todo{}` marker with clean, authoritative text:

#### 1. Abstract & Introduction
- State the research question honestly: *Do multi-criteria uncertainty signals improve accuracy-coverage trade-offs, and under what regimes?*
- State the concrete findings: MSP is near-optimal in-distribution; linear stacking excels under temporal shift; deep neural aggregators suffer from meta-overfitting; geometric signals fail on tabular features.

#### 2. Related Work (Section II)
- Convert [`paper/related_work_gap_table.md`](file:///c:/Users/Admin/Desktop/Selective_predict/paper/related_work_gap_table.md) into four structured narrative subsections:
  1. *Foundations of Selective Classification* (Chow 1970, Geifman & El-Yaniv 2017).
  2. *Learned Abstention vs. The Strong Baseline Phenomenon* (SelectiveNet, ConfidNet vs. Feng et al. 2023, Cattelan & Silva 2023).
  3. *Uncertainty & Out-of-Distribution Signals* (Lakshminarayanan 2017, Liu et al. 2020, Lee et al. 2018).
  4. *Distribution-Free Risk Control* (Vovk 2005, Angelopoulos et al. 2022).
- Include the **Gap Table** summarizing literature blind spots.

#### 3. Embed Empirical Results
- Embed `paper/figures/risk_coverage_curves.png` and `paper/figures/week2_gate_msp_vs_random.png`.
- Convert `paper/tables/aurc_summary.csv` and `paper/tables/friedman_mean_ranks.csv` into a LaTeX `table` using `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).

#### 4. Discussion & Limitations (Section VII)
- Explicitly discuss the "Tabular vs. Vision" dichotomy: why representation geometry behaves fundamentally differently in deep continuous spaces than in tabular spaces.
- Note the computational trade-off of Tier C signals (which require keeping the training set in memory at inference).

---

## 3. Checklist for Paper Submission

- [ ] Re-run pipeline at 5 seeds on Adult, German Credit, and Electricity.
- [ ] Add 2–3 additional tabular datasets to the registry and execute.
- [ ] Generate correlation heatmap figure and add to manuscript.
- [ ] Insert LaTeX tables generated directly from `results.parquet`.
- [ ] Replace all `\todo{}` placeholders in `paper/main.tex`.
- [ ] Verify that all citations in `paper/references.bib` are properly formatted.
- [ ] Ensure double-blind compliance (anonymize author name, institution, repository link).
- [ ] Compile and verify formatting with IEEE conference template.
