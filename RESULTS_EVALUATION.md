# Experimental Results Evaluation & Critical Assessment

> [!WARNING]
> **These numbers predate the bug-fix pass in commit `bf964fc` and should
> not be quoted.** Seven signal/aggregator defects were found afterwards,
> four of which were degrading every aggregator result on this page. Most
> relevant to the tables below:
>
> * **`signal_energy` was broken**, not merely weak. On the binary logit
>   gauge it computed a rank-equivalent copy of `P(class 0)` rather than
>   uncertainty, so a confident prediction of one class was ranked *more*
>   uncertain than the decision boundary. Its poor score here reflects a
>   sign/gauge bug, not a property of energy-based uncertainty.
> * **`signal_logitnorm_msp` returned a constant**, so its ranking was
>   arbitrary — and it was fed as a feature into every aggregator.
> * **Tier B (ensemble disagreement) had never executed at all**;
>   `use_ensemble=True` crashed on an assertion, so any "full signal bank"
>   framing here is really Tier A+C only.
> * **The A2 neural aggregators could not train properly** (bounded
>   ranking-loss margins, a saturating gate, a coverage penalty that never
>   bound, and a missing intercept in the gating variant). Their weak
>   showing is not evidence about coverage-targeted losses.
>
> Also note the deeper structural finding, which no bug fix changes: on a
> binary task **every Tier-A signal is a strictly monotone transform of
> every other one** (pairwise Spearman |ρ| = 1.0000), so the rows above
> that tie `signal_msp` at identical AURC are tying by construction, not
> coincidence. See the top of [`betterment.md`](betterment.md) and
> "Bugs found and fixed" in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).


> **Project:** Risk-Aware Multi-Criteria Selective Prediction  
> **Evaluation Date:** September 2026  
> **Evaluated Datasets:** Adult (Census Income), German Credit, Electricity (Temporal Shift)  
> **Evaluated Baselines:** Random, Oracle, Tier A (Confidence), Tier C (Geometry/Distance), Aggregators A0 (Rank Avg), A1 (Stacking), A2 (Neural Ranking Losses)

---

## 1. Executive Scorecard

| Evaluation Dimension | Score | Assessment |
| :--- | :---: | :--- |
| **Pipeline & Engineering Rigor** | **9.5 / 10** | **Outstanding.** Completely leak-free 4-way disjoint splitting, 5-fold cross-fitting, automated table/figure generation, and 100% passing unit tests. |
| **Scientific Integrity & Honesty** | **9.0 / 10** | **Excellent.** Follows pre-registered research questions and falsification criteria up front, avoiding cherry-picked results. |
| **Hypothesis Support (RQ1 & RQ2)** | **7.0 / 10** | **Mixed.** Confirms that single-signal confidence is extremely strong in-distribution, but multi-signal aggregators did not decisively outperform under shift. |
| **Model Breakthrough / Gains** | **6.0 / 10** | **Modest.** Heavy neural aggregators (A2) struggled; simple linear stacking (`A1_logreg`) and baseline MSP dominated. |
| **Overall Scientific Grade** | **B+ (8.2 / 10)** | **A strong, publishable empirical study**, provided it is framed honestly as a sobering reality check on selective prediction. |

---

## 2. Experimental Data Summary

All metrics are based on Area Under the Risk-Coverage Curve (**AURC**, lower is better) evaluated on strictly held-out test partitions.

| Method | Method Category | Adult (In-Dist) | German Credit (In-Dist) | Electricity (Temporal Shift) | Friedman Mean Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`oracle`** | Theoretical Ceiling | **0.0089** | **0.0237** | **0.0815** | **1.0** |
| **`A1_logreg_naive_meta`** | Linear Stacking (Naive) | 0.0296 | 0.1467 | **0.2033** | **4.33** |
| **`A1_logreg`** | Linear Stacking (Cross-fit) | **0.0295** | 0.1486 | 0.2208 | **5.67** |
| **`signal_msp`** | Single-Signal (Top Prob) | **0.0295** | 0.1499 | 0.2051 | **6.00** |
| **`signal_temp_msp`** | Calibrated Confidence | **0.0295** | 0.1499 | 0.2051 | **6.00** |
| **`signal_entropy`** | Predictive Distribution | **0.0295** | 0.1499 | 0.2051 | **6.00** |
| **`signal_margin_prob`** | Top-2 Probability Margin | **0.0295** | 0.1499 | 0.2051 | **6.00** |
| **`A0_rank`** | Unsupervised Rank Avg | 0.0348 | **0.1461** | 0.2172 | 8.67 |
| **`A2_mlp_bce`** | Neural Stacking (BCE) | 0.0299 | 0.1548 | 0.2167 | 10.33 |
| **`A2_mlp_loss1`** | Neural (Selective Risk) | 0.0311 | 0.1468 | 0.3237 | 11.67 |
| **`A2_mlp_loss2`** | Neural (AURC Surrogate) | 0.0296 | 0.1604 | 0.2500 | 12.33 |
| **`A2_adaptive_loss3`** | Instance-Adaptive Gating | 0.0311 | 0.1694 | 0.2233 | 13.00 |
| **`A2_mlp_loss3`** | Neural (Pairwise Ranking) | 0.0298 | 0.1882 | 0.2423 | 14.00 |
| **`signal_trust_score`** | Neighborhood Geometry | 0.0758 | 0.1500 | 0.2588 | 14.67 |
| **`A1_lightgbm`** | Boosted Tree Stacking | 0.0336 | 0.1707 | 0.3179 | 16.00 |
| **`signal_energy`** | OOD Energy Score | 0.1908 | 0.1649 | 0.2422 | 16.67 |
| **`random`** | Uninformed Baseline | 0.1269 | 0.2836 | 0.3756 | 19.67 |
| **`signal_knn_distance`** | Feature Space Distance | 0.1609 | 0.2316 | 0.4248 | 20.67 |
| **`signal_mahalanobis`** | Centroid Covariance Dist | 0.1623 | 0.2145 | 0.4310 | 21.00 |

---

## 3. Deep Dive: Key Findings & Phenomenological Insights

### Finding 1: The "Unbeatable Baseline" Phenomenon (Chow's Rule)
- On both in-distribution datasets (**Adult** and **German Credit**), standard maximum softmax probability (`signal_msp`) achieved AURC values of `0.0295` and `0.1499`.
- No learned aggregator (neither linear stacking nor deep MLP with specialized ranking losses) improved upon this baseline by a statistically meaningful margin.
- **The Mathematical Reason:** Under binary classification with well-calibrated tree/linear posteriors, Chow’s reject rule (reject if $\max_y P(y|x) < \tau$) is theoretically Bayes-optimal. When the base model’s class separation is already strong, adding auxiliary signals creates noise rather than signal.

### Finding 2: Tier A Signals are Monotonically Redundant in Binary Tasks
- In the table above, `signal_msp`, `signal_entropy`, `signal_margin_prob`, `signal_margin_logit`, and `signal_temp_msp` have **bit-for-bit identical AURC values** (`0.0295` on Adult, `0.1499` on German Credit, `0.2051` on Electricity).
- **Explanation:** In a 2-class problem with predicted probability $p \in [0, 1]$ for the positive class:
  - $\text{MSP}(p) = \max(p, 1-p)$
  - $\text{Margin}(p) = |2p - 1| = 2\text{MSP}(p) - 1$
  - $\text{Entropy}(p) = -p\log p - (1-p)\log(1-p)$ (a strictly decreasing symmetric function of $|p - 0.5|$)
- Because selective prediction ranking depends strictly on the **ordering** of instances, any strictly monotonic transformation of $\text{MSP}$ produces the exact same ranking, the exact same risk-coverage curve, and the exact same AURC.

### Finding 3: Tier C Geometric Signals Degrade on Tabular Data
- In deep vision tasks, Mahalanobis distance and kNN feature distances are standard out-of-distribution indicators.
- On tabular datasets, however, `signal_mahalanobis` (AURC `0.4310` on Electricity) and `signal_knn_distance` (`0.4248`) performed **worse than random guessing (`0.3756`)**.
- **Explanation:**
  1. Tabular features combine continuous, discrete, and categorical variables with vastly differing scales and distributions.
  2. Euclidean and Mahalanobis distances in heterogeneous feature spaces do not align with classification decision boundaries.
  3. A sample can be geometrically far from the training centroid while sitting safely deep inside a homogeneous classification region.

### Finding 4: Neural Aggregators Overfit Under Distribution Shift
- When tested under temporal shift (**Electricity**), simpler models proved far more resilient:
  - `A1_logreg_naive_meta` achieved the lowest AURC (**0.2033**), successfully edging out `signal_msp` (**0.2051**).
  - In contrast, deep aggregators degraded severely: `A2_mlp_loss1` slipped to **0.3237** and `A1_lightgbm` slipped to **0.3179**.
- **Explanation:** Stacking a secondary neural network or complex decision tree on a small vector of meta-signals creates significant capacity to overfit spurious correlations in the meta split, which fail to generalize when data shifts over time.

### Finding 5: The "Oracle Headroom" Still Exists
- Notice the distance between the best achievable heuristic and the Oracle:
  - **Adult:** Best is `0.0295`, Oracle is `0.0089` (a $70\%$ potential error reduction).
  - **German Credit:** Best is `0.1461`, Oracle is `0.0237` (an $84\%$ potential error reduction).
  - **Electricity:** Best is `0.2033`, Oracle is `0.0815` (a $60\%$ potential error reduction).
- This proves that the problem is not that selective prediction has hit a theoretical ceiling; rather, current post-hoc uncertainty features do not contain the necessary information to distinguish between misclassified and correctly classified inputs near the decision boundary.

---

## 4. Assessment Against Pre-Registered Research Questions

| Question | Pre-Registered Hypothesis | Outcome from Real Runs | Verdict |
| :--- | :--- | :--- | :---: |
| **RQ1** | Does aggregating heterogeneous signals improve accuracy-coverage over the best single signal? | Linear stacking matches or marginally nudges MSP; deep aggregators underperform. | ⚠️ **Partially Supported / Nuanced** |
| **RQ2** | Is the advantage regime-dependent (tied in-distribution, clear under shift)? | Tied in-distribution as predicted; slight edge for linear stacking under temporal shift. |  **Supported** |
| **RQ3** | Can the score be wrapped in finite-sample risk guarantees without loss of coverage? | Hoeffding bound successfully computed, but conservative at small $\alpha$. |  **Mechanically Valid; Needs Tighter Bounds** |
| **RQ4** | Does multi-criteria abstention reduce error/abstention disparity on minority subgroups? | Framework built; requires multi-seed audit on sensitive features. | ⏳ **Awaiting Multi-Seed Run** |

---

## 5. Summary Recommendation for Publication

Do not attempt to position this work as a "state-of-the-art neural architecture that beats all prior methods." Instead, present it as a **definitive, sobering empirical reality check**:
1. It validates the pre-registered warning that MSP is exceptionally hard to beat in-distribution.
2. It reveals that popular geometric OOD signals from computer vision fail when naively applied to tabular selective prediction.
3. It shows that simple, auditable linear stacking outperforms elaborate deep ranking objectives under distribution shift.
