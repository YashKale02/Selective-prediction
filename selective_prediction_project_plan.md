# Risk-Aware Multi-Criteria Selective Prediction
## End-to-end project plan (IEEE conference paper, final-year)

---

## 0. The one-paragraph version

Train a fixed base classifier. Extract a **vector** of uncertainty signals per test input
(confidence, calibrated confidence, ensemble disagreement, energy/OOD score, feature-space
density, local label agreement). Learn an **aggregator** that maps this vector to a single
abstention score, trained with a **coverage-targeted ranking loss** rather than plain
correctness classification. Wrap the resulting score in a **conformal risk-control** layer so the
deployed system carries a distribution-free guarantee on selective risk. Evaluate on 10+ tabular
datasets plus image benchmarks, in-distribution and under shift, against strong single-signal
baselines and published selective-classification methods.

---

## 1. Research question, stated so it can fail

> **RQ1.** Does aggregating heterogeneous uncertainty signals improve the accuracy–coverage
> trade-off over the best single-signal abstention rule?
>
> **RQ2.** Does the *advantage depend on the failure regime* – i.e. is multi-criteria abstention
> roughly equal to confidence thresholding in-distribution but clearly better under
> distribution shift and at low coverage?
>
> **RQ3.** Can the learned score be wrapped in a finite-sample guarantee on selective risk
> without losing coverage relative to a heuristic threshold?
>
> **RQ4.** Does multi-criteria abstention reduce the known tendency of selective prediction to
> concentrate abstentions (and residual error) on minority subgroups?

RQ2 is the load-bearing one. **Write it down before you run anything.** The single largest
threat to this paper is that a properly trained softmax confidence baseline is far stronger than
students expect, and several published papers have shown that elaborate learned-abstention
methods fail to beat it in-distribution. If you promise in-distribution gains and don't get them,
the paper dies. If you predict up front that gains are regime-dependent and then demonstrate
exactly that, a null result in-distribution becomes part of the contribution instead of a failure.

**Falsification criteria – decide now, honour them later:**

| Outcome | What you claim |
|---|---|
| Beats best single signal on ≥7/12 datasets under shift, paired test p<0.05 | Full claim, RQ1+RQ2 supported |
| Wins under shift only, ties in-distribution | Regime-dependent claim – still publishable, and more honest than most of the field |
| Ties everywhere | Negative-result paper: "heterogeneous signals are redundant; here is the correlation structure showing why" |
| Loses | You made a leakage error. See §6. |

---

## 2. Prior work you must read before writing code

Grouped by what each one does to your plan. **I don't have search access here, so treat every
citation below as a title I'm recalling from memory – verify each one on IEEE Xplore / arXiv /
Semantic Scholar before it enters your bibliography. Some may be misattributed or
misremembered.**

**Foundational – defines your baseline and the optimality argument**
- Chow, "On optimum recognition error and reject tradeoff" (1970). The reject rule: abstain if
  max posterior < threshold. **Chow's rule is Bayes-optimal if the posteriors are correct.**
  Your entire motivation is that they aren't – miscalibration, OOD inputs, and epistemic
  uncertainty all break the assumption. Say this explicitly in your introduction; it is the
  cleanest possible framing of why extra signals could help.
- Geifman & El-Yaniv, "Selective classification for deep neural networks" (NeurIPS 2017).
  Source of the risk-coverage evaluation protocol.
- El-Yaniv & Wiener, "On the foundations of noise-free selective classification" (JMLR 2010).

**Learned-abstention methods – these are your baselines, not your contribution**
- SelectiveNet (Geifman & El-Yaniv, ICML 2019) – abstention head with coverage constraint.
- Deep Gamblers (Liu et al., NeurIPS 2019) – abstention as an extra class with a reward.
- ConfidNet (Corbière et al., NeurIPS 2019) – auxiliary network predicting true-class prob.
- Self-Adaptive Training (Huang et al., 2020).
- Learning to Defer (Mozannar & Sontag, ICML 2020) – the human-in-the-loop variant.
- Confidence-Aware Learning / CRL (Moon et al., ICML 2020).

**The papers that could sink you – read these first**
- Feng et al., "Towards better selective classification" (ICLR 2023) – argues a well-tuned
  softmax baseline matches or beats the elaborate methods above.
- Cattelan & Silva, on repairing confidence estimators via logit normalisation (2023) – shows
  cheap post-hoc fixes recover most of the claimed gains.
- Jones et al., "Selective classification can magnify disparities across groups" (ICLR 2021) –
  the fairness failure mode. This is a threat *and* the source of RQ4.
- The 2024 benchmark of 18 selective-classification methods across 44 datasets that you already
  found (arXiv:2401.12708 per your notes – confirm the ID and authors). Its finding that no
  single method dominates across objectives is the strongest available motivation for an
  adaptive, multi-signal approach. Cite it in the first paragraph.

**Signal sources – where each candidate feature comes from**
- Temperature scaling / calibration: Guo et al. (ICML 2017).
- Deep Ensembles: Lakshminarayanan et al. (NeurIPS 2017).
- MC-Dropout: Gal & Ghahramani (ICML 2016).
- Energy score for OOD: Liu et al. (NeurIPS 2020).
- Mahalanobis distance detector: Lee et al. (NeurIPS 2018).
- kNN-based OOD: Sun et al. (ICML 2022).
- Trust Score: Jiang et al. (NeurIPS 2018).
- GradNorm: Huang et al. (NeurIPS 2021).

**Guarantee layer**
- Vovk / Shafer & Vovk – conformal prediction foundations.
- Angelopoulos et al., Conformal Risk Control (2022); Learn-then-Test (2021).
- Angelopoulos & Bates, "A gentle introduction to conformal prediction" – read this one first.

**Deliverable for this phase:** a 3-page related-work table with columns
`method | signals used | needs retraining? | guarantee? | evaluated under shift? | subgroup analysis?`
The empty cells in that table are your paper's justification. Build the table before you commit
to the contribution wording.

---

## 3. Method design

### 3.1 Signal bank

Per input `x`, given a fixed base model `f`, compute a feature vector `u(x) ∈ R^m`.

**Tier A – logit-derived (free, always available)**
1. Max softmax probability (MSP)
2. Predictive entropy
3. Top-1 minus top-2 margin (probability and logit space)
4. Negative energy: `−logsumexp(z)`
5. Logit-norm-normalised MSP (`softmax(z / ||z||_p)`)
6. Temperature-scaled MSP (T fit on the calibration split)

**Tier B – model-internal (needs multiple forward passes or ensembles)**
7. MC-dropout predictive entropy
8. MC-dropout mutual information (the epistemic component)
9. Ensemble vote entropy
10. Mean pairwise KL between ensemble members
11. Variance of the predicted top-class probability

**Tier C – representation / data geometry (needs the training set at inference)**
12. Mahalanobis distance to the predicted class centroid in penultimate-layer space
13. Mean distance to k nearest training points (k = 10, 50)
14. Local label agreement: fraction of the k nearest training neighbours sharing the predicted
    label – a proxy for local Bayes error, and the only signal here that targets *aleatoric*
    rather than epistemic uncertainty
15. Trust score
16. Local density estimate (kernel or normalising-flow log-density on features)

**Tier D – tree-specific (tabular models)**
17. Out-of-bag / cross-fold disagreement across boosting rounds or bagged members
18. Leaf co-occurrence distance to training points
19. Conformal nonconformity score (1 – softmax of true-class-agnostic APS score)

Start with Tier A + one signal from each of B and C. **Do not implement all 19 before you have
a working pipeline.** Add signals in order of expected marginal information, and measure
marginal contribution as you go (§7 ablations).

### 3.2 Aggregator

Input `u(x)`, output scalar risk score `s(x)`. Abstain when `s(x) > τ`.

Three aggregators, increasing in expressiveness:

- **A0 – z-score sum / rank average.** No learning. Sanity check and a legitimate cheap baseline.
- **A1 – linear + GBM on correctness.** Fit logistic regression and LightGBM to predict
  `1[f(x) ≠ y]` from `u(x)` on a disjoint meta-split. This is stacking; it is your own internal
  baseline, not the contribution.
- **A2 – coverage-targeted aggregator (the contribution).** Same function class as A1, but
  trained with a loss that targets the operating region you actually deploy at.

### 3.3 The loss – why A2 is different from stacking

Correctness-prediction accuracy is the wrong objective. A gater with 90% AUROC can still order
the top 10% of inputs badly, and the top 10% is precisely what matters when you deploy at 90%
coverage. What you want is a good *ranking* near the threshold.

**Loss 1 – soft selective risk at target coverage κ.**
With a soft gate `g(x) = σ((τ − s(x))/T)`:

```
L_κ = Σᵢ g(xᵢ)·ℓ(f(xᵢ), yᵢ) / Σᵢ g(xᵢ)  +  λ·max(0, κ − (1/n)Σᵢ g(xᵢ))²
```

First term is empirical selective risk on the accepted set; second penalises falling below the
coverage target. This is the SelectiveNet objective, repurposed to train a *post-hoc aggregator
over signals* instead of a jointly trained head – which is what makes it cheap and
model-agnostic.

**Loss 2 – AURC surrogate.** Sum `L_κ` over `κ ∈ {0.5, 0.6, …, 1.0}` for a single score good
across the whole curve.

**Loss 3 – pairwise ranking.** For pairs where one input is correctly classified and the other
isn't, penalise `σ(s(correct) − s(incorrect))`. Directly optimises the discrimination that AURC
measures, and is better behaved than Loss 1 on small tabular datasets.

Report all three. The comparison of A1 (BCE) against A2 (Losses 1–3) is one of your headline
ablations, and if it shows nothing you have lost a contribution – so run it early, on three
datasets, before building the rest.

### 3.4 Instance-adaptive signal weighting (second novelty axis)

The right signal depends on the failure mode. Mahalanobis distance is informative for OOD
inputs and near-useless for genuinely ambiguous in-distribution ones; local label agreement is
the reverse. So learn a gating network `w(x) = softmax(h(u(x)))` and set
`s(x) = Σⱼ wⱼ(x)·ũⱼ(x)` over normalised signals.

This gives a genuinely new mechanism *and* an interpretability story: plot the learned weight
distribution for in-distribution vs corrupted inputs and show that the model routes to different
signals. That figure, if it works, is the most memorable thing in the paper. If the weights turn
out to be nearly constant, report that too – it's evidence the regimes aren't separable from the
signal vector alone, which is a real finding.

### 3.5 Conformal risk-control wrapper (third novelty axis)

Everything above produces a heuristic threshold. Add a guarantee: on a held-out calibration set
of size `n`, choose the largest threshold `τ̂` such that an upper confidence bound on selective
risk stays below the user's target `α`. Under exchangeability this gives
`P(selective risk ≤ α) ≥ 1 − δ`, distribution-free, with no assumption about the aggregator.

This is the cheapest large win in the plan. It converts "our heuristic scored better" into "our
system meets a user-specified error budget, and does so at higher coverage than the
baselines' score allows." Reviewers respond to that framing. Report **coverage at guaranteed
risk ≤ α** for `α ∈ {1%, 2%, 5%, 10%}` as a primary table – this is the metric a deployer
actually cares about, and it's underused in the literature.

Caveat to state in the paper: exchangeability fails under distribution shift, so the guarantee is
valid in-distribution and is *evaluated empirically* under shift. Don't overclaim here; a reviewer
who knows conformal prediction will check.

---

## 4. Datasets

Target 12 tabular + 2 image. Tabular is the core; images demonstrate the method isn't
tabular-specific.

### Tabular – in-distribution
| Dataset | Source | n | Task | Why |
|---|---|---|---|---|
| Adult / Census Income | UCI, OpenML 1590 | 48k | Binary | Standard, subgroup attributes available |
| German Credit | UCI, OpenML 31 | 1k | Binary | Small-n stress test for meta-fitting |
| Bank Marketing | UCI | 45k | Binary | Class imbalance |
| Give Me Some Credit | Kaggle | 150k | Binary | Real cost asymmetry |
| Covertype | UCI, OpenML 1596 | 581k | 7-class | Multi-class, large |
| MiniBooNE | UCI, OpenML | 130k | Binary | Physics, clean |
| Higgs (subset) | UCI | 1M–200k | Binary | Irreducible aleatoric noise; MSP should do well here |
| Diabetes 130-US hospitals | UCI | 100k | 3-class | Clinical, natural human-review story |
| Telco churn | Kaggle | 7k | Binary | Business decision framing |
| ann-thyroid | UCI, OpenML | 7k | 3-class | Severe imbalance |
| Electricity | OpenML 151 | 45k | Binary | **Has temporal shift** |
| Jannis / Road-Safety | OpenML, Grinsztajn suite | varies | Multi | Reviewer-recognised suite |

Pull as many as possible from the **Grinsztajn et al. (2022) tabular benchmark suite** and
**OpenML-CC18** rather than hand-assembling. Reviewers trust curated suites and it removes
"you cherry-picked datasets" as an attack.

### Distribution shift
- **Electricity, Diabetes-130** – temporal split (train on early years, test on later). Free,
  realistic, and the most defensible shift you'll get on tabular data.
- **Adult** – split by geography or year across census extracts if available.
- **CIFAR-10-C / CIFAR-100-C** – 15 corruptions × 5 severities. The standard shift benchmark.
- **Synthetic covariate shift** – importance-weighted resampling of the test set on tabular data.
  Include but label clearly as synthetic; it's a controlled dial, not evidence about the real world.

### Image
- CIFAR-10 and CIFAR-100 with ResNet-18 or WideResNet-28-10. Use **published pretrained
  checkpoints** where you can to save GPU time.
- Skip ImageNet unless a lab GPU falls into your lap. If it does, use pretrained torchvision
  weights and only evaluate – never train.

### Fairness / subgroup analysis (RQ4)
Adult (sex, race), German Credit (age, foreign worker), Diabetes-130 (race), COMPAS if you
include it. **Note on COMPAS:** widely used but heavily criticised on data-quality and
construct-validity grounds. If you use it, use it only for the disparity analysis, and cite the
critiques. Don't build headline accuracy claims on it.

**Explicitly avoid:** anything requiring credentialed access (MIMIC, eICU) – the approval
timeline will not fit your schedule. Also avoid a single-dataset paper at all costs; the whole
strength of this project is breadth.

---

## 5. Base models

Keep the base model **frozen and identical** across all abstention methods. Any performance
difference must come from the abstention rule, not the classifier.

- **Tabular:** LightGBM or XGBoost (primary – it's the actual state of the art here), MLP
  (2–3 layers, needed for Tier B/C signals that assume a representation), logistic regression
  (interpretable floor).
- **Image:** ResNet-18, WideResNet-28-10.
- **Ensembles:** 5 members with different seeds, for the disagreement signals.

Two exceptions where retraining is required, and must be flagged as a cost in the paper:
SelectiveNet and Deep Gamblers modify the training objective. Your method does not – that
asymmetry ("post-hoc, model-agnostic, no retraining") is a selling point, so measure and report
it as wall-clock cost.

---

## 6. Data splits and leakage – read this twice

This is where this project most often silently fails, and a leakage bug produces beautiful
results, which is why it goes undetected.

**Four disjoint splits:**
```
D_train  (60%)  → fit base model f
D_meta   (15%)  → fit the aggregator (and temperature T)
D_cal    (10%)  → conformal threshold calibration
D_test   (15%)  → final evaluation, touched once
```

**The critical problem:** the base model is overconfident on its own training data. If you compute
signals for `D_meta` using a model that saw `D_meta`, the signal distribution is unrepresentative
and the aggregator learns nonsense that appears to work in validation.

**The fix – cross-fitting.** To get honest signal vectors on data you also want to train the
aggregator on:
1. Split `D_train ∪ D_meta` into K = 5 folds.
2. For each fold k: train base model on the other four, compute signals on fold k.
3. Concatenate the out-of-fold signal vectors – honest meta-training data.
4. Retrain the final base model on all of `D_train ∪ D_meta` for test-time deployment.

Cost: 5× base-model training. On tabular data with GBMs this is minutes. Worth it.

**Run this as an explicit ablation** (naive fitting vs cross-fitting). It will show a gap, and that
gap is a methodological contribution in its own right – the kind of practical finding that gets
cited.

**Other leakage traps:**
- Fit preprocessing (scalers, encoders, imputers) inside the training fold only.
- Fit temperature scaling on `D_meta`, never on test.
- Tier C signals index the *training* set only. Never let a test point be its own neighbour.
- Temporal shift datasets: split by time, never randomly.
- Touch `D_test` exactly once per dataset, at the end. Every hyperparameter decision comes
  from `D_meta`. Log the date you first evaluated on test as a self-discipline check.

---

## 7. Evaluation protocol

### Primary metrics
- **Risk–coverage curve** – the standard figure.
- **AURC** and **E-AURC** (excess over the optimal-ordering curve). Use E-AURC for
  cross-dataset aggregation, because raw AURC is confounded by base accuracy.
- **Selective risk at fixed coverage** – {100, 95, 90, 80, 70, 50}%.
- **Coverage at fixed target risk** – {1, 2, 5, 10}%. More deployment-relevant than the above.
- **Coverage at *guaranteed* risk ≤ α** (the conformal number). Your differentiator.
- **Failure-prediction AUROC** – discriminating correct from incorrect predictions.

### Secondary
- Calibration: ECE (equal-mass bins), Brier, NLL.
- Cost model: `total cost = c_error·(errors on accepted) + c_review·(abstentions)`. Sweep the
  ratio `c_error/c_review` from 1 to 100 and plot total cost. This reframes the whole paper in
  terms a practitioner cares about and costs you nothing to compute.
- Subgroup selective risk: max–min gap, and worst-group selective risk at fixed overall coverage.
- Runtime: inference overhead per signal, in ms, and total training cost.

### Statistics – non-negotiable
- **10 random seeds** for every configuration. Report mean ± std, not single runs.
- Per-dataset: **paired bootstrap** on the test set for AURC differences.
- Across datasets: **Wilcoxon signed-rank** for pairwise method comparison, **Friedman test +
  Nemenyi critical-difference diagram** for the full method set. A CD diagram over 12 datasets
  is worth more to a reviewer than any single-dataset win.
- Correct for multiple comparisons (Holm–Bonferroni) and say so.

### Baselines – the full ladder
| Baseline | Purpose |
|---|---|
| Random abstention | Lower bound – sanity floor |
| MSP threshold (Chow) | **The one to beat.** Tune it properly. |
| Entropy, margin | Cheap variants |
| Temperature-scaled MSP | The cheap fix that often closes the gap |
| Logit-norm-normalised MSP | The other cheap fix |
| MC-dropout MI | Epistemic single-signal |
| Ensemble disagreement | Strong but expensive |
| Energy / Mahalanobis / kNN, each alone | Isolates each signal family |
| SelectiveNet | Published learned-abstention |
| Deep Gamblers | Published learned-abstention |
| ConfidNet | Published failure predictor |
| Stacked correctness predictor (A1) | Your internal "is the loss doing anything?" control |
| **Oracle ordering** | Upper bound – abstain on exactly the misclassified points. Shows headroom. |

Reporting the oracle is a small honesty move with a large payoff: it tells the reader how much of
the achievable gap you closed, and pre-empts "the improvement is small" by showing the
ceiling is also close.

### Ablations
1. Leave-one-signal-out (all m signals).
2. Signal-family-only: Tier A only, +B, +C, +D.
3. Aggregator class: A0 vs linear vs GBM vs MLP vs adaptive gating.
4. Loss: BCE vs Loss 1 vs Loss 2 vs Loss 3.
5. With / without conformal wrapper.
6. Cross-fitting vs naive meta-fitting.
7. Meta-set size sweep: 100 / 500 / 1k / 5k / all – sample efficiency matters for small datasets.
8. Signal correlation matrix + PCA of the signal bank. If four signals carry all the variance, that
   is a finding, and it belongs in the paper whether or not it flatters the method.

---

## 8. Implementation requirements

### Environment
```
python 3.11
torch, lightgbm, xgboost, scikit-learn
numpy, pandas, pyarrow, scipy
openml            # dataset fetching
hydra-core        # config management – do not hand-roll argparse
matplotlib        # figures; avoid seaborn defaults for camera-ready
statsmodels       # Wilcoxon, Friedman
mapie or crepes   # conformal, or implement it yourself (~50 lines)
```

Pin versions in `requirements.txt` **on day one**. Reproducibility questions from reviewers are
common and a version drift mid-project will cost you a week.

### Repository layout
```
selective-prediction/
├── configs/            # hydra: dataset/, model/, signals/, aggregator/, experiment/
├── src/
│   ├── data/           # loaders, splits, cross-fitting, shift generators
│   ├── models/         # base model wrappers (uniform .fit/.predict_proba/.features)
│   ├── signals/        # one file per signal, common interface
│   ├── aggregators/    # A0, A1, A2, adaptive gating
│   ├── conformal/       # risk-control threshold selection
│   ├── metrics/        # AURC, E-AURC, risk@coverage, coverage@risk, ECE, subgroup
│   └── experiment/     # runner, seed control, result serialisation
├── scripts/            # run_all.sh, make_figures.py, make_tables.py
├── results/            # parquet, one row per (dataset, model, method, seed, coverage)
├── notebooks/          # exploration only – never the source of a paper number
├── paper/              # IEEEtran LaTeX
└── tests/
```

### Non-negotiable engineering rules
- **Every signal implements the same interface:** `fit(train_data, model) → self`,
  `score(X) → np.ndarray`. This makes leave-one-out ablations a config change, not a rewrite.
- **One long-format parquet for all results.** Columns:
  `dataset, base_model, method, seed, coverage, risk, accuracy, n_accepted, subgroup, runtime`.
  Every table and figure in the paper is a groupby over this file. No number in the paper should
  exist anywhere except as a query against it.
- **Seed everything:** python, numpy, torch, CUDA, and the dataset splits.
- **Cache signal vectors to disk.** They're expensive (ensembles, MC-dropout) and you'll
  re-aggregate them dozens of times.
- **Unit tests for the metrics.** Write a test where AURC is analytically known (perfect ordering,
  random ordering). A metric bug is the most likely source of a wrong headline number, and it's
  the cheapest possible bug to prevent.
- **Log every run** to a CSV with config hash, git SHA, and timestamp.

### Compute budget
Tabular is CPU-friendly. Full tabular grid – 12 datasets × 3 base models × 5 ensemble members
× 10 seeds – is roughly a few hundred CPU-hours, parallelisable, feasible on a decent laptop
over a weekend or a free Colab/Kaggle allowance. Images need a GPU: budget ~30 GPU-hours for
CIFAR-10/100 with 5-member ensembles, or near-zero if you use pretrained checkpoints. **The
image experiments are the droppable scope** if the schedule slips.

---

## 9. Timeline (14 weeks)

| Week | Work | Gate to pass before moving on |
|---|---|---|
| 1 | Literature: read the 6 "could sink you" + foundational papers. Build the related-work gap table. | Gap table complete; contribution wording drafted in one sentence |
| 2 | Repo skeleton, 3 datasets loading, base models training, MSP + entropy signals, AURC metric + unit tests | **Reproduce the known result that MSP is a strong baseline.** If you can't, your metric or splits are wrong. Do not proceed. |
| 3 | Tier A signals complete; risk–coverage plotting; cross-fitting implemented | Naive-vs-cross-fit gap measured on 3 datasets |
| 4 | Tier B + C signals; signal correlation analysis | Correlation matrix produced; redundancy understood |
| 5 | Aggregators A0, A1; the three losses; A1-vs-A2 comparison on 3 datasets | **Decision point.** If the coverage-targeted loss beats BCE, continue as planned. If not, pivot the paper's centre of gravity to the adaptive-weighting and conformal contributions. |
| 6 | Adaptive gating aggregator; weight-visualisation figure | Weights differ measurably between clean and shifted inputs – or you've established they don't |
| 7 | Conformal risk-control wrapper; coverage@guaranteed-risk table | Empirical coverage matches the nominal guarantee in-distribution |
| 8 | Scale to all 12 tabular datasets, 10 seeds, full method ladder | Full results parquet populated |
| 9 | Shift experiments: temporal splits + CIFAR-C | RQ2 answered one way or the other |
| 10 | Baselines requiring retraining (SelectiveNet, Deep Gamblers, ConfidNet) | Published baselines reproduced within a sane margin of their papers |
| 11 | All ablations; subgroup/fairness analysis; statistical testing + CD diagrams | Every claim you intend to make has a test behind it |
| 12 | Figures and tables finalised; write methods + experiments sections | Camera-ready figures; no number outside the parquet |
| 13 | Write intro, related work, discussion, limitations; internal review | Full draft, ≤ page limit |
| 14 | Buffer: reviewer-proofing, code cleanup, artifact release, submission | Submitted |

Weeks 2 and 5 are hard gates. Both exist to stop you spending three months on something that
was broken in week 2 or contributing nothing in week 5.

---

## 10. Paper structure (IEEE conference, typically 6–8 pages, IEEEtran)

1. **Introduction** – Chow's rule is optimal under correct posteriors; posteriors are not correct;
   the 2024 benchmark shows no method dominates; therefore adaptive multi-signal abstention.
   State contributions as a numbered list of exactly 3–4 items.
2. **Related work** – selective classification, uncertainty quantification, OOD detection,
   conformal risk control. Close with the gap table.
3. **Preliminaries** – selective risk, coverage, risk–coverage curve, AURC formalism.
4. **Method** – signal bank, aggregator, coverage-targeted loss, adaptive weighting, conformal
   wrapper. One clean architecture figure.
5. **Experimental setup** – datasets, base models, splits and cross-fitting, baselines, metrics,
   statistical protocol.
6. **Results** – main table (E-AURC across datasets), risk–coverage figures, coverage@guaranteed-risk
   table, shift results, cost curves, CD diagram.
7. **Ablations and analysis** – signal contribution, loss comparison, adaptive-weight
   visualisation, correlation/redundancy, subgroup disparity.
8. **Limitations** – exchangeability breaks under shift; Tier C signals need the training set at
   inference; meta-split cost on small datasets; results are tabular-and-CIFAR, not
   ImageNet-scale.
9. **Conclusion**

**Contribution list, drafted:**
1. A post-hoc, model-agnostic multi-criteria abstention framework requiring no retraining of the
   base model.
2. A coverage-targeted training objective for the aggregator, shown to outperform correctness-
   classification training in the low-coverage regime.
3. An instance-adaptive signal weighting mechanism, with analysis showing which signals
   dominate in which failure regime.
4. Distribution-free selective-risk control via a conformal wrapper, with an evaluation across 12
   tabular and 2 image benchmarks including distribution shift and subgroup disparity.

### Submission checklist
- Correct IEEEtran template for your specific venue (they differ) – download it in week 1, not week 13
- Page limit and whether references count against it
- Double-blind? – anonymise the repo, strip acknowledgements, check PDF metadata
- Figures legible at print size, in vector format, colour-blind-safe palette
- Ethics / broader-impact statement if required
- Reproducibility statement + public repo with a fixed release tag
- Every citation verified against the actual paper (see the caution in §2)

---

## 11. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tuned MSP baseline is unbeatable in-distribution | **High** | Pre-registered RQ2: claim regime-dependence, not universal gains. Emphasise shift, low coverage, and the guarantee. |
| Signals are highly redundant | Medium-high | Correlation + PCA analysis becomes a reported finding, not a hidden failure |
| Aggregator overfits the meta-split on small datasets | Medium | Cross-fitting; strong regularisation; the meta-set-size sweep quantifies exactly where it breaks |
| "This is just stacking" | **High** – expect it in review | The coverage-targeted loss, the adaptive weighting, and the conformal guarantee are the three answers. Have all three, and have the A1-vs-A2 ablation ready. |
| Tier B/C signals too slow to be practical | Medium | Report inference overhead honestly; offer a Tier-A-only variant as the cheap configuration |
| Multiple-comparison false positives across 12 datasets | Medium | Friedman + Nemenyi, Holm correction, 10 seeds |
| SelectiveNet / Deep Gamblers won't reproduce | Medium | Use authors' code where public; if reproduction fails, report your numbers alongside theirs and say so |
| Timeline slips | High | Drop scope in this order: (1) image experiments, (2) Tier D signals, (3) subgroup analysis, (4) adaptive weighting. Never drop cross-fitting, seeds, or statistical testing. |

---

## 12. What to do on Monday

1. Download the IEEEtran template for your target venue and create the empty paper skeleton.
2. Create the repo with the layout in §8.
3. Load Adult, German Credit, Covertype from OpenML. Train LightGBM on each.
4. Implement MSP and entropy signals, the AURC metric, and its unit test.
5. Produce one risk–coverage curve.

If by Friday you have a plot showing MSP thresholding clearly beating random abstention on
three datasets, with the numbers stored in a parquet file, the project is on track. That single plot
is the foundation everything else in this plan sits on.
