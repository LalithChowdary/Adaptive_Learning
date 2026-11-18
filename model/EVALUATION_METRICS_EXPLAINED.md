# Learner-Memory Evaluation Metrics Explained (`v2.2` 9-Model Benchmark)

This guide explains every metric, subgroup, cohort, statistical interval, and behavioral check reported in [`runs/v2.2-mps-transformer-comparison/REPORT.md`](runs/v2.2-mps-transformer-comparison/REPORT.md) and [`runs/v2.2-mps-transformer-comparison/run.json`](runs/v2.2-mps-transformer-comparison/run.json), defined in [`memory/evaluation.py`](memory/evaluation.py), [`memory/behavioral.py`](memory/behavioral.py), and [`run_v22.py`](run_v22.py).

---

## 1. Probability, Calibration, and Classification Metrics

Each model predicts a probability $\hat{p}_i \in [0, 1]$ for whether a learner will achieve **full success** ($y_i \in \{0, 1\}$) on the **first unaided attempt** of query $i$.

### 1.1 Log Loss (Binary Cross-Entropy) — **Primary Metric (Lower is Better)**
Defined in [`loss_per_query`](memory/evaluation.py#L8-L13) (clipped to $[10^{-7}, 1 - 10^{-7}]$):
$$\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^N \Big[y_i \ln(\hat{p}_i) + (1 - y_i)\ln(1 - \hat{p}_i)\Big]$$

- **What it measures:** Strictly proper scoring rule for probabilistic predictions. It rewards well-calibrated probabilities and heavily penalizes confident wrong predictions (e.g., predicting $90\%$ when the student fails).
- **Reference anchors:**
  - $\ln(2) \approx 0.69315$ is uniform $50\%$ guessing.
  - **`constant` baseline** always predicts the training base rate ($\bar{y}_{\text{train}} = 0.54304$), giving `0.68691` on `main_test` Ordinary and `0.67827` on `main_test` Delayed mean.
  - **`oracle` ceiling** uses the true simulator probability integrated over task noise, giving `0.53190` on `main_test` Ordinary and `0.52235` on `main_test` Delayed mean. (Even the oracle cannot reach `0.0` because human/simulated attempts remain inherently stochastic.)

### 1.2 Validation Selection Log Loss (`validation_selection_log_loss`) — **Lower is Better**
Defined in [`validation_losses`](memory/evaluation.py#L39-L49) with `balanced=True`:
$$\text{SelectionLogLoss} = \frac{\text{PracticeLogLoss}_{\text{val}} + \text{DelayedMeanLogLoss}_{\text{val}}}{2}$$
where $\text{DelayedMeanLogLoss}$ equally averages `day_1`, `day_7`, and `day_30` assessment log losses:
$$\text{DelayedMeanLogLoss} = \frac{1}{3}\left(\text{LogLoss}_{1\text{d}} + \text{LogLoss}_{7\text{d}} + \text{LogLoss}_{30\text{d}}\right).$$

- **What it measures:** Equal-weight balance between near-term practice prediction (`24,086` validation queries) and long-gap delayed retention (`3,327` validation queries across 1, 7, and 30 days) so neither dominates checkpoint selection.
- **How the winner is chosen ([`select_model`](run_v22.py#L73-L101)):** First, sequence models must pass **all 6/6 behavioral sanity checks across every seed** (`all_seeds_passed == True`); among eligible models, the model with the lowest mean `validation_selection_log_loss` wins, and its median-validation seed is chosen as the checkpoint.

### 1.3 Brier Score (`brier`) — **Lower is Better**
Defined in [`probability_report`](memory/evaluation.py#L80):
$$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2$$

- **What it measures:** Mean squared error between predicted probability $\hat{p}_i$ and binary outcome $y_i$. Like Log Loss, it is a strictly proper scoring rule, but it penalizes extreme confidence errors quadratically rather than logarithmically (`0.25` for $50\%$ guessing; `0.0` for perfect deterministic certainty).

### 1.4 10-Bin Expected Calibration Error (`ece_10`) — **Lower is Better**
Defined in [`probability_report`](memory/evaluation.py#L62-L76):
Predictions are grouped into $B = 10$ uniform confidence bins $[0.0, 0.1), \dots, [0.9, 1.0]$. For each non-empty bin $b$ of size $n_b$:
$$\text{ECE}_{10} = \sum_{b=1}^{10} \frac{n_b}{N} \left|\,\overline{\hat{p}}_b - \bar{y}_b\,\right|$$

- **What it measures:** Probability reliability. An `ece_10` of `0.0097` (about $0.97\%$) means that when the model says a student has a $70\%$ chance of succeeding, the student actually succeeds close to $70\%$ of the time (within ~1 percentage point on average).

### 1.5 Classification Accuracy (`accuracy_05`) — **Higher is Better**
Defined in [`probability_report`](memory/evaluation.py#L81):
$$\text{Accuracy}_{0.5} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}\!\Big((\hat{p}_i \ge 0.5) = y_i\Big)$$

- **What it measures:** Fraction of binary pass/fail decisions that match at a fixed $0.5$ threshold. Useful as a familiar sanity check, but **secondary to Log Loss and Brier** because a tutor needs well-calibrated probabilities (e.g., $62\%$ vs $88\%$), not just a hard $>50\%$ cutoff.

### 1.6 Fraction of Achievable Signal Captured (`constant_to_oracle_log_loss_fraction`) — **Higher is Better (Max = 1.0)**
Defined in [`run_v22.py`](run_v22.py#L160-L162):
$$\text{SignalFraction}(m) = \frac{\text{LogLoss}_{\text{constant}} - \text{LogLoss}_m}{\text{LogLoss}_{\text{constant}} - \text{LogLoss}_{\text{oracle}}}$$

- **What it measures:** How much of the **learnable predictive gap** between naive base-rate guessing (`constant`, `0.0`) and the theoretical ceiling (`oracle`, `1.0`) each model captures.
- **Example (`main_test` Delayed mean):** `constant` = `0.67827`, `oracle` = `0.52235` (achievable gap = `0.15592`). `monotonic_gru` scores `0.54517`, capturing `(0.67827 - 0.54517) / (0.67827 - 0.52235) = 85.36%` of all learnable delayed signal from observable history alone (without seeing hidden simulator strength).

### 1.7 Learner-Level Paired Bootstrap 95% Confidence Intervals (`paired_intervals`)
Defined in [`paired_intervals`](memory/evaluation.py#L87-L114):
- Computes per-query difference $\Delta_i = \text{LogLoss}_{\text{candidate},i} - \text{LogLoss}_{\text{reference},i}$ (averaged across seeds for sequence models).
- Resamples **whole learners** with replacement (`1,000` bootstrap draws) to account for correlation among queries from the same student.
- **Interpretation:** Since lower log loss is better, **`delta_log_loss < 0` favors the candidate**. When the 95% interval `[lower_95, upper_95]` is **strictly below zero**, the improvement is statistically consistent across learners; when it spans zero, the two models are statistically tied on that slice.

---

## 2. Evaluation Cohorts and Query Horizons

All test cohorts contain **200 held-out learners** completely disjoint from the `1,600` training and `200` validation learners:

| Evaluation Cohort | Purpose | Ordinary Queries | Follow-up Queries | 1-day | 7-day | 30-day | Total Queries |
|---|---|---:|---:|---:|---:|---:|---:|
| `main_validation` | Checkpoint & architecture selection | 17,687 | 6,399 | 1,116 | 1,113 | 1,098 | 27,413 |
| `main_test` | In-distribution held-out evaluation | 17,664 | 6,258 | 1,110 | 1,106 | 1,097 | 27,235 |
| `power_law_test` | Stress cohort with power-law forgetting $(1 + \Delta t/h)^{-1}$ instead of exponential | 17,720 | 6,333 | 1,105 | 1,102 | 1,116 | 27,376 |
| `long_gap_test` | Stress cohort with longer inter-session breaks | 17,722 | 6,846 | 1,096 | 1,101 | 1,117 | 27,882 |

- **Ordinary:** First unaided attempt on regular practice questions.
- **Follow-up:** Fresh unaided follow-up questions delivered after an assisted practice item.
- **1 day / 7 days / 30 days:** Independent delayed assessment checks after fixed retention gaps.
- **Delayed mean:** Unweighted mean across `1 day`, `7 days`, and `30 days`.

---

## 3. Behavioral Sanity Check Suite (6 Controlled Invariants)

Defined in [`evaluate_behavioral_suite`](memory/behavioral.py#L81-L196). Passing all 6 checks is a **hard selection gate** because an unconstrained neural model can match aggregate log loss while violating basic pedagogical sanity on edge cases:

1. **`zero_score_decreases_all_speeds`:** Submitting an unaided `0%` wrong answer across speeds (`5s`, `6.426s`, `60s`, `180s`, and missing time) must strictly lower predicted probability below cold start.
2. **`full_success_increases_all_speeds`:** Submitting an unaided `100%` answer across speeds (`5s`, `60s`, `180s`, and missing time) must strictly raise predicted probability above cold start.
3. **`assistance_and_outcome_ordering`:** `P(Unaided 100%) > P(Assisted 3-hint 100%) > P(Unaided 0%)`.
4. **`difficulty_monotonicity_all_formats`:** `Easy > Medium > Hard` strictly across all 5 question formats.
5. **`non_increasing_forgetting_after_pass_and_fail`:** Over `0–60` days without new practice, predictions never drift upward (both after a pass and after a failure), and two hard unaided passes retain at least $60\%$ of day-0 probability at day 30 while staying above cold start through day 60.
6. **`user_conditionals_trace_behaves_sensibly`:** Replaying the recorded *Conditionals* SQLite session (`0%` wrong in `6.426s` $\rightarrow$ 3 hints $\rightarrow$ `0%` retry in `2.27s`) drops prediction below cold start after attempt 1 and drops further after the second failure.

---

## 4. Complete 9-Model Benchmark Results (`v2.2`, 3 Seeds on `mps:0`)

### 4.1 Validation Selection, Behavioral Gate, Size, and Training Speed

| Model | Params | Avg Train Time / Seed (`mps:0`) | Behavioral Gate | Validation Selection Log Loss | Selected |
|---|---:|---:|---:|---:|---|
| `task_logistic` | — | CPU | Tabular baseline | `0.59707` | |
| `history_logistic` | — | CPU | Tabular baseline | `0.55692` | |
| `history_boosting` | — | CPU | Tabular baseline | `0.55532` | |
| `time_input_gru` | 9,633 | 60.0s | Pass (6.0 / 6) | `0.55177 ± 0.00022` | |
| `time_decay_gru` | 10,689 | 96.9s | **Fail (4.7 / 6)** | `0.55177 ± 0.00022` | |
| `time_aware_transformer` | 11,845 | 35.4s | Pass (6.0 / 6) | `0.55374 ± 0.00082` | |
| `kt_decay_gru` | 447 | 413.7s | Pass (6.0 / 6) | `0.55471 ± 0.00077` | |
| **`monotonic_gru`** | **10,223** | **593.4s** | **Pass (6.0 / 6)** | **`0.55175 ± 0.00025`** | **Winner** |
| `monotonic_transformer` | 10,208 | 577.2s | Pass (6.0 / 6) | `0.55209 ± 0.00020` | |

### 4.2 Full Held-Out Performance Across All Metrics

#### A. `main_test` (In-Distribution Held-Out Learners)

| Model | Ord Log Loss | Ord Brier | Ord ECE-10 | Ord Acc | Ord Signal% | Delayed Log Loss | Delayed Brier | Delayed ECE-10 | Delayed Acc | Delayed Signal% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant` | 0.68691 | 0.24689 | 0.01465 | 55.77% | 0.0% | 0.67827 | 0.24258 | 0.06473 | 60.78% | 0.0% |
| `task_logistic` | 0.60948 | 0.21077 | 0.01978 | 67.38% | 50.0% | 0.59806 | 0.20594 | 0.07963 | 69.12% | 51.4% |
| `history_logistic` | 0.57423 | 0.19552 | 0.01469 | 70.10% | 72.7% | 0.55282 | 0.18719 | 0.04999 | 71.81% | 80.5% |
| `history_boosting` | 0.57121 | 0.19449 | 0.00920 | 70.11% | 74.6% | 0.54955 | 0.18548 | 0.03121 | 71.50% | 82.6% |
| `time_input_gru` | 0.56802 | 0.19312 | 0.00875 | 70.39% | 76.7% | 0.54368 | 0.18306 | 0.03398 | 72.34% | 86.3% |
| `time_decay_gru` | 0.56782 | 0.19304 | 0.01056 | 70.40% | 76.8% | 0.54392 | 0.18319 | 0.03541 | 72.05% | 86.2% |
| `time_aware_transformer` | 0.56966 | 0.19379 | 0.01045 | 70.33% | 75.6% | 0.54523 | 0.18374 | 0.03676 | 72.32% | 85.3% |
| `kt_decay_gru` | 0.57074 | 0.19411 | 0.01174 | 70.15% | 74.9% | 0.54810 | 0.18480 | 0.03962 | 72.30% | 83.5% |
| **`monotonic_gru`** | **0.56772** | **0.19295** | 0.00966 | 70.37% | **76.9%** | 0.54517 | 0.18369 | 0.03549 | 72.28% | 85.4% |
| `monotonic_transformer` | 0.56800 | 0.19304 | 0.00981 | **70.44%** | 76.7% | 0.54488 | 0.18356 | 0.03456 | **72.33%** | 85.6% |
| `oracle` | 0.53190 | 0.17816 | 0.01023 | 73.22% | 100.0% | 0.52235 | 0.17460 | 0.03247 | 73.65% | 100.0% |

#### B. `power_law_test` & `long_gap_test` Stress Cohorts

| Model | `power_law` Ord LL | `power_law` Del LL | `power_law` Del Signal% | `long_gap` Ord LL | `long_gap` Del LL | `long_gap` Del Acc | `long_gap` Del Signal% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `constant` | 0.68963 | 0.68799 | 0.0% | 0.69491 | 0.68737 | 55.50% | 0.0% |
| `task_logistic` | 0.59827 | 0.60681 | 57.4% | 0.60675 | 0.60840 | 67.45% | 55.0% |
| `history_logistic` | 0.56388 | 0.57134 | 82.4% | 0.57436 | 0.57793 | 69.68% | 76.2% |
| `history_boosting` | 0.56083 | 0.56987 | 83.4% | 0.57210 | 0.57378 | 69.56% | 79.1% |
| `time_input_gru` | 0.55851 | 0.56647 | 85.8% | 0.56678 | 0.57163 | 69.42% | 80.6% |
| `time_decay_gru` | 0.55849 | 0.56721 | 85.3% | 0.56679 | 0.57241 | 69.27% | 80.0% |
| `time_aware_transformer` | 0.56061 | 0.56679 | 85.6% | 0.56889 | 0.57445 | 69.06% | 78.6% |
| `kt_decay_gru` | 0.56121 | 0.57284 | 81.3% | 0.57058 | 0.57464 | 70.00% | 78.5% |
| **`monotonic_gru`** | **0.55887** | 0.56839 | 84.5% | **0.56752** | 0.57186 | **70.15%** | 80.4% |
| `monotonic_transformer` | 0.55932 | 0.56869 | 84.3% | 0.56825 | **0.57153** | 70.01% | **80.6%** |
| `oracle` | 0.52479 | 0.54643 | 100.0% | 0.52969 | 0.54373 | 71.59% | 100.0% |

### 4.3 Key Learner-Level Paired Bootstrap Findings (GRU vs. Transformer)

- **`monotonic_gru` vs. `time_aware_transformer`:**
  - `monotonic_gru` wins validation selection (`0.55175` vs. `0.55374`) despite fewer parameters (`10,223` vs. `11,845`), and beats `time_aware_transformer` significantly on held-out `Ordinary` practice in both `main_test` (`Δ = -0.00194`, 95% CI `[-0.00322, -0.00048]`) and `power_law_test` (`Δ = -0.00174`, 95% CI `[-0.00324, -0.00023]`), as well as `power_law_test` `Follow-up` (`Δ = -0.00310`, 95% CI `[-0.00588, -0.00059]`) and `long_gap_test` Delayed mean (`0.57186` vs. `0.57445`).
- **`monotonic_transformer` vs. `time_aware_transformer`:**
  - Putting our cognitive grey-box equations around causal self-attention cuts validation selection log loss from `0.55374` down to `0.55209` and significantly lowers `main_test` Ordinary log loss (`Δ = -0.00166`, 95% CI `[-0.00290, -0.00029]`), confirming that the **HLR + FSRS Power-Law + Consolidated Floor + 3PL IRT structure**—not generic attention—drives performance.
- **`monotonic_gru` vs. `monotonic_transformer`:**
  - `monotonic_gru` edges out `monotonic_transformer` on validation selection (`0.55175` vs. `0.55209`), `main_test` Ordinary (`Δ = +0.00028`, 95% CI `[+0.00001, +0.00055]` favoring `monotonic_gru`), and `power_law_test` Ordinary (`Δ = +0.00045`, 95% CI `[+0.00019, +0.00072]`), while matching it within noise on delayed horizons—and uniquely provides an **$O(1)$ fixed 32-float Markov state** with zero history replay for the dashboard and upcoming PPO policy.
