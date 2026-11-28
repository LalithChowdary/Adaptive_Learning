# Research Evaluation Results & Report Reference (`v2.2` Offline Phase)

> **Scope Statement:** This document reports the verified **Retrospective** and **Pre-Registered** offline research evaluations across **Tier 1** (Learner Memory), **Tier 2** (Retrieval offline diagnostics), and **Tier 4 + Ablation 3** (Teaching Policy in simulation), alongside the exported blind review packs for **Tier 2/3 live-LLM & human adjudication**. All Tier 1 and Tier 4 claims are scoped strictly to the simulated learner environment (`3/data_Set/v2.2/simulator.py` and `3/ppo/env.py`); real-student effectiveness requires a separate IRB-approved Tier 5 pilot.

---

## Plain-English Guide for Readers (Beginner-Friendly Overview)

If you are reading this report for the first time (e.g., as an undergraduate student, thesis reviewer, or fresh graduate), here is what every part and metric means in plain English:

### What Each Tier Tests
1. **Our Main Goal (Primary Outcome & `Follow-up` vs. `1d/7d/30d`):** Imagine giving a Python tutor a fixed study budget (`32.0` units, roughly four short sessions spaced days apart). After studying finishes, we test the student **without any hints** at **1 day, 7 days, and 30 days later**. A good tutor should not just help a student cram for today—it should help them **still pass 30 days later**. Our primary outcome is the **equal-weight average pass rate across Days 1, 7, and 30 (`delayed_mean`)**, and we highlight **Day 30 (`day_30`)** on its own. (In Tier 1, `Follow-up` measures immediate unaided checks right after assisted practice, whereas `1d`/`7d`/`30d` measure retention after real time gaps.)
2. **Tier 1 — Memory Model (`monotonic_gru` & Comparators):** Before choosing what to teach, the tutor predicts: *"What is the chance this student will solve a fresh, unaided problem right now or 30 days from now?"*
   - **Log Loss (Lower is better):** Measures how accurate and honest the model's predicted probabilities are (punishes confident wrong guesses; `0.693` is random 50/50 guessing).
   - **ROC-AUC (Higher is better, `0.5` to `1.0`):** If you pick one student who passed and one who failed, how often did the model give the passing student a higher predicted probability?
   - **Brier Score & ECE-10 (Lower is better):** Checks calibration—if the model says "70% chance to pass," do students in that bin actually pass ~70% of the time? (`ECE-10` averages that calibration error across 10 equal-width probability bins `[0.0, 0.1)` ... `[0.9, 1.0]`.)
   - **Signal %:** How much of the maximum possible predictive signal (from naive `constant` base-rate `0%` up to the hidden simulator `oracle` ceiling `100%`) the model captures from observable history alone.
   - **6/6 Behavioral Gate:** Common-sense rules every tutor memory must obey (e.g., getting `0%` wrong must lower your predicted score, never raise it; 0–60 day forgetting curves must never creep upward without practice; passing without hints beats passing with 3 hints). Models failing any rule are disqualified from deployment.
3. **Tier 2 — Retrieval (`rag.py`):** Checks whether the search engine finds the **right** Python lessons and example questions inside the exact slot limits the LLM tutor receives (`4` chunks for chat; `3` exemplars + `2` curriculum chunks for question generation), quickly, and without ever leaking one student's mistakes into another student's session.
   - **`nDCG@k` (Higher is better, `0` to `1`):** Did the most relevant chunks appear near the *top* of the `k` retrieved slots?
   - **`Recall@k` (Higher is better, `0` to `1`):** Out of all truly relevant chunks available, what fraction made it into the top `k` slots?
   - **`p95 Latency (ms)` (Lower is better):** The 95th-percentile search speed—95% of queries finish faster than this time in milliseconds.
4. **Tier 3 — Tutor Replies, Question Generation & Grading (`llm_pedagogy.py`, `grader.py`):** Evaluates the live LLM teacher with blinded human reviewers.
   - **EduGuard Leakage vs. Over-Refusal (Tier 3a / Ablation 2):** *Leakage* is how often the tutor accidentally gives away the answer on protected answer-fishing prompts (lower is safer). *Over-Refusal* is how often the guardrail wrongly blocks or refuses a legitimate conceptual question (lower is more helpful).
   - **First-Pass / Post-Repair / Usable Rate (Tier 3b):** Out of `180` requests, *First-Pass* is valid JSON/AST on the initial call; *Post-Repair* includes 1-shot automatic fixing; *Usable Rate* is the fraction human reviewers confirm are actually correct, solvable, and match the requested skill/difficulty/format.
   - **MAE / QWK / Spearman $\rho$ (Tier 3c):** *MAE* is average grading error on `[0, 1]` (lower is better); *QWK* (Quadratic Weighted Kappa, up to `1.0`) measures agreement between the automated grader and human teachers on `0 / partial / full` grades; *Spearman $\rho$* checks whether higher grader confidence genuinely corresponds to lower error.
5. **Tier 4 — RL Teaching Policy (`delayed_retention_ppo` & Ablation 3):** Compares our reinforcement-learning tutor against a standard rule-based tutor (`threshold_heuristic`) at equal study budget.
   - **Permastore Floor $b_t$ (Higher is better):** The memory model's estimate of durable, consolidated knowledge that will *not* wash away even after a 30-day break.
   - **All 6 Skills at Day 30 & Budget Curves:** Makes sure the tutor doesn't neglect harder skills like `function_returns`, and checks whether PPO wins across tighter (`16`) and larger (`40`) study budgets.
   - **Ablation 3 (`ppo_no_longterm_state`):** Hides 30-day retention predictions and floor $b_t$ from PPO (shrinking its state from `36` to `23` numbers and restricting routing/masking to short-term `p_med`) to prove those long-term features cause the Day-30 gain.
6. **How Confidence Intervals Work (`Paired Bootstrap` vs. `Wilson 95% CI`):**
   - **Paired Bootstrap 95% CI (`B = 2,000`):** Compares two systems on the *exact same students* (or exact same prompts) by resampling students 2,000 times. If the entire 95% interval sits strictly above `0` for pass rates (or strictly below `0` for Log Loss), the improvement is consistent across students rather than luck.
   - **Wilson 95% CI:** A reliable confidence interval for simple yes/no percentages (like answer-leakage rate or usable-question rate), especially when rates are near `0%` or `100%`.
7. **Retrospective vs. Pre-Registered:** *Retrospective* numbers come from earlier 3-seed runs (`200` learners); *Pre-Registered* numbers come from fresh, locked **5-seed (Tier 1)** and **10-seed (`400` fresh learners, Tier 4)** runs executed under fixed protocols.

### Quick Table Map
| Table | What Question Does It Answer? |
|---|---|
| **T1-A** | Overall memory accuracy (`Log Loss`, `AUC`, `Brier`, `ECE-10`, `Signal %`) on ordinary practice vs. delayed exams. |
| **T1-B** | When `monotonic_gru` predicts e.g. `65%` chance to pass across 10 probability bins, do students actually pass ~`65%`? |
| **T1-C** | Memory accuracy broken down by exact gap (`followup`, `1d`, `7d`, `30d`, and `delayed_mean` with `Min–Max`). |
| **T1-D** | Paired 95% CIs (`B=2,000`) showing which memory architectures beat which baselines on the exact same students. |
| **T1-E** | Does each neural memory model pass all 6 common-sense behavioral sanity rules across every seed? |
| **T2** | How fast (`p95`), schema-accurate, leak-free (`0` violations), and relevant (`nDCG@k`, `Recall@k`) is each RAG retriever? |
| **T-A1** | Ablation 1: Does `BM25 + Dense` improve live-LLM tutor replies and generated questions over `Deployed BM25 + TF-IDF`? |
| **T3a** | Tutor reply quality (`Python Correct`, `Faithfulness`, 3 rubric dims) and Ablation 2 **EduGuard ON vs. OFF** safety. |
| **T3b** | Generated question validity (`First-Pass`, `Post-Repair`, `Fallback`, `Usable Rate`) overall and by skill/format/difficulty. |
| **T3c** | How closely does the automated self-explanation grader match human teachers (`MAE`, `P/R/F1`, `QWK`, `Spearman ρ`)? |
| **T4-A** | Primary policy benchmark: Does `delayed_retention_ppo` beat heuristic/bandit/fixed/random tutors at budget `32.0`? |
| **T4-B** | Day-30 unaided pass rates for every individual Python skill (and the weakest skill) under each teaching policy. |
| **T4-C** | Ablation 3: How much Day-30 retention is lost when hiding long-term 30-day memory & floor $b_t$ (`36` → `23` dims)? |
| **T4-D** | Budget curves: Does PPO still beat baselines across smaller and larger teaching budgets (`16`, `24`, `32`, `40`)? |
| **T4-E** | What mix of the 8 teaching actions does each policy use, and do PPO seeds pass the 5 behavioral sanity checks? |
| **T5** | Optional future IRB-approved human classroom pilot template (strictly separated from simulation). |

---

## Headline Summary (Pre-Registered Offline Findings)

- **Primary Outcome (`main_test`, Budget `32.0`, `N=400` fresh learners, 10 seeds):** `delayed_retention_ppo` achieves **62.77%** mean delayed unaided pass rate vs. **60.27%** for `threshold_heuristic` (**+2.50 pp**, 95% CI `[+2.12, +2.86]`).
- **Day-30 Durable Retention (`main_test`):** **59.66%** (`delayed_retention_ppo`) vs. **53.97%** (`threshold_heuristic`), a **+5.69 pp** (`[+5.05, +6.30]`) long-term retention gain.
- **Ablation 3 Proof (`ppo_no_longterm_state`, 23 dims):** Removing long-term retention & permastore features (and masking/routing with short-term `p_med` only) drops Day-30 pass rate from **59.66%** down to **58.82%** on `main_test` (`+0.84 pp`, 95% CI `[+0.61, +1.05]`), with **`8/10`** seeds passing `5/5` behavioral checks on `main_test`/`power_law_test` (`7/10` on `long_gap_test`) and **`10/10`** seeds beating all baselines on retention and floor.
- **Tier 1 Memory Selection & Gate:** In **Pre-Registered (5 seeds)**, `time_input_gru` (`0.55175 ± 0.00032` val selection LL, `6/6` behavioral gate) edged `monotonic_gru` (`0.55183 ± 0.00011`, `6/6` gate, tightest seed variance across all 5 seeds) by `0.00008`, whereas `monotonic_gru` won **Retrospective (3 seeds)** (`0.55175 ± 0.00025`) and remains the deployed $O(1)$ 32-float grey-box Markov state exposing explicit permastore floor $b_t$ and half-life $h_t$ for PPO/dashboard use; `time_decay_gru` was **rejected** by the gate (`5.0/6`, `0.55173 ± 0.00020` val LL, failing non-increasing forgetting).
- **Tier 2 Offline Retrieval Diagnostics:** Both `BM25 alone` and `Deployed BM25 + TF-IDF` achieve `100%` exemplar metadata match, `0` cross-learner episodic violations, `0` BM25 underfills, and `< 5.2 ms` `p95` latency across `66` dev and `132` held-out test queries.

---

## 1. Tier 1 Results: Learner Memory Model (`monotonic_gru` & Comparators)

Machine-readable log: [`3/eval_runs/v2.2-preregistered/tier1_run.json`](eval_runs/v2.2-preregistered/tier1_run.json)

### Tier 1 — Pre-Registered (5 seeds) (`5` seeds: `[20261201, 20261202, 20261203, 20261204, 20261205]`)

#### Validation Selection (`main_validation` · Pre-Registered (5 seeds))

- **Selected Model (`validation_selection` minimum among 6/6 gate passers):** `time_input_gru` · **Selected Checkpoint:** `time_input_gru/seed_20261201/model.pt`

| Model | Val Selection LL (Mean ± SD) | Val Min–Max | Behavioral Gate | Selection Status |
|---|---:|---|---|---|
| `task_logistic` | 0.59707 | — | Tabular | Baseline |
| `history_logistic` | 0.55692 | — | Tabular | Baseline |
| `history_boosting` | 0.55493 ± 0.00017 | 0.55467–0.55510 | Tabular | Baseline |
| `time_input_gru` | 0.55175 ± 0.00032 | 0.55138–0.55224 | 6.0/6 (Pass) | Selected (Lowest Passing Val LL) |
| `time_decay_gru` | 0.55173 ± 0.00020 | 0.55150–0.55197 | 5.0/6 (Fail) | Rejected (Behavioral Gate Fail) |
| `time_aware_transformer` | 0.55412 ± 0.00029 | 0.55378–0.55457 | 6.0/6 (Pass) | Eligible (Pass) |
| `kt_decay_gru` | 0.55469 ± 0.00038 | 0.55421–0.55519 | 6.0/6 (Pass) | Eligible (Pass) |
| `monotonic_gru` | 0.55183 ± 0.00011 | 0.55168–0.55198 | 6.0/6 (Pass) | Eligible (Pass) |
| `monotonic_transformer` | 0.55220 ± 0.00011 | 0.55208–0.55232 | 6.0/6 (Pass) | Eligible (Pass) |

#### Table T1-A (Pre-Registered (5 seeds) · `main_test`): Held-Out Memory Performance (Ordinary vs. Delayed)

| Model | Behav. Gate | Ord. LL (Mean±SD) | Ord. AUC | Ord. Brier | Ord. ECE-10 | Ord. Signal% | Del. LL (Mean±SD) | Del. AUC | Del. Brier | Del. ECE-10 | Del. Signal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant` | — | 0.68691 | 0.5000 | 0.24689 | 0.01465 | 0.0% | 0.67827 | 0.5000 | 0.24256 | 0.06487 | 0.0% |
| `task_logistic` | Tabular | 0.60948 | 0.7199 | 0.21077 | 0.01978 | 50.0% | 0.59806 | 0.7356 | 0.20593 | 0.07790 | 51.4% |
| `history_logistic` | Tabular | 0.57423 | 0.7630 | 0.19552 | 0.01469 | 72.7% | 0.55282 | 0.7769 | 0.18717 | 0.04755 | 80.5% |
| `history_boosting` | Tabular | 0.57122 ± 0.00010 | 0.7649 ± 0.0002 | 0.19451 ± 0.00005 | 0.01151 ± 0.00124 | 74.6% | 0.55037 ± 0.00068 | 0.7753 ± 0.0008 | 0.18586 ± 0.00028 | 0.02888 ± 0.00083 | 82.0% |
| `time_input_gru` | 6.0/6 (Pass) | 0.56782 ± 0.00028 | 0.7686 ± 0.0003 | 0.19306 ± 0.00012 | 0.01100 ± 0.00088 | 76.8% | 0.54354 ± 0.00086 | 0.7813 ± 0.0007 | 0.18301 ± 0.00038 | 0.02013 ± 0.00297 | 86.4% |
| `time_decay_gru` | 5.0/6 (Fail) | 0.56783 ± 0.00016 | 0.7686 ± 0.0002 | 0.19305 ± 0.00007 | 0.01057 ± 0.00050 | 76.8% | 0.54375 ± 0.00061 | 0.7813 ± 0.0005 | 0.18312 ± 0.00024 | 0.02194 ± 0.00094 | 86.3% |
| `time_aware_transformer` | 6.0/6 (Pass) | 0.56964 ± 0.00034 | 0.7668 ± 0.0004 | 0.19380 ± 0.00014 | 0.01170 ± 0.00233 | 75.7% | 0.54524 ± 0.00163 | 0.7804 ± 0.0009 | 0.18373 ± 0.00067 | 0.02730 ± 0.00488 | 85.3% |
| `kt_decay_gru` | 6.0/6 (Pass) | 0.57067 ± 0.00020 | 0.7662 ± 0.0002 | 0.19406 ± 0.00007 | 0.01245 ± 0.00090 | 75.0% | 0.54857 ± 0.00039 | 0.7764 ± 0.0001 | 0.18499 ± 0.00017 | 0.02671 ± 0.00183 | 83.2% |
| `monotonic_gru` | 6.0/6 (Pass) | 0.56806 ± 0.00026 | 0.7687 ± 0.0003 | 0.19307 ± 0.00010 | 0.01078 ± 0.00079 | 76.7% | 0.54473 ± 0.00066 | 0.7803 ± 0.0005 | 0.18350 ± 0.00029 | 0.02345 ± 0.00221 | 85.6% |
| `monotonic_transformer` | 6.0/6 (Pass) | 0.56806 ± 0.00015 | 0.7686 ± 0.0002 | 0.19307 ± 0.00008 | 0.01059 ± 0.00113 | 76.7% | 0.54560 ± 0.00100 | 0.7801 ± 0.0005 | 0.18383 ± 0.00041 | 0.02698 ± 0.00369 | 85.1% |
| `oracle` | — | 0.53190 | 0.8032 | 0.17816 | 0.01023 | 100.0% | 0.52235 | 0.8007 | 0.17459 | 0.02108 | 100.0% |

#### Table T1-C (Pre-Registered (5 seeds) · `main_test`): Follow-Up & Per-Horizon (`1d`, `7d`, `30d`) Log Loss

| Model | `followup` LL | `day_1` LL | `day_7` LL | `day_30` LL | `delayed_mean` LL | `delayed_mean` Min–Max |
|---|---:|---:|---:|---:|---:|---|
| `constant` | 0.69215 | 0.67463 | 0.67486 | 0.68530 | 0.67827 | — |
| `task_logistic` | 0.60885 | 0.58405 | 0.60806 | 0.60207 | 0.59806 | — |
| `history_logistic` | 0.57794 | 0.54817 | 0.54489 | 0.56540 | 0.55282 | — |
| `history_boosting` | 0.57300 ± 0.00048 | 0.55046 ± 0.00084 | 0.53681 ± 0.00054 | 0.56385 ± 0.00108 | 0.55037 ± 0.00068 | 0.54961–0.55130 |
| `time_input_gru` | 0.56595 ± 0.00068 | 0.54414 ± 0.00038 | 0.52682 ± 0.00160 | 0.55965 ± 0.00142 | 0.54354 ± 0.00086 | 0.54282–0.54501 |
| `time_decay_gru` | 0.56607 ± 0.00036 | 0.54356 ± 0.00040 | 0.52633 ± 0.00119 | 0.56136 ± 0.00051 | 0.54375 ± 0.00061 | 0.54304–0.54446 |
| `time_aware_transformer` | 0.56605 ± 0.00060 | 0.54473 ± 0.00139 | 0.53034 ± 0.00229 | 0.56065 ± 0.00169 | 0.54524 ± 0.00163 | 0.54352–0.54757 |
| `kt_decay_gru` | 0.57019 ± 0.00013 | 0.55255 ± 0.00072 | 0.52776 ± 0.00071 | 0.56542 ± 0.00073 | 0.54857 ± 0.00039 | 0.54810–0.54903 |
| `monotonic_gru` | 0.56642 ± 0.00061 | 0.54526 ± 0.00090 | 0.52884 ± 0.00092 | 0.56010 ± 0.00055 | 0.54473 ± 0.00066 | 0.54414–0.54574 |
| `monotonic_transformer` | 0.56644 ± 0.00083 | 0.54565 ± 0.00051 | 0.53007 ± 0.00216 | 0.56108 ± 0.00106 | 0.54560 ± 0.00100 | 0.54419–0.54658 |
| `oracle` | 0.53189 | 0.52376 | 0.50736 | 0.53592 | 0.52235 | — |

#### Table T1-D (Pre-Registered (5 seeds) · `main_test`): Learner-Clustered Paired Bootstrap 95% CIs (`B=2000`, Negative Favors Candidate)

| Contrast (`Candidate minus Reference`) | `ordinary` Δ (95% CI) | `followup` Δ (95% CI) | `day_1` Δ (95% CI) | `day_7` Δ (95% CI) | `day_30` Δ (95% CI) |
|---|---|---|---|---|---|
| `history_logistic minus task_logistic` | -0.03524 [-0.04182, -0.02921] | -0.03091 [-0.04197, -0.01968] | -0.03587 [-0.05018, -0.02027] | -0.06317 [-0.07799, -0.04824] | -0.03667 [-0.05132, -0.02284] |
| `history_boosting minus history_logistic` | -0.00301 [-0.00457, -0.00145] | -0.00494 [-0.00814, -0.00172] | +0.00228 [-0.00388, +0.00866] | -0.00807 [-0.01444, -0.00153] | -0.00155 [-0.00573, +0.00272] |
| `monotonic_gru minus kt_decay_gru` | -0.00261 [-0.00371, -0.00147] | -0.00377 [-0.00576, -0.00162] | -0.00729 [-0.01173, -0.00303] | +0.00109 [-0.00257, +0.00455] | -0.00532 [-0.00892, -0.00177] |
| `monotonic_gru minus time_decay_gru` | +0.00022 [-0.00066, +0.00108] | +0.00035 [-0.00105, +0.00188] | +0.00169 [-0.00126, +0.00456] | +0.00251 [-0.00057, +0.00541] | -0.00126 [-0.00441, +0.00175] |
| `monotonic_gru minus time_aware_transformer` | -0.00158 [-0.00291, -0.00020] | +0.00036 [-0.00163, +0.00252] | +0.00053 [-0.00411, +0.00506] | -0.00150 [-0.00524, +0.00198] | -0.00055 [-0.00489, +0.00350] |
| `monotonic_transformer minus time_aware_transformer` | -0.00158 [-0.00296, -0.00012] | +0.00039 [-0.00175, +0.00262] | +0.00092 [-0.00379, +0.00533] | -0.00027 [-0.00400, +0.00337] | +0.00043 [-0.00431, +0.00483] |
| `monotonic_transformer minus monotonic_gru` | +0.00001 [-0.00028, +0.00032] | +0.00002 [-0.00039, +0.00043] | +0.00040 [-0.00064, +0.00147] | +0.00123 [+0.00003, +0.00234] | +0.00098 [-0.00017, +0.00213] |
| `monotonic_gru minus history_boosting` | -0.00317 [-0.00490, -0.00154] | -0.00659 [-0.00915, -0.00404] | -0.00520 [-0.01206, +0.00138] | -0.00797 [-0.01339, -0.00279] | -0.00375 [-0.00811, +0.00029] |

#### Table T1-A (Pre-Registered (5 seeds) · `power_law_test`): Held-Out Memory Performance (Ordinary vs. Delayed)

| Model | Behav. Gate | Ord. LL (Mean±SD) | Ord. AUC | Ord. Brier | Ord. ECE-10 | Ord. Signal% | Del. LL (Mean±SD) | Del. AUC | Del. Brier | Del. ECE-10 | Del. Signal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant` | — | 0.68963 | 0.5000 | 0.24824 | 0.00111 | 0.0% | 0.68799 | 0.5000 | 0.24744 | 0.00827 | 0.0% |
| `task_logistic` | Tabular | 0.59827 | 0.7395 | 0.20570 | 0.01151 | 55.4% | 0.60681 | 0.7291 | 0.20927 | 0.02867 | 57.3% |
| `history_logistic` | Tabular | 0.56388 | 0.7782 | 0.19086 | 0.01315 | 76.3% | 0.57134 | 0.7685 | 0.19379 | 0.02312 | 82.4% |
| `history_boosting` | Tabular | 0.56058 ± 0.00012 | 0.7806 ± 0.0002 | 0.18971 ± 0.00005 | 0.01303 ± 0.00064 | 78.3% | 0.57048 ± 0.00048 | 0.7688 ± 0.0004 | 0.19374 ± 0.00014 | 0.02342 ± 0.00373 | 83.0% |
| `time_input_gru` | 6.0/6 (Pass) | 0.55894 ± 0.00044 | 0.7825 ± 0.0003 | 0.18907 ± 0.00019 | 0.01645 ± 0.00260 | 79.3% | 0.56705 ± 0.00091 | 0.7727 ± 0.0004 | 0.19256 ± 0.00035 | 0.02936 ± 0.00205 | 85.4% |
| `time_decay_gru` | 5.0/6 (Fail) | 0.55857 ± 0.00026 | 0.7827 ± 0.0001 | 0.18892 ± 0.00011 | 0.01455 ± 0.00281 | 79.5% | 0.56722 ± 0.00075 | 0.7722 ± 0.0003 | 0.19273 ± 0.00029 | 0.02816 ± 0.00386 | 85.3% |
| `time_aware_transformer` | 6.0/6 (Pass) | 0.56103 ± 0.00029 | 0.7807 ± 0.0003 | 0.18996 ± 0.00012 | 0.01901 ± 0.00330 | 78.0% | 0.56727 ± 0.00107 | 0.7716 ± 0.0006 | 0.19262 ± 0.00048 | 0.02208 ± 0.00519 | 85.3% |
| `kt_decay_gru` | 6.0/6 (Pass) | 0.56112 ± 0.00025 | 0.7806 ± 0.0001 | 0.18985 ± 0.00011 | 0.01744 ± 0.00213 | 78.0% | 0.57237 ± 0.00029 | 0.7667 ± 0.0003 | 0.19478 ± 0.00011 | 0.02758 ± 0.00170 | 81.7% |
| `monotonic_gru` | 6.0/6 (Pass) | 0.55927 ± 0.00018 | 0.7822 ± 0.0003 | 0.18913 ± 0.00007 | 0.01470 ± 0.00225 | 79.1% | 0.56856 ± 0.00053 | 0.7710 ± 0.0003 | 0.19319 ± 0.00021 | 0.02600 ± 0.00216 | 84.4% |
| `monotonic_transformer` | 6.0/6 (Pass) | 0.55896 ± 0.00019 | 0.7825 ± 0.0002 | 0.18899 ± 0.00008 | 0.01383 ± 0.00203 | 79.3% | 0.56841 ± 0.00055 | 0.7709 ± 0.0002 | 0.19313 ± 0.00023 | 0.02503 ± 0.00191 | 84.5% |
| `oracle` | — | 0.52479 | 0.8132 | 0.17495 | 0.00813 | 100.0% | 0.54643 | 0.7914 | 0.18431 | 0.02256 | 100.0% |

#### Table T1-C (Pre-Registered (5 seeds) · `power_law_test`): Follow-Up & Per-Horizon (`1d`, `7d`, `30d`) Log Loss

| Model | `followup` LL | `day_1` LL | `day_7` LL | `day_30` LL | `delayed_mean` LL | `delayed_mean` Min–Max |
|---|---:|---:|---:|---:|---:|---|
| `constant` | 0.69377 | 0.68273 | 0.68684 | 0.69439 | 0.68799 | — |
| `task_logistic` | 0.59805 | 0.60123 | 0.61628 | 0.60291 | 0.60681 | — |
| `history_logistic` | 0.55595 | 0.56965 | 0.57600 | 0.56838 | 0.57134 | — |
| `history_boosting` | 0.55188 ± 0.00037 | 0.56903 ± 0.00075 | 0.57658 ± 0.00065 | 0.56583 ± 0.00060 | 0.57048 ± 0.00048 | 0.56996–0.57100 |
| `time_input_gru` | 0.54654 ± 0.00071 | 0.56408 ± 0.00036 | 0.57158 ± 0.00136 | 0.56547 ± 0.00183 | 0.56705 ± 0.00091 | 0.56560–0.56791 |
| `time_decay_gru` | 0.54690 ± 0.00042 | 0.56553 ± 0.00095 | 0.57158 ± 0.00134 | 0.56456 ± 0.00058 | 0.56722 ± 0.00075 | 0.56637–0.56842 |
| `time_aware_transformer` | 0.54988 ± 0.00037 | 0.56409 ± 0.00174 | 0.57257 ± 0.00121 | 0.56515 ± 0.00119 | 0.56727 ± 0.00107 | 0.56570–0.56826 |
| `kt_decay_gru` | 0.54978 ± 0.00023 | 0.57388 ± 0.00027 | 0.57539 ± 0.00045 | 0.56785 ± 0.00033 | 0.57237 ± 0.00029 | 0.57199–0.57265 |
| `monotonic_gru` | 0.54690 ± 0.00013 | 0.56659 ± 0.00082 | 0.57357 ± 0.00053 | 0.56551 ± 0.00055 | 0.56856 ± 0.00053 | 0.56825–0.56950 |
| `monotonic_transformer` | 0.54698 ± 0.00033 | 0.56640 ± 0.00079 | 0.57357 ± 0.00092 | 0.56526 ± 0.00044 | 0.56841 ± 0.00055 | 0.56773–0.56907 |
| `oracle` | 0.51458 | 0.55089 | 0.54719 | 0.54121 | 0.54643 | — |

#### Table T1-D (Pre-Registered (5 seeds) · `power_law_test`): Learner-Clustered Paired Bootstrap 95% CIs (`B=2000`, Negative Favors Candidate)

| Contrast (`Candidate minus Reference`) | `ordinary` Δ (95% CI) | `followup` Δ (95% CI) | `day_1` Δ (95% CI) | `day_7` Δ (95% CI) | `day_30` Δ (95% CI) |
|---|---|---|---|---|---|
| `history_logistic minus task_logistic` | -0.03440 [-0.04042, -0.02834] | -0.04209 [-0.05227, -0.03292] | -0.03158 [-0.04718, -0.01639] | -0.04028 [-0.05535, -0.02525] | -0.03453 [-0.04971, -0.01879] |
| `history_boosting minus history_logistic` | -0.00330 [-0.00487, -0.00176] | -0.00407 [-0.00697, -0.00140] | -0.00063 [-0.00674, +0.00524] | +0.00058 [-0.00495, +0.00623] | -0.00255 [-0.00801, +0.00327] |
| `monotonic_gru minus kt_decay_gru` | -0.00185 [-0.00300, -0.00070] | -0.00288 [-0.00480, -0.00094] | -0.00728 [-0.01204, -0.00236] | -0.00182 [-0.00588, +0.00188] | -0.00234 [-0.00585, +0.00132] |
| `monotonic_gru minus time_decay_gru` | +0.00070 [-0.00030, +0.00163] | -0.00001 [-0.00166, +0.00164] | +0.00107 [-0.00193, +0.00403] | +0.00199 [-0.00099, +0.00498] | +0.00095 [-0.00169, +0.00358] |
| `monotonic_gru minus time_aware_transformer` | -0.00176 [-0.00319, -0.00041] | -0.00298 [-0.00533, -0.00068] | +0.00250 [-0.00199, +0.00688] | +0.00099 [-0.00323, +0.00539] | +0.00037 [-0.00345, +0.00429] |
| `monotonic_transformer minus time_aware_transformer` | -0.00206 [-0.00352, -0.00065] | -0.00290 [-0.00534, -0.00049] | +0.00230 [-0.00183, +0.00647] | +0.00099 [-0.00317, +0.00510] | +0.00012 [-0.00398, +0.00403] |
| `monotonic_transformer minus monotonic_gru` | -0.00031 [-0.00064, +0.00002] | +0.00008 [-0.00037, +0.00053] | -0.00020 [-0.00115, +0.00081] | -0.00000 [-0.00106, +0.00105] | -0.00025 [-0.00147, +0.00087] |
| `monotonic_gru minus history_boosting` | -0.00131 [-0.00301, +0.00040] | -0.00498 [-0.00795, -0.00188] | -0.00243 [-0.00914, +0.00417] | -0.00301 [-0.00851, +0.00245] | -0.00032 [-0.00567, +0.00493] |

#### Table T1-A (Pre-Registered (5 seeds) · `long_gap_test`): Held-Out Memory Performance (Ordinary vs. Delayed)

| Model | Behav. Gate | Ord. LL (Mean±SD) | Ord. AUC | Ord. Brier | Ord. ECE-10 | Ord. Signal% | Del. LL (Mean±SD) | Del. AUC | Del. Brier | Del. ECE-10 | Del. Signal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant` | — | 0.69491 | 0.5000 | 0.25088 | 0.03170 | 0.0% | 0.68737 | 0.5000 | 0.24713 | 0.01188 | 0.0% |
| `task_logistic` | Tabular | 0.60675 | 0.7322 | 0.20953 | 0.02909 | 53.4% | 0.60840 | 0.7263 | 0.21005 | 0.03511 | 55.0% |
| `history_logistic` | Tabular | 0.57436 | 0.7694 | 0.19577 | 0.02264 | 73.0% | 0.57793 | 0.7612 | 0.19714 | 0.03577 | 76.2% |
| `history_boosting` | Tabular | 0.57235 ± 0.00031 | 0.7700 ± 0.0004 | 0.19524 ± 0.00015 | 0.01693 ± 0.00065 | 74.2% | 0.57547 ± 0.00093 | 0.7616 ± 0.0008 | 0.19629 ± 0.00040 | 0.02177 ± 0.00394 | 77.9% |
| `time_input_gru` | 6.0/6 (Pass) | 0.56734 ± 0.00050 | 0.7753 ± 0.0004 | 0.19319 ± 0.00019 | 0.01859 ± 0.00255 | 77.2% | 0.57217 ± 0.00148 | 0.7639 ± 0.0007 | 0.19520 ± 0.00068 | 0.03056 ± 0.00522 | 80.2% |
| `time_decay_gru` | 5.0/6 (Fail) | 0.56711 ± 0.00028 | 0.7754 ± 0.0002 | 0.19307 ± 0.00011 | 0.01738 ± 0.00270 | 77.4% | 0.57238 ± 0.00055 | 0.7640 ± 0.0004 | 0.19526 ± 0.00023 | 0.02733 ± 0.00327 | 80.1% |
| `time_aware_transformer` | 6.0/6 (Pass) | 0.56923 ± 0.00054 | 0.7737 ± 0.0004 | 0.19403 ± 0.00020 | 0.02198 ± 0.00326 | 76.1% | 0.57445 ± 0.00058 | 0.7623 ± 0.0005 | 0.19615 ± 0.00028 | 0.03228 ± 0.00298 | 78.6% |
| `kt_decay_gru` | 6.0/6 (Pass) | 0.57054 ± 0.00021 | 0.7725 ± 0.0001 | 0.19433 ± 0.00008 | 0.01964 ± 0.00234 | 75.3% | 0.57482 ± 0.00060 | 0.7634 ± 0.0003 | 0.19602 ± 0.00027 | 0.02849 ± 0.00141 | 78.4% |
| `monotonic_gru` | 6.0/6 (Pass) | 0.56799 ± 0.00021 | 0.7749 ± 0.0003 | 0.19333 ± 0.00009 | 0.01811 ± 0.00232 | 76.8% | 0.57190 ± 0.00050 | 0.7654 ± 0.0005 | 0.19482 ± 0.00025 | 0.02642 ± 0.00251 | 80.4% |
| `monotonic_transformer` | 6.0/6 (Pass) | 0.56790 ± 0.00036 | 0.7748 ± 0.0002 | 0.19329 ± 0.00015 | 0.01636 ± 0.00355 | 76.9% | 0.57193 ± 0.00033 | 0.7655 ± 0.0003 | 0.19485 ± 0.00013 | 0.02743 ± 0.00334 | 80.4% |
| `oracle` | — | 0.52969 | 0.8095 | 0.17779 | 0.00920 | 100.0% | 0.54373 | 0.7903 | 0.18353 | 0.01464 | 100.0% |

#### Table T1-C (Pre-Registered (5 seeds) · `long_gap_test`): Follow-Up & Per-Horizon (`1d`, `7d`, `30d`) Log Loss

| Model | `followup` LL | `day_1` LL | `day_7` LL | `day_30` LL | `delayed_mean` LL | `delayed_mean` Min–Max |
|---|---:|---:|---:|---:|---:|---|
| `constant` | 0.69848 | 0.68537 | 0.68707 | 0.68968 | 0.68737 | — |
| `task_logistic` | 0.60505 | 0.59513 | 0.62258 | 0.60749 | 0.60840 | — |
| `history_logistic` | 0.56118 | 0.56219 | 0.59575 | 0.57586 | 0.57793 | — |
| `history_boosting` | 0.55876 ± 0.00024 | 0.55528 ± 0.00162 | 0.59288 ± 0.00095 | 0.57825 ± 0.00164 | 0.57547 ± 0.00093 | 0.57439–0.57643 |
| `time_input_gru` | 0.55197 ± 0.00068 | 0.55343 ± 0.00133 | 0.58892 ± 0.00174 | 0.57416 ± 0.00174 | 0.57217 ± 0.00148 | 0.57107–0.57473 |
| `time_decay_gru` | 0.55255 ± 0.00031 | 0.55395 ± 0.00063 | 0.58872 ± 0.00092 | 0.57447 ± 0.00057 | 0.57238 ± 0.00055 | 0.57160–0.57288 |
| `time_aware_transformer` | 0.55469 ± 0.00038 | 0.55781 ± 0.00127 | 0.59039 ± 0.00097 | 0.57516 ± 0.00143 | 0.57445 ± 0.00058 | 0.57364–0.57506 |
| `kt_decay_gru` | 0.55622 ± 0.00028 | 0.55597 ± 0.00043 | 0.59088 ± 0.00071 | 0.57760 ± 0.00083 | 0.57482 ± 0.00060 | 0.57414–0.57545 |
| `monotonic_gru` | 0.55251 ± 0.00032 | 0.55370 ± 0.00050 | 0.58866 ± 0.00074 | 0.57336 ± 0.00044 | 0.57190 ± 0.00050 | 0.57114–0.57236 |
| `monotonic_transformer` | 0.55263 ± 0.00030 | 0.55307 ± 0.00050 | 0.58829 ± 0.00038 | 0.57442 ± 0.00057 | 0.57193 ± 0.00033 | 0.57154–0.57244 |
| `oracle` | 0.52090 | 0.51428 | 0.56294 | 0.55395 | 0.54373 | — |

#### Table T1-D (Pre-Registered (5 seeds) · `long_gap_test`): Learner-Clustered Paired Bootstrap 95% CIs (`B=2000`, Negative Favors Candidate)

| Contrast (`Candidate minus Reference`) | `ordinary` Δ (95% CI) | `followup` Δ (95% CI) | `day_1` Δ (95% CI) | `day_7` Δ (95% CI) | `day_30` Δ (95% CI) |
|---|---|---|---|---|---|
| `history_logistic minus task_logistic` | -0.03239 [-0.03905, -0.02536] | -0.04387 [-0.05449, -0.03360] | -0.03294 [-0.04928, -0.01749] | -0.02683 [-0.04308, -0.01042] | -0.03163 [-0.04760, -0.01675] |
| `history_boosting minus history_logistic` | -0.00201 [-0.00357, -0.00052] | -0.00241 [-0.00513, +0.00020] | -0.00691 [-0.01286, -0.00079] | -0.00287 [-0.00950, +0.00383] | +0.00240 [-0.00310, +0.00822] |
| `monotonic_gru minus kt_decay_gru` | -0.00256 [-0.00364, -0.00144] | -0.00370 [-0.00575, -0.00163] | -0.00228 [-0.00624, +0.00167] | -0.00223 [-0.00648, +0.00240] | -0.00425 [-0.00842, -0.00019] |
| `monotonic_gru minus time_decay_gru` | +0.00088 [-0.00006, +0.00181] | -0.00003 [-0.00147, +0.00138] | -0.00025 [-0.00353, +0.00318] | -0.00007 [-0.00346, +0.00328] | -0.00111 [-0.00448, +0.00230] |
| `monotonic_gru minus time_aware_transformer` | -0.00124 [-0.00252, +0.00005] | -0.00217 [-0.00437, +0.00021] | -0.00411 [-0.00903, +0.00062] | -0.00174 [-0.00623, +0.00305] | -0.00180 [-0.00599, +0.00259] |
| `monotonic_transformer minus time_aware_transformer` | -0.00133 [-0.00266, +0.00006] | -0.00205 [-0.00430, +0.00027] | -0.00473 [-0.00985, +0.00008] | -0.00211 [-0.00654, +0.00254] | -0.00074 [-0.00497, +0.00385] |
| `monotonic_transformer minus monotonic_gru` | -0.00009 [-0.00038, +0.00022] | +0.00012 [-0.00029, +0.00048] | -0.00062 [-0.00166, +0.00035] | -0.00037 [-0.00145, +0.00064] | +0.00107 [-0.00003, +0.00215] |
| `monotonic_gru minus history_boosting` | -0.00436 [-0.00612, -0.00262] | -0.00625 [-0.00912, -0.00348] | -0.00159 [-0.00817, +0.00493] | -0.00422 [-0.01042, +0.00223] | -0.00490 [-0.01044, +0.00033] |

#### Table T1-B (Pre-Registered (5 seeds) · `main_test`): 10-Bin Calibration & Reliability Data (`monotonic_gru`, Seed-Averaged)

| Bin Range | Ordinary Queries ($n_b$) | Ord. Mean Pred | Ord. Observed | Delayed (`assessment`) Queries ($n_b$) | Del. Mean Pred | Del. Observed |
|---|---:|---:|---:|---:|---:|---:|
| `[0.0, 0.1)` | 327 | 0.0704 | 0.0550 | 58 | 0.0694 | 0.0345 |
| `[0.1, 0.2)` | 1178 | 0.1538 | 0.1341 | 173 | 0.1546 | 0.1850 |
| `[0.2, 0.3)` | 1599 | 0.2534 | 0.2558 | 256 | 0.2532 | 0.2383 |
| `[0.3, 0.4)` | 1784 | 0.3499 | 0.3593 | 330 | 0.3499 | 0.4030 |
| `[0.4, 0.5)` | 1962 | 0.4508 | 0.4633 | 362 | 0.4499 | 0.4751 |
| `[0.5, 0.6)` | 2191 | 0.5510 | 0.5486 | 374 | 0.5522 | 0.5615 |
| `[0.6, 0.7)` | 2513 | 0.6505 | 0.6450 | 479 | 0.6497 | 0.7077 |
| `[0.7, 0.8)` | 2671 | 0.7504 | 0.7282 | 535 | 0.7506 | 0.7551 |
| `[0.8, 0.9)` | 2549 | 0.8486 | 0.8348 | 515 | 0.8512 | 0.8583 |
| `[0.9, 1.0]` | 890 | 0.9242 | 0.9213 | 231 | 0.9249 | 0.9481 |

#### Table T1-E (Pre-Registered (5 seeds)): 6-Check Behavioral Sanity Gate & Conditionals Trace Replay

| Sequence Model | Checks Passed | All Seeds Passed? | Fast `0%` Wrong (`6.4s`) Δ | After 3 Hints + 2nd `0%` Wrong | Non-Increasing Forgetting (`0–60d`) | Gate Status |
|---|---:|---|---:|---:|---|---|
| `time_input_gru` | 6.0 / 6 | Yes | 41.0% → 32.3% (-8.72 pp) | 25.4% (-15.60 pp; post-hints 32.2%) | Pass | Eligible |
| `time_decay_gru` | 5.0 / 6 | No | 41.1% → 34.4% (-6.74 pp) | 27.7% (-13.41 pp; post-hints 31.2%) | Fail | Rejected |
| `time_aware_transformer` | 6.0 / 6 | Yes | 42.5% → 30.5% (-12.04 pp) | 26.2% (-16.35 pp; post-hints 34.5%) | Pass | Eligible |
| `kt_decay_gru` | 6.0 / 6 | Yes | 40.0% → 36.2% (-3.83 pp) | 31.4% (-8.62 pp; post-hints 36.2%) | Pass | Eligible |
| `monotonic_gru` | 6.0 / 6 | Yes | 39.7% → 29.1% (-10.60 pp) | 20.7% (-19.04 pp; post-hints 29.1%) | Pass | Eligible |
| `monotonic_transformer` | 6.0 / 6 | Yes | 41.0% → 31.2% (-9.78 pp) | 23.0% (-17.99 pp; post-hints 31.2%) | Pass | Eligible |

### Tier 1 — Retrospective (3 seeds) (`3` seeds: `[20261120, 20261121, 20261122]`)

#### Validation Selection (`main_validation` · Retrospective (3 seeds))

- **Selected Model (`validation_selection` minimum among 6/6 gate passers):** `monotonic_gru` · **Selected Checkpoint:** `monotonic_gru/seed_20261122/model.pt`

| Model | Val Selection LL (Mean ± SD) | Val Min–Max | Behavioral Gate | Selection Status |
|---|---:|---|---|---|
| `task_logistic` | 0.59707 | — | Tabular | Baseline |
| `history_logistic` | 0.55692 | — | Tabular | Baseline |
| `history_boosting` | 0.55532 | — | Tabular | Baseline |
| `time_input_gru` | 0.55177 ± 0.00022 | 0.55160–0.55202 | 6.0/6 (Pass) | Eligible (Pass) |
| `time_decay_gru` | 0.55177 ± 0.00022 | 0.55158–0.55201 | 4.7/6 (Fail) | Rejected (Behavioral Gate Fail) |
| `time_aware_transformer` | 0.55374 ± 0.00082 | 0.55279–0.55427 | 6.0/6 (Pass) | Eligible (Pass) |
| `kt_decay_gru` | 0.55471 ± 0.00077 | 0.55398–0.55552 | 6.0/6 (Pass) | Eligible (Pass) |
| `monotonic_gru` | 0.55175 ± 0.00025 | 0.55148–0.55196 | 6.0/6 (Pass) | Selected (Lowest Passing Val LL) |
| `monotonic_transformer` | 0.55209 ± 0.00020 | 0.55186–0.55220 | 6.0/6 (Pass) | Eligible (Pass) |

#### Table T1-A (Retrospective (3 seeds) · `main_test`): Held-Out Memory Performance (Ordinary vs. Delayed)

| Model | Behav. Gate | Ord. LL (Mean±SD) | Ord. AUC | Ord. Brier | Ord. ECE-10 | Ord. Signal% | Del. LL (Mean±SD) | Del. AUC | Del. Brier | Del. ECE-10 | Del. Signal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant` | — | 0.68691 | 0.5000 | 0.24689 | 0.01465 | 0.0% | 0.67827 | 0.5000 | 0.24256 | 0.06487 | 0.0% |
| `task_logistic` | Tabular | 0.60948 | 0.7199 | 0.21077 | 0.01978 | 50.0% | 0.59806 | 0.7356 | 0.20593 | 0.07790 | 51.4% |
| `history_logistic` | Tabular | 0.57423 | 0.7630 | 0.19552 | 0.01469 | 72.7% | 0.55282 | 0.7769 | 0.18717 | 0.04755 | 80.5% |
| `history_boosting` | Tabular | 0.57121 | 0.7649 | 0.19449 | 0.00920 | 74.6% | 0.54955 | 0.7766 | 0.18546 | 0.03004 | 82.6% |
| `time_input_gru` | 6.0/6 (Pass) | 0.56802 ± 0.00030 | 0.7684 ± 0.0003 | 0.19312 ± 0.00011 | 0.00875 ± 0.00144 | 76.7% | 0.54368 ± 0.00120 | 0.7814 ± 0.0011 | 0.18304 ± 0.00049 | 0.02068 ± 0.00150 | 86.3% |
| `time_decay_gru` | 4.7/6 (Fail) | 0.56782 ± 0.00023 | 0.7685 ± 0.0003 | 0.19304 ± 0.00011 | 0.01056 ± 0.00078 | 76.8% | 0.54392 ± 0.00061 | 0.7814 ± 0.0005 | 0.18317 ± 0.00027 | 0.02431 ± 0.00313 | 86.2% |
| `time_aware_transformer` | 6.0/6 (Pass) | 0.56966 ± 0.00038 | 0.7668 ± 0.0004 | 0.19379 ± 0.00016 | 0.01045 ± 0.00194 | 75.6% | 0.54523 ± 0.00139 | 0.7803 ± 0.0011 | 0.18372 ± 0.00053 | 0.02754 ± 0.00245 | 85.3% |
| `kt_decay_gru` | 6.0/6 (Pass) | 0.57074 ± 0.00061 | 0.7661 ± 0.0005 | 0.19411 ± 0.00025 | 0.01174 ± 0.00147 | 74.9% | 0.54810 ± 0.00046 | 0.7765 ± 0.0006 | 0.18478 ± 0.00024 | 0.02483 ± 0.00276 | 83.5% |
| `monotonic_gru` | 6.0/6 (Pass) | 0.56772 ± 0.00009 | 0.7689 ± 0.0001 | 0.19295 ± 0.00003 | 0.00966 ± 0.00042 | 76.9% | 0.54517 ± 0.00027 | 0.7803 ± 0.0003 | 0.18367 ± 0.00012 | 0.02493 ± 0.00026 | 85.4% |
| `monotonic_transformer` | 6.0/6 (Pass) | 0.56800 ± 0.00016 | 0.7687 ± 0.0002 | 0.19304 ± 0.00008 | 0.00981 ± 0.00044 | 76.7% | 0.54488 ± 0.00031 | 0.7803 ± 0.0001 | 0.18355 ± 0.00012 | 0.02380 ± 0.00116 | 85.5% |
| `oracle` | — | 0.53190 | 0.8032 | 0.17816 | 0.01023 | 100.0% | 0.52235 | 0.8007 | 0.17459 | 0.02108 | 100.0% |

#### Table T1-C (Retrospective (3 seeds) · `main_test`): Follow-Up & Per-Horizon (`1d`, `7d`, `30d`) Log Loss

| Model | `followup` LL | `day_1` LL | `day_7` LL | `day_30` LL | `delayed_mean` LL | `delayed_mean` Min–Max |
|---|---:|---:|---:|---:|---:|---|
| `constant` | 0.69215 | 0.67463 | 0.67486 | 0.68530 | 0.67827 | — |
| `task_logistic` | 0.60885 | 0.58405 | 0.60806 | 0.60207 | 0.59806 | — |
| `history_logistic` | 0.57794 | 0.54817 | 0.54489 | 0.56540 | 0.55282 | — |
| `history_boosting` | 0.57335 | 0.54912 | 0.53688 | 0.56265 | 0.54955 | — |
| `time_input_gru` | 0.56613 ± 0.00086 | 0.54378 ± 0.00086 | 0.52799 ± 0.00268 | 0.55928 ± 0.00116 | 0.54368 ± 0.00120 | 0.54282–0.54505 |
| `time_decay_gru` | 0.56592 ± 0.00067 | 0.54296 ± 0.00038 | 0.52779 ± 0.00099 | 0.56101 ± 0.00096 | 0.54392 ± 0.00061 | 0.54322–0.54432 |
| `time_aware_transformer` | 0.56663 ± 0.00008 | 0.54533 ± 0.00071 | 0.52969 ± 0.00304 | 0.56068 ± 0.00096 | 0.54523 ± 0.00139 | 0.54363–0.54610 |
| `kt_decay_gru` | 0.57028 ± 0.00062 | 0.55297 ± 0.00148 | 0.52671 ± 0.00118 | 0.56462 ± 0.00098 | 0.54810 ± 0.00046 | 0.54757–0.54837 |
| `monotonic_gru` | 0.56624 ± 0.00016 | 0.54516 ± 0.00082 | 0.52958 ± 0.00038 | 0.56076 ± 0.00067 | 0.54517 ± 0.00027 | 0.54486–0.54533 |
| `monotonic_transformer` | 0.56637 ± 0.00038 | 0.54544 ± 0.00112 | 0.52839 ± 0.00092 | 0.56081 ± 0.00115 | 0.54488 ± 0.00031 | 0.54454–0.54515 |
| `oracle` | 0.53189 | 0.52376 | 0.50736 | 0.53592 | 0.52235 | — |

#### Table T1-D (Retrospective (3 seeds) · `main_test`): Learner-Clustered Paired Bootstrap 95% CIs (`B=2000`, Negative Favors Candidate)

| Contrast (`Candidate minus Reference`) | `ordinary` Δ (95% CI) | `followup` Δ (95% CI) | `day_1` Δ (95% CI) | `day_7` Δ (95% CI) | `day_30` Δ (95% CI) |
|---|---|---|---|---|---|
| `history_logistic minus task_logistic` | -0.03524 [-0.04171, -0.02919] | -0.03091 [-0.04288, -0.01939] | -0.03587 [-0.05071, -0.02064] | -0.06317 [-0.07771, -0.04880] | -0.03667 [-0.05064, -0.02350] |
| `history_boosting minus history_logistic` | -0.00303 [-0.00454, -0.00150] | -0.00460 [-0.00758, -0.00151] | +0.00095 [-0.00453, +0.00661] | -0.00800 [-0.01383, -0.00152] | -0.00275 [-0.00727, +0.00167] |
| `monotonic_gru minus kt_decay_gru` | -0.00302 [-0.00424, -0.00188] | -0.00404 [-0.00614, -0.00186] | -0.00781 [-0.01269, -0.00311] | +0.00287 [-0.00075, +0.00630] | -0.00386 [-0.00730, -0.00061] |
| `monotonic_gru minus time_decay_gru` | -0.00010 [-0.00098, +0.00078] | +0.00032 [-0.00113, +0.00186] | +0.00220 [-0.00075, +0.00512] | +0.00180 [-0.00116, +0.00475] | -0.00025 [-0.00309, +0.00273] |
| `monotonic_gru minus time_aware_transformer` | -0.00194 [-0.00327, -0.00059] | -0.00039 [-0.00274, +0.00193] | -0.00017 [-0.00481, +0.00447] | -0.00010 [-0.00376, +0.00347] | +0.00008 [-0.00445, +0.00447] |
| `monotonic_transformer minus time_aware_transformer` | -0.00166 [-0.00297, -0.00034] | -0.00026 [-0.00264, +0.00208] | +0.00011 [-0.00474, +0.00491] | -0.00130 [-0.00518, +0.00225] | +0.00013 [-0.00438, +0.00458] |
| `monotonic_transformer minus monotonic_gru` | +0.00028 [+0.00001, +0.00054] | +0.00013 [-0.00035, +0.00059] | +0.00028 [-0.00067, +0.00121] | -0.00120 [-0.00213, -0.00032] | +0.00005 [-0.00071, +0.00088] |
| `monotonic_gru minus history_boosting` | -0.00349 [-0.00510, -0.00195] | -0.00711 [-0.00979, -0.00443] | -0.00396 [-0.01047, +0.00247] | -0.00730 [-0.01242, -0.00218] | -0.00189 [-0.00608, +0.00238] |

#### Table T1-A (Retrospective (3 seeds) · `power_law_test`): Held-Out Memory Performance (Ordinary vs. Delayed)

| Model | Behav. Gate | Ord. LL (Mean±SD) | Ord. AUC | Ord. Brier | Ord. ECE-10 | Ord. Signal% | Del. LL (Mean±SD) | Del. AUC | Del. Brier | Del. ECE-10 | Del. Signal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant` | — | 0.68963 | 0.5000 | 0.24824 | 0.00111 | 0.0% | 0.68799 | 0.5000 | 0.24744 | 0.00827 | 0.0% |
| `task_logistic` | Tabular | 0.59827 | 0.7395 | 0.20570 | 0.01151 | 55.4% | 0.60681 | 0.7291 | 0.20927 | 0.02867 | 57.3% |
| `history_logistic` | Tabular | 0.56388 | 0.7782 | 0.19086 | 0.01315 | 76.3% | 0.57134 | 0.7685 | 0.19379 | 0.02312 | 82.4% |
| `history_boosting` | Tabular | 0.56083 | 0.7804 | 0.18981 | 0.01282 | 78.1% | 0.56987 | 0.7690 | 0.19360 | 0.02403 | 83.4% |
| `time_input_gru` | 6.0/6 (Pass) | 0.55851 ± 0.00030 | 0.7827 ± 0.0003 | 0.18887 ± 0.00014 | 0.01454 ± 0.00083 | 79.5% | 0.56647 ± 0.00065 | 0.7732 ± 0.0004 | 0.19231 ± 0.00023 | 0.02675 ± 0.00122 | 85.8% |
| `time_decay_gru` | 4.7/6 (Fail) | 0.55849 ± 0.00036 | 0.7828 ± 0.0003 | 0.18886 ± 0.00014 | 0.01409 ± 0.00227 | 79.6% | 0.56721 ± 0.00021 | 0.7720 ± 0.0006 | 0.19270 ± 0.00012 | 0.02540 ± 0.00110 | 85.3% |
| `time_aware_transformer` | 6.0/6 (Pass) | 0.56061 ± 0.00013 | 0.7811 ± 0.0001 | 0.18977 ± 0.00007 | 0.01819 ± 0.00073 | 78.3% | 0.56679 ± 0.00100 | 0.7721 ± 0.0012 | 0.19241 ± 0.00050 | 0.02334 ± 0.00296 | 85.6% |
| `kt_decay_gru` | 6.0/6 (Pass) | 0.56121 ± 0.00069 | 0.7805 ± 0.0005 | 0.18989 ± 0.00028 | 0.01813 ± 0.00126 | 77.9% | 0.57284 ± 0.00108 | 0.7667 ± 0.0005 | 0.19492 ± 0.00032 | 0.03012 ± 0.00484 | 81.3% |
| `monotonic_gru` | 6.0/6 (Pass) | 0.55887 ± 0.00008 | 0.7825 ± 0.0001 | 0.18897 ± 0.00004 | 0.01310 ± 0.00010 | 79.3% | 0.56839 ± 0.00018 | 0.7710 ± 0.0002 | 0.19311 ± 0.00009 | 0.02537 ± 0.00149 | 84.5% |
| `monotonic_transformer` | 6.0/6 (Pass) | 0.55932 ± 0.00018 | 0.7823 ± 0.0002 | 0.18913 ± 0.00007 | 0.01558 ± 0.00277 | 79.1% | 0.56869 ± 0.00015 | 0.7710 ± 0.0001 | 0.19323 ± 0.00004 | 0.02613 ± 0.00245 | 84.3% |
| `oracle` | — | 0.52479 | 0.8132 | 0.17495 | 0.00813 | 100.0% | 0.54643 | 0.7914 | 0.18431 | 0.02256 | 100.0% |

#### Table T1-C (Retrospective (3 seeds) · `power_law_test`): Follow-Up & Per-Horizon (`1d`, `7d`, `30d`) Log Loss

| Model | `followup` LL | `day_1` LL | `day_7` LL | `day_30` LL | `delayed_mean` LL | `delayed_mean` Min–Max |
|---|---:|---:|---:|---:|---:|---|
| `constant` | 0.69377 | 0.68273 | 0.68684 | 0.69439 | 0.68799 | — |
| `task_logistic` | 0.59805 | 0.60123 | 0.61628 | 0.60291 | 0.60681 | — |
| `history_logistic` | 0.55595 | 0.56965 | 0.57600 | 0.56838 | 0.57134 | — |
| `history_boosting` | 0.55182 | 0.56757 | 0.57590 | 0.56613 | 0.56987 | — |
| `time_input_gru` | 0.54649 ± 0.00030 | 0.56417 ± 0.00087 | 0.57068 ± 0.00056 | 0.56457 ± 0.00131 | 0.56647 ± 0.00065 | 0.56608–0.56723 |
| `time_decay_gru` | 0.54682 ± 0.00047 | 0.56527 ± 0.00047 | 0.57199 ± 0.00034 | 0.56436 ± 0.00046 | 0.56721 ± 0.00021 | 0.56697–0.56733 |
| `time_aware_transformer` | 0.54976 ± 0.00041 | 0.56342 ± 0.00211 | 0.57208 ± 0.00193 | 0.56488 ± 0.00204 | 0.56679 ± 0.00100 | 0.56611–0.56795 |
| `kt_decay_gru` | 0.54994 ± 0.00067 | 0.57375 ± 0.00029 | 0.57604 ± 0.00161 | 0.56874 ± 0.00182 | 0.57284 ± 0.00108 | 0.57205–0.57407 |
| `monotonic_gru` | 0.54666 ± 0.00011 | 0.56603 ± 0.00042 | 0.57388 ± 0.00012 | 0.56524 ± 0.00050 | 0.56839 ± 0.00018 | 0.56817–0.56850 |
| `monotonic_transformer` | 0.54703 ± 0.00026 | 0.56654 ± 0.00048 | 0.57417 ± 0.00086 | 0.56535 ± 0.00071 | 0.56869 ± 0.00015 | 0.56853–0.56883 |
| `oracle` | 0.51458 | 0.55089 | 0.54719 | 0.54121 | 0.54643 | — |

#### Table T1-D (Retrospective (3 seeds) · `power_law_test`): Learner-Clustered Paired Bootstrap 95% CIs (`B=2000`, Negative Favors Candidate)

| Contrast (`Candidate minus Reference`) | `ordinary` Δ (95% CI) | `followup` Δ (95% CI) | `day_1` Δ (95% CI) | `day_7` Δ (95% CI) | `day_30` Δ (95% CI) |
|---|---|---|---|---|---|
| `history_logistic minus task_logistic` | -0.03440 [-0.04059, -0.02823] | -0.04209 [-0.05203, -0.03233] | -0.03158 [-0.04788, -0.01634] | -0.04028 [-0.05574, -0.02564] | -0.03453 [-0.05081, -0.01890] |
| `history_boosting minus history_logistic` | -0.00305 [-0.00469, -0.00152] | -0.00413 [-0.00680, -0.00129] | -0.00208 [-0.00759, +0.00339] | -0.00010 [-0.00627, +0.00563] | -0.00225 [-0.00718, +0.00286] |
| `monotonic_gru minus kt_decay_gru` | -0.00233 [-0.00343, -0.00115] | -0.00328 [-0.00515, -0.00138] | -0.00772 [-0.01295, -0.00244] | -0.00216 [-0.00623, +0.00174] | -0.00349 [-0.00679, -0.00006] |
| `monotonic_gru minus time_decay_gru` | +0.00038 [-0.00053, +0.00131] | -0.00016 [-0.00179, +0.00151] | +0.00076 [-0.00220, +0.00368] | +0.00189 [-0.00110, +0.00467] | +0.00089 [-0.00185, +0.00335] |
| `monotonic_gru minus time_aware_transformer` | -0.00174 [-0.00323, -0.00028] | -0.00310 [-0.00575, -0.00041] | +0.00261 [-0.00175, +0.00679] | +0.00180 [-0.00264, +0.00609] | +0.00036 [-0.00371, +0.00417] |
| `monotonic_transformer minus time_aware_transformer` | -0.00129 [-0.00275, +0.00015] | -0.00273 [-0.00530, -0.00007] | +0.00313 [-0.00119, +0.00744] | +0.00209 [-0.00233, +0.00633] | +0.00047 [-0.00356, +0.00434] |
| `monotonic_transformer minus monotonic_gru` | +0.00045 [+0.00019, +0.00072] | +0.00037 [-0.00014, +0.00084] | +0.00051 [-0.00039, +0.00136] | +0.00029 [-0.00061, +0.00114] | +0.00011 [-0.00065, +0.00087] |
| `monotonic_gru minus history_boosting` | -0.00195 [-0.00376, -0.00034] | -0.00517 [-0.00828, -0.00203] | -0.00154 [-0.00807, +0.00471] | -0.00202 [-0.00735, +0.00311] | -0.00088 [-0.00547, +0.00377] |

#### Table T1-A (Retrospective (3 seeds) · `long_gap_test`): Held-Out Memory Performance (Ordinary vs. Delayed)

| Model | Behav. Gate | Ord. LL (Mean±SD) | Ord. AUC | Ord. Brier | Ord. ECE-10 | Ord. Signal% | Del. LL (Mean±SD) | Del. AUC | Del. Brier | Del. ECE-10 | Del. Signal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant` | — | 0.69491 | 0.5000 | 0.25088 | 0.03170 | 0.0% | 0.68737 | 0.5000 | 0.24713 | 0.01188 | 0.0% |
| `task_logistic` | Tabular | 0.60675 | 0.7322 | 0.20953 | 0.02909 | 53.4% | 0.60840 | 0.7263 | 0.21005 | 0.03511 | 55.0% |
| `history_logistic` | Tabular | 0.57436 | 0.7694 | 0.19577 | 0.02264 | 73.0% | 0.57793 | 0.7612 | 0.19714 | 0.03577 | 76.2% |
| `history_boosting` | Tabular | 0.57210 | 0.7704 | 0.19510 | 0.01706 | 74.3% | 0.57378 | 0.7629 | 0.19557 | 0.01962 | 79.1% |
| `time_input_gru` | 6.0/6 (Pass) | 0.56678 ± 0.00021 | 0.7754 ± 0.0002 | 0.19294 ± 0.00012 | 0.01402 ± 0.00051 | 77.5% | 0.57163 ± 0.00024 | 0.7645 ± 0.0002 | 0.19491 ± 0.00007 | 0.03196 ± 0.00523 | 80.6% |
| `time_decay_gru` | 4.7/6 (Fail) | 0.56679 ± 0.00018 | 0.7756 ± 0.0003 | 0.19291 ± 0.00009 | 0.01677 ± 0.00258 | 77.5% | 0.57241 ± 0.00109 | 0.7643 ± 0.0007 | 0.19527 ± 0.00053 | 0.02978 ± 0.00537 | 80.0% |
| `time_aware_transformer` | 6.0/6 (Pass) | 0.56889 ± 0.00044 | 0.7739 ± 0.0002 | 0.19388 ± 0.00016 | 0.02108 ± 0.00099 | 76.3% | 0.57445 ± 0.00044 | 0.7625 ± 0.0005 | 0.19609 ± 0.00018 | 0.03429 ± 0.00409 | 78.6% |
| `kt_decay_gru` | 6.0/6 (Pass) | 0.57058 ± 0.00050 | 0.7726 ± 0.0003 | 0.19434 ± 0.00020 | 0.02037 ± 0.00220 | 75.3% | 0.57464 ± 0.00107 | 0.7634 ± 0.0007 | 0.19593 ± 0.00046 | 0.02710 ± 0.00058 | 78.5% |
| `monotonic_gru` | 6.0/6 (Pass) | 0.56752 ± 0.00014 | 0.7752 ± 0.0002 | 0.19313 ± 0.00007 | 0.01676 ± 0.00070 | 77.1% | 0.57186 ± 0.00034 | 0.7654 ± 0.0001 | 0.19480 ± 0.00013 | 0.02607 ± 0.00099 | 80.4% |
| `monotonic_transformer` | 6.0/6 (Pass) | 0.56825 ± 0.00044 | 0.7747 ± 0.0001 | 0.19345 ± 0.00019 | 0.01971 ± 0.00317 | 76.7% | 0.57153 ± 0.00032 | 0.7656 ± 0.0003 | 0.19467 ± 0.00016 | 0.02630 ± 0.00079 | 80.6% |
| `oracle` | — | 0.52969 | 0.8095 | 0.17779 | 0.00920 | 100.0% | 0.54373 | 0.7903 | 0.18353 | 0.01464 | 100.0% |

#### Table T1-C (Retrospective (3 seeds) · `long_gap_test`): Follow-Up & Per-Horizon (`1d`, `7d`, `30d`) Log Loss

| Model | `followup` LL | `day_1` LL | `day_7` LL | `day_30` LL | `delayed_mean` LL | `delayed_mean` Min–Max |
|---|---:|---:|---:|---:|---:|---|
| `constant` | 0.69848 | 0.68537 | 0.68707 | 0.68968 | 0.68737 | — |
| `task_logistic` | 0.60505 | 0.59513 | 0.62258 | 0.60749 | 0.60840 | — |
| `history_logistic` | 0.56118 | 0.56219 | 0.59575 | 0.57586 | 0.57793 | — |
| `history_boosting` | 0.55852 | 0.55621 | 0.59000 | 0.57514 | 0.57378 | — |
| `time_input_gru` | 0.55236 ± 0.00005 | 0.55282 ± 0.00106 | 0.58798 ± 0.00056 | 0.57408 ± 0.00091 | 0.57163 ± 0.00024 | 0.57136–0.57177 |
| `time_decay_gru` | 0.55256 ± 0.00052 | 0.55427 ± 0.00122 | 0.58858 ± 0.00149 | 0.57437 ± 0.00057 | 0.57241 ± 0.00109 | 0.57133–0.57350 |
| `time_aware_transformer` | 0.55371 ± 0.00014 | 0.55675 ± 0.00192 | 0.59046 ± 0.00087 | 0.57612 ± 0.00108 | 0.57445 ± 0.00044 | 0.57394–0.57475 |
| `kt_decay_gru` | 0.55644 ± 0.00070 | 0.55583 ± 0.00097 | 0.59079 ± 0.00137 | 0.57730 ± 0.00099 | 0.57464 ± 0.00107 | 0.57345–0.57553 |
| `monotonic_gru` | 0.55210 ± 0.00016 | 0.55344 ± 0.00038 | 0.58827 ± 0.00057 | 0.57386 ± 0.00040 | 0.57186 ± 0.00034 | 0.57149–0.57216 |
| `monotonic_transformer` | 0.55267 ± 0.00041 | 0.55285 ± 0.00025 | 0.58767 ± 0.00018 | 0.57408 ± 0.00069 | 0.57153 ± 0.00032 | 0.57124–0.57188 |
| `oracle` | 0.52090 | 0.51428 | 0.56294 | 0.55395 | 0.54373 | — |

#### Table T1-D (Retrospective (3 seeds) · `long_gap_test`): Learner-Clustered Paired Bootstrap 95% CIs (`B=2000`, Negative Favors Candidate)

| Contrast (`Candidate minus Reference`) | `ordinary` Δ (95% CI) | `followup` Δ (95% CI) | `day_1` Δ (95% CI) | `day_7` Δ (95% CI) | `day_30` Δ (95% CI) |
|---|---|---|---|---|---|
| `history_logistic minus task_logistic` | -0.03239 [-0.03952, -0.02576] | -0.04387 [-0.05485, -0.03349] | -0.03294 [-0.04760, -0.01746] | -0.02683 [-0.04335, -0.01056] | -0.03163 [-0.04685, -0.01596] |
| `history_boosting minus history_logistic` | -0.00227 [-0.00375, -0.00082] | -0.00265 [-0.00546, +0.00020] | -0.00598 [-0.01097, -0.00063] | -0.00575 [-0.01169, +0.00025] | -0.00071 [-0.00613, +0.00473] |
| `monotonic_gru minus kt_decay_gru` | -0.00305 [-0.00415, -0.00200] | -0.00435 [-0.00645, -0.00232] | -0.00238 [-0.00678, +0.00184] | -0.00252 [-0.00716, +0.00223] | -0.00344 [-0.00720, +0.00037] |
| `monotonic_gru minus time_decay_gru` | +0.00074 [-0.00027, +0.00170] | -0.00046 [-0.00188, +0.00096] | -0.00082 [-0.00450, +0.00279] | -0.00031 [-0.00375, +0.00318] | -0.00051 [-0.00388, +0.00288] |
| `monotonic_gru minus time_aware_transformer` | -0.00136 [-0.00267, -0.00002] | -0.00162 [-0.00396, +0.00061] | -0.00331 [-0.00819, +0.00139] | -0.00219 [-0.00706, +0.00299] | -0.00226 [-0.00704, +0.00230] |
| `monotonic_transformer minus time_aware_transformer` | -0.00064 [-0.00196, +0.00066] | -0.00104 [-0.00352, +0.00128] | -0.00390 [-0.00889, +0.00077] | -0.00279 [-0.00781, +0.00241] | -0.00205 [-0.00677, +0.00264] |
| `monotonic_transformer minus monotonic_gru` | +0.00072 [+0.00044, +0.00099] | +0.00058 [+0.00011, +0.00104] | -0.00059 [-0.00151, +0.00030] | -0.00060 [-0.00157, +0.00029] | +0.00022 [-0.00069, +0.00115] |
| `monotonic_gru minus history_boosting` | -0.00457 [-0.00628, -0.00287] | -0.00643 [-0.00905, -0.00367] | -0.00276 [-0.00879, +0.00312] | -0.00173 [-0.00789, +0.00457] | -0.00128 [-0.00637, +0.00337] |

#### Table T1-B (Retrospective (3 seeds) · `main_test`): 10-Bin Calibration & Reliability Data (`monotonic_gru`, Seed-Averaged)

| Bin Range | Ordinary Queries ($n_b$) | Ord. Mean Pred | Ord. Observed | Delayed (`assessment`) Queries ($n_b$) | Del. Mean Pred | Del. Observed |
|---|---:|---:|---:|---:|---:|---:|
| `[0.0, 0.1)` | 328 | 0.0707 | 0.0518 | 56 | 0.0704 | 0.0179 |
| `[0.1, 0.2)` | 1175 | 0.1537 | 0.1328 | 170 | 0.1541 | 0.1882 |
| `[0.2, 0.3)` | 1655 | 0.2538 | 0.2586 | 264 | 0.2517 | 0.2462 |
| `[0.3, 0.4)` | 1807 | 0.3509 | 0.3630 | 342 | 0.3503 | 0.3947 |
| `[0.4, 0.5)` | 1941 | 0.4507 | 0.4663 | 355 | 0.4501 | 0.4789 |
| `[0.5, 0.6)` | 2235 | 0.5511 | 0.5526 | 377 | 0.5515 | 0.5729 |
| `[0.6, 0.7)` | 2511 | 0.6511 | 0.6472 | 488 | 0.6498 | 0.7131 |
| `[0.7, 0.8)` | 2658 | 0.7501 | 0.7325 | 528 | 0.7507 | 0.7500 |
| `[0.8, 0.9)` | 2507 | 0.8483 | 0.8381 | 511 | 0.8511 | 0.8630 |
| `[0.9, 1.0]` | 847 | 0.9238 | 0.9221 | 222 | 0.9245 | 0.9459 |

#### Table T1-B (Retrospective (3 seeds) Archived `selected_checkpoint` · `main_test`): 10-Bin Calibration (`monotonic_gru/seed_20261122/model.pt`)

| Bin Range | Ordinary Queries ($n_b$) | Ord. Mean Pred | Ord. Observed | Delayed (`assessment`) Queries ($n_b$) | Del. Mean Pred | Del. Observed |
|---|---:|---:|---:|---:|---:|---:|
| `[0.0, 0.1)` | 335 | 0.0711 | 0.0567 | 54 | 0.0706 | 0.0185 |
| `[0.1, 0.2)` | 1183 | 0.1544 | 0.1344 | 171 | 0.1547 | 0.1871 |
| `[0.2, 0.3)` | 1600 | 0.2532 | 0.2550 | 264 | 0.2535 | 0.2462 |
| `[0.3, 0.4)` | 1840 | 0.3494 | 0.3609 | 333 | 0.3506 | 0.3934 |
| `[0.4, 0.5)` | 1955 | 0.4503 | 0.4701 | 354 | 0.4492 | 0.4774 |
| `[0.5, 0.6)` | 2240 | 0.5508 | 0.5509 | 391 | 0.5516 | 0.5703 |
| `[0.6, 0.7)` | 2519 | 0.6506 | 0.6503 | 483 | 0.6508 | 0.7081 |
| `[0.7, 0.8)` | 2690 | 0.7500 | 0.7320 | 538 | 0.7510 | 0.7546 |
| `[0.8, 0.9)` | 2492 | 0.8485 | 0.8399 | 510 | 0.8515 | 0.8647 |
| `[0.9, 1.0]` | 810 | 0.9236 | 0.9235 | 215 | 0.9243 | 0.9488 |

#### Table T1-E (Retrospective (3 seeds)): 6-Check Behavioral Sanity Gate & Conditionals Trace Replay

| Sequence Model | Checks Passed | All Seeds Passed? | Fast `0%` Wrong (`6.4s`) Δ | After 3 Hints + 2nd `0%` Wrong | Non-Increasing Forgetting (`0–60d`) | Gate Status |
|---|---:|---|---:|---:|---|---|
| `time_input_gru` | 6.0 / 6 | Yes | 40.7% → 33.3% (-7.39 pp) | 30.0% (-10.66 pp; post-hints 30.6%) | Pass | Eligible |
| `time_decay_gru` | 4.7 / 6 | No | 40.7% → 34.6% (-6.14 pp) | 31.1% (-9.65 pp; post-hints 34.8%) | Fail | Rejected |
| `time_aware_transformer` | 6.0 / 6 | Yes | 42.4% → 32.0% (-10.41 pp) | 30.2% (-12.21 pp; post-hints 34.5%) | Pass | Eligible |
| `kt_decay_gru` | 6.0 / 6 | Yes | 41.5% → 37.1% (-4.41 pp) | 32.3% (-9.20 pp; post-hints 37.1%) | Pass | Eligible |
| `monotonic_gru` | 6.0 / 6 | Yes | 40.7% → 30.0% (-10.71 pp) | 22.0% (-18.68 pp; post-hints 30.0%) | Pass | Eligible |
| `monotonic_transformer` | 6.0 / 6 | Yes | 41.4% → 31.0% (-10.38 pp) | 22.4% (-18.97 pp; post-hints 31.0%) | Pass | Eligible |

---

## 2. Tier 2 Results: Hybrid Retrieval & Ablation 1 (Offline Diagnostics + Pending Human/Dense)

Machine-readable logs: [`tier2_offline_diagnostics.json`](eval_runs/v2.2-preregistered/tier2_offline_diagnostics.json) · [`tier2_offline_eval.jsonl`](eval_runs/v2.2-preregistered/tier2_offline_eval.jsonl) · Blind annotation pack: [`tier2_relevance_annotation_pack.md`](eval_runs/v2.2-preregistered/blind_packs/tier2_relevance_annotation_pack.md) (`1,090` query-doc pairs across `66` Dev + `132` Test queries).

> **Dense Encoder Note:** Local `sentence_transformers`, `transformers`, and `huggingface_hub` weights are not installed/cached in this environment (`cached_hf_dense_models: []`), so `Dense` and `BM25 + Dense` are wired through `DenseEncoder` and marked `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]`. Relevance metrics (`nDCG@k`, `Recall@k`) await 2-reviewer blinded labeling on the exported annotation pack.

#### Table T2: Tier 2 Retrieval Comparison (Held-Out Test Split, `N = 132` Queries: `72` Chat + `60` Q-Gen)

| Arm | Query Input (`BM25` / `Dense`) | Chat (`k=4`) nDCG@4 (Raw) | Chat (`k=4`) Recall@4 (Raw) | Q-Gen Exemplar (`k=3`) nDCG@3 / Recall@3 | Q-Gen Curric. (`k=2`) nDCG@2 / Recall@2 | Exemplar Meta Match% | Cross-Learner Violations | BM25 Underfills | p95 Latency (ms) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| `BM25 alone` | `rewrite` / — | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | 100.0% | 0 (`+0` probe) | 0/192 | 3.551 ms |
| `Deployed BM25 + TF-IDF` | `rewrite` / `rewrite` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | 100.0% | 0 (`+0` probe) | 0/192 | 5.053 ms |
| `Dense` *(Dev-selected)* | — / Dev-sel. | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING]` | `[PENDING]` | — | `[PENDING]` |
| `BM25 + Dense` | `rewrite` / Dev-sel. | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |

*Dev split operational check (`66` queries / `96` retrieval tasks): `BM25 alone` p95 = `3.480 ms`, `Deployed BM25 + TF-IDF` p95 = `5.197 ms`, metadata match = `100.0%`, cross-learner violations = `0`, BM25 underfills = `0`. Out-of-scope abstention is `not_supported_deferred` (`3/adaptive_web/rag.py` always returns best available chunks).* 

#### Table T-A1: Ablation 1 Fixed-LLM Downstream Check (`Deployed BM25 + TF-IDF` vs. `BM25 + Dense`)
*Blind downstream packet exported (`132` items: `72` chat + `60` q-gen): [`ablation1_downstream_pack.jsonl`](eval_runs/v2.2-preregistered/blind_packs/ablation1_downstream_pack.jsonl) (`ablation1_downstream_pack.md`).*

| Retrieval Arm | Chat `live_n` / `72` | Chat Python Correct (`k/N`, %, CI) | Chat Claim Faithfulness (`k/N`, %, CI) | Chat Mean Rubric (0–2) | Q-Gen First-Pass Valid (`k/60`, %) | Q-Gen Post-Repair Valid (`k/60`, %) | Q-Gen Bank Fallback (`k/60`, %) | Q-Gen Usable Rate (`k/60`, %, CI) |
|---|---:|---|---|---|---|---|---|---|
| `Deployed BM25 + TF-IDF` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| `BM25 + Dense` *(Dev-sel.)* | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| **Paired Diff (`BM25+Dense - Deployed`)** | — | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |

---

## 3. Tier 3 Results: Tutor Quality, EduGuard, Question Generation & Self-Explanation Grading (Pending Live LLM & Human Review)

No live LLM API calls or fabricated human labels were executed in this offline phase. All blind prompt/annotation packs are ready under `3/eval_runs/v2.2-preregistered/blind_packs/`:
- **Tier 3a + Ablation 2 (`216` prompts = `96` adaptation + `60` protected + `60` legit doubts):** [`tier3a_ablation2_prompts.jsonl`](eval_runs/v2.2-preregistered/blind_packs/tier3a_ablation2_prompts.jsonl) · [`tier3a_ablation2_prompts.md`](eval_runs/v2.2-preregistered/blind_packs/tier3a_ablation2_prompts.md)
- **Tier 3b (`180` stratified requests = `6×5×3×2`):** [`tier3b_qgen_requests.jsonl`](eval_runs/v2.2-preregistered/blind_packs/tier3b_qgen_requests.jsonl) · [`tier3b_qgen_requests.md`](eval_runs/v2.2-preregistered/blind_packs/tier3b_qgen_requests.md)
- **Tier 3c (`120` prompts across `18` `se_prompt_bank` prompts, `rubric=None`):** [`tier3c_self_explanation_prompts.jsonl`](eval_runs/v2.2-preregistered/blind_packs/tier3c_self_explanation_prompts.jsonl) · [`tier3c_self_explanation_prompts.md`](eval_runs/v2.2-preregistered/blind_packs/tier3c_self_explanation_prompts.md)

#### Table T3a: Tier 3a Tutor Quality & Ablation 2 EduGuard Safety (Isolated `chat_doubt` Snapshots)

| Condition | `live_n` / `api_errors` | Python Correct (`k/N`, %, 95% CI) | Claim Faithfulness (`k/N`, %, 95% CI) | Protected Leakage (`k/60`, %, 95% CI) | Legit Over-Refusal (`k/60`, %, 95% CI) | Diff. Addressed (0–2) | Actionable Step (0–2) | Tier Adaptation (0–2) |
|---|---:|---|---|---|---|---:|---:|---:|
| `EduGuard ON` (Pinned LLM) | `[PENDING LIVE LLM]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING LIVE LLM & HUMAN]` | `[PENDING LIVE LLM & HUMAN]` | `[PENDING HUMAN]` | `[PENDING HUMAN]` | `[PENDING HUMAN]` |
| `EduGuard OFF` (Ablation 2) | `[PENDING LIVE LLM]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING LIVE LLM & HUMAN]` | `[PENDING LIVE LLM & HUMAN]` | `[PENDING HUMAN]` | `[PENDING HUMAN]` | `[PENDING HUMAN]` |

#### Table T3b: Tier 3b Generated Question Quality (Aggregate & Marginals, `N = 180`)

| Slice | Requests ($N$) | `api_error` (`k/N`, %) | First-Pass Valid (`k/N`, %, CI) | Post-Repair Valid (`k/N`, %, CI) | Repair Yield (`rep/att`) | Bank Fallback (`k/N`, %, CI) | Usable Generated Rate (`k/N`, %, CI) |
|---|---:|---|---|---|---|---|---|
| **Overall Aggregate** | 180 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Skill: `range_boundaries` | 30 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Skill: `loop_tracing` | 30 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Skill: `accumulator_initialization` | 30 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Skill: `conditional_evaluation` | 30 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Skill: `list_indexing` | 30 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Skill: `function_returns` | 30 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Format: `single_choice` | 36 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Format: `multiple_select` | 36 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Format: `output_prediction` | 36 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Format: `fill_in_the_blank` | 36 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Format: `code_completion (AST)` | 36 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Difficulty: `easy` | 60 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Difficulty: `medium` | 60 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |
| Difficulty: `hard` | 60 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |

#### Table T3c: Tier 3c Self-Explanation Grader (`N = 120`, `se_prompt_bank`, `rubric=None`, Threshold `score >= 1.0`)

| Grader Mode | Resolved Cov. (`n/120`) | MAE (`[0, 1]`, 95% CI) | Full-Success Precision | Full-Success Recall | Full-Success F1 | Conf.–Error Spearman $\rho$ | QWK (3-Cat.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic Rubric (`grade_self_explanation`) | `[PENDING HUMAN CORPUS]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` |
| Live LLM Confidence Check (`source == "llm"`, conditioned on `det_score`) | `[PENDING LIVE LLM]` | — *(score fixed)* | — | — | — | `[PENDING LIVE LLM & HUMAN]` | — |

---

## 4. Tier 4 Results: Teaching Policy (`delayed_retention_ppo`), Ablation 3 & Budget Curves

Machine-readable log: [`3/eval_runs/v2.2-preregistered/tier4_run.json`](eval_runs/v2.2-preregistered/tier4_run.json)

### Tier 4 — Pre-Registered (10 seeds, N=400) (`10` seeds, `N=400` learners `40000..40399`, Budget `32.0`)

#### Table T4-A (Pre-Registered (10 seeds, N=400)): Policy Benchmark at Budget `32.0`

| Cohort | Policy | 1-Day Pass | 7-Day Pass | **30-Day Pass** | **Delayed Mean (Obs)** | Paired Diff vs `threshold_heuristic` (95% CI) | Delayed Expected | Floor $b_t$ | Seed Range (Min–Max) |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `main_test` | `delayed_retention_ppo` | 65.39% ± 0.69% | 63.26% ± 1.02% | **59.66% ± 1.92%** | **62.77% ± 1.12%** | +2.50 pp [+2.12, +2.86] | 62.76% ± 1.12% | 0.4268 ± 0.0302 | 60.76%–63.98% |
| `main_test` | `threshold_heuristic` | 65.17% | 61.68% | **53.97%** | **60.27%** | `0.00 pp` (Ref) | 60.30% | 0.3239 | — |
| `main_test` | `myopic_rl_bandit` | 57.60% ± 0.02% | 55.05% ± 0.03% | **49.28% ± 0.02%** | **53.98% ± 0.01%** | -6.30 pp [-6.92, -5.73] | 53.94% ± 0.01% | 0.3525 ± 0.0009 | 53.95%–54.01% |
| `main_test` | `fixed_curriculum` | 62.39% | 57.76% | **49.64%** | **56.60%** | -3.68 pp [-4.32, -3.06] | 56.75% | 0.3129 | — |
| `main_test` | `random_valid` | 63.65% | 60.32% | **54.72%** | **59.56%** | -0.71 pp [-1.22, -0.18] | 59.42% | 0.3702 | — |
| `power_law_test` | `delayed_retention_ppo` | 64.92% ± 0.68% | 62.93% ± 0.94% | **60.29% ± 1.58%** | **62.71% ± 1.01%** | +2.30 pp [+1.96, +2.65] | 62.66% ± 1.04% | 0.4250 ± 0.0294 | 60.90%–63.88% |
| `power_law_test` | `threshold_heuristic` | 64.46% | 61.18% | **55.60%** | **60.41%** | `0.00 pp` (Ref) | 60.41% | 0.3244 | — |
| `power_law_test` | `myopic_rl_bandit` | 57.21% ± 0.02% | 55.08% ± 0.02% | **50.64% ± 0.03%** | **54.31% ± 0.02%** | -6.10 pp [-6.73, -5.52] | 54.18% ± 0.01% | 0.3519 ± 0.0009 | 54.28%–54.34% |
| `power_law_test` | `fixed_curriculum` | 61.47% | 56.92% | **51.15%** | **56.51%** | -3.90 pp [-4.53, -3.32] | 56.57% | 0.3124 | — |
| `power_law_test` | `random_valid` | 62.61% | 59.75% | **55.56%** | **59.31%** | -1.11 pp [-1.64, -0.57] | 59.19% | 0.3681 | — |
| `long_gap_test` | `delayed_retention_ppo` | 62.82% ± 0.82% | 61.04% ± 0.99% | **58.37% ± 1.74%** | **60.74% ± 1.13%** | +2.94 pp [+2.56, +3.33] | 60.65% ± 1.16% | 0.4169 ± 0.0264 | 58.77%–62.05% |
| `long_gap_test` | `threshold_heuristic` | 61.74% | 58.69% | **52.97%** | **57.80%** | `0.00 pp` (Ref) | 57.69% | 0.3251 | — |
| `long_gap_test` | `myopic_rl_bandit` | 54.79% ± 0.02% | 52.80% ± 0.03% | **48.43% ± 0.04%** | **52.01% ± 0.02%** | -5.79 pp [-6.40, -5.22] | 51.78% ± 0.01% | 0.3497 ± 0.0025 | 51.96%–52.04% |
| `long_gap_test` | `fixed_curriculum` | 57.97% | 54.50% | **48.36%** | **53.61%** | -4.19 pp [-4.87, -3.55] | 53.64% | 0.3126 | — |
| `long_gap_test` | `random_valid` | 60.18% | 57.32% | **53.36%** | **56.95%** | -0.85 pp [-1.38, -0.28] | 56.83% | 0.3632 | — |

#### Table T4-B (Pre-Registered (10 seeds, N=400)): All 6 Day-30 Per-Skill Pass Rates & Lowest-Skill Retention

| Cohort | Policy | `range_boundaries` | `loop_tracing` | `accumulator_init.` | `conditional_eval.` | `list_indexing` | `function_returns` | **Min (Lowest Skill)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `main_test` | `delayed_retention_ppo` | 81.55% ± 2.14% | 73.99% ± 3.49% | 56.73% ± 3.58% | 56.55% ± 5.38% | 52.42% ± 2.78% | 36.72% ± 2.50% | **36.72% ± 2.50%** (`function_returns`) |
| `main_test` | `threshold_heuristic` | 71.00% | 62.33% | 54.25% | 51.67% | 48.58% | 36.00% | **36.00%** (`function_returns`) |
| `main_test` | `myopic_rl_bandit` | 90.73% ± 0.12% | 47.47% ± 0.04% | 36.08% ± 0.00% | 40.67% ± 0.00% | 49.08% ± 0.00% | 31.67% ± 0.00% | **31.67% ± 0.00%** (`function_returns`) |
| `main_test` | `fixed_curriculum` | 58.92% | 49.17% | 46.58% | 51.00% | 50.75% | 41.42% | **41.42%** (`function_returns`) |
| `main_test` | `random_valid` | 74.92% | 61.25% | 51.50% | 53.42% | 51.25% | 36.00% | **36.00%** (`function_returns`) |
| `main_test` | `ppo_no_longterm_state` | 80.71% ± 3.25% | 70.45% ± 9.12% | 56.69% ± 7.98% | 56.39% ± 7.17% | 51.94% ± 4.30% | 36.74% ± 3.11% | **36.74% ± 3.11%** (`function_returns`) |
| `power_law_test` | `delayed_retention_ppo` | 81.82% ± 1.86% | 74.12% ± 3.32% | 57.39% ± 3.43% | 57.08% ± 4.84% | 53.54% ± 2.28% | 37.76% ± 2.28% | **37.76% ± 2.28%** (`function_returns`) |
| `power_law_test` | `threshold_heuristic` | 72.67% | 64.25% | 56.17% | 53.00% | 50.33% | 37.17% | **37.17%** (`function_returns`) |
| `power_law_test` | `myopic_rl_bandit` | 90.57% ± 0.15% | 48.72% ± 0.04% | 37.67% ± 0.00% | 42.83% ± 0.00% | 50.77% ± 0.04% | 33.25% ± 0.00% | **33.25% ± 0.00%** (`function_returns`) |
| `power_law_test` | `fixed_curriculum` | 60.92% | 50.83% | 47.58% | 52.50% | 52.17% | 42.92% | **42.92%** (`function_returns`) |
| `power_law_test` | `random_valid` | 75.58% | 62.08% | 52.58% | 54.08% | 52.25% | 36.75% | **36.75%** (`function_returns`) |
| `power_law_test` | `ppo_no_longterm_state` | 80.87% ± 3.12% | 70.64% ± 8.56% | 57.12% ± 7.57% | 57.02% ± 6.57% | 52.99% ± 3.75% | 38.00% ± 2.81% | **38.00% ± 2.81%** (`function_returns`) |
| `long_gap_test` | `delayed_retention_ppo` | 81.32% ± 2.13% | 72.93% ± 3.46% | 55.17% ± 3.56% | 54.52% ± 4.97% | 51.00% ± 2.20% | 35.25% ± 2.02% | **35.25% ± 2.02%** (`function_returns`) |
| `long_gap_test` | `threshold_heuristic` | 71.83% | 61.50% | 52.92% | 49.75% | 47.58% | 34.25% | **34.25%** (`function_returns`) |
| `long_gap_test` | `myopic_rl_bandit` | 89.97% ± 0.18% | 46.45% ± 0.09% | 35.00% ± 0.00% | 39.75% ± 0.00% | 48.33% ± 0.00% | 31.08% ± 0.00% | **31.08% ± 0.00%** (`function_returns`) |
| `long_gap_test` | `fixed_curriculum` | 58.08% | 47.08% | 45.67% | 50.08% | 49.17% | 40.08% | **40.08%** (`function_returns`) |
| `long_gap_test` | `random_valid` | 74.67% | 60.17% | 50.33% | 50.67% | 49.58% | 34.75% | **34.75%** (`function_returns`) |
| `long_gap_test` | `ppo_no_longterm_state` | 81.23% ± 2.91% | 69.00% ± 8.74% | 54.12% ± 7.73% | 54.12% ± 6.57% | 50.42% ± 3.30% | 35.37% ± 2.72% | **35.37% ± 2.72%** (`function_returns`) |

#### Table T4-C (Pre-Registered (10 seeds, N=400)): Ablation 3 — Full 36-Dim `delayed_retention_ppo` vs. 23-Dim Mask/Routing-Ablated `ppo_no_longterm_state`

| Cohort | Policy | State Dims | Delayed Mean (Obs) | **Day-30 Pass** | Paired Diff (`ppo_full - ablated`, Obs / Day-30 95% CI) | Floor $b_t$ | Min-Skill Day-30 | `self_explanation` Share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `main_test` | `delayed_retention_ppo` | 36 | 62.77% ± 1.12% | **59.66% ± 1.92%** | — | 0.4268 ± 0.0302 | 36.72% ± 2.50% | 16.9% |
| `main_test` | `ppo_no_longterm_state` | 23 | 61.76% ± 3.28% | **58.82% ± 4.24%** | +1.02 pp [+0.87, +1.16] / Day-30: +0.84 pp [+0.61, +1.05] | 0.4214 ± 0.0400 | 36.74% ± 3.11% | 20.7% |
| `power_law_test` | `delayed_retention_ppo` | 36 | 62.71% ± 1.01% | **60.29% ± 1.58%** | — | 0.4250 ± 0.0294 | 37.76% ± 2.28% | 16.9% |
| `power_law_test` | `ppo_no_longterm_state` | 23 | 61.73% ± 3.13% | **59.44% ± 3.87%** | +0.98 pp [+0.84, +1.12] / Day-30: +0.85 pp [+0.64, +1.05] | 0.4197 ± 0.0391 | 38.00% ± 2.81% | 20.7% |
| `long_gap_test` | `delayed_retention_ppo` | 36 | 60.74% ± 1.13% | **58.37% ± 1.74%** | — | 0.4169 ± 0.0264 | 35.25% ± 2.02% | 16.9% |
| `long_gap_test` | `ppo_no_longterm_state` | 23 | 59.69% ± 3.33% | **57.38% ± 4.04%** | +1.06 pp [+0.91, +1.20] / Day-30: +0.99 pp [+0.77, +1.20] | 0.4120 ± 0.0355 | 35.37% ± 2.72% | 20.8% |

#### Table T4-D (Pre-Registered (10 seeds, N=400)): Matched Budget Curves (`TOTAL_BUDGET` in `{16.0, 24.0, 32.0, 40.0}`, `SESSION_BUDGET = 8.0`)

| Cohort | Budget ($B$) | Policy | Delayed Mean (Obs) | **Day-30 Pass** | Floor $b_t$ | Paired Diff vs `threshold_heuristic` (95% CI) |
|---|---:|---|---:|---:|---:|---|
| `main_test` | 16.0 | `delayed_retention_ppo` | 57.06% ± 0.67% | **52.58% ± 1.14%** | 0.3522 ± 0.0166 | +1.89 pp [+1.60, +2.16] |
| `main_test` | 16.0 | `threshold_heuristic` | 55.17% | **48.14%** | 0.2896 | `0.00 pp` (Ref) |
| `main_test` | 16.0 | `myopic_rl_bandit` | 53.72% ± 0.10% | **48.10% ± 0.21%** | 0.3231 ± 0.0063 | -1.46 pp [-1.84, -1.10] |
| `main_test` | 24.0 | `delayed_retention_ppo` | 60.17% ± 0.89% | **56.53% ± 1.54%** | 0.3924 ± 0.0240 | +2.16 pp [+1.82, +2.51] |
| `main_test` | 24.0 | `threshold_heuristic` | 58.00% | **51.47%** | 0.3072 | `0.00 pp` (Ref) |
| `main_test` | 24.0 | `myopic_rl_bandit` | 54.02% ± 0.02% | **49.03% ± 0.05%** | 0.3414 ± 0.0027 | -3.98 pp [-4.49, -3.52] |
| `main_test` | 32.0 | `delayed_retention_ppo` | 62.77% ± 1.12% | **59.66% ± 1.92%** | 0.4268 ± 0.0302 | +2.50 pp [+2.12, +2.86] |
| `main_test` | 32.0 | `threshold_heuristic` | 60.27% | **53.97%** | 0.3239 | `0.00 pp` (Ref) |
| `main_test` | 32.0 | `myopic_rl_bandit` | 53.98% ± 0.01% | **49.28% ± 0.02%** | 0.3525 ± 0.0009 | -6.30 pp [-6.92, -5.73] |
| `main_test` | 40.0 | `delayed_retention_ppo` | 65.13% ± 1.39% | **62.57% ± 2.32%** | 0.4583 ± 0.0362 | +2.61 pp [+2.21, +3.00] |
| `main_test` | 40.0 | `threshold_heuristic` | 62.52% | **56.26%** | 0.3407 | `0.00 pp` (Ref) |
| `main_test` | 40.0 | `myopic_rl_bandit` | 53.83% ± 0.02% | **49.43% ± 0.02%** | 0.3590 ± 0.0003 | -8.69 pp [-9.40, -8.03] |
| `power_law_test` | 16.0 | `delayed_retention_ppo` | 57.14% ± 0.63% | **53.45% ± 0.97%** | 0.3516 ± 0.0163 | +1.76 pp [+1.46, +2.03] |
| `power_law_test` | 16.0 | `threshold_heuristic` | 55.38% | **49.85%** | 0.2897 | `0.00 pp` (Ref) |
| `power_law_test` | 16.0 | `myopic_rl_bandit` | 53.91% ± 0.12% | **49.34% ± 0.23%** | 0.3226 ± 0.0063 | -1.46 pp [-1.85, -1.11] |
| `power_law_test` | 24.0 | `delayed_retention_ppo` | 60.14% ± 0.82% | **57.26% ± 1.29%** | 0.3911 ± 0.0233 | +1.90 pp [+1.56, +2.23] |
| `power_law_test` | 24.0 | `threshold_heuristic` | 58.24% | **53.06%** | 0.3076 | `0.00 pp` (Ref) |
| `power_law_test` | 24.0 | `myopic_rl_bandit` | 54.22% ± 0.02% | **50.31% ± 0.05%** | 0.3408 ± 0.0027 | -4.02 pp [-4.56, -3.53] |
| `power_law_test` | 32.0 | `delayed_retention_ppo` | 62.71% ± 1.01% | **60.29% ± 1.58%** | 0.4250 ± 0.0294 | +2.30 pp [+1.96, +2.65] |
| `power_law_test` | 32.0 | `threshold_heuristic` | 60.41% | **55.60%** | 0.3244 | `0.00 pp` (Ref) |
| `power_law_test` | 32.0 | `myopic_rl_bandit` | 54.31% ± 0.02% | **50.64% ± 0.03%** | 0.3519 ± 0.0009 | -6.10 pp [-6.73, -5.52] |
| `power_law_test` | 40.0 | `delayed_retention_ppo` | 65.01% ± 1.24% | **63.02% ± 1.90%** | 0.4560 ± 0.0351 | +2.41 pp [+2.03, +2.77] |
| `power_law_test` | 40.0 | `threshold_heuristic` | 62.60% | **57.92%** | 0.3412 | `0.00 pp` (Ref) |
| `power_law_test` | 40.0 | `myopic_rl_bandit` | 54.16% ± 0.02% | **50.71% ± 0.02%** | 0.3586 ± 0.0003 | -8.44 pp [-9.15, -7.80] |
| `long_gap_test` | 16.0 | `delayed_retention_ppo` | 56.03% ± 0.73% | **51.90% ± 1.14%** | 0.3489 ± 0.0157 | +2.24 pp [+1.92, +2.54] |
| `long_gap_test` | 16.0 | `threshold_heuristic` | 53.79% | **47.40%** | 0.2894 | `0.00 pp` (Ref) |
| `long_gap_test` | 16.0 | `myopic_rl_bandit` | 52.60% ± 0.10% | **47.56% ± 0.21%** | 0.3204 ± 0.0060 | -1.19 pp [-1.58, -0.82] |
| `long_gap_test` | 24.0 | `delayed_retention_ppo` | 58.55% ± 0.92% | **55.48% ± 1.49%** | 0.3854 ± 0.0219 | +2.49 pp [+2.13, +2.84] |
| `long_gap_test` | 24.0 | `threshold_heuristic` | 56.06% | **50.51%** | 0.3073 | `0.00 pp` (Ref) |
| `long_gap_test` | 24.0 | `myopic_rl_bandit` | 52.44% ± 0.03% | **48.35% ± 0.08%** | 0.3383 ± 0.0036 | -3.62 pp [-4.11, -3.17] |
| `long_gap_test` | 32.0 | `delayed_retention_ppo` | 60.74% ± 1.13% | **58.37% ± 1.74%** | 0.4169 ± 0.0264 | +2.94 pp [+2.56, +3.33] |
| `long_gap_test` | 32.0 | `threshold_heuristic` | 57.80% | **52.97%** | 0.3251 | `0.00 pp` (Ref) |
| `long_gap_test` | 32.0 | `myopic_rl_bandit` | 52.01% ± 0.02% | **48.43% ± 0.04%** | 0.3497 ± 0.0025 | -5.79 pp [-6.40, -5.22] |
| `long_gap_test` | 40.0 | `delayed_retention_ppo` | 62.91% ± 1.33% | **60.96% ± 1.98%** | 0.4457 ± 0.0305 | +3.24 pp [+2.82, +3.67] |
| `long_gap_test` | 40.0 | `threshold_heuristic` | 59.67% | **55.24%** | 0.3427 | `0.00 pp` (Ref) |
| `long_gap_test` | 40.0 | `myopic_rl_bandit` | 51.59% ± 0.02% | **48.59% ± 0.02%** | 0.3569 ± 0.0016 | -8.07 pp [-8.78, -7.39] |

#### Table T4-E (Pre-Registered (10 seeds, N=400)): 8-Action Distribution Shares & 5-Check Behavioral Sanity Suite

**Part 1 — 8-Action Distribution Shares (% of Turns):**

| Cohort | Policy | `give_easier` | `give_same` | `give_harder` | `explain` | `worked_ex` | `self_explain` | `review_pre` | `advance` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `main_test` | `delayed_retention_ppo` | 3.8% | 25.8% | 15.1% | 5.4% | 16.4% | 16.9% | 8.1% | 8.5% |
| `main_test` | `threshold_heuristic` | 0.5% | 60.4% | 16.7% | 12.7% | 0.0% | 0.0% | 0.0% | 9.7% |
| `main_test` | `myopic_rl_bandit` | 0.0% | 71.9% | 0.0% | 0.0% | 0.0% | 28.1% | 0.0% | 0.0% |
| `main_test` | `fixed_curriculum` | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| `main_test` | `random_valid` | 9.5% | 22.7% | 19.5% | 12.4% | 11.4% | 8.1% | 6.8% | 9.6% |
| `main_test` | `ppo_no_longterm_state` | 7.5% | 26.5% | 18.4% | 7.0% | 6.1% | 20.7% | 4.4% | 9.4% |
| `power_law_test` | `delayed_retention_ppo` | 3.8% | 25.9% | 15.0% | 5.4% | 16.4% | 16.9% | 8.1% | 8.5% |
| `power_law_test` | `threshold_heuristic` | 0.5% | 60.4% | 16.5% | 12.9% | 0.0% | 0.0% | 0.0% | 9.7% |
| `power_law_test` | `myopic_rl_bandit` | 0.0% | 72.0% | 0.0% | 0.0% | 0.0% | 28.0% | 0.0% | 0.0% |
| `power_law_test` | `fixed_curriculum` | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| `power_law_test` | `random_valid` | 9.5% | 22.6% | 19.5% | 12.4% | 11.4% | 8.1% | 6.9% | 9.7% |
| `power_law_test` | `ppo_no_longterm_state` | 7.6% | 26.5% | 18.3% | 7.1% | 6.1% | 20.7% | 4.4% | 9.4% |
| `long_gap_test` | `delayed_retention_ppo` | 3.9% | 26.4% | 14.8% | 5.5% | 16.4% | 16.9% | 8.2% | 7.9% |
| `long_gap_test` | `threshold_heuristic` | 0.5% | 63.2% | 15.5% | 13.6% | 0.0% | 0.0% | 0.0% | 7.3% |
| `long_gap_test` | `myopic_rl_bandit` | 0.0% | 73.1% | 0.0% | 0.0% | 0.0% | 26.9% | 0.0% | 0.0% |
| `long_gap_test` | `fixed_curriculum` | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| `long_gap_test` | `random_valid` | 9.4% | 22.6% | 19.5% | 12.4% | 11.4% | 8.0% | 7.0% | 9.7% |
| `long_gap_test` | `ppo_no_longterm_state` | 7.6% | 26.0% | 18.0% | 7.0% | 6.1% | 20.8% | 5.7% | 8.9% |

**Part 2 — 5-Check Behavioral Sanity Suite (`delayed_retention_ppo` Across Seeds):**

| Cohort | Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%–25%`) | Floor > Heuristic | Beats All Baselines |
|---|---|---:|---:|---|---|---|---|
| `main_test` | `20261201` | 5 / 5 | Pass (59.1%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261202` | 5 / 5 | Pass (52.2%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261203` | 5 / 5 | Pass (43.8%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261204` | 5 / 5 | Pass (44.7%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261205` | 5 / 5 | Pass (52.7%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261206` | 4 / 5 | Fail (38.7%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261207` | 3 / 5 | Pass (51.4%) | Fail | Fail | Pass | Pass |
| `main_test` | `20261208` | 5 / 5 | Pass (78.7%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261209` | 5 / 5 | Pass (53.6%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261210` | 5 / 5 | Pass (52.8%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261201` | 5 / 5 | Pass (59.2%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261202` | 5 / 5 | Pass (52.3%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261203` | 5 / 5 | Pass (43.8%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261204` | 5 / 5 | Pass (44.7%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261205` | 5 / 5 | Pass (52.8%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261206` | 4 / 5 | Fail (38.8%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261207` | 3 / 5 | Pass (51.5%) | Fail | Fail | Pass | Pass |
| `power_law_test` | `20261208` | 5 / 5 | Pass (78.7%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261209` | 5 / 5 | Pass (53.5%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261210` | 5 / 5 | Pass (52.8%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261201` | 5 / 5 | Pass (60.3%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261202` | 5 / 5 | Pass (52.6%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261203` | 5 / 5 | Pass (44.0%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261204` | 5 / 5 | Pass (45.8%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261205` | 5 / 5 | Pass (53.0%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261206` | 4 / 5 | Fail (39.2%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261207` | 3 / 5 | Pass (52.2%) | Fail | Fail | Pass | Pass |
| `long_gap_test` | `20261208` | 5 / 5 | Pass (78.6%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261209` | 4 / 5 | Pass (53.7%) | Fail | Pass | Pass | Pass |
| `long_gap_test` | `20261210` | 5 / 5 | Pass (53.1%) | Pass | Pass | Pass | Pass |

> **Behavioral Sanity Note (Pre-Registered (10 seeds, N=400)):** `8/10` seeds pass `5/5` checks on `main_test` and `8/10` on `power_law_test` (`7/10` on `long_gap_test`), while **`10/10` seeds pass both outcome checks** (`consolidated_floor_beats_heuristic` and `delayed_retention_beats_all_baselines`) across all 3 cohorts; action-share bound misses occur only in `main_test`/`20261206` (question_turns_at_least_40pct); `main_test`/`20261207` (no_action_spam_over_32pct, curriculum_topic_advancement); `power_law_test`/`20261206` (question_turns_at_least_40pct); `power_law_test`/`20261207` (no_action_spam_over_32pct, curriculum_topic_advancement); `long_gap_test`/`20261206` (question_turns_at_least_40pct); `long_gap_test`/`20261207` (no_action_spam_over_32pct, curriculum_topic_advancement); `long_gap_test`/`20261209` (no_action_spam_over_32pct).

### Tier 4 — Retrospective (3 seeds, N=200) (`3` seeds, `N=200` learners `30000..30199`, Budget `32.0`)

#### Table T4-A (Retrospective (3 seeds, N=200)): Policy Benchmark at Budget `32.0`

| Cohort | Policy | 1-Day Pass | 7-Day Pass | **30-Day Pass** | **Delayed Mean (Obs)** | Paired Diff vs `threshold_heuristic` (95% CI) | Delayed Expected | Floor $b_t$ | Seed Range (Min–Max) |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `main_test` | `delayed_retention_ppo` | 66.05% ± 0.30% | 65.20% ± 0.75% | **61.69% ± 1.49%** | **64.31% ± 0.78%** | +2.93 pp [+2.33, +3.53] | 64.62% ± 0.71% | 0.4655 ± 0.0150 | 63.42%–64.81% |
| `main_test` | `threshold_heuristic` | 66.08% | 63.50% | **54.58%** | **61.39%** | `0.00 pp` (Ref) | 61.69% | 0.3321 | — |
| `main_test` | `myopic_rl_bandit` | 59.02% ± 0.04% | 56.08% ± 0.03% | **49.74% ± 0.03%** | **54.95% ± 0.02%** | -6.44 pp [-7.23, -5.62] | 55.51% ± 0.01% | 0.3609 ± 0.0001 | 54.94%–54.97% |
| `main_test` | `fixed_curriculum` | 64.22% | 60.28% | **51.47%** | **58.66%** | -2.73 pp [-3.61, -1.85] | 58.61% | 0.3228 | — |
| `main_test` | `random_valid` | 65.22% | 62.83% | **55.78%** | **61.28%** | -0.11 pp [-0.84, +0.68] | 61.24% | 0.3856 | — |
| `power_law_test` | `delayed_retention_ppo` | 65.73% ± 0.38% | 64.84% ± 0.61% | **62.00% ± 1.16%** | **64.19% ± 0.66%** | +2.68 pp [+2.10, +3.22] | 64.51% ± 0.60% | 0.4633 ± 0.0142 | 63.43%–64.58% |
| `power_law_test` | `threshold_heuristic` | 65.42% | 62.75% | **56.36%** | **61.51%** | `0.00 pp` (Ref) | 61.73% | 0.3325 | — |
| `power_law_test` | `myopic_rl_bandit` | 58.45% ± 0.02% | 56.12% ± 0.04% | **51.19% ± 0.03%** | **55.25% ± 0.02%** | -6.26 pp [-7.09, -5.42] | 55.76% ± 0.00% | 0.3604 ± 0.0002 | 55.23%–55.27% |
| `power_law_test` | `fixed_curriculum` | 63.31% | 59.11% | **52.83%** | **58.42%** | -3.09 pp [-3.91, -2.27] | 58.47% | 0.3227 | — |
| `power_law_test` | `random_valid` | 64.47% | 62.22% | **57.19%** | **61.30%** | -0.21 pp [-0.91, +0.52] | 61.13% | 0.3848 | — |
| `long_gap_test` | `delayed_retention_ppo` | 63.36% ± 0.39% | 62.80% ± 0.64% | **60.00% ± 1.06%** | **62.05% ± 0.63%** | +3.49 pp [+2.88, +4.08] | 62.59% ± 0.66% | 0.4523 ± 0.0114 | 61.32%–62.44% |
| `long_gap_test` | `threshold_heuristic` | 62.14% | 60.03% | **53.53%** | **58.56%** | `0.00 pp` (Ref) | 59.14% | 0.3337 | — |
| `long_gap_test` | `myopic_rl_bandit` | 55.31% ± 0.06% | 53.38% ± 0.06% | **48.73% ± 0.11%** | **52.48% ± 0.07%** | -6.09 pp [-6.96, -5.24] | 53.30% ± 0.02% | 0.3589 ± 0.0006 | 52.41%–52.55% |
| `long_gap_test` | `fixed_curriculum` | 59.94% | 56.31% | **49.56%** | **55.27%** | -3.30 pp [-4.17, -2.38] | 55.53% | 0.3222 | — |
| `long_gap_test` | `random_valid` | 61.39% | 59.56% | **54.50%** | **58.48%** | -0.08 pp [-0.84, +0.72] | 58.65% | 0.3766 | — |

#### Table T4-B (Retrospective (3 seeds, N=200)): All 6 Day-30 Per-Skill Pass Rates & Lowest-Skill Retention

| Cohort | Policy | `range_boundaries` | `loop_tracing` | `accumulator_init.` | `conditional_eval.` | `list_indexing` | `function_returns` | **Min (Lowest Skill)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `main_test` | `delayed_retention_ppo` | 82.22% ± 0.51% | 76.78% ± 0.98% | 62.72% ± 3.18% | 54.72% ± 5.14% | 52.89% ± 3.85% | 40.83% ± 3.06% | **40.83% ± 3.06%** (`function_returns`) |
| `main_test` | `threshold_heuristic` | 69.17% | 62.67% | 52.67% | 52.50% | 50.17% | 40.33% | **40.33%** (`function_returns`) |
| `main_test` | `myopic_rl_bandit` | 90.06% ± 0.10% | 50.72% ± 0.10% | 35.00% ± 0.00% | 39.33% ± 0.00% | 48.33% ± 0.00% | 35.00% ± 0.00% | **35.00% ± 0.00%** (`accumulator_init.`) |
| `main_test` | `fixed_curriculum` | 59.00% | 53.50% | 46.00% | 51.17% | 51.83% | 47.33% | **46.00%** (`accumulator_init.`) |
| `main_test` | `random_valid` | 75.67% | 65.17% | 52.00% | 51.83% | 50.17% | 39.83% | **39.83%** (`function_returns`) |
| `power_law_test` | `delayed_retention_ppo` | 82.17% ± 0.44% | 76.56% ± 1.17% | 62.78% ± 3.28% | 54.83% ± 4.07% | 54.00% ± 2.96% | 41.67% ± 2.96% | **41.67% ± 2.96%** (`function_returns`) |
| `power_law_test` | `threshold_heuristic` | 71.00% | 65.33% | 54.67% | 53.33% | 51.83% | 42.00% | **42.00%** (`function_returns`) |
| `power_law_test` | `myopic_rl_bandit` | 89.72% ± 0.10% | 52.22% ± 0.10% | 37.33% ± 0.00% | 41.33% ± 0.00% | 50.50% ± 0.00% | 36.00% ± 0.00% | **36.00% ± 0.00%** (`function_returns`) |
| `power_law_test` | `fixed_curriculum` | 60.83% | 55.50% | 48.17% | 52.50% | 51.83% | 48.17% | **48.17%** (`accumulator_init.`) |
| `power_law_test` | `random_valid` | 76.83% | 66.50% | 53.33% | 53.17% | 52.17% | 41.17% | **41.17%** (`function_returns`) |
| `long_gap_test` | `delayed_retention_ppo` | 82.06% ± 0.25% | 75.33% ± 1.17% | 61.17% ± 2.40% | 51.22% ± 3.82% | 51.17% ± 2.95% | 39.06% ± 2.80% | **39.06% ± 2.80%** (`function_returns`) |
| `long_gap_test` | `threshold_heuristic` | 68.17% | 63.50% | 52.50% | 49.00% | 48.67% | 39.33% | **39.33%** (`function_returns`) |
| `long_gap_test` | `myopic_rl_bandit` | 89.11% ± 0.54% | 49.78% ± 0.10% | 33.67% ± 0.00% | 38.50% ± 0.00% | 47.83% ± 0.00% | 33.50% ± 0.00% | **33.50% ± 0.00%** (`function_returns`) |
| `long_gap_test` | `fixed_curriculum` | 56.83% | 51.83% | 44.83% | 48.67% | 49.50% | 45.67% | **44.83%** (`accumulator_init.`) |
| `long_gap_test` | `random_valid` | 75.33% | 64.00% | 50.17% | 49.83% | 49.67% | 38.00% | **38.00%** (`function_returns`) |

#### Table T4-E (Retrospective (3 seeds, N=200)): 8-Action Distribution Shares & 5-Check Behavioral Sanity Suite

**Part 1 — 8-Action Distribution Shares (% of Turns):**

| Cohort | Policy | `give_easier` | `give_same` | `give_harder` | `explain` | `worked_ex` | `self_explain` | `review_pre` | `advance` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `main_test` | `delayed_retention_ppo` | 8.7% | 9.4% | 21.6% | 4.7% | 13.6% | 30.6% | 4.6% | 6.9% |
| `main_test` | `threshold_heuristic` | 0.5% | 58.3% | 17.6% | 13.2% | 0.0% | 0.0% | 0.0% | 10.4% |
| `main_test` | `myopic_rl_bandit` | 0.0% | 69.3% | 0.0% | 0.0% | 0.0% | 30.7% | 0.0% | 0.0% |
| `main_test` | `fixed_curriculum` | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| `main_test` | `random_valid` | 9.7% | 23.5% | 19.2% | 12.3% | 12.1% | 8.2% | 5.8% | 9.2% |
| `power_law_test` | `delayed_retention_ppo` | 8.7% | 9.4% | 21.6% | 4.7% | 13.6% | 30.6% | 4.6% | 6.8% |
| `power_law_test` | `threshold_heuristic` | 0.4% | 58.6% | 17.3% | 13.5% | 0.0% | 0.0% | 0.0% | 10.1% |
| `power_law_test` | `myopic_rl_bandit` | 0.0% | 69.4% | 0.0% | 0.0% | 0.0% | 30.6% | 0.0% | 0.0% |
| `power_law_test` | `fixed_curriculum` | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| `power_law_test` | `random_valid` | 9.7% | 23.6% | 19.2% | 12.3% | 12.1% | 8.1% | 5.8% | 9.2% |
| `long_gap_test` | `delayed_retention_ppo` | 8.8% | 9.7% | 21.2% | 4.8% | 13.9% | 30.3% | 4.8% | 6.5% |
| `long_gap_test` | `threshold_heuristic` | 0.5% | 61.6% | 16.6% | 14.3% | 0.0% | 0.0% | 0.0% | 7.1% |
| `long_gap_test` | `myopic_rl_bandit` | 0.0% | 70.3% | 0.0% | 0.0% | 0.0% | 29.7% | 0.0% | 0.0% |
| `long_gap_test` | `fixed_curriculum` | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| `long_gap_test` | `random_valid` | 9.8% | 23.4% | 19.4% | 12.2% | 12.1% | 8.0% | 6.1% | 9.0% |

**Part 2 — 5-Check Behavioral Sanity Suite (`delayed_retention_ppo` Across Seeds):**

| Cohort | Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%–25%`) | Floor > Heuristic | Beats All Baselines |
|---|---|---:|---:|---|---|---|---|
| `main_test` | `20261120` | 5 / 5 | Pass (43.5%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261121` | 5 / 5 | Pass (46.8%) | Pass | Pass | Pass | Pass |
| `main_test` | `20261122` | 5 / 5 | Pass (42.6%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261120` | 5 / 5 | Pass (43.5%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261121` | 5 / 5 | Pass (46.9%) | Pass | Pass | Pass | Pass |
| `power_law_test` | `20261122` | 5 / 5 | Pass (42.6%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261120` | 5 / 5 | Pass (43.7%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261121` | 5 / 5 | Pass (47.1%) | Pass | Pass | Pass | Pass |
| `long_gap_test` | `20261122` | 5 / 5 | Pass (42.7%) | Pass | Pass | Pass | Pass |

---

## 5. Tier 5: Optional Human Pilot Protocol Outline (Strictly Separated from Simulation)

#### Table T5: Optional Tier 5 Human Pilot Summary

| Cohort | Enrolled ($N$) | Completed ($n$, %, 95% CI) | Unaided Pretest | Delayed Posttest (`d` days) | Paired Change `post - pre` (95% CI) |
|---|---:|---|---:|---:|---|
| Human Pilot (Optional, IRB-gated) | `[NOT RUN — OPTIONAL]` | `[NOT RUN]` | `[NOT RUN]` | `[NOT RUN]` | `[NOT RUN]` |

---

## Next-Step Reminder: Live-LLM Evaluation Tests Pending

> **Action Reminder:** Now that the offline evaluation phase (Tier 1 5-seed, Tier 2 offline diagnostics, Tier 4 10-seed + Ablation 3 + Budget Curves, and Retrospective compilations) is complete, **remember to run the live-LLM evaluation tests next**:
> 1. Set a single pinned `(provider, model)` in `3/adaptive_web/.env` with `LLM_FALLBACK_PROVIDERS=""` (no silent failover).
> 2. Optionally configure a dense bi-encoder (`sentence-transformers`) for Tier 2 `Dense` and `BM25 + Dense`, then complete 2-reviewer blinded relevance scoring on [`tier2_relevance_annotation_pack.jsonl`](eval_runs/v2.2-preregistered/blind_packs/tier2_relevance_annotation_pack.jsonl).
> 3. Execute the live-LLM generation runs (`source == "llm"` verified) on the exported packs for **Ablation 1 Downstream** (`ablation1_downstream_pack.jsonl`), **Tier 3a + Ablation 2 EduGuard ON/OFF** (`tier3a_ablation2_prompts.jsonl`), **Tier 3b Question Synthesis** (`tier3b_qgen_requests.jsonl`), and **Tier 3c Self-Explanation Grading** (`tier3c_self_explanation_prompts.jsonl`), followed by 2-reviewer blinded human adjudication.
