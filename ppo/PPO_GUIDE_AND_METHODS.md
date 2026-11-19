# PPO Adaptive Teaching Policy: Complete Guide, Methods, and Results

This guide explains **what a PPO policy is**, **why we need it on top of our frozen `monotonic_gru` memory model**, **how our 8-action closed-loop PPO training environment works**, and **how our trained policy performs** across held-out learners and retention stress cohorts.

---

## 1. What Is a PPO Policy?

**PPO (Proximal Policy Optimization)** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) is a **deep reinforcement learning** algorithm that learns a sequential decision-making policy $\pi_\theta(a_t \mid s_t)$ through trial-and-error interaction with an environment.

Think of the system as having **two neural networks** inside `MaskedActorCritic` ([`agent.py`](agent.py)):

1. **The Actor (Policy Network $\pi_\theta$):** Looks at the current student observation state $s_t \in \mathbb{R}^{36}$ and outputs probabilities over all valid teaching actions $a_t \in \{0, \dots, 7\}$ (for example, *give harder question*, *explain concept*, *request self-explanation*, or *advance to next topic*).
2. **The Critic (Value Network $V_\phi$):** Looks at the same state $s_t$ and predicts the expected cumulative future reward (how well the student will score on 1-, 7-, and 30-day delayed tests if the tutor follows the current policy from state $s_t$ onward).

### How PPO Updates Without Crashing

In reinforcement learning, making too huge a policy jump from one noisy batch of student episodes can destroy everything the agent previously learned. PPO solves this in four steps:

1. **Advantage Estimation (`compute_gae`):** Using Generalized Advantage Estimation ($\gamma = 0.99$, $\lambda = 0.95$), PPO asks: *"Did taking action $a_t$ in state $s_t$ turn out better or worse than what the Critic $V_\phi(s_t)$ expected?"*
   $$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).$$
   - If $\hat{A}_t > 0$, action $a_t$ led to better-than-expected delayed learning $\rightarrow$ **increase** its probability.
   - If $\hat{A}_t < 0$, action $a_t$ led to worse-than-expected retention $\rightarrow$ **decrease** its probability.

2. **Clipped Surrogate Objective (`ppo_update`):** Let $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ be the probability ratio between the updated policy and the rollout policy. PPO maximizes:
   $$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\Big(r_t(\theta)\hat{A}_t,\;\operatorname{clip}\big(r_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon\big)\hat{A}_t\Big)\right], \qquad \epsilon = 0.20.$$
   Clipping keeps $r_t(\theta)$ inside $[0.8, 1.2]$, guaranteeing **smooth, stable policy improvement** across updates.

3. **Value Loss & Entropy Bonus:**
   - The Critic minimizes MSE against empirical returns $\hat{R}_t = \hat{A}_t + V_\phi(s_t)$ (`vf_coef = 0.5`).
   - An entropy bonus (linearly annealed from `0.025` down to `0.005` in `train_ppo.py`) encourages early exploration across all 8 teaching choices and sharpens the converged policy.

4. **Exact Invalid-Action Masking:** Before taking `softmax` / `Categorical`, invalid actions (e.g., `give_harder_question` when already at `Hard`, or consecutive passive explanations without testing) are masked to `-1e9`, giving them strictly **`0.0` probability**.

---

## 2. Why Do We Need PPO When We Already Have `monotonic_gru`?

A common question is: *"Our frozen `monotonic_gru` already predicts student success accurately — why can't we just write a simple `if/else` rule around its predictions?"*

The answer comes down to the difference between a **Diagnostic Observer** and a **Pedagogical Controller**:

- **`monotonic_gru` is the Observer (Diagnosis):** Given past attempts, hints, and elapsed days, it tracks the student's 32-dim cognitive state and answers *"What happens if I ask question X right now or 30 days from now?"* It does **not** decide which teaching action produces the highest long-term learning across a 32-unit study budget.
- **The PPO Policy is the AI Teacher (Decision-Maker):** It decides *which skill, what difficulty, and which intervention* maximizes **1-, 7-, and 30-day independent retention** under a fixed teaching budget.

Simple `if/else` rules and greedy immediate-reward RL fail on four phenomena built into realistic tutoring:

| Challenge | Why Simple `if/else` Rules & Myopic RL Fail | How Delayed-Reward PPO Solves It |
|---|---|---|
| **1. The Illusion of Competence** | A student who crams Easy items or passes using hints has high short-term `P(Med)` today (`~70%`), fooling `ThresholdHeuristicPolicy` into calling `advance_next_topic` immediately. | PPO reads **both** `P(Med)` and `Floor b_t` / `Fragility = P_0d - P_30d`, recognizing fragile transient fluency and triggering **`request_self_explanation`** to lock knowledge into durable permastore memory before advancing. |
| **2. LLM Hint Leakage ($l_t \in [0, 1]$)** | When an LLM hint leaks most of the answer ($l_t \approx 0.8$), the student passes the retry! Greedy rules treat that `100%` score as mastery. | PPO observes continuous `leakage l_t` and `unverified_assisted_flag`, choosing **`give_same_difficulty_question`** (unaided verification) to test real independent recall before moving on. |
| **3. Prerequisite Bottlenecks** | Repeatedly drilling `function_returns` when `loop_tracing` or `accumulator_initialization` is weak causes repeated failure and wasted budget. | PPO observes prerequisite `P(Med)` and `Floor b_t` in $s_t$ and chooses **`review_prerequisite`** to unblock downstream learning. |
| **4. Cross-Session Forgetting** | As the student advances across multi-day sessions, earlier topics decay unless well-consolidated or spaced. | PPO optimizes **1-, 7-, and 30-day delayed retention** across all 6 Python skills simultaneously (`gamma = 0.99`), balancing depth of consolidation against curriculum breadth. |

---

## 3. How Our 8-Action Closed-Loop PPO Environment Works

Implemented in [`env.py`](env.py) (`AdaptiveTutorEnv`) matching our 3-stage architecture:

### Stage 1: Within-Question Turn (`Box 1`)
- The student attempts the presented question on `(skill s, difficulty d)` using `v2.2` rubric mechanics across `5` formats (`single_choice`, `multiple_select`, `output_prediction`, `fill_in_the_blank`, `code_completion`).
- Up to `3` attempts (`1` initial + `2` retries) and `0..3` LLM hints.
- The Evaluator grades `(score, full_success)` and measures **LLM Hint Leakage** $l_t \in [0, 1]$ (high $l_t$ helps the immediate retry pass but discounts retained learning by `1 - 0.75 * l_t` and sets `unverified_assisted_flag = True`).

### Stage 2: Frozen Memory & 36-Dim PPO Observation State (`Box 2`)
- Hints, instruction events, and paired response/grade attempts flow into the **frozen `v2.2` `monotonic_gru`** ([`seed_20261122/model.pt`](../model/runs/v2.2-mps-main/monotonic_gru/seed_20261122/model.pt)).
- Querying `TaskReadoutHead` at $\Delta t = 0\text{d}, 7\text{d}, 30\text{d}$ extracts a **36-dimensional pure observable state $s_t \in [0, 1]^{36}$** (zero hidden simulator variables):
  - Current topic: `P(Easy)`, `P(Med)`, `P(Hard)`, `P_7d(Med)`, `P_30d(Med)`, **`Fragility = P_Med(0d) - P_Med(30d)`**, **`Floor b_t`**, `active_k`.
  - Turn evidence: `unverified_assisted_flag`, `last_score`, `last_hints / 3`, `last_leakage`, `attempts_on_topic`.
  - Gated Option B context: `awaiting_question`, `primed_concept`, `primed_example`, difficulty one-hot (`Easy/Med/Hard`).
  - Curriculum & budget: prerequisite `min P(Med)` and `min Floor b_t`, remaining budget fraction, min/mean `P_30d` across all 6 skills, per-skill `P_30d` vector, and active topic one-hot.

### Stage 3: 8-Action Gated Option B Policy Menu (`Box 3`)

| ID | Action | Cost | Role & Pedagogical Effect |
|---:|---|---:|---|
| `0` | `give_easier_question` | `1.0` | Steps down `Hard -> Med -> Easy`; builds foundational competence after failure |
| `1` | `give_same_difficulty_question` | `1.0` | Drills at current difficulty; if `unverified_assisted_flag=True`, administers an **unaided verification check** |
| `2` | `give_harder_question` | `1.0` | Steps up `Easy -> Med -> Hard`; delivers higher learning dose and stronger consolidation on pass |
| `3` | `explain_concept` | `0.6` | Strongest boost for beginners (`(1-k)^2`); sets `primed_mode="concept"` (`1.45×` dose on next question) |
| `4` | `show_worked_example` | `0.7` | Procedural scaffolding (`1-k`); sets `primed_mode="example"` (`+0.11` readiness on next question) |
| `5` | `request_self_explanation` | `0.9` | Post-question active recall prompt (unlocks after a question turn, locks after one use); consolidates transient skill into **permastore floor $b_t$** & boosts HLR half-life $h_t$ |
| `6` | `review_prerequisite` | `1.0` | Unlocks only when a prerequisite or visited earlier skill is genuinely weak/decayed; jumps there and administers a refresher question |
| `7` | `advance_next_topic` | `0.2` | Advances to next unvisited (or lowest-retention) Python skill once at least 1 task was attempted |

- **Gated Option B & Anti-Spam Rules:** After a passive explanation (`3` or `4`), `awaiting_question = True` masks out consecutive lectures and forces PPO to choose the follow-up question difficulty (`0`, `1`, or `2`). Additionally, `request_self_explanation` (`5`) requires a preceding question turn and locks until the next question turn, and `review_prerequisite` (`6`) only unmasks when a prerequisite or visited skill falls below target retention/floor thresholds.
- **Session Spacing:** Every `8.0` units of budget spent (out of `TOTAL_BUDGET = 32.0`), the clock advances (`1.5` days in `main`/`power_law`, `4.0` days in `long_gap`) so real forgetting occurs between sessions.
- **Reward:** Step-level progress on `P_30d`, `Floor b_t`, and worst-skill `min P_30d` (anti-neglect floor shaping) plus a terminal reward of `10.0 * delayed_mean` evaluated across **54 independent cloned assessment tasks** (6 skills × 3 delays `(1d, 7d, 30d)` × 3 difficulties).


---

## 4. How We Are Doing — Complete 5-Policy Benchmark Results

Both RL policies (`myopic_rl_bandit` and `delayed_retention_ppo`) were trained across **3 random seeds (`20261120`, `20261121`, `20261122`)** (`100` updates × `24` episodes per update, linear LR annealing and entropy decay from `0.025 -> 0.005`, post-question gating on `request_self_explanation`, targeted prerequisite unmasking without free priming on `review_prerequisite`, and min-skill floor shaping) and evaluated alongside `random_valid`, `fixed_curriculum`, and `threshold_heuristic` on **200 disjoint held-out test learners** across all 3 cohorts ([`runs/v2.2-ppo-main/REPORT.md`](runs/v2.2-ppo-main/REPORT.md) · [`run.json`](runs/v2.2-ppo-main/run.json)).

### 4.1 Held-Out Independent Retention Performance (Higher Is Better)

#### A. `main_test` (200 Held-Out Learners, 1.5-Day Inter-Session Spacing)

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| `random_valid` | 65.22% | 62.83% | 55.78% | 61.28% | 61.24% | 0.3856 |
| `fixed_curriculum` | 64.22% | 60.28% | 51.47% | 58.66% | 58.61% | 0.3228 |
| `threshold_heuristic` | 66.08% | 63.50% | 54.58% | 61.39% | 61.69% | 0.3321 |
| `myopic_rl_bandit` | 59.02% ± 0.04% | 56.08% ± 0.03% | 49.74% ± 0.03% | 54.95% ± 0.02% | 55.51% ± 0.01% | 0.3609 ± 0.0001 |
| **`delayed_retention_ppo`** | **66.05% ± 0.30%** | **65.20% ± 0.75%** | **61.69% ± 1.49%** | **64.31% ± 0.78%** | **64.62% ± 0.71%** | **0.4655 ± 0.0150** |

#### B. `power_law_test` (200 Held-Out Learners, Power-Law Forgetting Stress Cohort)

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| `random_valid` | 64.47% | 62.22% | 57.19% | 61.30% | 61.13% | 0.3848 |
| `fixed_curriculum` | 63.31% | 59.11% | 52.83% | 58.42% | 58.47% | 0.3227 |
| `threshold_heuristic` | 65.42% | 62.75% | 56.36% | 61.51% | 61.73% | 0.3325 |
| `myopic_rl_bandit` | 58.45% ± 0.02% | 56.12% ± 0.04% | 51.19% ± 0.03% | 55.25% ± 0.02% | 55.76% ± 0.00% | 0.3604 ± 0.0002 |
| **`delayed_retention_ppo`** | **65.73% ± 0.38%** | **64.84% ± 0.61%** | **62.00% ± 1.16%** | **64.19% ± 0.66%** | **64.51% ± 0.60%** | **0.4633 ± 0.0142** |

#### C. `long_gap_test` (200 Held-Out Learners, 4.0-Day Inter-Session Spacing)

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| `random_valid` | 61.39% | 59.56% | 54.50% | 58.48% | 58.65% | 0.3766 |
| `fixed_curriculum` | 59.94% | 56.31% | 49.56% | 55.27% | 55.53% | 0.3222 |
| `threshold_heuristic` | 62.14% | 60.03% | 53.53% | 58.56% | 59.14% | 0.3337 |
| `myopic_rl_bandit` | 55.31% ± 0.06% | 53.38% ± 0.06% | 48.73% ± 0.11% | 52.48% ± 0.07% | 53.30% ± 0.02% | 0.3589 ± 0.0006 |
| **`delayed_retention_ppo`** | **63.36% ± 0.39%** | **62.80% ± 0.64%** | **60.00% ± 1.06%** | **62.05% ± 0.63%** | **62.59% ± 0.66%** | **0.4523 ± 0.0114** |

---

### 4.2 Learner-Level Paired Bootstrap 95% Confidence Intervals

Across all 3 cohorts and all 4 baselines, **`delayed_retention_ppo`'s gains are statistically significant (every 95% confidence interval lies strictly above `0%`)**:

| Contrast (`delayed_retention_ppo` minus Baseline) | `main_test` Observed (95% CI) | `power_law_test` Observed (95% CI) | `long_gap_test` Observed (95% CI) |
|---|---|---|---|
| vs. `threshold_heuristic` | **+2.93%** `[+2.31%, +3.51%]` | **+2.68%** `[+2.09%, +3.24%]` | **+3.49%** `[+2.89%, +4.09%]` |
| vs. `random_valid` | **+3.04%** `[+2.37%, +3.73%]` | **+2.90%** `[+2.26%, +3.53%]` | **+3.57%** `[+2.85%, +4.23%]` |
| vs. `fixed_curriculum` | **+5.66%** `[+4.75%, +6.59%]` | **+5.77%** `[+4.89%, +6.63%]` | **+6.78%** `[+5.85%, +7.64%]` |
| vs. `myopic_rl_bandit` | **+9.37%** `[+8.63%, +10.13%]` | **+8.94%** `[+8.17%, +9.68%]` | **+9.58%** `[+8.76%, +10.38%]` |

---

### 4.3 Per-Seed Stability & 5-Check Behavioral Sanity Suite (`main_test`)

Every trained PPO seed (`20261120`, `20261121`, `20261122`) independently passes **`5 / 5`** behavioral sanity checks across **all 3 test cohorts** (`main_test`, `power_law_test`, `long_gap_test`):

| Seed | Delayed Obs (`main_test`) | Delayed Exp (`main_test`) | Floor $b_t$ | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%–25%`) | Floor > Heuristic | Beats All Baselines | Suite |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| `20261120` | 64.81% | 64.98% | 0.4722 | Pass (43.5%) | Pass (`se=30.9%`, `prereq=4.4%`, `explain+worked=18.0%`) | Pass (7.6%) | Pass | Pass | **5 / 5** |
| `20261121` | 64.72% | 65.08% | 0.4760 | Pass (46.8%) | Pass (`se=31.4%`, `prereq=7.8%`, `explain+worked=14.0%`) | Pass (7.9%) | Pass | Pass | **5 / 5** |
| `20261122` | 63.42% | 63.80% | 0.4484 | Pass (42.6%) | Pass (`se=29.6%`, `prereq=1.6%`, `explain+worked=22.7%`) | Pass (5.1%) | Pass | Pass | **5 / 5** |

---

### 4.4 Why PPO Wins: Action Strategy & The Illusion of Competence

1. **Overcoming the Day-30 Cramming Drop:** On `main_test`, `threshold_heuristic` ties PPO at **Day 1** (`66.08%` vs. `66.05%`) because short-term cramming passes a 1-day check. By **Day 30**, `threshold_heuristic` drops by **`-11.50` percentage points** (`66.08% -> 54.58%`) because its durable consolidated floor is only `0.3321`. `delayed_retention_ppo` loses only **`-4.36` percentage points** (`66.05% -> 61.69%`), builds a **`+40%` higher consolidated floor (`0.4655`)**, and beats `threshold_heuristic` by **`+7.11` percentage points at Day 30** (and `fixed_curriculum` by **`+10.22` percentage points**).
2. **All 8 Actions Used Intelligently (`main_test` Mean Action Shares):**
   - **`delayed_retention_ppo`:** `30.6%` `request_self_explanation`, `21.6%` `give_harder_question`, `13.6%` `show_worked_example`, `9.4%` `give_same_difficulty_question`, `8.7%` `give_easier_question`, `6.9%` `advance_next_topic`, `4.7%` `explain_concept`, `4.6%` `review_prerequisite` (`39.7%` direct questions, `44.3%` total question turns including `review_prerequisite`, `18.3%` conceptual scaffolding).
   - **`myopic_rl_bandit`:** `69.3%` `give_same_difficulty_question`, `30.7%` `request_self_explanation`, `0.0%` `advance_next_topic`, and `0.0%` `review_prerequisite` — proving that immediate-reward RL gets trapped farming easy local rewards on a single skill while ignoring long-term curriculum retention.
3. **Zero-Retraining Stress Robustness:** Trained only on `main` exponential dynamics, `delayed_retention_ppo` transfers cleanly to both `power_law_test` (`64.19%` vs. `61.51%`, Day 30 `62.00%` vs. `56.36%`) and `long_gap_test` (`62.05%` vs. `58.56%`, Day 30 `60.00%` vs. `53.53%`).
