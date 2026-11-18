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
   - An entropy bonus (`ent_coef = 0.015`) prevents the Actor from prematurely collapsing to a single action before exploring all 8 teaching choices.

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
| `5` | `request_self_explanation` | `0.9` | Active unaided micro-assessment; strongly consolidates transient skill into **permastore floor $b_t$** & boosts HLR half-life $h_t$ |
| `6` | `review_prerequisite` | `1.0` | Jumps to weakest prerequisite (or decayed earlier skill) with a concept-primed refresher task |
| `7` | `advance_next_topic` | `0.2` | Advances to next unvisited (or lowest-retention) Python skill once at least 1 task was attempted |

- **Gated Option B Rule:** After a passive explanation (`3` or `4`), `awaiting_question = True` masks out consecutive lectures and forces PPO to choose the follow-up question difficulty (`0`, `1`, or `2`).
- **Session Spacing:** Every `8.0` units of budget spent (out of `TOTAL_BUDGET = 32.0`), the clock advances (`1.5` days in `main`/`power_law`, `4.0` days in `long_gap`) so real forgetting occurs between sessions.
- **Reward:** Step-level progress on `P_30d` and `Floor b_t` plus a terminal reward of `10.0 * delayed_mean` evaluated across **54 independent cloned assessment tasks** (6 skills × 3 delays `(1d, 7d, 30d)` × 3 difficulties).


---

## 4. How We Are Doing — Complete 5-Policy Benchmark Results

Both RL policies (`myopic_rl_bandit` and `delayed_retention_ppo`) were trained across **3 random seeds (`20261120`, `20261121`, `20261122`)** (`90` updates × `16` episodes per update) and evaluated alongside `random_valid`, `fixed_curriculum`, and `threshold_heuristic` on **200 disjoint held-out test learners** across all 3 cohorts ([`runs/v2.2-ppo-main/REPORT.md`](runs/v2.2-ppo-main/REPORT.md) · [`run.json`](runs/v2.2-ppo-main/run.json)).

### 4.1 Held-Out Independent Retention Performance (Higher Is Better)

#### A. `main_test` (200 Held-Out Learners, 1.5-Day Inter-Session Spacing)

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| `random_valid` | 64.33% | 61.81% | 54.72% | 60.29% | 60.55% | 0.3820 |
| `fixed_curriculum` | 64.22% | 60.28% | 51.47% | 58.66% | 58.61% | 0.3228 |
| `threshold_heuristic` | 66.08% | 63.50% | 54.58% | 61.39% | 61.69% | 0.3321 |
| `myopic_rl_bandit` | 59.25% ± 0.05% | 56.40% ± 0.11% | 50.21% ± 0.23% | 55.29% ± 0.13% | 55.81% ± 0.09% | 0.3699 ± 0.0026 |
| **`delayed_retention_ppo`** | **66.36% ± 1.83%** | **64.43% ± 1.72%** | **59.44% ± 1.96%** | **63.41% ± 1.83%** | **63.53% ± 1.74%** | **0.4388 ± 0.0167** |

#### B. `power_law_test` (200 Held-Out Learners, Power-Law Forgetting Stress Cohort)

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| `random_valid` | 63.64% | 61.47% | 55.75% | 60.29% | 60.49% | 0.3810 |
| `fixed_curriculum` | 63.31% | 59.11% | 52.83% | 58.42% | 58.47% | 0.3227 |
| `threshold_heuristic` | 65.42% | 62.75% | 56.36% | 61.51% | 61.73% | 0.3325 |
| `myopic_rl_bandit` | 58.69% ± 0.06% | 56.44% ± 0.12% | 51.70% ± 0.22% | 55.61% ± 0.13% | 56.07% ± 0.10% | 0.3697 ± 0.0028 |
| **`delayed_retention_ppo`** | **65.93% ± 1.81%** | **64.16% ± 1.73%** | **60.12% ± 1.82%** | **63.40% ± 1.78%** | **63.45% ± 1.69%** | **0.4373 ± 0.0171** |

#### C. `long_gap_test` (200 Held-Out Learners, 4.0-Day Inter-Session Spacing)

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| `random_valid` | 60.64% | 59.14% | 53.86% | 57.88% | 58.33% | 0.3778 |
| `fixed_curriculum` | 59.94% | 56.31% | 49.56% | 55.27% | 55.53% | 0.3222 |
| `threshold_heuristic` | 62.14% | 60.03% | 53.53% | 58.56% | 59.14% | 0.3337 |
| `myopic_rl_bandit` | 55.56% ± 0.06% | 53.69% ± 0.17% | 49.31% ± 0.25% | 52.86% ± 0.16% | 53.62% ± 0.14% | 0.3685 ± 0.0037 |
| **`delayed_retention_ppo`** | **63.04% ± 2.33%** | **61.72% ± 2.32%** | **57.88% ± 2.49%** | **60.88% ± 2.38%** | **61.11% ± 2.39%** | **0.4278 ± 0.0173** |

---

### 4.2 Learner-Level Paired Bootstrap 95% Confidence Intervals

Across all 3 cohorts and all 4 baselines, **`delayed_retention_ppo`'s gains are statistically significant (every 95% confidence interval lies strictly above `0%`)**:

| Contrast (`delayed_retention_ppo` minus Baseline) | `main_test` Observed (95% CI) | `power_law_test` Observed (95% CI) | `long_gap_test` Observed (95% CI) |
|---|---|---|---|
| vs. `threshold_heuristic` | **+2.02%** `[+1.44%, +2.66%]` | **+1.89%** `[+1.31%, +2.47%]` | **+2.31%** `[+1.66%, +2.93%]` |
| vs. `random_valid` | **+3.12%** `[+2.35%, +3.83%]` | **+3.11%** `[+2.36%, +3.83%]` | **+3.00%** `[+2.21%, +3.73%]` |
| vs. `fixed_curriculum` | **+4.75%** `[+3.84%, +5.68%]` | **+4.98%** `[+4.12%, +5.87%]` | **+5.61%** `[+4.73%, +6.56%]` |
| vs. `myopic_rl_bandit` | **+8.12%** `[+7.36%, +8.91%]` | **+7.79%** `[+7.06%, +8.54%]` | **+8.02%** `[+7.26%, +8.83%]` |

---

### 4.3 Why PPO Wins: Action Strategy & The Illusion of Competence

1. **Overcoming the Day-30 Cramming Drop:** On `main_test`, `threshold_heuristic` nearly ties PPO at **Day 1** (`66.08%` vs. `66.36%`) because short-term cramming passes a 1-day check. By **Day 30**, `threshold_heuristic` collapses by **`-11.50` percentage points** (`66.08% -> 54.58%`) because its durable consolidated floor is only `0.3321`. `delayed_retention_ppo` builds a **`+32%` higher consolidated floor (`0.4388`)** and stays at **`59.44%` at Day 30** (**`+4.86%` ahead of `threshold_heuristic` and `+7.97%` ahead of `fixed_curriculum`**).
2. **All 8 Actions Used Intelligently (`main_test` Action Shares):**
   - **`delayed_retention_ppo`:** `47.8%` `request_self_explanation`, `20.6%` `review_prerequisite`, `11.4%` `give_harder_question`, `6.1%` `advance_next_topic`, `5.1%` `give_same_difficulty_question`, `5.1%` `give_easier_question`, `2.4%` `explain_concept`, `1.4%` `show_worked_example`.
   - **`myopic_rl_bandit`:** `0.0%` `advance_next_topic` and `0.0%` `review_prerequisite` — proving that immediate-reward RL gets trapped farming easy local rewards on a single skill while ignoring long-term curriculum retention.
3. **Zero-Retraining Stress Robustness:** Trained only on `main` exponential dynamics, `delayed_retention_ppo` transfers cleanly to both `power_law_test` (`63.40%` vs. `61.51%`) and `long_gap_test` (`60.88%` vs. `58.56%`).
