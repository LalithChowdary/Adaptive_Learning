# Dataset `v2.2` and the Monotonic Grey-Box GRU (`monotonic_gru`): Complete Technical Specification

## 1. Executive Summary

During interactive evaluation of the local Python tutoring dashboard (`3/dashboard/`), three behavioral anomalies were identified in the earlier `v2.1` time-decayed GRU (`time_decay_gru`) and single-component exponential decay models:

1. **Fast wrong answers increased predicted success (`v2.1` anomaly):** On *Conditionals*, submitting an unaided `0%` wrong answer in `6.426s` raised the predicted probability from `42.96%` to `51.33%` (`+8.37%`). Controlled replay isolated two additive causes in `v2.1`:
   - `+4.74%` from `v2.1`'s simulator granting an unconditional practice learning dose on attempt 1 even when `score = 0` and `hints = 0`.
   - `+3.63%` from `v2.1`'s response-time generator scaling duration strictly with $(1.25 - 0.5k)$, teaching the unconstrained GRU that fast responses always imply high latent skill $k$.
2. **Upward-drifting forgetting curves after failures (`v2.1` anomaly):** Because `time_decay_gru` decayed its hidden state toward the shared cold-start prior $\mathbf{h}_0$ ($\approx 43\%$), any skill whose prediction dropped below `43%` after wrong answers saw its 0–60 day retention curve **rise** over time during a break.
3. **Overly steep 30–60 day forgetting cliff under single-component exponential decay:** When decay was changed to a naive single-component exponential $k(\Delta t) = k \cdot e^{-\lambda \Delta t} \to 0$ fed into a $\operatorname{logit}(k(\Delta t))$ readout, $\operatorname{logit}(k(\Delta t)) \to -\infty$ linearly in $\Delta t$. After solving two **Hard** questions `100%` unaided in *Function returns* (`72.1%` at Day 0), the predicted retention curve plunged to `52.2%` (7d), `20.3%` (30d), and `6.6%` (60d) — far below even an unassessed beginner (`44.2%`).

To solve all three issues rigorously without ad-hoc heuristics, we grounded both **Dataset `v2.2`** (`3/data_Set/v2.2/`) and the **Monotonic Grey-Box GRU (`monotonic_gru`)** (`3/model/memory/recurrent.py`) in verified equations from the cognitive-science, spaced-repetition, and psychometric literature.

---

## 2. Verified Literature Formulas Used

| Mechanism | Primary Research Reference | Verified Mathematical Formula | Role in `v2.2` Simulator & `monotonic_gru` |
|---|---|---|---|
| **Trainable Memory Half-Life Growth (HLR)** | **Settles & Meeder (ACL 2016)**, *A Trainable Spaced Repetition Model for Language Learning* (Duolingo), Section 3.3, Eq. 3–4 | $$h_t = h_0 \cdot 2^{\,\theta_+(\sqrt{1 + n_+} - 1) \;-\; \theta_-(\sqrt{1 + n_-} - 1) \;+\; \phi(\mathbf{c}_t)}$$ | Memory half-life $h_t$ (in days) grows with cumulative successful retrievals $n_+$ and contracts with failures $n_-$. |
| **Power-Law Retention Parameterized by Exact Half-Life** | **Wickelgren (1974)**, *Single-trace fragility theory of memory dynamics*; **Rubin & Wenzel (1996)**; **Pavlik & Anderson (2005, ACT-R)**; **Ye et al. (KDD 2022, FSRS)** | $$R(\Delta t, h_t) = \left(1 + \left(2^{1/w} - 1\right)\frac{\Delta t}{h_t}\right)^{-w}, \quad w > 0.5$$ | Satisfies $R(0, h_t) = 1$ and $R(h_t, h_t) = 0.5$ (**$h_t$ is the exact half-life in days**), while exhibiting a realistic power-law tail over 30–60 days instead of an exponential cliff to zero. |
| **Two-Component Consolidated ("Permastore") Baseline** | **Bahrick (1984)**, *Semantic memory content in permastore*; **Mozer & Lindsey (2016, DASH)** | $$\mathbf{k}(t + \Delta t) = \mathbf{b}(t + \Delta t) + (\mathbf{k}_t - \tilde{\mathbf{b}}_t) \odot R(\Delta t, \mathbf{h}_t)$$ $$\mathbf{b}(t + \Delta t) = \tilde{\mathbf{b}}_t \odot \big((1 - \boldsymbol{\rho}) + \boldsymbol{\rho} \odot R(\Delta t, 4\mathbf{h}_t)\big)$$ | Separates durable consolidated conceptual knowledge $\tilde{\mathbf{b}}_t = \min(\mathbf{k}_t, \mathbf{b}_t)$ from transient practice fluency $\Delta\mathbf{k}_t = \mathbf{k}_t - \tilde{\mathbf{b}}_t \ge \mathbf{0}$. Guarantees strictly non-increasing retention while preventing collapse below consolidated competence. |
| **Outcome-Signed Evidence & Assistance Weighting** | **Corbett & Anderson (1995)**, *Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge* | $$\mathbf{k}_{\text{drop}} = \mathbf{k} \odot (\mathbf{1} - u_- \mathbf{g}_-)$$ $$\mathbf{k}_{\text{new}} = \mathbf{k}_{\text{drop}} + (\mathbf{1} - \mathbf{k}_{\text{drop}}) \odot (u_+ \mathbf{g}_+)$$ | Separates negative assessment evidence ($u_- > 0$ on non-full-success) from positive learning transitions ($u_+ > 0$ when $\text{score} > 0$, weighted by unaided vs. assisted status). |
| **3-Parameter Logistic (3PL) Item Response Readout** | **Birnbaum (1968)**, *Some Latent Trait Models and Their Use in Inferring an Examinee's Ability* | $$P(\text{pass}) = g_{\text{fmt}} + (1 - g_{\text{fmt}})\,\sigma\!\Big(\alpha\big(\theta_{\text{skill}} - d(\mathbf{x}_{\text{task}})\big)\Big)$$ | Guarantees positive monotonicity in skill $\theta_{\text{skill}}$, strict difficulty ordering (`Easy > Medium > Hard`), and format-appropriate guessing floors $g_{\text{fmt}}$. |

---

## 3. Dataset `v2.2` Specification (`3/data_Set/v2.2/`)

All earlier dataset versions (`v1`, `v2.0`, `v2.1` — **226 files**) are preserved intact and verified by SHA-256 in [`3/data_Set/v2.2/reports/RELEASE_MANIFEST.json`](../data_Set/v2.2/reports/RELEASE_MANIFEST.json). Version `2.2.0` updates [`simulator.py`](../data_Set/v2.2/simulator.py) in three places:

### 3.1 Outcome- and Assistance-Sensitive Learning Dose
In `Student.attempt` ([`simulator.py`](../data_Set/v2.2/simulator.py#L149-L163)):
$$\text{factor} = \begin{cases} \text{score} \cdot \mathbb{I}(\text{full\_success}) + 0.5 \cdot \text{score} \cdot \mathbb{I}(\neg\text{full\_success}), & \text{if } \text{hints} = 0, \\ f_{\text{retained}} \cdot (0.20 + 0.80 \cdot \text{score}), & \text{if } \text{hints} > 0, \end{cases}$$
$$\text{dose} = \frac{d_{\text{format}} \cdot \text{factor}}{\text{attempt\_number}}, \qquad k_{\text{after}} = k_{\text{before}} + (1 - k_{\text{before}})\left(1 - e^{-\eta \cdot \text{dose}}\right).$$
- An unaided zero-score failure ($\text{hints} = 0, \text{score} = 0$) yields $\text{factor} = 0$, $\text{dose} = 0$, and $k_{\text{after}} = k_{\text{before}}$.

### 3.2 Rapid-Guess Timing Mixture on Non-Successful Attempts
In `Student.attempt` ([`simulator.py`](../data_Set/v2.2/simulator.py#L137-L142)):
- On failed or partial attempts ($\neg\text{full\_success}$), with probability $0.35$ the response duration is drawn from a rapid-guess regime $\text{Uniform}(0.05, 0.25) \times t_{\text{ref}}$ (clipped to $[3, 900]$ seconds); with probability $0.65$ it is drawn from the engaged duration log-normal model.

### 3.3 Duolingo HLR Half-Life Growth & Consolidated Baseline Retention
In `retained`, `Student.half_life`, and `Student.strength` ([`simulator.py`](../data_Set/v2.2/simulator.py#L26-L97)):
- Each learner-skill initializes a durable consolidated baseline $b_0 = 0.65 k_0$ and HLR counters $n_+ = 0, n_- = 0$.
- On each practice attempt, $b_t = \min\!\big(k_{\text{after}},\, \min(k_{\text{before}}, b_{t-1}) + \gamma (k_{\text{after}} - k_{\text{before}})\big)$ with $\gamma = 0.45$ on unaided full success and $0.25$ otherwise; $n_+$ increments on unaided full success and $n_-$ increments on non-full-success.
- Effective half-life follows Settles & Meeder (2016):
  $$h_t = \text{clamp}\!\left(h_0 \cdot 2^{0.35(\sqrt{1 + n_+} - 1) - 0.15(\sqrt{1 + n_-} - 1)},\; 2.0,\; 240.0\right).$$
- Retained strength after $\Delta t$ days decays toward the consolidated baseline $b_t$:
  $$k(t + \Delta t) = b_t + (k_t - b_t) \cdot R(\Delta t, h_t),$$
  with $R(\Delta t, h_t) = 2^{-\Delta t / h_t}$ for `exponential` cohorts and $R(\Delta t, h_t) = (1 + \Delta t / h_t)^{-1}$ for the `power_law` cohort.

### 3.4 Verified `v2.2` Dataset Summary
- **Automated Tests:** **57 / 57 passed** ([`run_tests.py`](../data_Set/v2.2/run_tests.py)).
- **Cohort Audits:** **28 / 28 categories passed** on each of `pilot` (200 learners), `main` (2,000 learners: 1,600 train / 200 val / 200 test; 192,779 eligible practice + 26,505 eligible delayed training targets = 219,284 total fitting targets), `power_law` (200 test learners), and `long_gap` (200 test learners).

---

## 4. Model Architecture: Monotonic Grey-Box GRU (`monotonic_gru`)

Implemented in [`3/model/memory/recurrent.py`](memory/recurrent.py):

### 4.1 State Decomposition ($D = 32$, $M = 6$ skill channels)
$$\mathbf{s}_t = \big[\,\underbrace{\mathbf{k}_t}_{6\text{ dims}} \in (0,1)^6,\;\; \underbrace{\mathbf{b}_t}_{6\text{ dims}} \in (0,1)^6,\;\; \underbrace{n_{+,t}}_{1\text{ dim}},\;\; \underbrace{n_{-,t}}_{1\text{ dim}},\;\; \underbrace{\mathbf{c}_t}_{18\text{ dims}}\,\big]$$
- Cold-start initialization: $\mathbf{k}_0 = \mathbf{b}_0 = \sigma(\mathbf{p}_{\text{skill}})$, $n_{+,0} = n_{-,0} = 0$, $\mathbf{c}_0 = \mathbf{0}$.

### 4.2 Time Decay Step (`_hlr_power_law_decay`)
Given elapsed days $\Delta t \ge 0$:
1. **Duolingo HLR Half-Life ($h_{t,m} \in [5, 365]$ days):**
   $$h_{t,m} = \text{clamp}\!\left(\big(6.0 + \operatorname{softplus}(\tilde{h}_{0,m})\big) \cdot 2^{\,\operatorname{softplus}(\tilde{\theta}_{+,m})(\sqrt{1 + n_{+,t}} - 1) \;-\; \operatorname{softplus}(\tilde{\theta}_{-,m})(\sqrt{1 + n_{-,t}} - 1) \;+\; 0.4\tanh(\mathbf{w}_{h,m}^\top \mathbf{c}_t)},\; 5.0,\; 365.0\right).$$
2. **Power-Law Retention Factor:**
   $$w_m = 0.5 + \operatorname{softplus}(\tilde{w}_m), \qquad \kappa_m = 2^{1/w_m} - 1,$$
   $$R_{\text{trans}}(\Delta t, h_{t,m}) = \left(1 + \kappa_m \frac{\Delta t}{h_{t,m}}\right)^{-w_m}, \qquad R_{\text{perm}}(\Delta t, h_{t,m}) = \left(1 + \kappa_m \frac{\Delta t}{4 h_{t,m}}\right)^{-w_m}.$$
3. **Two-Component Skill Decay:**
   $$\tilde{\mathbf{b}}_t = \min(\mathbf{k}_t, \mathbf{b}_t), \qquad \boldsymbol{\rho} = 0.12\,\sigma(\tilde{\boldsymbol{\rho}}),$$
   $$\mathbf{b}(t + \Delta t) = \tilde{\mathbf{b}}_t \odot \big((1 - \boldsymbol{\rho}) + \boldsymbol{\rho} \odot R_{\text{perm}}(\Delta t, \mathbf{h}_t)\big),$$
   $$\mathbf{k}(t + \Delta t) = \mathbf{b}(t + \Delta t) + (\mathbf{k}_t - \tilde{\mathbf{b}}_t) \odot R_{\text{trans}}(\Delta t, \mathbf{h}_t).$$

### 4.3 Evidence Update Step (`_update_skill`)
1. **Timing de-confounding (`_clean_token`):** Response-duration features (indices `35..39`) are multiplied by `full_success` (`token[..., 41:42]`), so rapid wrong guesses cannot inflate skill updates.
2. **Bounded negative/positive updates & consolidation:**
   $$\mathbf{k}_{\text{drop}} = \mathbf{k} \odot (\mathbf{1} - u_- \mathbf{g}_-), \qquad \mathbf{b}_{\text{drop}} = \min\!\big(\mathbf{k}_{\text{drop}},\; \tilde{\mathbf{b}} \odot (\mathbf{1} - 0.90\,u_- \mathbf{g}_-)\big),$$
   $$\Delta_{\text{gain}} = (\mathbf{1} - \mathbf{k}_{\text{drop}}) \odot (u_+ \mathbf{g}_+), \qquad \mathbf{k}_{\text{new}} = \mathbf{k}_{\text{drop}} + \Delta_{\text{gain}},$$
   $$\mathbf{b}_{\text{new}} = \min\!\big(\mathbf{k}_{\text{new}},\; \mathbf{b}_{\text{drop}} + \boldsymbol{\gamma} \odot \Delta_{\text{gain}}\big), \qquad \boldsymbol{\gamma} = 0.50 + 0.25\,\sigma(\tilde{\boldsymbol{\gamma}}),$$
   $$n_{+,\text{new}} = n_+ + u_+, \qquad n_{-,\text{new}} = n_- + u_-.$$

### 4.4 Monotonic 3PL Readout (`TaskReadoutHead`)
$$\theta_{\text{skill}} = \sum_{m=1}^M w_m(\text{fmt}) \operatorname{logit}(k_m(t + \Delta t)), \qquad w_m(\text{fmt}) > 0, \quad \sum_{m=1}^M w_m(\text{fmt}) = 1,$$
$$d(\mathbf{x}_{\text{task}}) = d_{\text{base}}(\mathbf{x}_{\text{non-diff}}) + \delta_{\text{med}}(\text{fmt})\,\mathbb{I}(\text{medium}) + \big(\delta_{\text{med}}(\text{fmt}) + \delta_{\text{hard}}(\text{fmt})\big)\mathbb{I}(\text{hard}), \qquad \delta_{\text{med}}, \delta_{\text{hard}} \ge 0.05,$$
$$P(\text{full\_success}) = g_{\text{fmt}} + (1 - g_{\text{fmt}})\,\sigma\!\Big(\alpha(\mathbf{x}_{\text{non-diff}}) \cdot \big(\theta_{\text{skill}} - d(\mathbf{x}_{\text{task}})\big)\Big).$$

---

## 5. Experimental Results (`3/model/runs/v2.2-mps-main/`)

All 7 models were trained and evaluated on `v2.2` ([`REPORT.md`](runs/v2.2-mps-main/REPORT.md)), with recurrent models trained across 3 seeds (`20261120`, `20261121`, `20261122`) on Apple Metal (`mps:0`):

### 5.1 Validation Selection & Behavioral Sanity Gate

| Model | Validation Selection Log Loss | Behavioral Checks Passed | Fast 0% Wrong (6.4s) Delta | Non-Increasing & Realistic Forgetting |
|---|---:|---:|---:|---:|
| **`monotonic_gru` (Selected)** | **`0.55175 ± 0.00025`** | **`6.0 / 6`** | `40.7% → 30.0% (-10.7%)` | **Pass (Guaranteed)** |
| `time_input_gru` | `0.55177 ± 0.00022` | `6.0 / 6` | `40.7% → 33.3% (-7.4%)` | Pass on probe (unconstrained head) |
| `time_decay_gru` | `0.55177 ± 0.00022` | `4.7 / 6` | `40.7% → 34.6% (-6.1%)` | **Fail** (rises after failures) |
| `kt_decay_gru` | `0.55471 ± 0.00077` | `6.0 / 6` | `41.5% → 37.1% (-4.4%)` | **Pass (Guaranteed)** |
| `history_boosting` | `0.55532` | — | — | — |
| `history_logistic` | `0.55692` | — | — | — |
| `task_logistic` | `0.59707` | — | — | — |

### 5.2 Held-Out Test Performance (Log Loss — Lower is Better)

| Model | `main_test` Ordinary | `main_test` Follow-up | `main_test` Delayed Mean | `power_law_test` Delayed Mean | `long_gap_test` 30-Day Delayed |
|---|---:|---:|---:|---:|---:|
| **`monotonic_gru`** | **`0.56772 ± 0.00009`** | `0.56624 ± 0.00016` | `0.54517 ± 0.00027` | `0.56839 ± 0.00018` | **`0.57386 ± 0.00040`** |
| `time_decay_gru` | `0.56782 ± 0.00023` | `0.56592 ± 0.00067` | `0.54392 ± 0.00061` | `0.56721 ± 0.00021` | `0.57437 ± 0.00057` |
| `time_input_gru` | `0.56802 ± 0.00030` | `0.56613 ± 0.00086` | `0.54368 ± 0.00120` | `0.56647 ± 0.00065` | `0.57408 ± 0.00091` |
| `kt_decay_gru` | `0.57074 ± 0.00061` | `0.57028 ± 0.00062` | `0.54810 ± 0.00046` | `0.57284 ± 0.00108` | `0.57730 ± 0.00099` |
| `history_boosting` | `0.57121` | `0.57335` | `0.54955` | `0.56987` | `0.57514` |
| `history_logistic` | `0.57423` | `0.57794` | `0.55282` | `0.57134` | `0.57586` |
| `task_logistic` | `0.60948` | `0.60885` | `0.59806` | `0.60681` | `0.60749` |

---

## 6. Verification on Your Real Dashboard Profile (`3/dashboard/data/sessions.sqlite3`)

Replaying your saved profile (`Lalith`) under the selected checkpoint ([`runs/v2.2-mps-main/monotonic_gru/seed_20261122/model.pt`](runs/v2.2-mps-main/monotonic_gru/seed_20261122/model.pt)) shows how both failure and mastery curves behave across `0` to `60` days:

| Skill | Your Recorded Attempts | Earlier Steep Decay Curve (0d → 7d → 30d → 60d) | Upgraded HLR + Power-Law + Consolidated Floor Curve (0d → 7d → 30d → 60d) |
|---|---:|---|---|
| **Function returns** | 2 Hard attempts (`100%` unaided) | `72.1% → 52.2% → 20.3% → 6.6%` (collapsed to 7%) | **`72.6% → 69.8% → 65.7% → 63.4%`** (smooth, realistic retention above cold start) |
| **Range boundaries** | 15 attempts (mixed practice) | `38.8% → 26.3% → 10.9% → 4.6%` | **`62.2% → 56.7% → 50.7% → 48.2%`** |
| **List indexing** | 2 attempts (`100%` unaided) | — | **`62.7% → 60.6% → 57.6% → 56.2%`** |
| **Loop tracing** | 1 failed attempt | `32.2% → 12.4% → 2.3% → 1.9%` | **`36.1% → 35.9% → 35.6% → 35.2%`** |
| **Accumulators** | 1 failed attempt | `24.7% → 9.3% → 2.2% → 1.9%` | **`26.8% → 26.7% → 26.4% → 26.2%`** |
| **Conditionals** | 5 attempts (failures + hints) | `24.4% → 5.1% → 1.9% → 1.8%` (`v2.1` jumped UP to `51.3%`) | **`24.8% → 23.9% → 22.9% → 22.5%`** |
