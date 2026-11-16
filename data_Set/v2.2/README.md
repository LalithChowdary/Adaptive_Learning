# Python learner-memory dataset v2.2

Version `2.2.0` preserves `v1`, `v2.0`, and `v2.1` untouched (all 226 historical file hashes verified in [`reports/RELEASE_MANIFEST.json`](reports/RELEASE_MANIFEST.json)) and upgrades three core simulation mechanisms in [`simulator.py`](simulator.py) based on cognitive memory literature and diagnostic findings from interactive dashboard evaluation:

---

## 1. Mathematical & Behavioral Upgrades in `v2.2`

### A. Outcome- and Assistance-Sensitive Practice Learning (`simulator.py` lines 142–158)
- **Issue in `v2.0` / `v2.1`:** Attempting a practice item granted an unconditional positive learning dose (`dose = practice_dose / attempt_number`) even when the learner answered **unaided (`hints == 0`) with `score == 0.0`**. As a result, trained models inferred that an unaided `0%` first attempt increased underlying competence.
- **Fix in `v2.2`:**
  $$\text{factor} = \begin{cases} \text{score} \cdot \mathbb{I}(\text{full\_success}) + 0.5 \cdot \text{score} \cdot \mathbb{I}(\neg\text{full\_success}), & \text{if } \text{hints} = 0, \\ f_{\text{retained}} \cdot (0.20 + 0.80 \cdot \text{score}), & \text{if } \text{hints} > 0, \end{cases}$$
  $$\text{dose} = \frac{d_{\text{format}} \cdot \text{factor}}{\text{attempt\_number}}, \qquad k_{\text{after}} = k_{\text{before}} + (1 - k_{\text{before}})\left(1 - e^{-\eta \cdot \text{dose}}\right).$$
  When $\text{hints} = 0$ and $\text{score} = 0$, $\text{dose} = 0$ and $k_{\text{after}} = k_{\text{before}}$ (zero retained skill gain).
  - Verified by **Test 56** (`test_unaided_zero_score_failure_grants_no_retained_learning_gain`) and **Audit Category 27**.

### B. Rapid-Guessing / Fast-Slip Timing Mixture (`simulator.py` lines 130–135)
- **Issue in `v2.0` / `v2.1`:** Active response duration scaled deterministically with $(1.25 - 0.5 k)$, so very fast responses occurred almost exclusively when hidden strength $k$ was high. Unconstrained sequence models learned a spurious shortcut that a 5–6 second wrong answer implied high latent skill.
- **Fix in `v2.2`:** Non-successful attempts ($\neg\text{full\_success}$) sample from a two-component timing mixture (Wise & Kong, 2005, *Applied Measurement in Education*):
  - **35% rapid-guess / slip regime:** $t_{\text{active}} \sim \text{Uniform}(0.05, 0.25) \times t_{\text{ref}}$, clipped to $[3, 900]$ seconds.
  - **65% engaged effort regime:** $t_{\text{active}} \sim t_{\text{ref}} \cdot s_{\text{learner}} \cdot (1.15 - 0.35 k) \cdot \text{LogNormal}(0, 0.35) / (1 + 0.12 \cdot \text{hints})$.

### C. Duolingo HLR Half-Life Growth & Consolidated Permastore Baseline (`simulator.py` lines 26–31, 79–95, 151–158)
- **Issue in `v2.0` / `v2.1`:**
  1. Each skill's half-life $h_0$ was static and never increased with successful practice.
  2. Retention decayed the entire skill state $k$ all the way toward $0.0$ ($k \cdot 2^{-\Delta t / h_0}$), wiping out even pre-existing background competence over a 30–60 day break.
- **Fix in `v2.2` (Verified Literature Formulas):**
  1. **Duolingo Half-Life Regression (Settles & Meeder, ACL 2016, *"A Trainable Spaced Repetition Model for Language Learning"*, Eq. 3–4):**
     Effective memory half-life $h_t$ (in days) grows with cumulative unaided full successes $n_+$ and decreases with failures $n_-$:
     $$h_t = \text{clamp}\!\left(h_0 \cdot 2^{0.35(\sqrt{1 + n_+} - 1) - 0.15(\sqrt{1 + n_-} - 1)},\; 2.0,\; 240.0\right).$$
  2. **Two-Component Retention with Consolidated Skill Floor (Bahrick, 1984, *"Semantic memory content in permastore"*; Pavlik & Anderson, 2005, *ACT-R*; Ye et al., KDD 2022, *FSRS*):**
     Each learner-skill maintains a durable consolidated floor $b_t \in [0, k_t]$ (initialized at $b_0 = 0.65 k_0$ and updated on each practice gain via $b_t = \min(k_{\text{after}}, b_{t-1} + \gamma (k_{\text{after}} - k_{\text{before}}))$ with $\gamma = 0.45$ for unaided full success and $0.25$ otherwise). Over an elapsed interval of $\Delta t$ days:
     $$k(t + \Delta t) = b_t + (k_t - b_t) \cdot R(\Delta t, h_t),$$
     where the retention factor $R(\Delta t, h_t)$ is:
     - **Exponential HLR (`main`, `pilot`, `long_gap`):** $R(\Delta t, h_t) = 2^{-\Delta t / h_t}$ (Settles & Meeder, 2016).
     - **Power-Law (`power_law` stress cohort):** $R(\Delta t, h_t) = \left(1 + \frac{\Delta t}{h_t}\right)^{-1}$ (Wickelgren, 1974; Rubin & Wenzel, 1996).
  - Verified by **Test 57** (`test_hlr_half_life_grows_with_success_and_consolidated_floor_prevents_zero_collapse`) and **Audit Category 28**.

---

## 2. Verified Cohort Counts (`v2.2.0`)

| Cohort | Learners | Total Events | Responses (incl. retries) | Eligible Practice Targets | Eligible Delayed Targets | Audit Categories Passed |
|---|---:|---:|---:|---:|---:|---:|
| `pilot` | 200 | 137,838 | 37,876 | 23,982 | 3,332 | **28 / 28** |
| `main` (train / val / test: 1600 / 200 / 200) | 2,000 | 1,390,948 | 382,429 | 241,021 (192,779 train) | 33,157 (26,505 train) | **28 / 28** |
| `power_law` (test-only stress) | 200 | 138,839 | 38,185 | 24,053 | 3,323 | **28 / 28** |
| `long_gap` (test-only stress) | 200 | 144,791 | 39,544 | 24,568 | 3,314 | **28 / 28** |

- **Automated Test Suite (`run_tests.py`):** **57 / 57 passed** (0 failures, 0 errors, 0 skips).
- **Preserved Earlier Release Files (`v1`, `v2`, `v2.1`):** **226 files** verified by SHA-256 in [`reports/RELEASE_MANIFEST.json`](reports/RELEASE_MANIFEST.json).
