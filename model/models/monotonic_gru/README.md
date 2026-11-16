# Monotonic Grey-Box GRU (`monotonic_gru`)

## Purpose and Motivation

The unconstrained time-decayed GRU (`time_decay_gru`) exhibited three behavioral failure modes when deployed interactively in the learner dashboard (`3/dashboard/`):

1. **Wrong answers could increase predicted success** when answered rapidly (e.g. a 6.4s `0%` attempt on *Conditionals* increased predicted success from `42.96%` to `51.33%`).
2. **Retention curves drifted upward after failures** because the hidden state decayed exponentially toward the shared population prior $h_0$ ($\approx 43\%$ on a medium output-prediction question).
3. **Pure exponential decay toward zero in logit space caused an unrealistically steep forgetting cliff** over 30–60 days after mastery (e.g. dropping from `72%` to `20%` at 30 days and `7%` at 60 days after two hard unaided successes).

The **Monotonic Grey-Box GRU (`monotonic_gru`)** resolves all three issues by combining a recurrent context encoder (`GRUCell`) with an explicit two-component monotonic skill state governed by verified cognitive-science and spaced-repetition equations.

---

## Mathematical Formulation

### 1. State Representation ($D = 32$)

For each learner and canonical skill, the 32-dimensional state vector at step $t$ is partitioned into four interpretable blocks:

$$\mathbf{s}_t = \big[\,\mathbf{k}_t \in (0,1)^M,\;\; \mathbf{b}_t \in (0,1)^M,\;\; n_{+,t} \in \mathbb{R}_{\ge 0},\;\; n_{-,t} \in \mathbb{R}_{\ge 0},\;\; \mathbf{c}_t \in \mathbb{R}^{D - 2M - 2}\,\big]$$

where $M = 6$ multi-scale skill channels:
- $\mathbf{k}_t$: current active skill vector across $M$ timescales.
- $\mathbf{b}_t$: consolidated long-term baseline ("permastore") skill vector, satisfying $0 < \mathbf{b}_t \le \mathbf{k}_t < 1$.
- $n_{+,t}, n_{-,t}$: cumulative positive and negative retrieval evidence counters for Duolingo Half-Life Regression.
- $\mathbf{c}_t \in \mathbb{R}^{18}$: recurrent context vector updated by a `GRUCell`.

At cold start ($t = 0$, zero recorded attempts):
$$\mathbf{k}_0 = \mathbf{b}_0 = \sigma(\mathbf{p}_{\text{skill}}), \qquad n_{+,0} = 0, \qquad n_{-,0} = 0, \qquad \mathbf{c}_0 = \mathbf{0}.$$

---

### 2. Verified Retention & Forgetting Equations (`_hlr_power_law_decay`)

Over an elapsed time gap of $\Delta t \ge 0$ days without practice, `monotonic_gru` combines three literature-verified mechanisms:

1. **Duolingo Half-Life Regression (Settles & Meeder, ACL 2016, *"A Trainable Spaced Repetition Model for Language Learning"*, Eq. 3–4):**
   Each channel's effective memory half-life $h_{t,m} > 0$ (in days) grows with cumulative successful retrievals $n_{+,t}$ and decreases with failures $n_{-,t}$:
   $$h_{t,m} = \text{clamp}\!\left(h_{0,m} \cdot 2^{\,\theta_{+,m}\left(\sqrt{1 + n_{+,t}} - 1\right) \;-\; \theta_{-,m}\left(\sqrt{1 + n_{-,t}} - 1\right) \;+\; 0.4\tanh(\mathbf{w}_{h,m}^\top \mathbf{c}_t)},\; 5.0,\; 365.0\right)$$
   where $h_{0,m} = 6.0 + \operatorname{softplus}(\tilde{h}_{0,m}) > 6.0$ days, $\theta_{+,m} = \operatorname{softplus}(\tilde{\theta}_{+,m}) > 0$, and $\theta_{-,m} = \operatorname{softplus}(\tilde{\theta}_{-,m}) > 0$.

2. **Power-Law Forgetting Parameterized by Exact Half-Life (Wickelgren, 1974; Rubin & Wenzel, 1996; Pavlik & Anderson, 2005 *ACT-R*; Ye et al., KDD 2022 *FSRS*):**
   $$R(\Delta t, h_{t,m}) = \left(1 + \left(2^{1/w_m} - 1\right)\frac{\Delta t}{h_{t,m}}\right)^{-w_m}, \qquad w_m = 0.5 + \operatorname{softplus}(\tilde{w}_m) > 0.5.$$
   - At $\Delta t = 0$: $R(0, h_{t,m}) = 1$.
   - At $\Delta t = h_{t,m}$: $R(h_{t,m}, h_{t,m}) = (2^{1/w_m})^{-w_m} = 0.5$ (**$h_{t,m}$ is the exact half-life in days**).
   - For $\Delta t \gg h_{t,m}$: power-law tail prevents unrealistic exponential collapse to zero.

3. **Two-Component Consolidation / Permastore Baseline (Bahrick, 1984, *"Semantic memory content in permastore"*; Mozer & Lindsey, 2016 *DASH*):**
   Let $\tilde{\mathbf{b}}_t = \min(\mathbf{k}_t, \mathbf{b}_t)$ be the consolidated baseline and $\Delta\mathbf{k}_t = \mathbf{k}_t - \tilde{\mathbf{b}}_t \ge \mathbf{0}$ be the transient practice gain above baseline:
   $$\mathbf{b}(t + \Delta t) = \tilde{\mathbf{b}}_t \odot \Big((1 - \boldsymbol{\rho}) + \boldsymbol{\rho} \odot R(\Delta t, 4\mathbf{h}_t)\Big), \qquad \boldsymbol{\rho} = 0.12\,\sigma(\tilde{\boldsymbol{\rho}}) \in (0, 0.12)^M,$$
   $$\mathbf{k}(t + \Delta t) = \mathbf{b}(t + \Delta t) + (\mathbf{k}_t - \tilde{\mathbf{b}}_t) \odot R(\Delta t, \mathbf{h}_t).$$
   Because $\tilde{\mathbf{b}}_t > 0$, $\mathbf{k}_t - \tilde{\mathbf{b}}_t \ge 0$, and $R(\Delta t, \cdot)$ is strictly decreasing in $\Delta t$, **every channel $\mathbf{k}(t + \Delta t)$ is strictly non-increasing over elapsed time $\Delta t \ge 0$**, while retaining its consolidated floor $\ge 0.88\,\tilde{\mathbf{b}}_t$ over 30–60 days.

---

### 3. Bounded Outcome-Signed Evidence Update (`_update_skill`)

When a resolved attempt arrives with rubric score $y \in [0, 1]$, binary full-success indicator $f \in \{0, 1\}$, and assistance indicators $a_{\text{indep}}, a_{\text{assist}} \in \{0, 1\}$:
1. **Timing de-confounding (`_clean_token`):** Response-speed features (`active_response_seconds`, `relative_response_time`) are masked to zero whenever $f = 0$, preventing rapid guesses from being misinterpreted as high skill.
2. **Signed evidence signals:**
   $$u_+ = \mathbb{I}(\text{resolved}) \cdot (a_{\text{indep}} + w_{\text{assist}} a_{\text{assist}}) \cdot \big(f + (1 - f) w_{\text{partial}} y\big) \cdot \mathbb{I}(y > 0),$$
   $$u_- = \mathbb{I}(\text{resolved}) \cdot (1 - f) \cdot (1 - 0.5 y),$$
   where $w_{\text{partial}} \in (0.20, 0.70)$ and $w_{\text{assist}} \in (0.10, 0.45)$.
3. **Bounded `[0, 0.45]` update gates and consolidation:**
   $$\mathbf{g}_+ = 0.45\,\sigma(\text{MLP}_+(\tilde{\mathbf{x}}_t, \mathbf{c}_t, \operatorname{logit}(\mathbf{k}), \operatorname{logit}(\mathbf{b}))), \qquad \mathbf{g}_- = 0.45\,\sigma(\text{MLP}_-(\tilde{\mathbf{x}}_t, \mathbf{c}_t, \operatorname{logit}(\mathbf{k}), \operatorname{logit}(\mathbf{b}))),$$
   $$\mathbf{k}_{\text{drop}} = \mathbf{k} \odot (\mathbf{1} - u_- \mathbf{g}_-), \qquad \mathbf{b}_{\text{drop}} = \min\!\big(\mathbf{k}_{\text{drop}},\; \mathbf{b} \odot (\mathbf{1} - 0.90\,u_- \mathbf{g}_-)\big),$$
   $$\Delta_{\text{gain}} = (\mathbf{1} - \mathbf{k}_{\text{drop}}) \odot (u_+ \mathbf{g}_+), \qquad \mathbf{k}_{\text{new}} = \mathbf{k}_{\text{drop}} + \Delta_{\text{gain}},$$
   $$\mathbf{b}_{\text{new}} = \min\!\big(\mathbf{k}_{\text{new}},\; \mathbf{b}_{\text{drop}} + \boldsymbol{\gamma} \odot \Delta_{\text{gain}}\big), \qquad \boldsymbol{\gamma} = 0.50 + 0.25\,\sigma(\tilde{\boldsymbol{\gamma}}) \in (0.50, 0.75)^M.$$

---

### 4. Monotonic 3PL Psychometric Readout (`TaskReadoutHead`)

Given decayed skill $\mathbf{k}(t + \Delta t) \in (0, 1)^M$ and target task features $\mathbf{x}_{\text{task}}$:
1. **Normalized positive channel weights:**
   $$\mathbf{w}(\text{fmt}) = \frac{\operatorname{softplus}(\mathbf{w}_0) + \operatorname{softplus}(W_{\text{fmt}})\mathbf{e}_{\text{fmt}}}{\sum_{m=1}^M \big(\operatorname{softplus}(\mathbf{w}_0) + \operatorname{softplus}(W_{\text{fmt}})\mathbf{e}_{\text{fmt}}\big)_m}, \qquad \theta_{\text{skill}} = \sum_{m=1}^M w_m(\text{fmt}) \operatorname{logit}(k_m(t + \Delta t)).$$
2. **Ordered difficulty demand (`Easy < Medium < Hard`):**
   $$d(\mathbf{x}_{\text{task}}) = d_{\text{base}}(\mathbf{x}_{\text{non-diff}}) + \delta_{\text{med}}(\text{fmt})\,\mathbb{I}(\text{medium}) + \big(\delta_{\text{med}}(\text{fmt}) + \delta_{\text{hard}}(\text{fmt})\big)\mathbb{I}(\text{hard}),$$
   where $\delta_{\text{med}}(\text{fmt}), \delta_{\text{hard}}(\text{fmt}) \ge 0.05 > 0$.
3. **3-Parameter Logistic (3PL) Item Response Probability (Birnbaum, 1968):**
   $$P(\text{full\_success} \mid \mathbf{s}_t, \mathbf{x}_{\text{task}}, \Delta t) = g_{\text{fmt}} + (1 - g_{\text{fmt}})\,\sigma\!\Big(\alpha(\mathbf{x}_{\text{non-diff}}) \cdot \big(\theta_{\text{skill}} - d(\mathbf{x}_{\text{task}})\big)\Big),$$
   where $\alpha(\mathbf{x}_{\text{non-diff}}) = 0.4 + \operatorname{softplus}(\tilde{\alpha}) > 0$ and $g_{\text{fmt}} = 0.35\,\sigma(\tilde{g}_{\text{fmt}}) \in (0, 0.35)$ is the format-specific guessing floor.
