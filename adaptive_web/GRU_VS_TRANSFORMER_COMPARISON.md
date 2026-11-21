# Monotonic GRU vs. Transformer Comparison

This note summarizes the empirical comparison between **`monotonic_gru`** and Transformer-based knowledge tracing models in this project (see [`PROJECT_DECISIONS.md`](file:///Users/lalith/snu/sem7/minor_pro/code/PROJECT_DECISIONS.md) and [`3/model/runs/v2.2-mps-transformer-comparison/REPORT.md`](file:///Users/lalith/snu/sem7/minor_pro/code/3/model/runs/v2.2-mps-transformer-comparison/REPORT.md)).

---

## 1. Benchmark Comparison (`v2.2` 9-Model Comparison, 3 Seeds on `mps:0`)

| Model | Validation Log Loss *(lower is better)* | Behavioral Sanity Checks | Runtime State Complexity |
| :--- | :---: | :---: | :--- |
| **`monotonic_gru`** *(Ours)* | **`0.55175 ± 0.00025`** *(Best)* | **6.0 / 6** | **$O(1)$** fixed 32-float Markov state |
| `monotonic_transformer` | `0.55209 ± 0.00020` | 6.0 / 6 | $O(L)$ full-history replay ($O(L^2)$ attention) |
| `time_aware_transformer` | `0.55374 ± 0.00082` | 6.0 / 6 | $O(L)$ full-history replay ($O(L^2)$ attention) |
| `time_decay_gru` *(vanilla decay)* | `0.55177 ± 0.00022` | 4.7 / 6 *(fails non-increasing forgetting)* | $O(1)$ |

- **Statistical Significance (`v2.2` Held-Out Practice):** `monotonic_gru` beat `time_aware_transformer` with 95% learner-bootstrap confidence intervals strictly below zero on both `main_test` ($\Delta = -0.00194$, $[-0.00322, -0.00048]$) and `power_law_test` ($\Delta = -0.00174$, $[-0.00324, -0.00023]$).
- **Earlier Large-Scale Benchmark (ASSISTments, 6.1M rows):** Recurrent `lstm_dkt` (`0.5028` test log loss, `0.7852` AUC) also outperformed `sakt_transformer` (`0.5231` test log loss, `0.7698` AUC).

---

## 2. Why `monotonic_gru` is Better Here (Simple Explanation)

1. **No "mixed-skill needle in a haystack" problem:**
   - Transformers excel in Knowledge Tracing when a student's log mixes 100+ different skills in one long sequence, because self-attention can look back 80 steps to find the last attempt on the *same* skill.
   - In our architecture, each `(learner_id, skill_id)` stream is **already isolated** (median ~33 events, all on the same skill). There is no cross-skill clutter to filter out, so Transformer attention adds cost without improving accuracy.

2. **Grey-Box Cognitive Memory, not a plain 2014 GRU:**
   - A standard GRU forgets unpredictably. `monotonic_gru` explicitly encodes cognitive memory laws:
     - **6 active strength channels ($k_t$)** and **6 consolidated permastore floor channels ($b_t$)** (Bahrick, 1984)
     - **Power-law forgetting** and **half-life growth** driven by retrieval counts $(n_+, n_-)$ (Duolingo HLR / FSRS)
     - **Guaranteed non-increasing forgetting curves** over 0–60 days
   - The small GRU cell only updates an auxiliary context vector.

3. **Fast $O(1)$ Streaming State for PPO & Live Tutoring:**
   - `monotonic_gru` maintains a fixed **32-float Markov state** per `(learner, skill)` that updates in **$O(1)$** time on each new answer and feeds directly into `delayed_retention_ppo`.
   - A Transformer has no fixed-size Markov state and must re-attend over the entire history ($O(L^2)$) whenever the learner answers or PPO queries 30-day retention.

---

## 3. One-Sentence Presentation Summary

> *"Transformers shine on long, mixed-skill logs, but because our system isolates each skill into short, single-skill streams, `monotonic_gru` combines cognitive forgetting laws with $O(1)$ streaming state to achieve better accuracy (`0.55175` vs. `0.55374`), guaranteed monotonic curves, and an exact state representation for PPO."*
