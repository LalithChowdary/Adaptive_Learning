# Knowledge-Tracing + HLR/Power-Law Decay Baseline (`kt_decay_gru`)

## Purpose

`kt_decay_gru` is a lightweight, fully interpretable grey-box baseline in `LearnerGRU(mode="kt_decay")` that uses the exact same **Duolingo Half-Life Regression (Settles & Meeder, 2016) + Power-Law Retention (Wickelgren, 1974; Ye et al., 2022) + Consolidated Permastore Baseline (Bahrick, 1984)** and **Monotonic 3PL Readout (`TaskReadoutHead`)** as `monotonic_gru`, **without** a recurrent `GRUCell` context vector ($M = 2$ skill channels, linear task-feature update gates).

Comparing `monotonic_gru` against `kt_decay_gru` isolates the predictive contribution of the recurrent `GRUCell` context vector while holding the psychometric readout and forgetting equations constant.
