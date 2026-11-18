# Monotonic Grey-Box Transformer (`monotonic_transformer`)

## Purpose

`monotonic_transformer` is a neuro-symbolic hybrid implemented in `LearnerGRU(mode="monotonic_transformer")` (`3/model/memory/recurrent.py`). It isolates whether replacing the auxiliary 18-dimensional `GRUCell` context encoder in `monotonic_gru` with causal time-aware self-attention improves accuracy while preserving all cognitive memory guarantees.

## Architecture

1. **Identical Interpretable Skill State ($D = 32$, $M = 6$ channels):**
   $$\mathbf{s}_t = \big[\,\mathbf{k}_t \in (0,1)^6,\;\; \mathbf{b}_t \in (0,1)^6,\;\; n_{+,t},\;\; n_{-,t},\;\; \mathbf{c}_t \in \mathbb{R}^{18}\,\big]$$
2. **Parallel Causal Self-Attention Auxiliary Context (`CausalTimeAttentionEncoder`):**
   - Replaces `GRUCell` with a $H = 3$-head (`head_dim = 6`) causal self-attention block with learnable continuous-time log-gap decay bias $-\operatorname{softplus}(\gamma_h)\ln(1 + \Delta t_{i,j})$ over timing-cleaned tokens (`_clean_token`).
3. **Preserved Grey-Box Equations & Behavioral Guarantees:**
   - Retains Duolingo Half-Life Regression (`Settles & Meeder, 2016`), Wickelgren/FSRS Power-Law forgetting (`Ye et al., 2022`), Bahrick (`1984`) consolidated permastore baseline, timing de-confounding on failed attempts, bounded `[0, 0.45]` outcome-signed update gates, and the Birnbaum (`1968`) monotonic 3PL IRT readout (`TaskReadoutHead`).
   - Like `monotonic_gru`, every seed is mathematically guaranteed to satisfy non-increasing forgetting and difficulty ordering (`Easy > Medium > Hard`).
