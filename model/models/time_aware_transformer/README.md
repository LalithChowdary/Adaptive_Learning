# Time-Aware Transformer Baseline (`time_aware_transformer`)

## Purpose

`time_aware_transformer` is an unconstrained SAKT/AKT-style causal self-attention baseline implemented in `LearnerGRU(mode="time_aware_transformer")` (`3/model/memory/recurrent.py`). It serves as a head-to-head comparison against recurrent and grey-box models under the exact same structured per-skill token allowlist, hidden width (`D = 32`), and training budget.

## Architecture

1. **Token + Elapsed-Time Projection:**
   Each observed evidence token $\mathbf{x}_t \in \mathbb{R}^{47}$ is concatenated with $\ln(1 + \Delta t_t)$ (elapsed days since the previous encoded event) and projected into $d_{\text{model}} = 32$.
2. **Causal Time-Aware Multi-Head Self-Attention (`CausalTimeAttentionEncoder`):**
   - $H = 4$ heads (`head_dim = 8`), Pre-LN residual self-attention + `SiLU` feed-forward block.
   - Pairwise elapsed time $\Delta t_{i,j} = \max(0, t_i - t_j)$ applies a learnable per-head decay bias inside the attention logits before softmax:
     $$\text{AttentionScore}_{h}(i, j) = \frac{\mathbf{q}_{h,i}^\top \mathbf{k}_{h,j}}{\sqrt{d_h}} - \operatorname{softplus}(\gamma_h)\ln(1 + \Delta t_{i,j}), \qquad j \le i.$$
3. **Cold Start and Readout Head:**
   - Queries prior to any observed event use a learned cold-start vector $\mathbf{h}_0 \in \mathbb{R}^{32}$.
   - Prediction concatenates the causal representation $\mathbf{z}_t$, target task features $\mathbf{x}_{\text{task}}$, and $\ln(1 + \Delta t_{\text{query}})$ into a 2-layer `SiLU` MLP matched to `time_input_gru`.
