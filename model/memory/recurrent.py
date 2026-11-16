"""Sequence memory models: unconstrained GRUs, interpretable KT-decay, and Monotonic Grey-Box GRU."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .features import TASK_SIZE, TOKEN_SIZE


def _logit(p, eps=1e-5):
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class TaskReadoutHead(nn.Module):
    """Monotonic psychometric readout: higher skill channels -> higher probability; Easy > Medium > Hard."""

    def __init__(self, skill_channels=1, hidden_dim=24):
        super().__init__()
        self.skill_channels = skill_channels
        self.skill_weights = nn.Parameter(torch.full((skill_channels,), -0.5))
        self.format_skill_weights = nn.Parameter(torch.zeros(5, skill_channels))

        # Task structural encoder (excludes difficulty indices 5, 6, 7 so difficulty monotonicity is exact)
        # Task indices: 0..5 formats, 5..8 difficulty, 8..19 structural features (11 dims -> total 16 non-diff dims)
        self.task_mlp = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.med_base = nn.Parameter(torch.tensor(-0.3))
        self.hard_base = nn.Parameter(torch.tensor(-0.3))
        self.med_by_fmt = nn.Linear(5, 1, bias=False)
        self.hard_by_fmt = nn.Linear(5, 1, bias=False)
        self.guess = nn.Linear(5, 1, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.med_by_fmt.weight)
            nn.init.zeros_(self.hard_by_fmt.weight)
            # Formats: single_choice, multiple_select, output_prediction, fill_in_the_blank, code_completion
            self.guess.weight.copy_(torch.tensor([[0.4, -1.8, -2.8, -2.8, -3.2]]))

    def forward(self, skill_probs, task):
        fmt = task[..., 0:5]
        is_med = task[..., 6:7]
        is_hard = task[..., 7:8]
        non_diff = torch.cat((task[..., 0:5], task[..., 8:19]), dim=-1)

        demand_and_slope = self.task_mlp(non_diff)
        base_demand = demand_and_slope[..., 0:1]
        slope_raw = demand_and_slope[..., 1:2]

        med_step = F.softplus(self.med_base + self.med_by_fmt(fmt)) + 0.05
        hard_step = med_step + F.softplus(self.hard_base + self.hard_by_fmt(fmt)) + 0.05
        demand = base_demand + med_step * is_med + hard_step * is_hard

        # Positive weights on every skill channel guarantee monotonicity in skill_probs
        w_base = F.softplus(self.skill_weights)
        w_fmt = F.softplus(self.format_skill_weights)
        channel_weights = w_base + torch.matmul(fmt, w_fmt)
        channel_weights = channel_weights / (channel_weights.sum(dim=-1, keepdim=True) + 1e-6)

        skill_logit = (_logit(skill_probs) * channel_weights).sum(dim=-1, keepdim=True)
        alpha = 0.4 + F.softplus(slope_raw)
        q = torch.sigmoid(alpha * (skill_logit - demand))
        g = 0.35 * torch.sigmoid(self.guess(fmt))
        prob = g + (1.0 - g) * q
        return _logit(prob).squeeze(-1)


class LearnerGRU(nn.Module):
    def __init__(self, *, mode, hidden_size=32):
        super().__init__()
        if mode not in {"time_input", "time_decay", "kt_decay", "monotonic"}:
            raise ValueError(f"Unknown sequence model {mode!r}")
        self.mode = mode
        self.hidden_size = hidden_size

        if mode in {"time_input", "time_decay"}:
            self.prior = nn.Parameter(torch.zeros(hidden_size))
            self.cell = nn.GRUCell(TOKEN_SIZE + 1, hidden_size)
            self.head = nn.Sequential(
                nn.Linear(hidden_size + TASK_SIZE + 1, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1),
            )
            if mode == "time_decay":
                self.rate = nn.Linear(hidden_size, hidden_size)
                nn.init.zeros_(self.rate.weight)
                half_lives = torch.logspace(math.log10(0.5), math.log10(120), hidden_size)
                initial_rates = math.log(2) / half_lives
                with torch.no_grad():
                    self.rate.bias.copy_(torch.log(torch.expm1(initial_rates)))
        elif mode == "kt_decay":
            # Interpretable 2-channel HLR + Power-Law + Consolidated-Floor KT model (no black-box GRU context)
            self.skill_channels = 2
            self.skill_prior = nn.Parameter(torch.zeros(self.skill_channels))
            self.base_hl = nn.Parameter(torch.log(torch.expm1(torch.tensor([14.0, 45.0]) - 6.0)))
            self.hlr_pos = nn.Parameter(torch.full((self.skill_channels,), -0.8))
            self.hlr_neg = nn.Parameter(torch.full((self.skill_channels,), -1.8))
            self.power_w = nn.Parameter(torch.zeros(self.skill_channels))
            self.permastore_drift = nn.Parameter(torch.full((self.skill_channels,), -1.5))
            self.consolidation = nn.Parameter(torch.full((self.skill_channels,), 0.2))
            self.gain_gate = nn.Linear(TASK_SIZE + 4, self.skill_channels)
            self.drop_gate = nn.Linear(TASK_SIZE + 4, self.skill_channels)
            self.assist_weight = nn.Parameter(torch.tensor(-1.1))
            self.partial_weight = nn.Parameter(torch.tensor(-0.4))
            self.readout = TaskReadoutHead(skill_channels=self.skill_channels, hidden_dim=16)
        elif mode == "monotonic":
            # Multi-scale Monotonic Grey-Box GRU with Duolingo HLR + Power-Law + Consolidated Floor
            self.skill_channels = min(6, max(1, (hidden_size - 4) // 3))
            ctx_size = hidden_size - (2 * self.skill_channels + 2)
            self.ctx_size = ctx_size
            self.skill_prior = nn.Parameter(torch.zeros(self.skill_channels))
            hl_init = torch.logspace(math.log10(10.0), math.log10(75.0), self.skill_channels)
            self.base_hl = nn.Parameter(torch.log(torch.expm1(hl_init - 6.0)))
            self.hlr_pos = nn.Parameter(torch.full((self.skill_channels,), -0.8))
            self.hlr_neg = nn.Parameter(torch.full((self.skill_channels,), -1.8))
            self.hlr_ctx = nn.Linear(ctx_size, self.skill_channels)
            self.power_w = nn.Parameter(torch.zeros(self.skill_channels))
            self.permastore_drift = nn.Parameter(torch.full((self.skill_channels,), -1.5))
            self.consolidation = nn.Parameter(torch.full((self.skill_channels,), 0.2))
            self.cell = nn.GRUCell(TOKEN_SIZE + 1, ctx_size)
            self.ctx_rate = nn.Linear(ctx_size, ctx_size)
            gate_in_dim = TOKEN_SIZE + ctx_size + 2 * self.skill_channels
            self.gain_gate = nn.Sequential(
                nn.Linear(gate_in_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.skill_channels),
            )
            self.drop_gate = nn.Sequential(
                nn.Linear(gate_in_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.skill_channels),
            )
            self.assist_weight = nn.Parameter(torch.tensor(-1.1))
            self.partial_weight = nn.Parameter(torch.tensor(-0.4))
            self.readout = TaskReadoutHead(skill_channels=self.skill_channels, hidden_dim=32)
            with torch.no_grad():
                nn.init.zeros_(self.hlr_ctx.weight)
                nn.init.zeros_(self.hlr_ctx.bias)
                nn.init.zeros_(self.ctx_rate.weight)
                ctx_hl = torch.logspace(math.log10(1.0), math.log10(120.0), ctx_size)
                self.ctx_rate.bias.copy_(torch.log(torch.expm1(math.log(2) / ctx_hl)))

    def initial_state(self, batch_size=1):
        if self.mode in {"time_input", "time_decay"}:
            return self.prior.unsqueeze(0).expand(batch_size, -1)
        device = self.skill_prior.device
        k0 = torch.sigmoid(self.skill_prior).unsqueeze(0).expand(batch_size, -1)
        # Initialize both active skill k and consolidated floor to k0; counters and context start at 0
        zeros = torch.zeros((batch_size, self.hidden_size - 2 * self.skill_channels), device=device, dtype=k0.dtype)
        return torch.cat((k0, k0, zeros), dim=-1)

    def _clean_token(self, token):
        """Mask out raw response speed on non-full-success tokens to prevent fast-guess confounding."""
        full = token[..., 41:42]
        mask = torch.ones_like(token)
        # Indices 35..39 are active_response_seconds and relative_response_time
        mask[..., 35:39] = full
        return token * mask

    def _hlr_power_law_decay(self, k, floor, n_pos, n_neg, gap, ctx_mod=None):
        """
        Verified two-component retention combining:
        1. Duolingo Half-Life Regression (Settles & Meeder, ACL 2016):
           h_t = h_0 * 2^(theta_+ * (sqrt(1 + n_+) - 1) - theta_- * (sqrt(1 + n_-) - 1) + ctx_mod)
        2. Wickelgren (1974) / Pavlik & Anderson (2005) / FSRS (Ye et al., KDD 2022) Power-Law Retention:
           R(d, h_t) = (1 + (2^(1/w) - 1) * (d / h_t))^(-w), so R(0, h_t) = 1 and R(h_t, h_t) = 0.5.
        3. Bahrick (1984) Consolidated Permastore Baseline:
           Transient gain (k - floor) >= 0 decays with R(d, h_t); consolidated floor has >=88% permastore stability.
        """
        h0 = 6.0 + F.softplus(self.base_hl)
        theta_pos = F.softplus(self.hlr_pos)
        theta_neg = F.softplus(self.hlr_neg)
        exponent = theta_pos * (torch.sqrt(1.0 + n_pos) - 1.0) - theta_neg * (torch.sqrt(1.0 + n_neg) - 1.0)
        if ctx_mod is not None:
            exponent = exponent + 0.4 * torch.tanh(ctx_mod)
        half_life = torch.clamp(h0 * torch.pow(2.0, exponent), min=5.0, max=365.0)

        w = 0.5 + F.softplus(self.power_w)
        kappa = torch.pow(2.0, 1.0 / w) - 1.0
        r_transient = torch.pow(1.0 + kappa * (gap / half_life), -w)
        r_permastore = torch.pow(1.0 + kappa * (gap / (4.0 * half_life)), -w)

        rho = 0.12 * torch.sigmoid(self.permastore_drift)
        safe_floor = torch.minimum(k, floor)
        transient = torch.clamp(k - safe_floor, min=0.0)

        floor_decayed = safe_floor * ((1.0 - rho) + rho * r_permastore)
        k_decayed = floor_decayed + transient * r_transient
        return torch.clamp(k_decayed, 1e-5, 1.0 - 1e-5), torch.clamp(floor_decayed, 1e-5, 1.0 - 1e-5)

    def _at_time(self, state, elapsed_days):
        gap = elapsed_days.unsqueeze(-1)
        if self.mode == "time_input":
            return state, torch.log1p(gap)
        if self.mode == "time_decay":
            factor = torch.exp(-F.softplus(self.rate(state)) * gap)
            decayed = self.prior + factor * (state - self.prior)
            return decayed, torch.zeros_like(gap)
        m = self.skill_channels
        k = state[..., :m]
        floor = state[..., m:2 * m]
        n_pos = state[..., 2 * m:2 * m + 1]
        n_neg = state[..., 2 * m + 1:2 * m + 2]
        rest = state[..., 2 * m + 2:]
        if self.mode == "kt_decay":
            k_dec, floor_dec = self._hlr_power_law_decay(k, floor, n_pos, n_neg, gap, ctx_mod=None)
            return torch.cat((k_dec, floor_dec, n_pos, n_neg, rest), dim=-1), torch.zeros_like(gap)
        # monotonic mode
        ctx = rest
        k_dec, floor_dec = self._hlr_power_law_decay(k, floor, n_pos, n_neg, gap, ctx_mod=self.hlr_ctx(ctx))
        c_rate = F.softplus(self.ctx_rate(ctx))
        ctx_dec = ctx * torch.exp(-c_rate * gap)
        return torch.cat((k_dec, floor_dec, n_pos, n_neg, ctx_dec), dim=-1), torch.log1p(gap)

    def _update_skill(self, k, floor, n_pos, n_neg, token, gain_raw, drop_raw):
        is_resolved_attempt = token[..., 0:1] * token[..., 29:30]
        is_indep = token[..., 26:27]
        is_assist = token[..., 27:28]
        score = token[..., 39:40]
        full = token[..., 41:42]

        w_partial = 0.20 + 0.50 * torch.sigmoid(self.partial_weight)
        w_assist = 0.10 + 0.35 * torch.sigmoid(self.assist_weight)

        # Positive credit requires score > 0; full > partial; independent > assisted
        raw_credit = full + (1.0 - full) * w_partial * score
        pos_signal = is_resolved_attempt * (is_indep + is_assist * w_assist) * raw_credit * (score > 0).to(score.dtype)
        # Negative penalty when not full success; strongest when score == 0
        neg_signal = is_resolved_attempt * (1.0 - full) * (1.0 - 0.5 * score)

        g_plus = 0.45 * torch.sigmoid(gain_raw)
        g_minus = 0.45 * torch.sigmoid(drop_raw)

        safe_floor = torch.minimum(k, floor)
        k_after_drop = k * (1.0 - neg_signal * g_minus)
        floor_after_drop = torch.minimum(k_after_drop, safe_floor * (1.0 - 0.90 * neg_signal * g_minus))

        gain = (1.0 - k_after_drop) * pos_signal * g_plus
        k_new = k_after_drop + gain
        gamma = 0.50 + 0.25 * torch.sigmoid(self.consolidation)
        floor_new = torch.minimum(k_new, floor_after_drop + gamma * gain)

        n_pos_new = n_pos + pos_signal
        n_neg_new = n_neg + neg_signal
        return (
            torch.clamp(k_new, 1e-5, 1.0 - 1e-5),
            torch.clamp(floor_new, 1e-5, 1.0 - 1e-5),
            n_pos_new,
            n_neg_new,
        )

    def observe(self, state, token, elapsed_days):
        ready, time_feature = self._at_time(state, elapsed_days)
        if self.mode in {"time_input", "time_decay"}:
            return self.cell(torch.cat((token, time_feature), dim=-1), ready)
        m = self.skill_channels
        k = ready[..., :m]
        floor = ready[..., m:2 * m]
        n_pos = ready[..., 2 * m:2 * m + 1]
        n_neg = ready[..., 2 * m + 1:2 * m + 2]
        rest = ready[..., 2 * m + 2:]
        clean = self._clean_token(token)
        if self.mode == "kt_decay":
            task_and_meta = torch.cat((clean[..., 4:23], clean[..., 26:28], clean[..., 31:35:2]), dim=-1)
            k_new, floor_new, n_pos_new, n_neg_new = self._update_skill(
                k, floor, n_pos, n_neg, clean, self.gain_gate(task_and_meta), self.drop_gate(task_and_meta),
            )
            return torch.cat((k_new, floor_new, n_pos_new, n_neg_new, rest), dim=-1)
        # monotonic mode
        ctx = rest
        gate_input = torch.cat((clean, ctx, _logit(k), _logit(floor)), dim=-1)
        k_new, floor_new, n_pos_new, n_neg_new = self._update_skill(
            k, floor, n_pos, n_neg, clean, self.gain_gate(gate_input), self.drop_gate(gate_input),
        )
        ctx_new = self.cell(torch.cat((clean, time_feature), dim=-1), ctx)
        return torch.cat((k_new, floor_new, n_pos_new, n_neg_new, ctx_new), dim=-1)

    def predict_logit(self, state, task, elapsed_days):
        ready, time_feature = self._at_time(state, elapsed_days)
        if self.mode in {"time_input", "time_decay"}:
            return self.head(torch.cat((ready, task, time_feature), dim=-1)).squeeze(-1)
        m = self.skill_channels
        k = ready[..., :m]
        return self.readout(k, task)

    def forward(self, tokens, gaps, positions, tasks, query_gaps):
        """Gather the state at each query's exact evidence prefix."""
        state = self.initial_state(tokens.shape[0])
        states = [state]
        for token, gap in zip(tokens.unbind(dim=1), gaps.unbind(dim=1)):
            state = self.observe(state, token, gap)
            states.append(state)
        trajectory = torch.stack(states, dim=1)
        indices = positions.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        query_states = torch.gather(trajectory, dim=1, index=indices)
        return self.predict_logit(query_states, tasks, query_gaps)
