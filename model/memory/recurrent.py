"""The two sequence models share event encodings and a prediction head."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .features import TASK_SIZE, TOKEN_SIZE


class LearnerGRU(nn.Module):
    def __init__(self, *, mode, hidden_size=32):
        super().__init__()
        if mode not in {"time_input", "time_decay"}:
            raise ValueError(f"Unknown sequence model {mode!r}")
        self.mode = mode
        self.hidden_size = hidden_size
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
            # Start with channels spanning short and long horizons.
            half_lives = torch.logspace(math.log10(0.5), math.log10(120), hidden_size)
            initial_rates = math.log(2) / half_lives
            with torch.no_grad():
                self.rate.bias.copy_(torch.log(torch.expm1(initial_rates)))

    def initial_state(self, batch_size=1):
        return self.prior.unsqueeze(0).expand(batch_size, -1)

    def _at_time(self, state, elapsed_days):
        gap = elapsed_days.unsqueeze(-1)
        if self.mode == "time_input":
            return state, torch.log1p(gap)
        factor = torch.exp(-F.softplus(self.rate(state)) * gap)
        decayed = self.prior + factor * (state - self.prior)
        return decayed, torch.zeros_like(gap)

    def observe(self, state, token, elapsed_days):
        ready, time_feature = self._at_time(state, elapsed_days)
        return self.cell(torch.cat((token, time_feature), dim=-1), ready)

    def predict_logit(self, state, task, elapsed_days):
        ready, time_feature = self._at_time(state, elapsed_days)
        return self.head(torch.cat((ready, task, time_feature), dim=-1)).squeeze(-1)

    def forward(self, tokens, gaps, positions, tasks, query_gaps):
        """Gather the state at each query's exact evidence prefix."""
        state = self.initial_state(tokens.shape[0])
        states = [state]
        for token, gap in zip(tokens.unbind(dim=1), gaps.unbind(dim=1)):
            state = self.observe(state, token, gap)
            states.append(state)
        trajectory = torch.stack(states, dim=1)
        # Real query positions stop at the true sequence length, before padding.
        indices = positions.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        query_states = torch.gather(trajectory, dim=1, index=indices)
        return self.predict_logit(query_states, tasks, query_gaps)
