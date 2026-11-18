"""Masked Actor-Critic PPO agent and pedagogical baseline policies."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .env import ACTIONS, OBS_DIM


def _init_linear(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class MaskedActorCritic(nn.Module):
    """MLP Actor-Critic with exact invalid-action masking for the 8-action menu."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=len(ACTIONS), hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.actor = nn.Sequential(
            _init_linear(nn.Linear(obs_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.SiLU(),
            _init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.SiLU(),
            _init_linear(nn.Linear(hidden_dim, act_dim), gain=0.01),
        )
        self.critic = nn.Sequential(
            _init_linear(nn.Linear(obs_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.SiLU(),
            _init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.SiLU(),
            _init_linear(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, obs, mask):
        logits = self.actor(obs)
        masked = logits.masked_fill(~mask, -1e9)
        value = self.critic(obs).squeeze(-1)
        return masked, value

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        logits, value = self(obs_t, mask_t)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
            dist = Categorical(logits=logits)
            log_prob = float(dist.log_prob(torch.tensor([action])).item())
            return action, log_prob, float(value.item())
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        return int(action_t.item()), float(dist.log_prob(action_t).item()), float(value.item())

    def evaluate_actions(self, obs, masks, actions):
        logits, values = self(obs, masks)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


def compute_gae(rewards, values, dones, *, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation across concatenated episodes."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 0.0 if dones[t] else 1.0
        next_val = 0.0 if (t == n - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model,
    optimizer,
    obs,
    masks,
    actions,
    old_log_probs,
    returns,
    advantages,
    *,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.015,
    ppo_epochs=4,
    minibatch_size=128,
    rng=None,
):
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    masks_t = torch.as_tensor(masks, dtype=torch.bool)
    actions_t = torch.as_tensor(actions, dtype=torch.long)
    old_lp_t = torch.as_tensor(old_log_probs, dtype=torch.float32)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    adv_t = torch.as_tensor(advantages, dtype=torch.float32)
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    n = obs_t.shape[0]
    rng = rng or np.random.default_rng(0)
    stats = {"policy_loss": [], "value_loss": [], "entropy": []}
    model.train()

    for _ in range(ppo_epochs):
        order = rng.permutation(n)
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            new_lp, entropy, values = model.evaluate_actions(obs_t[idx], masks_t[idx], actions_t[idx])
            ratio = torch.exp(new_lp - old_lp_t[idx])
            mb_adv = adv_t[idx]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns_t[idx])
            ent_loss = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * ent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            stats["policy_loss"].append(float(policy_loss.item()))
            stats["value_loss"].append(float(value_loss.item()))
            stats["entropy"].append(float(ent_loss.item()))

    return {k: float(np.mean(v)) for k, v in stats.items()}


class RandomValidPolicy:
    """Uniformly samples among valid masked actions."""

    def select_action(self, obs, mask, rng):
        valid = np.flatnonzero(mask)
        return int(rng.choice(valid))


class FixedCurriculumPolicy:
    """Deterministic per-topic progression: explain -> Easy -> Medium -> Hard -> advance."""

    def __init__(self):
        self.stage = 0

    def reset(self):
        self.stage = 0

    def select_action(self, obs, mask, rng=None):
        # obs[13] is awaiting_question; obs[16:19] is difficulty one-hot (easy, med, hard)
        awaiting = obs[13] > 0.5
        if awaiting:
            self.stage = 1
            return 1
        if self.stage == 0:
            if mask[3]:
                return 3
            self.stage = 1
            return 1
        if self.stage == 1:
            self.stage = 2
            return 2 if mask[2] else 1
        if self.stage == 2:
            self.stage = 3
            return 2 if mask[2] else 1
        if mask[7]:
            self.stage = 0
            return 7
        return 1


class ThresholdHeuristicPolicy:
    """Classic short-term mastery-threshold tutor (vulnerable to Illusion of Competence)."""

    def select_action(self, obs, mask, rng=None):
        p_med = float(obs[1])
        awaiting = obs[13] > 0.5
        if awaiting:
            if p_med >= 0.55 and mask[2]:
                return 2
            if p_med < 0.35 and mask[0]:
                return 0
            return 1
        if mask[7] and p_med >= 0.65:
            return 7
        if p_med < 0.42 and mask[3]:
            return 3
        if p_med >= 0.52 and mask[2]:
            return 2
        if p_med < 0.30 and mask[0]:
            return 0
        return 1
