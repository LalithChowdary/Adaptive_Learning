"""Unit tests for the 8-action closed-loop PPO environment and MaskedActorCritic agent."""
import unittest

import numpy as np
import torch

from ppo.agent import (
    FixedCurriculumPolicy,
    MaskedActorCritic,
    RandomValidPolicy,
    ThresholdHeuristicPolicy,
    compute_gae,
    ppo_update,
)
from ppo.env import ACTIONS, AdaptiveTutorEnv, FrozenMemoryTracker, OBS_DIM, SKILLS_ORDER


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracker = FrozenMemoryTracker()

    def test_reset_observation_bounds_and_initial_mask(self):
        env = AdaptiveTutorEnv(tracker=self.tracker)
        obs, mask = env.reset(seed=42)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(((obs >= 0.0) & (obs <= 1.0)).all())
        self.assertEqual(mask.shape, (len(ACTIONS),))
        # At topic 0 (range_boundaries, root, difficulty Easy, 0 attempts):
        # 0 (easier), 5 (self-expl), 6 (prereq), 7 (advance) must be masked out.
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertTrue(mask[4])
        self.assertFalse(mask[5])
        self.assertFalse(mask[6])
        self.assertFalse(mask[7])

    def test_gated_option_b_prevents_consecutive_passive_lectures(self):
        env = AdaptiveTutorEnv(tracker=self.tracker)
        obs, mask = env.reset(seed=7)
        self.assertTrue(mask[3])
        obs, reward, done, mask, info = env.step(3)  # explain_concept
        self.assertFalse(done)
        self.assertTrue(env.awaiting_question)
        self.assertEqual(env.primed_mode, "concept")
        # Immediately after explain_concept, only valid question difficulties (1, 2) are allowed
        self.assertFalse(mask[3])
        self.assertFalse(mask[4])
        self.assertFalse(mask[5])
        self.assertFalse(mask[6])
        self.assertFalse(mask[7])
        self.assertTrue(mask[1] and mask[2])

        obs, reward, done, mask, info = env.step(1)  # give_same_difficulty_question
        self.assertFalse(env.awaiting_question)
        self.assertIsNone(env.primed_mode)
        self.assertTrue(mask[5])
        self.assertTrue(mask[7])

    def test_hint_leakage_sets_and_unaided_check_clears_unverified_flag(self):
        env = AdaptiveTutorEnv(tracker=self.tracker)
        env.reset(seed=101)
        skill = SKILLS_ORDER[0]
        # Force an assisted retry with leakage by lowering strength and raising help dependence
        env.latent[skill]["k"] = 0.05
        env.params[skill]["help_dependence"] = 1.0
        env.params[skill]["help_effect"] = 2.0
        saw_unverified = False
        for s in range(15):
            env.reset(seed=200 + s)
            env.latent[skill]["k"] = 0.12
            env.params[skill]["help_dependence"] = 1.0
            env.params[skill]["help_effect"] = 2.5
            env.step(2)  # Hard question -> likely triggers hint + assisted retry
            if env.unverified_assisted[skill]:
                saw_unverified = True
                # Now raise skill so unaided verification check passes cleanly and verify flag clears
                env.latent[skill]["k"] = 0.95
                env.step(1)  # give_same_difficulty_question triggers unaided verification
                self.assertFalse(env.unverified_assisted[skill])
                break
        self.assertTrue(saw_unverified)

    def test_self_explanation_consolidates_floor_and_does_not_leak_delayed_clones(self):
        env = AdaptiveTutorEnv(tracker=self.tracker)
        env.reset(seed=11)
        skill = SKILLS_ORDER[0]
        env.step(1)  # at least 1 attempt so self_explanation is unlocked
        env.latent[skill]["k"] = 0.85
        env.latent[skill]["floor"] = 0.25
        before_record_anchor = env.records[skill]["anchor"]
        before_clock = env.clock
        delayed = env.evaluate_delayed_retention()
        self.assertEqual(env.records[skill]["anchor"], before_record_anchor)
        self.assertEqual(env.clock, before_clock)
        self.assertIn("day_30", delayed)

        floor_before_se = env.latent[skill]["floor"]
        _, _, _, mask_after_se, _ = env.step(5)  # request_self_explanation
        self.assertGreaterEqual(env.latent[skill]["floor"], floor_before_se)
        self.assertFalse(mask_after_se[5])

    def test_masked_actor_critic_never_selects_invalid_actions_and_updates(self):
        torch.manual_seed(0)
        model = MaskedActorCritic()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        env = AdaptiveTutorEnv(tracker=self.tracker)
        obs, mask = env.reset(seed=5)
        obs_buf, mask_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], []
        while True:
            act, lp, val = model.act(obs, mask)
            self.assertTrue(mask[act])
            next_obs, rew, done, next_mask, _ = env.step(act)
            obs_buf.append(obs)
            mask_buf.append(mask)
            act_buf.append(act)
            lp_buf.append(lp)
            rew_buf.append(rew)
            val_buf.append(val)
            done_buf.append(done)
            obs, mask = next_obs, next_mask
            if done:
                break

        adv, ret = compute_gae(
            np.asarray(rew_buf, dtype=np.float32),
            np.asarray(val_buf, dtype=np.float32),
            np.asarray(done_buf, dtype=bool),
        )
        stats = ppo_update(
            model,
            optimizer,
            np.asarray(obs_buf),
            np.asarray(mask_buf),
            np.asarray(act_buf),
            np.asarray(lp_buf, dtype=np.float32),
            ret,
            adv,
            minibatch_size=16,
        )
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all().item())

    def test_baseline_policies_complete_full_episode(self):
        env = AdaptiveTutorEnv(tracker=self.tracker)
        rng = np.random.default_rng(123)
        for policy in (RandomValidPolicy(), FixedCurriculumPolicy(), ThresholdHeuristicPolicy()):
            if hasattr(policy, "reset"):
                policy.reset()
            obs, mask = env.reset(seed=88)
            steps = 0
            while True:
                action = policy.select_action(obs, mask, rng)
                self.assertTrue(mask[action])
                obs, _, done, mask, info = env.step(action)
                steps += 1
                if done:
                    self.assertIn("delayed", info)
                    break
            self.assertGreater(steps, 10)


if __name__ == "__main__":
    unittest.main()
