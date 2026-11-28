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
from ppo.env import (
    ABLATED_OBS_DIM,
    ABLATED_OBS_INDICES,
    ACTIONS,
    AdaptiveTutorEnv,
    FrozenMemoryTracker,
    OBS_DIM,
    SKILLS_ORDER,
)
from ppo.train_ppo import _bootstrap_diff, _summarize


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

    def test_ablated_env_obs_dim_23_budget_and_per_skill_day_30(self):
        env = AdaptiveTutorEnv(tracker=self.tracker, total_budget=16.0, no_longterm_state=True)
        self.assertEqual(env.obs_dim, 23)
        obs, mask = env.reset(seed=77)
        self.assertEqual(obs.shape, (ABLATED_OBS_DIM,))
        self.assertEqual(ABLATED_OBS_DIM, 23)
        self.assertTrue(np.isfinite(obs).all())
        model = MaskedActorCritic(obs_dim=ABLATED_OBS_DIM)
        while True:
            action, _, _ = model.act(obs, mask, deterministic=True)
            obs, _, done, mask, info = env.step(action)
            self.assertEqual(obs.shape, (23,))
            if done:
                self.assertLessEqual(env.budget_used, 16.0 + 1.0)
                self.assertGreaterEqual(env.budget_used, 16.0 - 1e-6)
                delayed = info["delayed"]
                self.assertIn("per_skill_day_30", delayed)
                self.assertEqual(set(delayed["per_skill_day_30"].keys()), set(SKILLS_ORDER))
                self.assertAlmostEqual(
                    float(np.mean(list(delayed["per_skill_day_30"].values()))),
                    delayed["day_30"],
                    places=12,
                )
                break

    def test_total_budget_scaling_16_and_40(self):
        for budget in (16.0, 40.0):
            env = AdaptiveTutorEnv(tracker=self.tracker, total_budget=budget)
            self.assertEqual(env.obs_dim, OBS_DIM)
            obs, mask = env.reset(seed=55)
            self.assertAlmostEqual(float(obs[21]), 1.0, places=6)
            obs, _, done, mask, _ = env.step(1)  # cost 1.0
            self.assertFalse(done)
            self.assertAlmostEqual(float(obs[21]), (budget - 1.0) / budget, places=6)
            policy = FixedCurriculumPolicy()
            policy.reset()
            rng = np.random.default_rng(0)
            while not done:
                act = policy.select_action(obs, mask, rng)
                obs, _, done, mask, info = env.step(act)
            self.assertGreaterEqual(env.budget_used, budget - 1e-6)
            self.assertLessEqual(env.budget_used, budget + 1.0)
            self.assertEqual(set(info["delayed"]["per_skill_day_30"].keys()), set(SKILLS_ORDER))
            self.assertAlmostEqual(
                float(np.mean(list(info["delayed"]["per_skill_day_30"].values()))),
                info["delayed"]["day_30"],
                places=12,
            )

    def test_no_longterm_state_exact_slice_and_zero_leakage(self):
        env_full = AdaptiveTutorEnv(tracker=self.tracker, no_longterm_state=False)
        env_abl = AdaptiveTutorEnv(tracker=self.tracker, no_longterm_state=True)
        self.assertEqual(env_full.obs_dim, 36)
        self.assertEqual(env_abl.obs_dim, 23)
        self.assertEqual(
            list(ABLATED_OBS_INDICES),
            [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 30, 31, 32, 33, 34, 35],
        )

        obs_full, _ = env_full.reset(seed=91)
        obs_abl, _ = env_abl.reset(seed=91)
        self.assertTrue(np.array_equal(obs_abl, obs_full[list(ABLATED_OBS_INDICES)]))

        # Advance to loop_tracing (prereq: range_boundaries) and set high p_med everywhere
        env_abl.topic_idx = SKILLS_ORDER.index("loop_tracing")
        env_abl.visited_topics = set(SKILLS_ORDER)
        env_full.topic_idx = SKILLS_ORDER.index("loop_tracing")
        env_full.visited_topics = set(SKILLS_ORDER)
        for s in SKILLS_ORDER:
            env_abl.metrics_cache[s]["p_med"] = 0.80
            env_abl.metrics_cache[s]["p_30d"] = 0.80
            env_abl.metrics_cache[s]["floor_b"] = 0.80
            env_full.metrics_cache[s]["p_med"] = 0.80
            env_full.metrics_cache[s]["p_30d"] = 0.80
            env_full.metrics_cache[s]["floor_b"] = 0.80

        obs_before = env_abl._observation().copy()
        mask_before = env_abl.valid_action_mask().copy()
        self.assertFalse(mask_before[6])  # prereq p_med=0.80 >= 0.60, visited >= 0.55
        self.assertFalse(env_full.valid_action_mask()[6])

        # Mutate p_30d, floor_b, p_7d, fragility on all skills: ablated obs & mask must NOT change!
        for s in SKILLS_ORDER:
            env_abl.metrics_cache[s]["p_30d"] = 0.05
            env_abl.metrics_cache[s]["floor_b"] = 0.05
            env_abl.metrics_cache[s]["p_7d"] = 0.10
            env_abl.metrics_cache[s]["fragility"] = 0.75
            env_full.metrics_cache[s]["p_30d"] = 0.05
            env_full.metrics_cache[s]["floor_b"] = 0.05

        self.assertTrue(np.array_equal(env_abl._observation(), obs_before))
        self.assertTrue(np.array_equal(env_abl.valid_action_mask(), mask_before))
        self.assertTrue(env_full.valid_action_mask()[6])  # full env DOES respond to p_30d/floor_b

        # Verify Action 6 & Action 7 target selection in no_longterm_state=True uses p_med, not p_30d
        env_abl.attempts_on_topic["loop_tracing"] = 1
        env_abl.metrics_cache["range_boundaries"]["p_med"] = 0.50  # lowest p_med prereq (< 0.60)
        env_abl.metrics_cache["range_boundaries"]["p_30d"] = 0.99  # highest p_30d
        env_abl.metrics_cache["list_indexing"]["p_med"] = 0.52     # higher p_med than range_boundaries
        env_abl.metrics_cache["list_indexing"]["p_30d"] = 0.01     # lowest p_30d
        self.assertTrue(env_abl.valid_action_mask()[6])
        env_abl.step(6)
        self.assertEqual(SKILLS_ORDER[env_abl.topic_idx], "range_boundaries")

        # Now test Action 7 (all visited) routes to lowest p_med, ignoring lowest p_30d
        env_abl.reset(seed=92)
        env_abl.topic_idx = 0  # range_boundaries
        env_abl.visited_topics = set(SKILLS_ORDER)
        env_abl.attempts_on_topic["range_boundaries"] = 1
        for s in SKILLS_ORDER:
            env_abl.metrics_cache[s]["p_med"] = 0.85
            env_abl.metrics_cache[s]["p_30d"] = 0.90
        env_abl.metrics_cache["function_returns"]["p_med"] = 0.25  # lowest p_med
        env_abl.metrics_cache["function_returns"]["p_30d"] = 0.95  # highest p_30d
        env_abl.metrics_cache["loop_tracing"]["p_med"] = 0.70
        env_abl.metrics_cache["loop_tracing"]["p_30d"] = 0.02      # lowest p_30d
        env_abl.step(7)
        self.assertEqual(SKILLS_ORDER[env_abl.topic_idx], "function_returns")

    def test_summarize_and_bootstrap_diff_defaults(self):
        s = _summarize([0.1, 0.3, 0.2])
        self.assertAlmostEqual(s["mean"], 0.2)
        self.assertAlmostEqual(s["min"], 0.1)
        self.assertAlmostEqual(s["max"], 0.3)
        self.assertIsNotNone(s["sd"])

        res = _bootstrap_diff([0.6, 0.7, 0.8], [0.5, 0.5, 0.6], seed=20261201)
        self.assertEqual(res["bootstrap_draws"], 2000)
        self.assertLessEqual(res["lower_95"], res["delta"])
        self.assertLessEqual(res["delta"], res["upper_95"])


if __name__ == "__main__":
    unittest.main()
