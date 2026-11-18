# PPO Adaptive Teaching Policy (`3/ppo/`)

Phase 1 & Phase 2 implementation of the closed-loop **8-action PPO Teaching Policy** using the frozen `v2.2` Monotonic Grey-Box GRU (`monotonic_gru`, `seed_20261122/model.pt`).

## Structure

| File | Responsibility |
|---|---|
| [`env.py`](env.py) | `AdaptiveTutorEnv` (3-box loop: within-question turn + hint leakage $l_t$, frozen `monotonic_gru` memory & dual-memory `P_now - P_30d` fragility state, 8-action Gated Option B decision space) and `FrozenMemoryTracker` |
| [`agent.py`](agent.py) | `MaskedActorCritic` (invalid-action masking), `compute_gae`, `ppo_update`, and baseline policies (`RandomValidPolicy`, `FixedCurriculumPolicy`, `ThresholdHeuristicPolicy`) |
| [`train_ppo.py`](train_ppo.py) | Multi-seed training (`delayed_retention_ppo` & `myopic_rl_bandit`) and 200-learner held-out evaluation across `main_test`, `power_law_test`, and `long_gap_test` |
| [`tests/test_ppo.py`](tests/test_ppo.py) | Regression tests for observation bounds, Gated Option B action masking, hint leakage, self-explanation consolidation, delayed-clone isolation, and PPO updates |
| [`PPO_GUIDE_AND_METHODS.md`](PPO_GUIDE_AND_METHODS.md) | Detailed guide explaining what a PPO policy is, why it is needed on top of `monotonic_gru`, how training works, and benchmark results |

## Run Tests and Training

From `3/`:

```sh
model/.venv/bin/python -m unittest discover -s ppo/tests -v
model/.venv/bin/python ppo/train_ppo.py --output ppo/runs/v2.2-ppo-main
```
