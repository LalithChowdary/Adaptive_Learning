# PPO Adaptive Teaching Policy Benchmark (`v2.2` Frozen `monotonic_gru`)

Mode: `full` · Seeds: `[20261120, 20261121, 20261122]` · Test learners per cohort: `200` · Budget: `32.0`.

## main_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 64.33% | 61.81% | 54.72% | 60.29% | 60.55% | 0.3820 |
| fixed_curriculum | 64.22% | 60.28% | 51.47% | 58.66% | 58.61% | 0.3228 |
| threshold_heuristic | 66.08% | 63.50% | 54.58% | 61.39% | 61.69% | 0.3321 |
| myopic_rl_bandit | 59.25% ± 0.05% | 56.40% ± 0.11% | 50.21% ± 0.23% | 55.29% ± 0.13% | 55.81% ± 0.09% | 0.3699 ± 0.0026 |
| delayed_retention_ppo | 66.36% ± 1.83% | 64.43% ± 1.72% | 59.44% ± 1.96% | 63.41% ± 1.83% | 63.53% ± 1.74% | 0.4388 ± 0.0167 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 10.1% | 22.1% | 18.9% | 11.9% | 11.9% | 8.7% | 7.5% | 8.8% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.5% | 58.3% | 17.6% | 13.2% | 0.0% | 0.0% | 0.0% | 10.4% |
| myopic_rl_bandit | 0.0% | 35.9% | 0.0% | 0.0% | 0.0% | 64.1% | 0.0% | 0.0% |
| delayed_retention_ppo | 5.1% | 5.1% | 11.4% | 2.4% | 1.4% | 47.8% | 20.6% | 6.1% |

## power_law_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 63.64% | 61.47% | 55.75% | 60.29% | 60.49% | 0.3810 |
| fixed_curriculum | 63.31% | 59.11% | 52.83% | 58.42% | 58.47% | 0.3227 |
| threshold_heuristic | 65.42% | 62.75% | 56.36% | 61.51% | 61.73% | 0.3325 |
| myopic_rl_bandit | 58.69% ± 0.06% | 56.44% ± 0.12% | 51.70% ± 0.22% | 55.61% ± 0.13% | 56.07% ± 0.10% | 0.3697 ± 0.0028 |
| delayed_retention_ppo | 65.93% ± 1.81% | 64.16% ± 1.73% | 60.12% ± 1.82% | 63.40% ± 1.78% | 63.45% ± 1.69% | 0.4373 ± 0.0171 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 10.1% | 22.1% | 18.9% | 11.9% | 11.9% | 8.7% | 7.5% | 8.8% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.4% | 58.6% | 17.3% | 13.5% | 0.0% | 0.0% | 0.0% | 10.1% |
| myopic_rl_bandit | 0.0% | 36.0% | 0.0% | 0.0% | 0.0% | 64.0% | 0.0% | 0.0% |
| delayed_retention_ppo | 5.2% | 5.2% | 11.4% | 2.4% | 1.4% | 47.7% | 20.6% | 6.0% |

## long_gap_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 60.64% | 59.14% | 53.86% | 57.88% | 58.33% | 0.3778 |
| fixed_curriculum | 59.94% | 56.31% | 49.56% | 55.27% | 55.53% | 0.3222 |
| threshold_heuristic | 62.14% | 60.03% | 53.53% | 58.56% | 59.14% | 0.3337 |
| myopic_rl_bandit | 55.56% ± 0.06% | 53.69% ± 0.17% | 49.31% ± 0.25% | 52.86% ± 0.16% | 53.62% ± 0.14% | 0.3685 ± 0.0037 |
| delayed_retention_ppo | 63.04% ± 2.33% | 61.72% ± 2.32% | 57.88% ± 2.49% | 60.88% ± 2.38% | 61.11% ± 2.39% | 0.4278 ± 0.0173 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 10.1% | 22.0% | 19.1% | 11.9% | 11.8% | 8.9% | 7.4% | 8.8% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.5% | 61.6% | 16.6% | 14.3% | 0.0% | 0.0% | 0.0% | 7.1% |
| myopic_rl_bandit | 0.0% | 37.8% | 0.0% | 0.0% | 0.0% | 62.2% | 0.0% | 0.0% |
| delayed_retention_ppo | 5.2% | 5.3% | 10.9% | 2.1% | 1.4% | 49.4% | 20.2% | 5.5% |

