# PPO Adaptive Teaching Policy Benchmark (`v2.2` Frozen `monotonic_gru`)

Mode: `full` · Seeds: `[20261201, 20261202, 20261203, 20261204, 20261205, 20261206, 20261207, 20261208, 20261209, 20261210]` · Test learners per cohort: `400` · Budget: `32.0`.

## main_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 63.65% | 60.32% | 54.72% | 59.56% | 59.42% | 0.3702 |
| fixed_curriculum | 62.39% | 57.76% | 49.64% | 56.60% | 56.75% | 0.3129 |
| threshold_heuristic | 65.17% | 61.68% | 53.97% | 60.27% | 60.30% | 0.3239 |
| myopic_rl_bandit | 57.60% ± 0.02% | 55.05% ± 0.03% | 49.28% ± 0.02% | 53.98% ± 0.01% | 53.94% ± 0.01% | 0.3525 ± 0.0009 |
| delayed_retention_ppo | 65.39% ± 0.69% | 63.26% ± 1.02% | 59.66% ± 1.92% | 62.77% ± 1.12% | 62.76% ± 1.12% | 0.4268 ± 0.0302 |
| ppo_no_longterm_state | 64.20% ± 2.68% | 62.24% ± 2.99% | 58.82% ± 4.24% | 61.76% ± 3.28% | 61.77% ± 3.31% | 0.4214 ± 0.0400 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 9.5% | 22.7% | 19.5% | 12.4% | 11.4% | 8.1% | 6.8% | 9.6% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.5% | 60.4% | 16.7% | 12.7% | 0.0% | 0.0% | 0.0% | 9.7% |
| myopic_rl_bandit | 0.0% | 71.9% | 0.0% | 0.0% | 0.0% | 28.1% | 0.0% | 0.0% |
| delayed_retention_ppo | 3.8% | 25.8% | 15.1% | 5.4% | 16.4% | 16.9% | 8.1% | 8.5% |
| ppo_no_longterm_state | 7.5% | 26.5% | 18.4% | 7.0% | 6.1% | 20.7% | 4.4% | 9.4% |

### 5-Check PPO Behavioral Sanity Suite (`delayed_retention_ppo`)

| Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%-25%`) | Floor > Heuristic | Beats All Baselines |
|---|---:|---:|---:|---:|---:|---:|
| `20261201` | 5 / 5 | Pass (59.1%) | Pass | Pass | Pass | Pass |
| `20261202` | 5 / 5 | Pass (52.2%) | Pass | Pass | Pass | Pass |
| `20261203` | 5 / 5 | Pass (43.8%) | Pass | Pass | Pass | Pass |
| `20261204` | 5 / 5 | Pass (44.7%) | Pass | Pass | Pass | Pass |
| `20261205` | 5 / 5 | Pass (52.7%) | Pass | Pass | Pass | Pass |
| `20261206` | 4 / 5 | Fail (38.7%) | Pass | Pass | Pass | Pass |
| `20261207` | 3 / 5 | Pass (51.4%) | Fail | Fail | Pass | Pass |
| `20261208` | 5 / 5 | Pass (78.7%) | Pass | Pass | Pass | Pass |
| `20261209` | 5 / 5 | Pass (53.6%) | Pass | Pass | Pass | Pass |
| `20261210` | 5 / 5 | Pass (52.8%) | Pass | Pass | Pass | Pass |

## power_law_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 62.61% | 59.75% | 55.56% | 59.31% | 59.19% | 0.3681 |
| fixed_curriculum | 61.47% | 56.92% | 51.15% | 56.51% | 56.57% | 0.3124 |
| threshold_heuristic | 64.46% | 61.18% | 55.60% | 60.41% | 60.41% | 0.3244 |
| myopic_rl_bandit | 57.21% ± 0.02% | 55.08% ± 0.02% | 50.64% ± 0.03% | 54.31% ± 0.02% | 54.18% ± 0.01% | 0.3519 ± 0.0009 |
| delayed_retention_ppo | 64.92% ± 0.68% | 62.93% ± 0.94% | 60.29% ± 1.58% | 62.71% ± 1.01% | 62.66% ± 1.04% | 0.4250 ± 0.0294 |
| ppo_no_longterm_state | 63.76% ± 2.72% | 62.00% ± 2.87% | 59.44% ± 3.87% | 61.73% ± 3.13% | 61.71% ± 3.19% | 0.4197 ± 0.0391 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 9.5% | 22.6% | 19.5% | 12.4% | 11.4% | 8.1% | 6.9% | 9.7% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.5% | 60.4% | 16.5% | 12.9% | 0.0% | 0.0% | 0.0% | 9.7% |
| myopic_rl_bandit | 0.0% | 72.0% | 0.0% | 0.0% | 0.0% | 28.0% | 0.0% | 0.0% |
| delayed_retention_ppo | 3.8% | 25.9% | 15.0% | 5.4% | 16.4% | 16.9% | 8.1% | 8.5% |
| ppo_no_longterm_state | 7.6% | 26.5% | 18.3% | 7.1% | 6.1% | 20.7% | 4.4% | 9.4% |

### 5-Check PPO Behavioral Sanity Suite (`delayed_retention_ppo`)

| Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%-25%`) | Floor > Heuristic | Beats All Baselines |
|---|---:|---:|---:|---:|---:|---:|
| `20261201` | 5 / 5 | Pass (59.2%) | Pass | Pass | Pass | Pass |
| `20261202` | 5 / 5 | Pass (52.3%) | Pass | Pass | Pass | Pass |
| `20261203` | 5 / 5 | Pass (43.8%) | Pass | Pass | Pass | Pass |
| `20261204` | 5 / 5 | Pass (44.7%) | Pass | Pass | Pass | Pass |
| `20261205` | 5 / 5 | Pass (52.8%) | Pass | Pass | Pass | Pass |
| `20261206` | 4 / 5 | Fail (38.8%) | Pass | Pass | Pass | Pass |
| `20261207` | 3 / 5 | Pass (51.5%) | Fail | Fail | Pass | Pass |
| `20261208` | 5 / 5 | Pass (78.7%) | Pass | Pass | Pass | Pass |
| `20261209` | 5 / 5 | Pass (53.5%) | Pass | Pass | Pass | Pass |
| `20261210` | 5 / 5 | Pass (52.8%) | Pass | Pass | Pass | Pass |

## long_gap_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 60.18% | 57.32% | 53.36% | 56.95% | 56.83% | 0.3632 |
| fixed_curriculum | 57.97% | 54.50% | 48.36% | 53.61% | 53.64% | 0.3126 |
| threshold_heuristic | 61.74% | 58.69% | 52.97% | 57.80% | 57.69% | 0.3251 |
| myopic_rl_bandit | 54.79% ± 0.02% | 52.80% ± 0.03% | 48.43% ± 0.04% | 52.01% ± 0.02% | 51.78% ± 0.01% | 0.3497 ± 0.0025 |
| delayed_retention_ppo | 62.82% ± 0.82% | 61.04% ± 0.99% | 58.37% ± 1.74% | 60.74% ± 1.13% | 60.65% ± 1.16% | 0.4169 ± 0.0264 |
| ppo_no_longterm_state | 61.72% ± 2.95% | 59.96% ± 3.03% | 57.38% ± 4.04% | 59.69% ± 3.33% | 59.63% ± 3.38% | 0.4120 ± 0.0355 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 9.4% | 22.6% | 19.5% | 12.4% | 11.4% | 8.0% | 7.0% | 9.7% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.5% | 63.2% | 15.5% | 13.6% | 0.0% | 0.0% | 0.0% | 7.3% |
| myopic_rl_bandit | 0.0% | 73.1% | 0.0% | 0.0% | 0.0% | 26.9% | 0.0% | 0.0% |
| delayed_retention_ppo | 3.9% | 26.4% | 14.8% | 5.5% | 16.4% | 16.9% | 8.2% | 7.9% |
| ppo_no_longterm_state | 7.6% | 26.0% | 18.0% | 7.0% | 6.1% | 20.8% | 5.7% | 8.9% |

### 5-Check PPO Behavioral Sanity Suite (`delayed_retention_ppo`)

| Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%-25%`) | Floor > Heuristic | Beats All Baselines |
|---|---:|---:|---:|---:|---:|---:|
| `20261201` | 5 / 5 | Pass (60.3%) | Pass | Pass | Pass | Pass |
| `20261202` | 5 / 5 | Pass (52.6%) | Pass | Pass | Pass | Pass |
| `20261203` | 5 / 5 | Pass (44.0%) | Pass | Pass | Pass | Pass |
| `20261204` | 5 / 5 | Pass (45.8%) | Pass | Pass | Pass | Pass |
| `20261205` | 5 / 5 | Pass (53.0%) | Pass | Pass | Pass | Pass |
| `20261206` | 4 / 5 | Fail (39.2%) | Pass | Pass | Pass | Pass |
| `20261207` | 3 / 5 | Pass (52.2%) | Fail | Fail | Pass | Pass |
| `20261208` | 5 / 5 | Pass (78.6%) | Pass | Pass | Pass | Pass |
| `20261209` | 4 / 5 | Pass (53.7%) | Fail | Pass | Pass | Pass |
| `20261210` | 5 / 5 | Pass (53.1%) | Pass | Pass | Pass | Pass |

