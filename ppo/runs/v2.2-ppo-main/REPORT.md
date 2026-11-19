# PPO Adaptive Teaching Policy Benchmark (`v2.2` Frozen `monotonic_gru`)

Mode: `full` · Seeds: `[20261120, 20261121, 20261122]` · Test learners per cohort: `200` · Budget: `32.0`.

## main_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 65.22% | 62.83% | 55.78% | 61.28% | 61.24% | 0.3856 |
| fixed_curriculum | 64.22% | 60.28% | 51.47% | 58.66% | 58.61% | 0.3228 |
| threshold_heuristic | 66.08% | 63.50% | 54.58% | 61.39% | 61.69% | 0.3321 |
| myopic_rl_bandit | 59.02% ± 0.04% | 56.08% ± 0.03% | 49.74% ± 0.03% | 54.95% ± 0.02% | 55.51% ± 0.01% | 0.3609 ± 0.0001 |
| delayed_retention_ppo | 66.05% ± 0.30% | 65.20% ± 0.75% | 61.69% ± 1.49% | 64.31% ± 0.78% | 64.62% ± 0.71% | 0.4655 ± 0.0150 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 9.7% | 23.5% | 19.2% | 12.3% | 12.1% | 8.2% | 5.8% | 9.2% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.5% | 58.3% | 17.6% | 13.2% | 0.0% | 0.0% | 0.0% | 10.4% |
| myopic_rl_bandit | 0.0% | 69.3% | 0.0% | 0.0% | 0.0% | 30.7% | 0.0% | 0.0% |
| delayed_retention_ppo | 8.7% | 9.4% | 21.6% | 4.7% | 13.6% | 30.6% | 4.6% | 6.9% |

### 5-Check PPO Behavioral Sanity Suite (`delayed_retention_ppo`)

| Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%-25%`) | Floor > Heuristic | Beats All Baselines |
|---|---:|---:|---:|---:|---:|---:|
| `20261120` | 5 / 5 | Pass (43.5%) | Pass | Pass | Pass | Pass |
| `20261121` | 5 / 5 | Pass (46.8%) | Pass | Pass | Pass | Pass |
| `20261122` | 5 / 5 | Pass (42.6%) | Pass | Pass | Pass | Pass |

## power_law_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 64.47% | 62.22% | 57.19% | 61.30% | 61.13% | 0.3848 |
| fixed_curriculum | 63.31% | 59.11% | 52.83% | 58.42% | 58.47% | 0.3227 |
| threshold_heuristic | 65.42% | 62.75% | 56.36% | 61.51% | 61.73% | 0.3325 |
| myopic_rl_bandit | 58.45% ± 0.02% | 56.12% ± 0.04% | 51.19% ± 0.03% | 55.25% ± 0.02% | 55.76% ± 0.00% | 0.3604 ± 0.0002 |
| delayed_retention_ppo | 65.73% ± 0.38% | 64.84% ± 0.61% | 62.00% ± 1.16% | 64.19% ± 0.66% | 64.51% ± 0.60% | 0.4633 ± 0.0142 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 9.7% | 23.6% | 19.2% | 12.3% | 12.1% | 8.1% | 5.8% | 9.2% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.4% | 58.6% | 17.3% | 13.5% | 0.0% | 0.0% | 0.0% | 10.1% |
| myopic_rl_bandit | 0.0% | 69.4% | 0.0% | 0.0% | 0.0% | 30.6% | 0.0% | 0.0% |
| delayed_retention_ppo | 8.7% | 9.4% | 21.6% | 4.7% | 13.6% | 30.6% | 4.6% | 6.8% |

### 5-Check PPO Behavioral Sanity Suite (`delayed_retention_ppo`)

| Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%-25%`) | Floor > Heuristic | Beats All Baselines |
|---|---:|---:|---:|---:|---:|---:|
| `20261120` | 5 / 5 | Pass (43.5%) | Pass | Pass | Pass | Pass |
| `20261121` | 5 / 5 | Pass (46.9%) | Pass | Pass | Pass | Pass |
| `20261122` | 5 / 5 | Pass (42.6%) | Pass | Pass | Pass | Pass |

## long_gap_test

| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |
|---|---:|---:|---:|---:|---:|---:|
| random_valid | 61.39% | 59.56% | 54.50% | 58.48% | 58.65% | 0.3766 |
| fixed_curriculum | 59.94% | 56.31% | 49.56% | 55.27% | 55.53% | 0.3222 |
| threshold_heuristic | 62.14% | 60.03% | 53.53% | 58.56% | 59.14% | 0.3337 |
| myopic_rl_bandit | 55.31% ± 0.06% | 53.38% ± 0.06% | 48.73% ± 0.11% | 52.48% ± 0.07% | 53.30% ± 0.02% | 0.3589 ± 0.0006 |
| delayed_retention_ppo | 63.36% ± 0.39% | 62.80% ± 0.64% | 60.00% ± 1.06% | 62.05% ± 0.63% | 62.59% ± 0.66% | 0.4523 ± 0.0114 |

### Action Distribution (`delayed_retention_ppo` vs. Baselines)

| Policy | give_easier_question | give_same_difficulty_question | give_harder_question | explain_concept | show_worked_example | request_self_explanation | review_prerequisite | advance_next_topic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_valid | 9.8% | 23.4% | 19.4% | 12.2% | 12.1% | 8.0% | 6.1% | 9.0% |
| fixed_curriculum | 0.0% | 21.4% | 38.1% | 21.4% | 0.0% | 0.0% | 0.0% | 19.0% |
| threshold_heuristic | 0.5% | 61.6% | 16.6% | 14.3% | 0.0% | 0.0% | 0.0% | 7.1% |
| myopic_rl_bandit | 0.0% | 70.3% | 0.0% | 0.0% | 0.0% | 29.7% | 0.0% | 0.0% |
| delayed_retention_ppo | 8.8% | 9.7% | 21.2% | 4.8% | 13.9% | 30.3% | 4.8% | 6.5% |

### 5-Check PPO Behavioral Sanity Suite (`delayed_retention_ppo`)

| Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%-25%`) | Floor > Heuristic | Beats All Baselines |
|---|---:|---:|---:|---:|---:|---:|
| `20261120` | 5 / 5 | Pass (43.7%) | Pass | Pass | Pass | Pass |
| `20261121` | 5 / 5 | Pass (47.1%) | Pass | Pass | Pass | Pass |
| `20261122` | 5 / 5 | Pass (42.7%) | Pass | Pass | Pass | Pass |

