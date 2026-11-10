# pilot dataset audit

Status: **passed**.

200 learners; 26 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 20 | 14430 | 3932 | 2441 | 340 |
| train | 160 | 117121 | 31862 | 19671 | 2667 |
| validation | 20 | 14777 | 4011 | 2455 | 335 |

Delayed training labels use only the train learners' frozen practice histories.

## Labeled gaps from the last model evidence

| Partition/phase | Non-cold queries | Cold starts | Median days | 99th percentile | Max days | Over 5 days | Over 30 days |
|---|---:|---:|---:|---:|---:|---:|---:|
| test/practice | 2327 | 114 | 0.001 | 38.72 | 58.74 | 212 | 41 |
| test/assessment | 340 | 0 | 7.010 | 33.05 | 40.75 | 226 | 93 |
| train/practice | 18792 | 879 | 0.001 | 49.01 | 92.12 | 1616 | 530 |
| train/assessment | 2667 | 0 | 7.010 | 49.56 | 78.59 | 1835 | 762 |
| validation/practice | 2345 | 110 | 0.001 | 50.53 | 70.13 | 210 | 43 |
| validation/assessment | 335 | 0 | 7.010 | 39.57 | 62.58 | 225 | 98 |

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 117 | 64 |
| test/7d | 112 | 46 |
| test/30d | 111 | 33 |
| train/1d | 897 | 470 |
| train/7d | 887 | 401 |
| train/30d | 883 | 243 |
| validation/1d | 113 | 48 |
| validation/7d | 110 | 37 |
| validation/30d | 112 | 39 |

## Checks

- Passed: question-first default with no routine instruction.
- Passed: fresh followup for every assisted practice task.
- Passed: configurable caps and per-hint/per-retry chronology.
- Passed: assessment interruption and review semantics.
- Passed: five-format skill coverage.
- Passed: shared task and lesson catalog checksums.
- Passed: hidden trace attribution and bounded response-only gains.
- Passed: file checksums.
- Passed: generation source fingerprints.
- Passed: unique disjoint learner assignments.
- Passed: balanced declared schedules.
- Passed: six-skill budgets.
- Passed: strict event schema.
- Passed: event and grade causality.
- Passed: assistance consistency.
- Passed: rubric derivation and unresolved labels.
- Passed: simulated grading provenance.
- Passed: independent delayed assessment branches.
- Passed: assessment delays and coverage.
- Passed: separate task pools.
- Passed: all prepared records match causal replay.
- Passed: delayed fitting labels use frozen practice prefixes.
- Passed: spaced training gap and horizon coverage.
- Passed: stress cohorts are test-only.
- Passed: model feature allowlist.
- Passed: no missing/extra observation, model, assessment or administration learner rows.

## Interpretation

Internal consistency of synthetic data, not human fidelity or model effectiveness.

Independent branches share a learner's practice history; they are correlated, not extra independent learners. No training or PPO effectiveness result is reported.
