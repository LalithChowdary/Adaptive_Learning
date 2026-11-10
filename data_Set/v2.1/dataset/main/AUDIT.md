# main dataset audit

Status: **passed**.

2,000 learners; 26 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 200 | 149869 | 40569 | 24782 | 3343 |
| train | 1600 | 1197598 | 324446 | 198651 | 26573 |
| validation | 200 | 151981 | 41089 | 24950 | 3323 |

Delayed training labels use only the train learners' frozen practice histories.

## Labeled gaps from the last model evidence

| Partition/phase | Non-cold queries | Cold starts | Median days | 99th percentile | Max days | Over 5 days | Over 30 days |
|---|---:|---:|---:|---:|---:|---:|---:|
| test/practice | 23686 | 1096 | 0.001 | 46.49 | 115.02 | 2076 | 639 |
| test/assessment | 3343 | 0 | 7.013 | 57.84 | 86.85 | 2304 | 1001 |
| train/practice | 189820 | 8831 | 0.001 | 49.16 | 115.54 | 17058 | 5111 |
| train/assessment | 26573 | 0 | 7.012 | 59.58 | 88.74 | 18399 | 7860 |
| validation/practice | 23853 | 1097 | 0.001 | 46.70 | 104.52 | 2056 | 579 |
| validation/assessment | 3323 | 0 | 7.012 | 51.86 | 80.50 | 2312 | 959 |

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 1114 | 589 |
| test/7d | 1111 | 451 |
| test/30d | 1118 | 253 |
| train/1d | 8859 | 4357 |
| train/7d | 8865 | 3575 |
| train/30d | 8849 | 2296 |
| validation/1d | 1099 | 555 |
| validation/7d | 1124 | 470 |
| validation/30d | 1100 | 270 |

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
