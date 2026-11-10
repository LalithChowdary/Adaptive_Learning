# pilot dataset audit

Status: **passed**.

200 learners; 17 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 20 | 9207 | 2650 | 1831 | 345 |
| train | 160 | 73513 | 21177 | 14579 | 0 |
| validation | 20 | 9212 | 2654 | 1824 | 345 |

Training learners' assessment outcomes are retained in raw assessment files for audit only; they are absent from model fitting exports.

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 114 | 79 |
| test/7d | 115 | 70 |
| test/30d | 116 | 34 |
| train/1d | 917 | 656 |
| train/7d | 912 | 519 |
| train/30d | 906 | 256 |
| validation/1d | 113 | 80 |
| validation/7d | 117 | 62 |
| validation/30d | 115 | 24 |

## Checks

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
- Passed: no delayed fitting labels.
- Passed: model feature allowlist.
- Passed: no missing/extra observation, model, assessment or administration learner rows.

## Interpretation

Internal consistency of synthetic data, not human fidelity or model effectiveness.

Independent branches share a learner's practice history; they are correlated, not extra independent learners. No training or PPO effectiveness result is reported.
