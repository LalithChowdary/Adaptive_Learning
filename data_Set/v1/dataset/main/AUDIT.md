# main dataset audit

Status: **passed**.

2,000 learners; 17 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 200 | 92866 | 26797 | 18261 | 3418 |
| train | 1600 | 739366 | 213117 | 146035 | 0 |
| validation | 200 | 91947 | 26485 | 18284 | 3424 |

Training learners' assessment outcomes are retained in raw assessment files for audit only; they are absent from model fitting exports.

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 1140 | 813 |
| test/7d | 1135 | 642 |
| test/30d | 1143 | 344 |
| train/1d | 9116 | 6314 |
| train/7d | 9111 | 5110 |
| train/30d | 9081 | 2614 |
| validation/1d | 1151 | 816 |
| validation/7d | 1141 | 656 |
| validation/30d | 1132 | 315 |

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
