# main dataset audit

Status: **passed**.

2,000 learners; 24 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 200 | 127659 | 35628 | 23122 | 3309 |
| train | 1600 | 1032864 | 287382 | 185510 | 0 |
| validation | 200 | 128423 | 35783 | 23101 | 3319 |

Training learners' assessment outcomes are retained in raw assessment files for audit only; they are absent from model fitting exports.

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 1096 | 678 |
| test/7d | 1108 | 547 |
| test/30d | 1105 | 370 |
| train/1d | 8882 | 5413 |
| train/7d | 8831 | 4486 |
| train/30d | 8838 | 2552 |
| validation/1d | 1118 | 722 |
| validation/7d | 1094 | 594 |
| validation/30d | 1107 | 348 |

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
- Passed: no delayed fitting labels.
- Passed: model feature allowlist.
- Passed: no missing/extra observation, model, assessment or administration learner rows.

## Interpretation

Internal consistency of synthetic data, not human fidelity or model effectiveness.

Independent branches share a learner's practice history; they are correlated, not extra independent learners. No training or PPO effectiveness result is reported.
