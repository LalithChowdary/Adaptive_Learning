# pilot dataset audit

Status: **passed**.

200 learners; 24 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 20 | 13086 | 3634 | 2333 | 332 |
| train | 160 | 103342 | 28801 | 18564 | 0 |
| validation | 20 | 13209 | 3659 | 2363 | 335 |

Training learners' assessment outcomes are retained in raw assessment files for audit only; they are absent from model fitting exports.

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 115 | 74 |
| test/7d | 105 | 58 |
| test/30d | 112 | 35 |
| train/1d | 887 | 563 |
| train/7d | 898 | 444 |
| train/30d | 888 | 253 |
| validation/1d | 110 | 73 |
| validation/7d | 115 | 65 |
| validation/30d | 110 | 33 |

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
