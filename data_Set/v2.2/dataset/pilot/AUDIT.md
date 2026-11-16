# pilot dataset audit

Status: **passed**.

200 learners; 28 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 20 | 13082 | 3629 | 2317 | 339 |
| train | 160 | 111824 | 30743 | 19370 | 2649 |
| validation | 20 | 14378 | 3923 | 2441 | 331 |

Delayed training labels use only the train learners' frozen practice histories.

## Labeled gaps from the last model evidence

| Partition/phase | Non-cold queries | Cold starts | Median days | 99th percentile | Max days | Over 5 days | Over 30 days |
|---|---:|---:|---:|---:|---:|---:|---:|
| test/practice | 2204 | 113 | 0.001 | 48.32 | 76.32 | 207 | 70 |
| test/assessment | 339 | 0 | 7.009 | 48.78 | 61.95 | 238 | 99 |
| train/practice | 18485 | 885 | 0.001 | 50.54 | 96.69 | 1778 | 525 |
| train/assessment | 2649 | 0 | 7.009 | 51.76 | 81.84 | 1846 | 780 |
| validation/practice | 2332 | 109 | 0.001 | 41.18 | 60.36 | 198 | 46 |
| validation/assessment | 331 | 0 | 7.008 | 38.41 | 42.20 | 226 | 90 |

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 114 | 75 |
| test/7d | 111 | 69 |
| test/30d | 114 | 60 |
| train/1d | 879 | 529 |
| train/7d | 874 | 529 |
| train/30d | 896 | 478 |
| validation/1d | 114 | 57 |
| validation/7d | 107 | 63 |
| validation/30d | 110 | 58 |

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
- Passed: unaided zero-score failures grant no retained learning gain.
- Passed: duolingo HLR half-life growth and consolidated baseline retention.

## Interpretation

Internal consistency of synthetic data, not human fidelity or model effectiveness.

Independent branches share a learner's practice history; they are correlated, not extra independent learners. No training or PPO effectiveness result is reported.
