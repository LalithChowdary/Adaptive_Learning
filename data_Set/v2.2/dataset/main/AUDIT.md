# main dataset audit

Status: **passed**.

2,000 learners; 28 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 200 | 137161 | 37784 | 23922 | 3313 |
| train | 1600 | 1113623 | 306291 | 192779 | 26505 |
| validation | 200 | 139331 | 38284 | 24086 | 3327 |

Delayed training labels use only the train learners' frozen practice histories.

## Labeled gaps from the last model evidence

| Partition/phase | Non-cold queries | Cold starts | Median days | 99th percentile | Max days | Over 5 days | Over 30 days |
|---|---:|---:|---:|---:|---:|---:|---:|
| test/practice | 22830 | 1092 | 0.001 | 47.05 | 83.14 | 2050 | 646 |
| test/assessment | 3313 | 0 | 7.009 | 56.97 | 88.29 | 2309 | 982 |
| train/practice | 183908 | 8871 | 0.001 | 48.51 | 111.28 | 16510 | 5150 |
| train/assessment | 26505 | 0 | 7.009 | 60.74 | 89.50 | 18348 | 7880 |
| validation/practice | 22976 | 1110 | 0.001 | 47.51 | 111.08 | 2144 | 609 |
| validation/assessment | 3327 | 0 | 7.008 | 62.71 | 89.01 | 2296 | 992 |

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 1110 | 698 |
| test/7d | 1106 | 694 |
| test/30d | 1097 | 622 |
| train/1d | 8857 | 5390 |
| train/7d | 8853 | 5105 |
| train/30d | 8795 | 4751 |
| validation/1d | 1116 | 706 |
| validation/7d | 1113 | 661 |
| validation/30d | 1098 | 606 |

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
