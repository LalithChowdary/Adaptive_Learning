# power_law dataset audit

Status: **passed**.

200 learners; 28 validation categories passed.

| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |
|---|---:|---:|---:|---:|---:|
| test | 200 | 138839 | 38185 | 24053 | 3323 |

Delayed training labels use only the train learners' frozen practice histories.

## Labeled gaps from the last model evidence

| Partition/phase | Non-cold queries | Cold starts | Median days | 99th percentile | Max days | Over 5 days | Over 30 days |
|---|---:|---:|---:|---:|---:|---:|---:|
| test/practice | 22945 | 1108 | 0.001 | 50.85 | 115.56 | 2109 | 677 |
| test/assessment | 3323 | 0 | 7.009 | 56.61 | 88.57 | 2292 | 980 |

## Delayed outcomes (descriptive)

| Partition/delay | Eligible | Full successes |
|---|---:|---:|
| test/1d | 1105 | 643 |
| test/7d | 1102 | 615 |
| test/30d | 1116 | 574 |

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
