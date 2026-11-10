# Python learner-memory dataset v2.1

Version 2.1.0 adds training evidence at longer elapsed gaps. The v2.0 experiment had 185,510 eligible training targets, with a maximum gap of 3.06 days since the last model evidence. Its 7- and 30-day validation assessments therefore asked the model to extrapolate beyond the fitted target gaps. V2.1 retains the [v2 task, grading, and event contract](../v2/DATASET_DOCUMENTATION.md) and changes the practice calendar and fitting export.

## Scheduling and fitting labels

- A mixed round still visits each skill once. A blocked round now visits each skill in four consecutive ordinary tasks and revisits each skill in later rounds. Every learner has 16 ordinary tasks per skill plus the existing fresh follow-ups when applicable.
- Between sessions, pilot, main, and power-law learners draw a 5–60-day gap with probability 0.30, uniform on the log scale. The remaining gaps are sampled from 0.25, 1, and 3 days. The long-gap cohort uses the original equal-weight mixture of 0.25, 1, 3, 14, and 60 days, with more 60-day gaps. Gaps from the last evidence for a given skill can accumulate over several sessions.
- Train learners contribute eligible ordinary and follow-up practice labels plus their 1-, 7-, and 30-day assessment labels. Each assessment query reads the same frozen practice history for that learner and skill. An assessment outcome cannot enter that history or affect another assessment branch.
- Validation and test learners remain disjoint from fitting learners. The power-law and long-gap cohorts are test-only. The power-law cohort changes the simulator's retention law while using the same practice calendar as main. The task catalog and assessment bank are shared across learner splits.
- The raw trace now stores `strength_at_attempt` separately from `strength_before`, which was measured after response time elapsed. This field stays in private administration records and supports an evaluation-only simulator reference. Response time still depends on latent strength through the generator's timing formula, so timing carries predictive signal by construction.

The clock in the gap audit uses the last hybrid evidence token, including grades, hints, and reveals. Cold starts have no evidence clock and appear separately. The 1/7/30-day labels specify the time after the end of the learner's practice sequence. An individual skill's latest evidence can be earlier.

## Verified coverage

| Main training measure | Count |
|---|---:|
| Learners | 1,600 |
| Eligible practice labels | 198,651 |
| Eligible delayed labels | 26,573 |
| Total fitting labels | 225,224 |
| Practice labels over 5 days since same-skill evidence | 17,058 |
| Practice labels over 30 days since same-skill evidence | 5,111 |

The largest labeled training practice gap is 115.54 days since same-skill evidence. All four cohort audits passed 26 categories each, and all 55 dataset regression tests passed. The [main audit](dataset/main/AUDIT.md) has full gap statistics and horizon counts. The [release summary](reports/RELEASE_SUMMARY.json) identifies the three empty assessment coverage cells in the small pilot. Main and both stress cohorts have complete nominal horizon × skill × format query coverage.

## Reproduce

From the project root, using the existing environment:

```sh
3/model/.venv/bin/python 3/data_Set/v2.1/run_tests.py
3/model/.venv/bin/python 3/data_Set/v2.1/build_dataset.py --cohort all --output /tmp/learner-memory-v21-reproduction
```

The build refuses to overwrite cohort folders and requires a passing pilot before generating larger cohorts. The existing cohort manifest, audit, and test artifacts are in this release. `release.py` records checksums for the audited release and verifies the preservation inventory for v1 and v2.0. [Model training guide](../../model/README.md)
