# Python learner-memory dataset v2

This is the active synthetic dataset release for the first Python memory experiment. V1 is archived in [the sibling v1 folder](../v1/README.md). Use the `dataset/main` directory **inside this v2 folder** for the next memory experiment.

Read [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md) for the complete methods, field dictionary, assumptions, examples, verification results, and limitations. Read [the central project record](../../../PROJECT_DECISIONS.md) for current architecture and decisions across chats.

The main changes are question-first practice, five question formats with variable rubrics, and separately logged multiple hints/retries. The default is three hints and two retries per practice question. The target is full success on a later first unaided response. No memory model or PPO policy is trained by these scripts.

Verified release totals are 2,600 learners, 1,700,389 events, and 185,510 eligible main training targets. All 54 tests passed and each of the four cohorts passed 24 audit categories. These checks establish internal synthetic-data consistency.

## Files to inspect first

- `dataset/task_catalog.json` contains all 2,160 authored task variants and their rubrics, hints, reference answers, and behavior tests.
- `examples/controlled_histories.json` demonstrates three hints, fresh follow-up, help requested before answering, interrupted assessment, requested introduction, and assessment followed by review.
- `dataset/main/example_learner.json` contains one generated learner's opening events and one complete structured skill record.
- `dataset/main/model_data/train.jsonl.gz` contains the memory fitting sequences. Load strict historical prefixes through `loader.py`.
- `reports/RELEASE_SUMMARY.json` contains measured totals and pass counts. Each cohort has `AUDIT.md` and detailed `audit.json`.
- `reports/test_results.json` records each executed test's status.

## Reproduction

Run from `/Users/lalith/snu/sem7/minor_pro/code`. The existing release refuses overwrites. Choose a new output directory to reproduce it.

```sh
2/.venv/bin/python 3/data_Set/v2/run_tests.py
2/.venv/bin/python 3/data_Set/v2/build_dataset.py --output /tmp/python_memory_v2_reproduction --cohort all
```

The `all` command builds and audits the pilot before the main and stress cohorts. Alternatively run `--cohort pilot`, followed by `main`, `power_law`, and `long_gap`. Change `--max-hints` from 0 through 4 and `--max-retries` from 0 through 10 for a separate experiment. The authored hint bank currently has four levels; higher hint caps require extending it first.

```sh
2/.venv/bin/python 3/data_Set/v2/audit_dataset.py 3/data_Set/v2/dataset/main
```

Generation uses the Python standard library. `2/.venv` supplies the existing interpreter; this v2 implementation does not import the older project simulators.
