# Python learner-memory dataset

Start with [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md) for the complete why/how, simulator equations, fields, splitting, limitations, commands, and actual test results.

- `dataset/pilot`: 200-learner development cohort.
- `dataset/main`: 2,000-learner initial model-comparison cohort.
- `dataset/power_law`, `dataset/long_gap`: separate 200-learner stress cohorts, test-only.
- `dataset/task_catalog.json`: authored Python tasks and reference outputs; not network input.
- `fixtures/`: three authored histories for behavior checks.
- `reports/test_results.json`, `reports/test_output.txt`: actual executed automated-test results.
- Each cohort's `AUDIT.md` and `audit.json`: actual full-data audit results.

Numerical synthetic outcomes only. No real student records, Jev calls, model training, or PPO results. Use `loader.load_training_examples("dataset/main")` to avoid accidentally feeding future labels or simulator internals into memory.

Requires Python 3.12 (tested on 3.12.14); standard library only. See the main document for reproduction using a fresh output directory. Existing cohorts are not overwritten.
