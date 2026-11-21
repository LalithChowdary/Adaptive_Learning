# Verification for the full v2.1 MPS run

- All 55 dataset regression tests passed. The four full-data audits each passed 26 categories.
- All 12 model tests passed on CPU and in the MPS test configuration.
- The v2.1 GPU pilot passed. All six full recurrent fits recorded finite gradients on `mps:0` with CPU fallback disabled.
- The implementation hashes, cohort manifests, and complete dataset release inventory match the saved run.
- Model loader gap counts match the independent dataset audit for practice and assessments.
- The oracle rejects the training split. Its predicted success rates agreed with 24,000 simulated attempts per format within 1.2 binomial standard errors across five diagnostic items.

## Independent metric check

Recomputed the probability metrics from the saved prediction CSVs and checked the delayed means and sample standard deviations.

- `main_validation` had 28,273 unique query IDs and reproduced every group log loss, Brier score, and accuracy to within 1e-12.
- `main_test` had 28,125 unique query IDs and reproduced every group log loss, Brier score, and accuracy to within 1e-12.
- `power_law_test` had 28,161 unique query IDs and reproduced every group log loss, Brier score, and accuracy to within 1e-12.
- `long_gap_test` had 28,715 unique query IDs and reproduced every group log loss, Brier score, and accuracy to within 1e-12.

## Checkpoint reload on CPU

Reloaded all nine checkpoints and compared all 28,273 main validation probabilities to the exports.

| Checkpoint | Maximum absolute probability difference |
|---|---:|
| `task_logistic` | 0 |
| `history_logistic` | 0 |
| `history_boosting` | 0 |
| `time_input_gru_seed_20261110` | 2.09e-07 |
| `time_input_gru_seed_20261111` | 1.79e-07 |
| `time_input_gru_seed_20261112` | 1.79e-07 |
| `time_decay_gru_seed_20261110` | 8.34e-07 |
| `time_decay_gru_seed_20261111` | 1.43e-06 |
| `time_decay_gru_seed_20261112` | 8.94e-07 |
