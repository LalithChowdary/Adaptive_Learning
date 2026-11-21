# Verification for the full v2 MPS run

- All nine model regression tests passed on CPU and in the MPS test configuration.
- The five-model pilot completed before the main fit.
- Both main recurrent fits reported parameters and recurrent gradients on `mps:0`. CPU fallback was disabled.
- The current implementation and cohort manifest hashes match the saved run.

## Independent metric check

Recomputed log loss and Brier score with scikit-learn from each saved prediction CSV.

- `main_validation` had 26,420 unique query IDs and reproduced every reported group log loss and Brier score to within 1e-12.
- `main_test` had 26,431 unique query IDs and reproduced every reported group log loss and Brier score to within 1e-12.
- `power_law_test` had 26,558 unique query IDs and reproduced every reported group log loss and Brier score to within 1e-12.
- `long_gap_test` had 28,242 unique query IDs and reproduced every reported group log loss and Brier score to within 1e-12.

## Checkpoint reload on CPU

Reloaded every checkpoint and compared all 26,420 main validation probabilities to the saved outputs.

| Checkpoint | Maximum absolute probability difference |
|---|---:|
| `task_logistic` | 0 |
| `history_logistic` | 0 |
| `history_boosting` | 0 |
| `time_input_gru` | 1.19e-07 |
| `time_decay_gru` | 1.79e-07 |
