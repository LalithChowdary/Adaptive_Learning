# v2.1 learner-memory comparison

Selected **`history_logistic`** on main validation for the full run. Representative checkpoint `history_logistic/model.joblib`.
This is the pilot smoke result.
Recurrent device `mps`. Training used 1,470 practice and 205 delayed labels.

Checkpoint selection weights practice log loss and equal-horizon delayed log loss equally. The score below aggregates recurrent fits across seeds. ± denotes sample standard deviation.

| Model | Validation selection log loss |
|---|---:|
| task_logistic | 0.60606 |
| history_logistic | 0.57543 |
| history_boosting | 0.59595 |
| time_input_gru | 0.68488 |
| time_decay_gru | 0.68786 |

## pilot_validation

Lower log loss is better. The constant and simulator oracle are evaluation references.

| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days | Delayed mean |
|---|---:|---:|---:|---:|---:|---:|
| task_logistic | 0.62564 | 0.61263 | 0.63815 | 0.59495 | 0.53761 | 0.59023 |
| history_logistic | 0.60858 | 0.59293 | 0.57765 | 0.54375 | 0.51897 | 0.54679 |
| history_boosting | 0.63158 | 0.61443 | 0.61898 | 0.48581 | 0.59099 | 0.56526 |
| time_input_gru | 0.69458 | 0.68946 | 0.69260 | 0.67259 | 0.66479 | 0.67666 |
| time_decay_gru | 0.69575 | 0.68934 | 0.69451 | 0.67673 | 0.67425 | 0.68183 |
| constant | 0.69333 | 0.69469 | 0.69493 | 0.69867 | 0.69993 | 0.69784 |
| oracle | 0.51900 | 0.51403 | 0.51941 | 0.44463 | 0.46580 | 0.47662 |

The JSON contains per-seed probability metrics, calibration bins, paired learner intervals, and the fraction of the log-loss improvement from the constant to the simulator oracle.
Each paired comparison averages each query's loss over seeds before resampling learners.
The simulator oracle uses private pre-attempt strength and difficulty. It is a reference for this synthetic benchmark.
