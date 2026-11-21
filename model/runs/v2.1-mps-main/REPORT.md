# v2.1 learner-memory comparison

Selected **`time_decay_gru`** on main validation for the full run. Representative checkpoint `time_decay_gru/seed_20261111/model.pt`.

Recurrent device `mps`. Training used 198,651 practice and 26,573 delayed labels.

Checkpoint selection weights practice log loss and equal-horizon delayed log loss equally. The score below aggregates recurrent fits across seeds. ± denotes sample standard deviation.

| Model | Validation selection log loss |
|---|---:|
| task_logistic | 0.60023 |
| history_logistic | 0.55881 |
| history_boosting | 0.55241 |
| time_input_gru | 0.53856 ± 0.00024 |
| time_decay_gru | 0.53734 ± 0.00047 |

## main_validation

Lower log loss is better. The constant and simulator oracle are evaluation references.

| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days | Delayed mean |
|---|---:|---:|---:|---:|---:|---:|
| task_logistic | 0.60403 | 0.59309 | 0.61314 | 0.61199 | 0.57370 | 0.59961 |
| history_logistic | 0.57862 | 0.57460 | 0.58603 | 0.58664 | 0.44785 | 0.54017 |
| history_boosting | 0.57487 | 0.57189 | 0.57385 | 0.57681 | 0.44178 | 0.53082 |
| time_input_gru | 0.55913 ± 0.00035 | 0.55044 ± 0.00044 | 0.55755 ± 0.00275 | 0.56911 ± 0.00025 | 0.43491 ± 0.00139 | 0.52053 ± 0.00047 |
| time_decay_gru | 0.55868 ± 0.00030 | 0.54870 ± 0.00059 | 0.55684 ± 0.00010 | 0.56902 ± 0.00067 | 0.43088 ± 0.00308 | 0.51891 ± 0.00097 |
| constant | 0.69208 | 0.69287 | 0.69477 | 0.68646 | 0.66992 | 0.68372 |
| oracle | 0.51793 | 0.51142 | 0.52362 | 0.52329 | 0.40415 | 0.48368 |

## main_test

Lower log loss is better. The constant and simulator oracle are evaluation references.

| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days | Delayed mean |
|---|---:|---:|---:|---:|---:|---:|
| task_logistic | 0.60744 | 0.59660 | 0.61596 | 0.59088 | 0.58332 | 0.59672 |
| history_logistic | 0.57702 | 0.57390 | 0.59750 | 0.55329 | 0.45265 | 0.53448 |
| history_boosting | 0.57356 | 0.57138 | 0.58303 | 0.54321 | 0.44677 | 0.52434 |
| time_input_gru | 0.55961 ± 0.00038 | 0.54841 ± 0.00061 | 0.56767 ± 0.00171 | 0.53438 ± 0.00200 | 0.44447 ± 0.00123 | 0.51551 ± 0.00118 |
| time_decay_gru | 0.55876 ± 0.00021 | 0.54713 ± 0.00038 | 0.56666 ± 0.00112 | 0.53343 ± 0.00093 | 0.44389 ± 0.00074 | 0.51466 ± 0.00064 |
| constant | 0.69354 | 0.69311 | 0.69704 | 0.68529 | 0.66809 | 0.68347 |
| oracle | 0.52013 | 0.50943 | 0.53314 | 0.48831 | 0.41239 | 0.47795 |

## power_law_test

Lower log loss is better. The constant and simulator oracle are evaluation references.

| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days | Delayed mean |
|---|---:|---:|---:|---:|---:|---:|
| task_logistic | 0.60549 | 0.59088 | 0.62740 | 0.57719 | 0.58877 | 0.59779 |
| history_logistic | 0.58264 | 0.57772 | 0.60143 | 0.55755 | 0.49119 | 0.55006 |
| history_boosting | 0.57997 | 0.57602 | 0.59148 | 0.54857 | 0.49072 | 0.54359 |
| time_input_gru | 0.56943 ± 0.00020 | 0.56060 ± 0.00124 | 0.57288 ± 0.00108 | 0.53867 ± 0.00189 | 0.49157 ± 0.00051 | 0.53437 ± 0.00073 |
| time_decay_gru | 0.56868 ± 0.00041 | 0.56015 ± 0.00037 | 0.57190 ± 0.00054 | 0.53701 ± 0.00195 | 0.48876 ± 0.00123 | 0.53256 ± 0.00034 |
| constant | 0.69279 | 0.69407 | 0.69586 | 0.68449 | 0.67246 | 0.68427 |
| oracle | 0.52916 | 0.52288 | 0.53923 | 0.50722 | 0.45408 | 0.50018 |

## long_gap_test

Lower log loss is better. The constant and simulator oracle are evaluation references.

| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days | Delayed mean |
|---|---:|---:|---:|---:|---:|---:|
| task_logistic | 0.60575 | 0.58601 | 0.59704 | 0.58359 | 0.58261 | 0.58775 |
| history_logistic | 0.57224 | 0.56220 | 0.55971 | 0.52659 | 0.44220 | 0.50950 |
| history_boosting | 0.56654 | 0.55728 | 0.55331 | 0.52151 | 0.43013 | 0.50165 |
| time_input_gru | 0.54503 ± 0.00084 | 0.52814 ± 0.00122 | 0.54158 ± 0.00105 | 0.52195 ± 0.00051 | 0.42044 ± 0.00092 | 0.49466 ± 0.00077 |
| time_decay_gru | 0.54380 ± 0.00038 | 0.52708 ± 0.00026 | 0.53665 ± 0.00218 | 0.52015 ± 0.00055 | 0.41901 ± 0.00075 | 0.49194 ± 0.00100 |
| constant | 0.68810 | 0.68822 | 0.68762 | 0.68063 | 0.66894 | 0.67906 |
| oracle | 0.50299 | 0.49007 | 0.50438 | 0.47961 | 0.39631 | 0.46010 |

The JSON contains per-seed probability metrics, calibration bins, paired learner intervals, and the fraction of the log-loss improvement from the constant to the simulator oracle.
Each paired comparison averages each query's loss over seeds before resampling learners.
The simulator oracle uses private pre-attempt strength and difficulty. It is a reference for this synthetic benchmark.
