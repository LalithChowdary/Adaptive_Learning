# Learner-memory model comparison

Run type: smoke. Recurrent device: `mps`.

Validation choice: **`task_logistic`**. Models within 0.001 of the lowest validation delayed log loss are ranked by simplicity.

| Model | Training device | Validation delayed log loss |
|---|---|---:|
| task_logistic | cpu | 0.62420 |
| history_logistic | cpu | 0.66526 |
| history_boosting | cpu | 0.68566 |
| time_input_gru | mps:0 | 0.69173 |
| time_decay_gru | mps:0 | 0.69112 |

## pilot_validation

Each cell gives log loss / Brier score / 10-bin calibration error.

| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days |
|---|---:|---:|---:|---:|---:|
| task_logistic | 0.61430 / 0.21082 / 0.04573 | 0.61541 / 0.21027 / 0.11134 | 0.53268 / 0.17892 / 0.10207 | 0.63419 / 0.22096 / 0.14832 | 0.70574 / 0.24450 / 0.26947 |
| history_logistic | 0.57742 / 0.19613 / 0.03991 | 0.55576 / 0.18655 / 0.07516 | 0.48272 / 0.15949 / 0.07724 | 0.65327 / 0.22782 / 0.12528 | 0.85978 / 0.30466 / 0.35843 |
| history_boosting | 0.60539 / 0.20413 / 0.06344 | 0.62625 / 0.20687 / 0.08821 | 0.58059 / 0.19967 / 0.10266 | 0.61621 / 0.21210 / 0.14289 | 0.86017 / 0.30544 / 0.33048 |
| time_input_gru | 0.68861 / 0.24774 / 0.09139 | 0.69087 / 0.24887 / 0.05548 | 0.68421 / 0.24553 / 0.12092 | 0.69287 / 0.24986 / 0.06626 | 0.69811 / 0.25248 / 0.17295 |
| time_decay_gru | 0.69104 / 0.24895 / 0.09661 | 0.69115 / 0.24900 / 0.05610 | 0.69397 / 0.25041 / 0.13932 | 0.70060 / 0.25372 / 0.09221 | 0.67880 / 0.24283 / 0.14037 |

The JSON report includes query counts, calibration bins, learner-level paired confidence intervals, run settings, and the recurrent learning curves.
Performance is measured on the synthetic v2 benchmark.
