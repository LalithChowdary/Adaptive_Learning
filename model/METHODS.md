# Training and evaluation procedure

This file records the original v2.0 protocol and shared feature definitions. The [v2.1 retraining protocol](METHODS_V21.md) supersedes the fitting, checkpoint selection, and reporting settings below.

## 1. Load the v2 fitting and validation records

`memory/data.py` calls the existing v2 `checked_records` loader. It checks the dataset version, file digest, and learner assignments. The model reads the approved `model_data` exports. The model runner fingerprints the cohort manifests and the implementation files in each run.

The full run requires 1,600 fitting learners, 200 validation learners, and 185,510 labeled fitting queries. It checks learner isolation between fitting, validation, and each evaluation partition. Each record represents one learner and skill. The model shares weights across those records and resets to its learned prior at the start of each record.

## 2. Build causal hybrid evidence

`memory/features.py` joins a projected response to the adjacent matching grade. It checks the pairing fields and releases the attempt at the grade's availability time. An explicitly unresolved grade carries the known attempt and timing information and marks the outcome missing. A pending response contributes no attempt token before its grade arrives.

Hints, solution reveals, and delivered instruction form separate tokens. Task presentations, standalone help requests, retry declarations, and other administrative events do not create tokens. Their elapsed time still advances the source clock. The attempt preserves the response's timing and the grade's availability and outcome.

`memory/data.py` advances the source cursor to each query's original `history_end` and records how many evidence tokens are available at that cursor. The query uses the new task's metadata and the elapsed time from the latest encoded evidence. It never observes the query's answer, grade, or later help. Later evidence can enter later queries in the same record.

The label is `full_success` from the dataset's eligible first unaided response. Ineligible query outcomes supply no loss. Eligible practice includes ordinary and fresh follow-up queries. Held-out assessment queries serve evaluation. Assisted attempts remain historical evidence.

## 3. Encode task, event, and summary features

- **Task features.** One-hot format and difficulty, reference time, snippet count, option count, criterion count, and the size, sum, and maximum of the credit and penalty weight lists.
- **Evidence features.** Token type, past task structure, hint type, assistance category, grade status, attempt number, hint count, active duration, relative duration, score, full success, reveal status, and session flag.
- **Tabular history features.** Counts of attempts, retries, unknown assistance, unresolved grades, hint types, reveals, instruction, known scores, and known response times. Separate means for unaided and assisted outcomes, a recent unaided result and average of up to three unaided results, timing averages, and recency of the latest response, encoded evidence, and resolved unaided grade.
- **Numeric encoding.** Counts and time inputs for the tabular models use fixed logarithmic transforms. Optional numeric values are paired with a presence bit. Missing and zero therefore have different representations. The recurrent decay uses actual elapsed days.
- **Routing and reporting.** Learner, skill, query IDs, query purpose, and nominal assessment horizon support grouping and auditing. The feature arrays contain structured observations and task features. They omit identity, item text, answer text, raw code, and simulator hidden variables.

Both history-aware tabular models receive exactly the same columns. The logistic pipelines fit scaling on training rows. The tree receives the same feature information without scaling. The models predict a probability for a specified question.

## 4. Fit the tabular comparators

Use regularized logistic regression for the task-only and history-aware models, comparing `C=0.1` and `C=1.0`. Fit histogram gradient boosting with 15 leaves and 120 iterations or 31 leaves and 160 iterations, learning rate 0.08, and L2 regularization 1.0. Select the setting with lower mean validation log loss across 1, 7, and 30 days. The tree uses the existing validation learners for model selection and disables automatic early stopping that could create a random row split.

## 5. Fit both recurrent models on MPS

Each sequence model uses a single 32-unit GRU cell and a 32-unit SiLU prediction head. A query reads the state at its prefix position and combines it with the new task features. Records are batched by similar sequence length, padded, and gathered at exact query positions. Only real eligible query slots contribute to the loss. Batches are shuffled each epoch using the recorded seed.

Both GRUs minimize binary cross-entropy using AdamW with learning rate 0.003, weight decay 0.001, and gradient clipping at 5.0. The default fit runs up to eight epochs. It restores the checkpoint with the best mean validation delayed log loss and stops after three epochs without improvement. The pilot smoke fit defaults to one epoch.

The recurrent implementation transfers tokens, gaps, query positions, query features, labels, and masks to MPS. Model parameters and backpropagation stay on MPS. Metrics and report construction use CPU. The first recurrent batch verifies a finite gradient on the requested device. MPS training uses float32.

The time-input comparator sees log-transformed gaps at both update and query time. The decay model applies each gap to the state before the update or query, and the direct time input slots carry zero. The two models share token encodings and use equal GRU and head widths.

## 6. Select a model and evaluate held-out learners

Select on the mean of validation log loss at 1, 7, and 30 days. Within 0.001 of the lowest mean, prefer the first model in the order task-only logistic regression, history-aware logistic regression, history boosting, time-input GRU, then time-decayed GRU. This is a fixed engineering tolerance for the first experiment.

After selection, score the main test partition and both stress cohorts using the saved fits. The power-law cohort changes the simulator's retention rule. The long-gap cohort includes inter-session gaps of up to 60 days. The models are neither fit nor recalibrated on these cohorts.

Report log loss, Brier score, and calibration using ten equal-width probability bins. Break out ordinary practice, fresh follow-up, each assessment horizon, and each question format. Also record group sizes. Paired intervals bootstrap whole learners with 1,000 draws and compare models on the same queries. Negative delta log loss favors the first model named in the contrast.

A tree-to-GRU comparison tests the value of the sequence model beyond nonlinear history summaries. The two GRUs compare how elapsed time enters the model. A study that attributes an improvement specifically to event order would need an order-insensitive comparator using the same tokens.

## 7. Verify the important failure modes

The regression tests cover future-grade isolation, missing grades, mismatched pairs, timing through skipped events, identity exclusion, training query restrictions, incremental versus batch replay, repeated predictions, and finite gradients in the decay and GRU parameters. Run them on both CPU and MPS with the commands in the [README](README.md).

The pilot run tests the complete GPU fitting and reporting path before the full experiment. The full run saves individual model checkpoints and an aligned prediction export for each evaluation partition. The dataset is simulated, so the results measure prediction on the v2 synthetic benchmark.
