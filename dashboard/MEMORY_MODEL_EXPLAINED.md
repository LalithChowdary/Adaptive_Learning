# How the dashboard's learner memory works

Verified against the current implementation and selected v2.1 checkpoint on 2026-11-10. This document explains existing behavior. The proposed display improvements at the end have **not** been implemented by this documentation update.

## 1. The answer to the two dashboard questions

The model already accepts **difficulty and question format**. The current memory explorer exposes only a skill selector and always queries a medium-difficulty output-prediction reference task. Its single curve is a display choice, not a model limitation. The practice page already predicts the actual chosen question, including its difficulty and format.

There is **no hard-coded 50% starting prediction**. A new learner–skill record uses a shared, learned initial state. The prediction head combines that state with the target question's structured features. The result can be far above or below 50%.

Identical predictions across unassessed skills are expected when their reference tasks have identical structured features. Skill names and identities are deliberately absent from the neural input. An unassessed record does not mean the person has never studied the skill; it means this application has not recorded relevant evidence yet.

## 2. Measured starting predictions

These are actual CPU inference results with **empty history**, using the first assessment-pool task in each skill × difficulty × format combination. They are not example values or measurements from a student's answers. The model hash is recorded in [cold_start_analysis.json](reports/cold_start_analysis.json), which contains all 90 queries and their exact task features.

| Format | Easy | Medium | Hard |
|---|---:|---:|---:|
| Single-choice | 79.82% | 70.88% | 60.58% |
| Multiple-select | 52.66% | 36.66% | 23.26% |
| Output prediction | 61.97% | 42.96% | 27.12% |
| Fill-in-the-blank | 60.96% | 42.25% | 26.78% |
| Code completion | 49.57% | 29.94% | 15.97% |

The six skills give the same values for these matched references because their structured task features match. The current explorer's no-history reference gives **42.96%**, displayed as **43%**, for every skill. An easy code-completion reference gives 49.57%, which the UI rounds to 50%. A value seen after an answer may instead be a personalized prediction; it is no longer the no-history prior.

These predictions concern **full success on all required criteria on the first unaided attempt**. They are not percentage of knowledge, partial credit, or confidence in a grade. Different formats have different completion requirements and guessing opportunities in the simulator. A table difference is not a causal estimate of format difficulty in real people. Across easy/medium/hard references, other authored metadata such as reference time can also change.

### Why initialization is not “50% mastery”

`LearnerGRU.prior` begins as a zero vector when constructing an untrained model, but it is a trainable parameter. The dashboard loads the trained checkpoint, replacing those initialization values. Its current 32-dimensional prior is nonzero: minimum approximately −0.8215, maximum 0.6922, Euclidean norm 1.7013. Even a zero hidden vector would not imply 50% because the prediction head also uses task features and learned biases. The sigmoid equals 50% only when its input logit equals zero.

## 3. Shared weights versus personal state

**We do not store a different set of model weights for each user or skill.** We store one trained model, then maintain a separate evolving state for each learner–skill pair.

| Object | Shared or separate? | Updated while answering? | Storage |
|---|---|---|---|
| GRU weights, decay-layer weights, prediction-head weights | Shared across all learners and skills | No | Selected `model.pt` checkpoint, loaded into process memory |
| Learned initial state `h0` | Shared starting point | No during dashboard use; learned during offline training | Same checkpoint |
| Hidden state `h(user, skill)` | Separate for each learner–skill pair | Yes, as that pair receives evidence | In-process cache; reconstructed from events after restart |
| Evidence timestamp and encoded-event cursor | Separate for each learner–skill pair | Yes, on new events | Cache, with underlying timestamps/events retained in SQLite |
| Questions, grades, hints, answers, attempts | Separate records with learner/skill/task linkage | Appended by actual interaction | `data/sessions.sqlite3` |
| Success probability | Derived for a particular state, question, and query time | Recomputed when queried | Pre-answer probability saved with each presented task; exploratory queries are not journal writes |

For example:

| Record | Evidence used | State |
|---|---|---|
| Learner A / range boundaries | Only A's range-boundary history | `h(A, range)` |
| Learner A / function returns | Only A's function-return history | `h(A, functions)` |
| Learner B / range boundaries | Only B's range-boundary history | `h(B, range)` |

All three use the **same neural weights**. A new range-boundary answer from A updates only A's range state. It does not change B's state, A's function state, or any trained weights. If two records have exactly the same structured history and target features, they can still have identical predictions: separate storage does not imply artificially different values.

The 32 hidden numbers are a learned summary, not 32 named subskills. We cannot label an individual component “loop mastery” or “hint dependence” without additional analysis.

## 4. What happens for one question

1. The application chooses an unseen task from the requested skill, format, difficulty, and practice/assessment pool.
2. It loads that learner's earlier evidence for that skill and brings the cached state up to date. The model predicts the new task **before** its response, timing, or grade exists.
3. The task event records this probability and the model hash. The interface hides the number until submission to avoid influencing the answer.
4. A separate deterministic grader checks the submitted answer against keys or behavior tests and produces criterion flags, a weighted score, full-success status, and grading status.
5. The application saves the response and grade chronologically. The encoder combines the linked response/grade into one attempt token at grade availability.
6. The GRU processes that new evidence when the state is next accessed. Its hidden state changes; its weights do not.
7. The next new question receives a prediction from the updated state and that next question's own features.

Hints are separate evidence tokens. Their delivery does not mechanically add mastery points or establish that learning occurred. A learned prediction change after a hint is an association learned from the training histories, not proof of a causal teaching benefit.

A retry can earn full rubric credit, but it is never a first-attempt training target. Help before the first response also prevents an independent target label. Independent-check results and subsequent review remain distinct.

## 5. Structured inputs and learner–skill routing

IDs select the record in storage. They are not learned user or skill embeddings. The memory network sees:

- Past partial grades, full-success flags, missing/unresolved outcome indicators.
- Attempt numbers, hint counts and kinds, assistance/reveal status.
- Response-time estimates and their missingness, relative response time, and elapsed gaps.
- Event type and the past task's structural characteristics.
- At query time: the new task's difficulty, format, reference duration, snippet/option/criterion counts, and summaries of rubric credit/penalty weights.

It does **not** read names, question wording, submitted code, answer text, course titles, or hidden simulator strength. The grader can use answer text; the memory model only receives its structured output. Removing text supports a reusable interface, but it does not prove performance transfers to biology, chemistry, or real students.

There is one state per skill, **not one state per difficulty or format**. Evidence from an easy range question belongs to the same range state queried for a hard range question. The query head accounts for the new task's demands. Creating separate easy/medium/hard networks would change this architecture and is unnecessary merely to add difficulty controls to the dashboard.

## 6. How time changes a prediction

The time-decayed GRU keeps a state at its last evidence anchor. For a gap of `d` days, it computes a temporary state:

```text
rate = softplus(decay_layer(saved_state))
state_at_query_time = h0 + exp(-rate * d) * (saved_state - h0)
probability = sigmoid(prediction_head(state_at_query_time, task_features))
```

Before an evidence update, the model also applies the gap, then feeds the new evidence through the GRU and saves the resulting state at the new anchor. At prediction time, it only uses a temporary copy. Repeated chart queries cannot double-apply forgetting.

Each state dimension approaches the learned prior at its own positive rate. Because the output head combines those dimensions nonlinearly, the **probability is not guaranteed to decrease monotonically** or approach zero. In the mathematical long-gap limit, it approaches the selected task's prior prediction. A 60-day curve can be below that prior while different dimensions are still decaying. This is a model limitation to inspect, not a reason to overwrite displayed predictions with a hand-drawn decay curve.

An untouched state already equal to `h0` remains there under decay. A flat curve for a completely unassessed skill is therefore expected. There is no recorded personal evidence to forget.

## 7. Files and persistence

Shared checkpoint:

```text
3/model/runs/v2.1-mps-main/time_decay_gru/seed_20261111/model.pt
```

SHA-256:

```text
353ccee669f66f60acf1a6e08f9cae410f2c1ea7744c7c56e2a5523a834866de
```

The checkpoint contains `dataset_version`, `mode`, `hidden_size`, and `weights`. `MemoryService` loads it, calls `.eval()`, disables parameter gradients, and uses inference mode. No optimizer or online training runs in the dashboard.

The cache key is `(learner_id, canonical_skill_id, model_hash)`. Each entry contains the hidden vector, evidence stream, anchor, consumed-event count, and last event ID. It belongs to the server process. On restart the cache is empty and the application reconstructs states from the SQLite journal. It does not save separate `.pt` files per person.

SQLite stores profiles, interactions, events, and request IDs. Events retain actual observations and provenance; the selected model hash is attached to each event. The original pre-answer prediction is preserved for later evaluation. The original synthetic dataset is not rewritten and real observations are not automatically added to training.

Current performance boundary: although recurrence processes only newly appended evidence when cached, the service currently reads the full same-skill event list from SQLite on each call to validate/access the prefix. For very long histories, incremental database reads and durable state snapshots would be a separate optimization. The dashboard does not yet provide those optimizations or a model-migration workflow. A future replacement checkpoint requires state replay and reevaluation; historical predictions must retain their original provenance.

## 8. Why the three dashboard views can show different percentages

| Display | What is being predicted? |
|---|---|
| Before this question | The actual selected task, at its presentation time, before its answer |
| Skill card | Equal-weight mean over five fixed medium-difficulty reference tasks, one per format, using current evidence |
| Memory explorer | One fixed medium-difficulty output-prediction task, using current evidence at several future gaps |

These need not agree. A correct easy multiple-choice answer may coexist with a lower probability for a harder output question. A 100% grade on one answer does not imply a 100% success probability on every future question. Likewise, a prediction near 50% is not a confidence interval or a claim that half the skill is mastered.

## 9. Recommended display changes — proposed, not implemented

Keep the trained weights and separate learner–skill states. Improve the explorer instead:

1. Add a **question-format selector**.
2. Show **easy, medium, and hard** reference curves together for that format, with distinct labels and colours.
3. Keep the reference task fixed within each curve across all time gaps and disclose the reference definition. Use complete task metadata for each difficulty instead of merely changing a label on the same task.
4. For empty histories, prominently label results **“Unassessed — model prior”**. After evidence, show the number of attempts supporting the state; this is evidence coverage, not calibrated uncertainty.
5. Retain observed performance separately from predictions. Do not force cold-start values to 0%, 50%, or 100% to make the display look intuitive.

A broader alternative is averaging a declared set of reference questions for each difficulty and format. That would change the chart's meaning from a specific reference-task prediction to a reference-set average, and should be documented explicitly. Difficulty labels and reference times remain authored estimates, not calibrated human difficulty measurements.

No model retraining is necessary to expose the existing inputs. A model change should be justified by evaluation on real observations, not by a desired chart shape.

## 10. Source map

- [Memory inference and cache](memory_service.py)
- [Selection, predictions, and learner–skill filtering](service.py)
- [Frozen GRU architecture](../model/memory/recurrent.py)
- [Evidence encoder and numeric feature definitions](../model/memory/features.py)
- [SQLite storage](store.py)
- [Current UI/reference labels](static/app.js)
- [Measured cold-start predictions](reports/cold_start_analysis.json)
- [Original dashboard verification](reports/VERIFICATION.md)

This analysis loaded the selected checkpoint and queried empty histories only. It did not read or change the user's real answers, retrain the model, restart the running app, or change the dashboard UI.
