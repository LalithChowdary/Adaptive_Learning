# Python learner-memory dataset v2 — design, fields, and verification

Version 2.0.0. Built on 2026-11-09. This document describes the implemented synthetic dataset in `/Users/lalith/snu/sem7/minor_pro/code/3/data_Set/v2/`. The central architecture and handoff record remains `../../../PROJECT_DECISIONS.md`. Measured release results appear at the end of this document and in `reports/RELEASE_SUMMARY.json`.

## 1. Purpose and scope

The memory experiment will predict the probability of **full success on a new first unaided attempt**, conditioned on earlier available evidence for the same learner and skill and structured characteristics of the new task. The dataset supplies binary outcomes to fit that probability. It does not contain an invented observed “mastery percentage.” No memory model, PPO policy, Jev agent, or live tutor was trained or deployed in this build.

All learners, response outcomes, help requests, durations, and gaps are numerical simulation. The authored Python material anchors task and rubric definitions. No real student responses or generated student answer text are present. A numerical grading adapter converts sampled option choices or criterion results to a grade. Its provenance is `simulated_rubric_v2`; it does not claim to be Jev.

The intended shared memory interface excludes subject names, question/answer text, code embeddings, and identity embeddings. One shared model can later process separate learner-skill histories. This Python-only release tests that interface and declared simulation assumptions. It does not establish transfer to biology, chemistry, common sense, or real learners.

## 2. Changes from v1 and their reasons

| Concern in v1 | Implemented v2 behavior | Reason |
|---|---|---|
| Instruction before roughly 65% of questions, with automatic latent gain | Default generation starts with a question. No routine instruction events. A requested introduction has a request, reason, stored content, and no direct retained gain. | A bare teaching label is not evidence that learning occurred. |
| Two output snippets and two equally weighted criteria everywhere | Five formats, 0–3 snippets, 1–6 criteria, and format-specific weighted scoring | The memory experiment must encounter varied activities and different definitions of full success. |
| At most one hint and one retry | Up to three hints and two retries by default; every hint, request, retry, response, and grade is a separate event | Help quantity and the order of attempts need to remain observable. |
| Narrow numerical outcome process | Format-specific guessing, criterion counts, timing, temporary hint effects, and practice doses | Changing wording alone would leave the numerical experiment unchanged. |
| No explicit interrupted-assessment flow | Preserve the first unaided grade; help before submission closes assessment as incomplete and switches to review | Assisted success must not replace or manufacture independent success. |
| Subsequent task not explicitly linked to assistance | Every assisted ordinary practice item receives a different, linked, unaided same-skill follow-up | Repeating the just-assisted question can measure recall of the help instead of transfer to a fresh task. |

V1 remains in `3/data_Set/v1` with its original sources, docs, and reports. `reports/v1_preservation.json` checks archived v1 files using paths relative to the shared dataset directory. V2 is a new implementation under `3/data_Set/v2`; no v1 rows are mixed into it.

## 3. Cohorts, skills, and activity budgets

| Cohort | Learners | Partition | Purpose |
|---|---:|---|---|
| `pilot` | 200 | 160 train, 20 validation, 20 test | Gate for generation, contracts, coverage, and causal replay |
| `main` | 2,000 | 1,600 train, 200 validation, 200 test | Principal future fitting and evaluation |
| `power_law` | 200 | Test only | Sensitivity to a different hidden forgetting function |
| `long_gap` | 200 | Test only | Sensitivity to much longer gaps between sessions |

These are engineering sample sizes, not a statistical power calculation or evidence of human representativeness. Pilot learners are separate from main learners. Never add the stress cohorts to training when reporting stress evaluation. Each cohort has a separate seed and learner-ID namespace.

The six skills are `range_boundaries`, `loop_tracing`, `accumulator_initialization`, `conditional_evaluation`, `function_returns`, and `list_indexing`. A task targets exactly one skill. No automatic prerequisite transfer is simulated.

Each learner has 16 ordinary practice questions per skill, or 96 ordinary questions in total. Assisted ordinary questions each add one fresh unaided follow-up. This produces a variable total number of practice questions. A question can have several response attempts, and each attempt creates response and grade events. Thus learners, questions, responses, events, and eligible training targets have different counts.

Half the learners in each split receive blocked practice, and half receive mixed practice. Blocked practice completes one skill's ordinary budget before moving to the next. Mixed practice shuffles all six skills in each round. Extra follow-ups occur immediately after the parent interaction, after a 60-second simulated pause. A session begins every 12 **ordinary** questions; follow-ups do not consume that ordinary-question counter.

Practice format selection cycles through the five formats with a learner-specific offset; each presented follow-up also advances this cycle. An unused task of the required format and skill is sampled from its practice pool. Difficulty is sampled through that pool. Assessment items are sampled independently from the separate assessment pool. Task IDs never repeat for a learner's practice, and the three assessment items per skill have distinct IDs.

Every learner also has 18 delayed assessment opportunities, covering six skills at 1, 7, and 30 days after the end of the complete practice history. These are **separate simulation branches** from the same final practice state. The 7-day branch never learns from the 1-day branch. Their outcomes are correlated through the same learner and history; they are not extra learners. The true gap since the last response on a particular skill may exceed the named delay, especially with blocked practice. Query features retain that actual gap.

## 4. Question-first practice and assessment behavior

The default ordinary interaction is

1. Present a new question with no help.
2. Record the first response, including its time and assistance snapshot.
3. Record the grade after its grading delay.
4. After a resolved non-full-success result, optionally provide requested or offered help and permit a retry.
5. Continue within the per-question budgets, stopping on full success, an unresolved grade, a sampled decision to stop, or budget exhaustion.
6. When assistance occurred, present a different same-skill question as a fresh unaided follow-up.

A learner can explicitly request help before the first submission. This is a simulated 3% request probability where help is enabled. The resulting first response is assisted and supplies no independent target. It is retained as historical evidence. New records start with varying hidden initial strengths; the first recorded task does not imply the learner has never studied the skill.

Default limits are **three hints total across the question** and **two retries**, giving at most three submissions. Limits do not reset at each retry. A help block requests one hint with probability 2/3 and two hints with probability 1/3, clipped to the remaining budget. After difficulty, a retry round continues with probability 0.75. Hints can therefore accumulate as 1, 2, or 3; there is no rule that emits all hints on every failure. A retry can occur with no new hint after the hint budget is exhausted. Known zero-hint retries are still excluded as targets because they are not first attempts.

Each hint has an ordinal, type, reason, and a reference to actual authored text in the task catalog. Hint levels are conceptual, targeted, scaffold, and an optional fourth scaffold. Hint-request events and delivered-hint events are separate. Offered help requires an earlier resolved non-full-success grade. Explicitly requested help requires a request event. A solution reveal is a separate event with a reference to the task's answer material. It does not count as another hint, produce another submission, or create a success label. Its reason is `practice_ended_after_difficulty`, which also covers early stopping.

The fresh follow-up has one first attempt and no hints or retries. It may succeed, partially succeed, or fail. Its result is not forced to improve. Follow-ups are proximal practice outcomes, distinguishable by `purpose=followup`; they are not substitutes for delayed evaluation. Failure on a follow-up does not recursively generate more follow-ups.

### Independent assessments and help

The **scored assessment phase** allows one unaided submission and zero hints/retries. Support is still available in a separate review phase.

- A help request before submission emits `assessment_end` with `status=incomplete`, then `mode_switch` to `review`. The assisted first response supplies no independent label. Incomplete is not encoded as failure.
- A first unaided submission receives its grade before `assessment_end` with `status=complete`. Completion means the attempt was submitted; its grade can still be unresolved and therefore ineligible.
- After a resolved failure, post-assessment review is offered with probability 0.35. Review can contain hints and retries. The earlier independent grade remains the label even if a later assisted response succeeds.
- Branch review events stay in raw assessment files for inspection. They are never appended to the original practice history or to another branch's context. No retained learning updates occur anywhere in these isolated assessment branches, including their optional review.

An explicit `requested_lesson(skill)` entry point records `lesson_request` and `instruction`, a reason `learner_requested_intro`, and a content reference into `lesson_catalog.json`. Ordinary cohort generation never calls this entry point automatically. A controlled example exercises it. Delivery consumes 45 simulated seconds but performs no retained learning update.

## 5. Task bank, answers, and grading

The bank has **2,160 tasks**, with 24 variants for each combination of six skills, three authored difficulty levels, and five formats. In each combination, 16 variants belong to practice and eight to assessment. The pools contain 1,440 and 720 tasks respectively.

These are parameterized authored families. Separate IDs and numeric variants do not establish a task-family generalization test. The same families are shared across learner splits, and some underlying concepts or equivalent examples may overlap across formats. A genuinely new task-family or new-subject evaluation remains future work.

| Format | Presentation | Grading and full success | Numerical response representation |
|---|---|---|---|
| Single choice | Four options; either prose without a snippet or one snippet | Exactly the correct option earns 1, otherwise 0 | One option ID |
| Multiple select | Four, five, or six concrete example statements; no separate snippet block | Correct selections earn weighted credit; wrong selections subtract penalties; exact answer set required for full success | List of selected option IDs |
| Output prediction | One, two, or three independent snippets | Per-output exact-match criteria with weights; all required outputs correct for full success | One Boolean per output criterion |
| Fill in the blank | One snippet with one, two, or three missing integer literals and an explicit specification | Each canonical literal has an exact-match criterion; all blanks required | One Boolean per blank criterion |
| Code completion | One function-body stub with behavior specification and two, three, or four behavior tests | Weighted test outcomes; every included required test must pass for full success | One Boolean per behavior test |

For output, blank, and completion tasks, weights are proportional to `1, 2, ..., m`, normalized to sum to one. This is a transparent authoring convention for varied criterion importance, not an empirically validated pedagogical weight. All criteria are required. Full success is determined by the required criteria, not a universal threshold such as score ≥0.8.

For multiple-select tasks, correct options have positive weights proportional to their correct-option order, summing to one. Incorrect-option penalties are equal and sum to one. The grade is

`score = clamp(sum(weights of selected correct options) - sum(penalties of selected incorrect options), 0, 1)`.

Missing a correct option forfeits its credit. Selecting every option earns zero, and selecting nothing earns zero. `full_success` requires the exact correct selection set. `criteria[j]` records whether the select/do-not-select decision for option j was correct. The score is not the weighted average of those flags; correct rejections avoid penalties without adding positive credit. The adapter recomputes the score from the selected option IDs.

The raw grade preserves numerical evidence and per-criterion correctness. The memory export keeps the score and full-success outcome plus structural rubric information, not the answer key, selected option IDs, reference solution, or criterion identities.

No student program is generated or executed. Tests execute only our authored reference snippets and completion solutions to verify their keys and expected behavior. Numerical completion outcomes can represent different combinations of passing and failing behaviors without specifying a concrete program that produced them. This is an explicit simulator limitation. A future adapter using real answers must implement exact-answer parsing, accepted alternatives where appropriate, and sandboxed execution of student code. Jev integration and its calibration require a separate validation exercise.

## 6. How the numerical simulators work

All parameter ranges below are explicit synthetic assumptions. They are not fitted to real Python interaction logs. Random streams for parameters, policy, task selection, outcomes, timing, missingness, and gaps are seeded separately using a SHA-256-derived seed from version, cohort seed, learner identity, and stream name. This keeps generation reproducible and separates different sources of randomness.

### Initial learner variation

A learner-level base draws initial strength from Beta(2,2), learning rate uniformly from 0.08–0.28, half-life log-uniformly from 5–60 days, and speed from a lognormal distribution with log mean 0 and log standard deviation 0.25. Each skill varies around that base.

| Hidden learner-skill parameter | Generation |
|---|---|
| `initial_strength` | Base strength plus Uniform(-0.2,0.2), clipped to [0.02,0.95] |
| `learning_rate` | Base rate × LogNormal(0,0.25), clipped to [0.04,0.4] |
| `half_life_days` | Base half-life × LogNormal(0,0.35), clipped to [2,120] days |
| `speed` | Base speed × LogNormal(0,0.15), clipped to [0.4,2.5] |
| `help_effect` | Uniform(0.6,1.8), controlling temporary answer assistance |
| `retained_practice_fraction` | Zero for 25% of sampled learner-skill pairs; otherwise Uniform(0.05,0.5), scaling retained gains from assisted attempts |

These parameters and hidden states live only in `administration/`. They are excluded from predictors and primary labels.

### Forgetting and retained practice gains

The main and pilot use exponential half-life decay

`k(t + gap) = k(t) × 2^(-gap_days / half_life_days)`.

The power-law cohort uses

`k(t + gap) = k(t) / (1 + gap_days / half_life_days)`.

Both halve strength after one half-life; their long tails differ. Repeatedly querying a state does not write it or apply decay twice. Skill states update only on practice responses. The model will later learn how elapsed time affects prediction; no memory architecture or extra inference-time forgetting multiplier was implemented here.

After a practice response, retained strength updates as

`k_after = k_before + (1 - k_before) × (1 - exp(-learning_rate × dose))`.

The dose is the format's base practice dose divided by the attempt number. Assisted attempts additionally multiply it by `retained_practice_fraction`. Unassisted failures can still receive a practice gain under this effort-based assumption. The grade does not manufacture a mastery label. Grade availability is later than the response, and the latent update is attributed to the response effort. There is no direct retained gain from a hint, instruction label, request, or solution reveal. There is also no automatic perfect strength after success or exposure.

### Format-aware outcomes and hint effects

A hidden task demand is the authored difficulty value (0.25, 0.45, or 0.65) plus a stable per-task, per-cohort offset Uniform(-0.12,0.12). Each attempt adds shared Normal(0,0.65) noise so its criteria are correlated.

For criterion index j starting at zero,

`z_j = 5 × (k - demand) + 0.65 + noise + format_help × help_effect × log(1 + hints) - 0.12 × j`

and `q_j = 0.97 × sigmoid(z_j)`.

| Format | Base seconds | Temporary help multiplier | Base practice dose | Guessing rule |
|---|---:|---:|---:|---|
| Single choice | 25 | 0.75 | 0.30 | Use the correct option with probability q; otherwise choose uniformly among all four. Success probability q+(1-q)/4. |
| Multiple select | 45 | 0.90 | 0.40 | Each select/reject decision is correct with probability 0.5+0.5q. Exact-set success depends on all decisions. |
| Output prediction | 35 | 1.00 | 0.45 | Per-criterion probability 0.02+0.98q |
| Fill in the blank | 40 | 1.10 | 0.45 | Per-criterion probability 0.04+0.96q |
| Code completion | 75 | 1.25 | 0.55 | Per-test probability 0.01+0.99q |

The logarithmic hint term gives diminishing increments. Assistance raises the current response probability under matched random conditions. It does not establish independent success or guarantee a retained benefit. A numerical “scaffold” uses the catalog's corresponding hint level; no LLM generates it.

### Timing and elapsed gaps

Authored reference time is

`base_seconds × (1 + 0.4 × difficulty_index) × (1 + 0.18 × (criterion_count - 1))`.

The response duration is

`reference_seconds × speed × (1.25 - 0.5k) × LogNormal(0,0.35) / (1 + 0.12 × hints)`, clipped to 3–900 seconds.

The active response clock excludes time spent reading prior hints. Delivered hints advance the wall clock by 8, 12, 16, or 20 seconds by level; grading advances it by a sampled 0.1, 2, or 30 seconds. Event timestamps include these delays. Missing active time does not remove elapsed wall-clock time.

Regular between-session gaps are sampled from 0.25, 1, or 3 days. Long-gap stress uses 0.25, 1, 3, 14, or 60 days. The synthetic Unix time origin is 1794009600; timestamps are not actual student dates. Long-gap histories span multiple months, but a full-year course is not guaranteed or specifically modeled.

Reference times and difficulty levels are authored, unvalidated task metadata. A future real-data pipeline must validate them or estimate them using training data only. These numerical assumptions allow speed to correlate with ability while remaining noisy. They do not justify interpreting “fast” as synonymous with understanding.

### Missingness and grading availability

The default independent probabilities are 3% unresolved grades, 4% missing active response times, 4% missing exposed reference times, and 2% unknown response-assistance snapshots. All are synthetic missingness mechanisms.

An unresolved grade has null evidence, criteria, score, and full-success. Unknown assistance has null hint count and answer-revealed fields. Even when surrounding synthetic events might suggest the answer, the first experiment conservatively excludes unknown-assistance first responses from the target set. Null timing remains null; it is never converted to zero. No grader confidence is invented; `grader_confidence` is always null.

The generator emits occurred and available timestamps synchronously for each logged event. The grading delay is represented by a later grade event. This release does not simulate out-of-order or retrospectively arriving grades. The validator rejects such ordering. A production event adapter will need an explicit availability-order replay policy before using asynchronously arriving evidence.

## 7. Files and what a row represents

Compressed JSON Lines is used because events and rubrics have variable structure and histories have variable length. One line is one JSON object. The `.gz` wrapper is deterministic gzip with fixed compression timestamp. “Columns” below mean JSON keys; nested objects and arrays are described separately.

| Path inside a cohort | One row represents | Who should use it |
|---|---|---|
| `observations/{split}.jsonl.gz` | One chronological practice event | Event storage, audit, causal feature preparation |
| `assessments/{split}.jsonl.gz` | One event inside one delayed branch | Evaluation construction and inspection; group by learner and branch |
| `model_data/{split}.jsonl.gz` | One learner-skill sequence plus query pointers and separately stored targets | Memory loading through `loader.py` |
| `administration/{split}.jsonl.gz` | One learner's hidden parameters, state, and generation traces | Simulator auditing only; never a model feature source |
| `config.json` | Cohort configuration and learner assignments | Reproduction and split checks |
| `manifest.json` | SHA-256 fingerprints of data, catalogs, and source | Integrity and reproduction |
| `example_learner.json` | Opening raw events and one full skill record | Human inspection |
| `audit.json`, `AUDIT.md` | Executed checks and measured counts | Release verification |

At the dataset root, `task_catalog.json` and `lesson_catalog.json` contain teaching/grading material, and `schema_fields.json` lists the implemented field allowlists. `examples/controlled_histories.json` is outside the cohorts and never used for fitting.

Raw practice events are chronological within a learner. Delayed branches are alternatives, so concatenated branch files are not one chronological student trajectory. Do not merge all their timestamps into a learner training sequence.

## 8. Raw event column dictionary

### Common fields on every event

| Field | Type | Meaning and reason |
|---|---|---|
| `schema_version` | string | `2.0.0`; rejects accidental v1 mixing |
| `event_id` | string | Unique event link within learner/branch |
| `learner_id` | string | Routes a learner's records; excluded from network features |
| `skill_id` | string | Canonical skill record key; excluded from network features |
| `course_id` | string | `python_basics_v2`; provenance, not a network subject cue |
| `session_id` | string | Session grouping; network sees only derived session-change flag |
| `branch` | string | `practice` or a named delayed assessment fork |
| `interaction_id` | string | Groups one task and its attempts/help, or a requested lesson |
| `event_type` | string | Discriminates the event-specific fields below |
| `phase` | string | `practice`, `assessment`, or `review`; distinguishes scored assessment from subsequent help |
| `occurred_at` | number | Synthetic Unix seconds when the event occurred |
| `available_at` | number | Synthetic Unix seconds when its evidence became available |

### Event-specific fields

Only the listed fields for that event type are allowed, in addition to common fields. This prevents accidental hidden state or answer text entering raw observation contracts.

| Event type | Fields and types | Meaning |
|---|---|---|
| `task` | `task_id` string; `purpose` string; `parent_interaction_id` string/null; `reference_seconds` number/null; `max_hints`, `max_retries` integers | Presentation and query point. Purpose is ordinary, followup, or delayed. Parent links a fresh follow-up to its assisted practice item. Caps are the interaction's configured budgets; scored assessment/follow-up rules additionally forbid help and retries. |
| `help_request` | `task_id`, `reason` strings | Learner explicitly requests assistance; reason is `learner_request`. |
| `hint` | `task_id` string; `hint_number` integer; `hint_type`, `reason`, `content_ref` strings | One delivered hint. Number is cumulative for this question. Reason is learner request or observed difficulty; content reference resolves in the task's `hints` array. |
| `retry` | `task_id` string; `attempt_number` integer; `reason` string | A new submission opportunity after difficulty, recorded before its response. Reason is `retry_after_difficulty`. |
| `response` | `task_id` string; `attempt_number` integer; `active_response_seconds` number/null; `hint_count` integer/null; `answer_revealed` Boolean/null; `assistance` string | Attempt starts at 1. Assistance is independent, assisted, or unknown. Known hint count is the cumulative delivered count. There is no student answer text. |
| `grade` | `task_id`, `response_id`, `grading_status`, `rubric_id`, `grader` strings; `grader_confidence` null; `evidence` string/list/null; `criteria` Boolean list/null; `score` number/null; `full_success` Boolean/null | Grade links exactly one response. Status is resolved or unresolved. Evidence is an option ID, selection list, or simulated criterion flags. Score is bounded [0,1], rounded to 10 decimals. Full success follows the rubric. |
| `solution_reveal` | `task_id`, `reason`, `content_ref` strings | Reveal separate from hint count; reference resolves to answer key/reference solution. No response follows the reveal in this release. |
| `assessment_end` | `task_id`, `status`, `reason` strings | Complete after a submitted first attempt, or incomplete when help was requested before submission. Stores why the scored phase ended. |
| `mode_switch` | `task_id`, `reason` strings | Switches to phase review. Reason is requested help or post-assessment review. |
| `lesson_request` | `reason` string | Explicit request for an introductory lesson. Present only in the controlled example by default. |
| `instruction` | `reason`, `content_ref`, `teaching_action` strings | Requested explanation with stored content; no direct retained-state gain. |

`answer_revealed=False` on known response snapshots because this release never submits another response after revealing a solution. The later `solution_reveal` event is still visible in historical features before a fresh question. Missing assistance retains null values.

## 9. Task and lesson catalog columns

| Task field | Type | Definition |
|---|---|---|
| `task_id`, `skill_id`, `version` | strings | Task routing, canonical skill, and authored task version `2` |
| `pool` | string | Practice or assessment allocation |
| `task_type` | string | One of the five formats |
| `difficulty`, `difficulty_source` | strings | Easy/medium/hard and `authored_unvalidated` provenance |
| `reference_seconds`, `reference_source` | number, string | Authored expected duration and `authored_unvalidated` provenance |
| `prompt` | string | Student-facing specification |
| `snippets` | string array | Zero, one, two, or three code blocks |
| `options` | object array | Each has `id` and `text`; empty outside choice formats |
| `answer_key` | string/list/null | Correct option ID, exact correct selection IDs, expected output strings, or canonical blank literals; completion uses tests instead |
| `reference_solution` | string/null | Full trusted fill or completion solution when applicable |
| `tests` | object array | Completion cases with `input` and `expected`; empty for other formats |
| `rubric_id` | string | Per-task, versioned scoring reference |
| `criteria` | object array | Each has `id`, `weight`, `required`, and `kind` |
| `hints` | object array | Each has `level`, `kind`, and authored `text`; four available levels |

Criterion kinds are `exact_match`, `behavior_test`, `correct_selection`, and `incorrect_selection_penalty`. For multiple-select, criterion order aligns with option order and weights include both positive credits and subtraction penalties. For completion, order aligns with `tests`; for output/blank it aligns with the answer list. All `required` flags are true in this release. The full rubric is held outside the network interface.

`lesson_catalog.json` is a mapping from each `python_intro_v2/{skill_id}` content reference to an actual short introductory explanation. It is stored for the explicit-request path even though ordinary generation does not emit lessons.

## 10. Memory export and leakage prevention

Each model-data row has `learner_id`, `skill_id`, `split`, `events`, and `queries`. IDs and split are routing metadata. `events` contains the entire prepared practice sequence for storage efficiency; **it is not the input to every query**.

Each query contains

| Query field | Type | Meaning |
|---|---|---|
| `query_id` | string | The presentation event's ID, for tracing |
| `phase` | string | Practice or assessment; used to choose training versus evaluation |
| `purpose` | string | Ordinary, followup, or delayed; useful for separate evaluation reporting |
| `history_end` | integer | Exclusive end index of the available historical prefix |
| `input` | object | Target task metadata and current elapsed gaps, defined below |
| `target` | object/null | Eligible outcome with integer `full_success` 0/1 and numerical partial `score` |
| `target_exclusion` | string/null | Assisted/review first attempt, unknown assistance, unresolved grade, or no eligible first response; null when eligible |
| `delay_days` | number, delayed queries only | Nominal 1/7/30-day branch horizon, for evaluation metadata |

A query is made at task presentation **before adding even that presentation to its history**. The target task is supplied separately. It cannot see the current answer's time, hint requests, retries, or grade. Past assisted responses remain useful history. The first query on a new skill can have an empty history; that represents a cold-start prediction.

A label requires a resolved grade for attempt 1, known independent assistance, and a response outside review. Later retries never become new primary targets. All independent first attempts in ordinary practice and fresh follow-ups can supply training labels. The separate delayed branches are omitted from training learners' `model_data` exports, even though their raw files remain available for audits.

For a validation/test delayed query, `history_end` points to the end of that learner-skill's original practice sequence. The query's input includes the elapsed gap at the requested assessment time. The same base history can therefore yield different 1/7/30-day predictions without seeing any other assessment response.

### Input returned by `materialize_query` and `loader.py`

Only four top-level fields are supplied to a predictor.

| Field | Type | Meaning |
|---|---|---|
| `history` | object array | `record.events[:history_end]`; a strict earlier-evidence prefix |
| `target_task` | object | Structural characteristics of the question being predicted |
| `since_last_event_seconds` | number/null | Elapsed time since the latest earlier available event on this skill |
| `since_last_response_seconds` | number/null | Elapsed time since the latest earlier response on this skill |

Each history event has the same allowlisted keys. Nonapplicable entries remain null.

| Historical feature | Type | Meaning |
|---|---|---|
| `event_type` | string | Event category |
| `gap_seconds` | number/null | Gap from previous available event on this skill; other skills do not erase the gap |
| `new_session` | Boolean/null | Whether this skill's previous event was in a different session |
| `task` | object/null | Current interaction's structural task features |
| `attempt_number` | integer/null | Which response attempt or retry this event concerns |
| `active_response_seconds` | number/null | Available only on historical response events |
| `relative_response_time` | number/null | Active response seconds / exposed reference seconds when both are known |
| `hint_count` | integer/null | Hint ordinal on hint events or snapshot on response/grade events |
| `hint_type` | string/null | Conceptual, targeted, or scaffold on a delivered hint |
| `answer_revealed` | Boolean/null | Historical response/grade snapshot or true on a reveal event |
| `assistance` | string/null | Independent, assisted, or unknown where applicable |
| `score`, `full_success` | number/Boolean/null | Available historical grade evidence |
| `grading_status` | string/null | Resolved or unresolved on grade events |

The same task-characteristic dictionary is used for `target_task` and historical `task`.

| Task feature | Type | Reason |
|---|---|---|
| `difficulty` | string | Authored demand category available before answering |
| `task_type` | string | Format changes guessing, partial credit, time, and success requirements |
| `reference_seconds` | number/null | Task-relative timing, with missingness preserved |
| `snippet_count` | integer | Structural activity size, without reading code |
| `option_count` | integer | Choice count, including zero for non-choice tasks |
| `criterion_count` | integer | Number of scoring requirements |
| `credit_weights` | number array | Sorted positive rubric weights |
| `penalty_weights` | number array | Sorted wrong-selection penalties; empty where inapplicable |

Sorting weight groups retains rubric structure while withholding which option is correct. There are no answer keys, selected option IDs, criterion identities, task IDs, subject names, raw timestamps, hidden ability, probabilities, or future response durations in network inputs. Grade confidence and raw grading provenance also remain outside this first feature set. Instruction event type can appear in a requested-lesson history, but no field states a measured gain from instruction.

The materializer rejects unexpected keys and invalid prefix ranges. A full causal audit independently rebuilds model records from raw observations and compares every record and query. This catches a future-prefix edit even if its data checksum is updated. Always use audited releases and the loader; passing an entire stored `events` array alongside its earlier labels would introduce leakage.

### Loading one fitting example

Run in a fresh Python process from the project root so v1 modules with the same names are not already imported.

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("3/data_Set/v2").resolve()))
from loader import load_training_examples, load_evaluation_examples

cohort = Path("3/data_Set/v2/dataset/main")
x, y = next(load_training_examples(cohort))
# x has only the four allowed input fields; y is the binary unaided outcome.
# This code loads examples; it does not train a model.
```

`load_evaluation_examples(cohort, split="validation", phase="assessment")` selects eligible delayed labels. The default loader returns input and binary label; for subgroup analysis, use record/query routing metadata alongside `materialize_query`, keeping that metadata outside model features. Fit normalization, imputation, encoders, or learned time references on the main training partition only.

## 11. Hidden administration and manifests

Administration rows contain `learner_id`, `config`, `parameters`, `final_states`, `trace`, and `assessments`. `parameters` maps skill IDs to the six hidden parameter names in section 6. `final_states` maps each skill to `[strength, last_update_unix_seconds]`.

Each response trace has `response_id`, `skill_id`, `probabilities`, `hidden_demand`, `evidence_truth`, `strength_before`, and `strength_after`. Probabilities refer to criterion success or correct select/reject decisions, depending on format; they are not necessarily the probability of full success for the whole task. They are stored solely for simulator checks. A branch record has `delay_days`, `skill_id`, and its own `trace`.

The cohort configuration has `name`, `learners`, `evaluation_only`, `assignments`, and nested `config`. Each assignment is `[learner_id, split, schedule]`. Config contains all dataclass defaults described in sections 3–6, including seed, budgets, caps, mechanism, gap stress flag, missingness probabilities, help/review probabilities, and assessment horizons. Exact settings are preserved in each generated `config.json`.

Cohort seeds are 2026110901 for pilot, 2026110902 for main, 2026110903 for power-law, and 2026110904 for long-gap. The manifest contains `version`, interpreter `python` version, `files` hashes, `catalog_files` hashes, and `source_files` hashes including the tests. Audit reports record the checksum of the audited manifest. `RELEASE_MANIFEST.json` inventories release files, including documentation and executed reports.

## 12. Tests, pilot gate, and full-data audit

The automated suite exercises grading, trusted authored references, event ordering, caps, causal inputs, branch isolation, reproducibility, and disk tampering. It includes controlled examples for three hints/two retries, an explicit pre-answer hint, an interrupted assessment, and review after an independent failure. It also exercises four-hint and zero-hint configurations.

The full-data audit reloads every learner across observations, assessments, prepared model sequences, and hidden administration. It verifies

- Checksums for data, source, and shared catalogs, and unique disjoint learner assignments.
- Balanced schedules, the six ordinary skill budgets, practice/assessment task-pool separation, and no repeated practice IDs.
- Exact event schemas, chronological availability, response-to-grade links, valid help reasons, assistance snapshots, hint/retry caps, and valid phase transitions.
- Every resolved grade recomputed from its rubric evidence, null unresolved grades, and honest simulated grading provenance.
- One fresh follow-up for each assisted ordinary practice item, and no routine instruction.
- All expected delayed skill/horizon branches, interruption/review semantics, and no retained branch learning.
- Every model record rebuilt from causal replay, no delayed fitting labels, and feature allowlists.
- One-to-one hidden trace attribution, bounded retained updates, and consistency between resolved observations and simulated evidence.
- Coverage and measured distributions by split, skill, format, delay, hint count, attempt number, and response time.

Coverage reports show missing delayed split/skill/format/horizon cells rather than silently treating them as evaluated. A small pilot can have empty cells after random assessment-format selection. The main and stress coverage results are reported below. Coverage does not establish precise subgroup performance; the memory model has not been fitted.

`reports/test_results.json` stores every actual test result, with `test_output.txt` retaining execution output. `reports/pilot_iteration_01.json` preserves the first development pilot's measured audit. That initial pilot passed 23 categories, but manual review found that the reveal reason was overly specific and the optional introduction needed stored content. Those were corrected, a catalog-integrity check was added, and the final pilot was rebuilt and audited before main generation. Intermediate development-pilot counts are excluded from release totals.

## 13. Reproduction and safe changes

Use the commands in `README.md`. The existing destination refuses overwrite. Generate a separate output directory for another cap, seed, or simulator change. Rebuild and re-audit dependent model exports whenever raw events, rubrics, or causal replay change. The CLI audits each newly built cohort and requires a passed pilot before larger cohort generation.

Generation needs only the Python standard library. The existing interpreter is `2/.venv/bin/python`. Sources are self-contained in this v2 directory. No external package or service, paid API, Jev call, LLM generation, or network access is used.

The recommended next experiment is to fit a simple structured baseline and compare a small time-aware shared memory candidate using the main split, then report log loss, Brier score, calibration, and AUC when defined. Break down results by format, ordinary/follow-up purpose, hint history, and elapsed gap. Keep delayed and stress evaluation separate. Use learner-level uncertainty because a learner contributes many correlated events and assessment branches. This is a proposed next step, not completed training.

## 14. Interpretation limits

Passing the checks means the release is internally consistent with its documented assumptions. It does not mean a human learner behaves this way. The synthetic help policy and its continuation probabilities determine which assisted histories and fresh follow-ups are observed; they can introduce selection effects. More generated learners do not remove simulator bias.

The first experiment does not cover real student answer quality, Jev reliability, unexpected code behavior, richer multi-skill tasks, spontaneous real-world study, a validated forgetting law for Python, or full-year deployed use. Task weights, time references, difficulty labels, learning doses, and help multipliers remain unvalidated. Completion flags are numerical abstractions; assessment branches are counterfactual forks. The architecture permits replacement of the simulator, grading adapter, and memory model through their documented boundaries, but each replacement requires revalidation.

## 15. Executed release results

The results below were read from the completed release reports after full generation and audit.

### Verified totals

Across the four cohorts there are **2,600 simulated learners**, **1,700,389 raw events**, and **471,999 response attempts**. There are 373,249 task presentations, 138,046 delivered hints, and 98,750 retry events. Raw event counts include practice and delayed branches, including review. The main fitting export contains **185,510 eligible first-unaided targets** from 1,600 training learners.

| Cohort / split | Learners | Raw events | Responses | Eligible practice targets | Eligible delayed targets in model export |
|---|---:|---:|---:|---:|---:|
| long_gap / test | 200 | 151,840 | 40,961 | 24,909 | 3,333 |
| main / test | 200 | 127,659 | 35,628 | 23,122 | 3,309 |
| main / train | 1,600 | 1,032,864 | 287,382 | 185,510 | 0 |
| main / validation | 200 | 128,423 | 35,783 | 23,101 | 3,319 |
| pilot / test | 20 | 13,086 | 3,634 | 2,333 | 332 |
| pilot / train | 160 | 103,342 | 28,801 | 18,564 | 0 |
| pilot / validation | 20 | 13,209 | 3,659 | 2,363 | 335 |
| power_law / test | 200 | 129,966 | 36,151 | 23,231 | 3,327 |

Training exports contain zero delayed queries by design. Raw training-learner assessment files remain available solely for auditing. All four full-data audits passed **24 categories each**. All **54 automated tests passed**, with zero failures, errors, or skips. The current release inventory checks **73 v1 files**. Finder metadata is excluded.

### Main training targets by format

| Format | Eligible targets | Full successes |
|---|---:|---:|
| code_completion | 37,016 | 18,110 |
| fill_in_the_blank | 37,101 | 23,221 |
| multiple_select | 36,787 | 19,466 |
| output_prediction | 37,246 | 22,867 |
| single_choice | 37,360 | 30,512 |

These are descriptive sampled outcomes, not prediction-model performance.

### Multiple-hint and retry coverage in main

| Hints recorded on a response | Response count |
|---|---:|
| 0 | 273,101 |
| 1 | 43,842 |
| 2 | 26,143 |
| 3 | 8,441 |
| Unknown | 7,266 |

| Submission number | Response count |
|---|---:|
| 1 | 285,588 |
| 2 | 58,024 |
| 3 | 15,181 |

Counts include practice and delayed branches. There are 1,108 interrupted assessment opportunities in main (875 train, 116 validation, 117 test); their assisted first answers are excluded from independent targets.

### Delayed coverage and pilot limits

The main, power-law, and long-gap cohorts have no empty validation/test skill × format × delay query cells. The smaller pilot has five empty cells in its 20-learner test partition. The complete list is preserved in `reports/RELEASE_SUMMARY.json`; these cells are not treated as evaluated. These are coverage counts, not guarantees of sufficient sample size per subgroup.

### Readable example of the revised sequence

The controlled `fixture_three_hints` history records the following order. Its first query target is failure even though the third assisted response succeeds.

| Event | Attempt | Cumulative hints | Outcome or role |
|---|---:|---:|---|
| Present original question | — | 0 | Predict using earlier history |
| Response, then grade | 1 | 0 | Independent failure; primary target 0 |
| Hint 1, then hint 2 (requests recorded when applicable) | — | 1, then 2 | Separate chronological help events |
| Retry, response, grade | 2 | 2 | Assisted failure |
| Hint 3 | — | 3 | Another separate hint event |
| Retry, response, grade | 3 | 3 | Assisted success; does not replace target 0 |
| New same-skill question after 60 seconds | 1 | 0 | Fresh independent target, with the earlier assisted history as context |

### Executed automated test inventory

Each entry below passed in the recorded run. The machine-readable result and execution output are retained under `reports/`.

#### CatalogTests — 11 passed

| Test | Result |
|---|---|
| `test_catalog_covers_all_skill_format_difficulty_pool_combinations` | Passed |
| `test_completion_reference_passes_every_authored_case` | Passed |
| `test_fill_keys_reconstruct_reference_code` | Passed |
| `test_invalid_grade_evidence_rejected` | Passed |
| `test_multiple_select_all_and_none_earn_zero` | Passed |
| `test_multiple_select_wrong_selection_reduces_credit` | Passed |
| `test_output_keys_match_trusted_reference_execution` | Passed |
| `test_single_choice_exact_scoring` | Passed |
| `test_snippets_and_criteria_are_variable` | Passed |
| `test_weight_normalization_and_positive_penalty_totals` | Passed |
| `test_weighted_partial_credit_does_not_imply_full_success` | Passed |

#### DiskTests — 9 passed

| Test | Result |
|---|---|
| `test_checksum_tampering_detected` | Passed |
| `test_evaluation_loader_has_all_clean_delayed_labels` | Passed |
| `test_full_audit_passes` | Passed |
| `test_future_prefix_tampering_detected_even_with_new_checksum` | Passed |
| `test_overwrite_refused` | Passed |
| `test_regeneration_byte_identical` | Passed |
| `test_splits_disjoint_and_balanced` | Passed |
| `test_training_loader_returns_only_input_and_binary_label` | Passed |
| `test_training_records_exclude_delayed_queries` | Passed |

#### EvidenceTests — 16 passed

| Test | Result |
|---|---|
| `test_assessment_help_after_answer_does_not_replace_label` | Passed |
| `test_assessment_request_before_answer_is_incomplete` | Passed |
| `test_assisted_retry_keeps_original_independent_failure_label` | Passed |
| `test_delayed_queries_share_only_original_practice_prefix` | Passed |
| `test_hidden_or_text_fields_in_features_rejected` | Passed |
| `test_hint_before_first_answer_has_no_independent_label` | Passed |
| `test_invalid_event_mutations_rejected` | Passed |
| `test_missing_times_remain_null` | Passed |
| `test_other_skill_events_do_not_erase_elapsed_gap` | Passed |
| `test_prediction_cannot_see_current_response_or_grade` | Passed |
| `test_query_identity_feature_rejected` | Passed |
| `test_renaming_learner_does_not_change_network_inputs` | Passed |
| `test_retry_without_retry_event_rejected` | Passed |
| `test_unjustified_instruction_rejected` | Passed |
| `test_unknown_assistance_excludes_target` | Passed |
| `test_unresolved_grade_excludes_target` | Passed |

#### SimulationTests — 18 passed

| Test | Result |
|---|---|
| `test_assessment_branch_does_not_modify_practice_or_learn` | Passed |
| `test_each_assisted_practice_has_fresh_unaided_followup` | Passed |
| `test_formats_have_distinct_guessing_time_and_help_rules` | Passed |
| `test_four_hint_configuration` | Passed |
| `test_help_changes_current_probability` | Passed |
| `test_hint_event_has_no_retained_gain` | Passed |
| `test_initial_strength_varies_and_is_not_assumed_zero` | Passed |
| `test_invalid_configuration_rejected` | Passed |
| `test_learning_is_bounded_and_zero_dose_is_noop` | Passed |
| `test_no_routine_instruction_and_question_first` | Passed |
| `test_querying_strength_does_not_change_state` | Passed |
| `test_requested_lesson_content_exists` | Passed |
| `test_requested_lesson_has_no_direct_learning_gain` | Passed |
| `test_retention_half_life_and_alternative_tail` | Passed |
| `test_seed_reproduces_full_history_and_hidden_state` | Passed |
| `test_strength_changes_current_probability` | Passed |
| `test_three_hints_and_two_retries_are_chronological` | Passed |
| `test_zero_help_and_retry_caps` | Passed |
