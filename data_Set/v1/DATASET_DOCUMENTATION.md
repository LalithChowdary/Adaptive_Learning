# Python learner-memory dataset: design, construction, and validation

Version: 1.0.0. Built for the project restart authorized on 7 November 2026.

This document explains what we generate, why each part exists, how the simulator works, how to use the files, and what the executed checks establish. The final validation section is populated from the actual reports after generation. The central cross-chat decision record remains [PROJECT_DECISIONS.md](../../../PROJECT_DECISIONS.md); this document is the technical record for this dataset.

## 1. What this dataset is

This is **numerical simulated learner-history data**, organized around six Python skills. It is not collected student behavior, generated Python answer text, executed student code, or output from Jev. No LLM/API calls, model fitting, or PPO training are part of this build.

We want to train a shared memory predictor to answer: **given earlier available evidence for one learner and skill, and the characteristics of a new task, will the learner fully succeed on their first attempt without help?**

The label is the later simulated independent response outcome, 0 or 1. It is not a hand-assigned mastery percentage or the simulator's hidden knowledge value. A probabilistic predictor can learn from binary outcomes. Prior partial results, failed attempts, assistance, and instruction remain useful history.

Python is the initial content domain. The predictor's feature format remains independent of task wording and subject identities. Numerical behavior is produced by general equations, not by interpreting Python code: skill names and authored tasks do not establish that simulated students reproduce Python reasoning. Cross-subject and real-student validity remain untested.

## 2. Approved choices and implementation defaults

The user confirmed simulated numerical outcomes without answer generation/Jev calls, the six skills below, and a small pilot before expansion. They then authorized creating the dataset in `code/3` and requested complete methods and executed test results.

The following counts and parameter ranges are implementation choices for this version, not empirically established requirements or measurements of people:

| Cohort | Learners | Purpose | Train / validation / test |
|---|---:|---|---|
| `pilot` | 200 | Check generator, schemas, and distributions | 160 / 20 / 20 |
| `main` | 2,000 | Initial memory-model training and comparison | 1,600 / 200 / 200 |
| `power_law` | 200 | Test-only changed forgetting mechanism | 0 / 0 / 200 |
| `long_gap` | 200 | Test-only longer gaps under exponential decay | 0 / 0 / 200 |

There are **2,600 distinct learner IDs**, with different cohort seeds and identity prefixes. Do not merge the pilot into the main experiment or train on stress cohorts. Different synthetic seeds do not make these independently validated human populations.

Why expand beyond 200? With an 80/10/10 split, a 200-person pilot leaves only 20 test learners. Two thousand provides 200 main test learners and more varied histories while remaining manageable locally. It is an initial experimental scale, not a guarantee of sufficient statistical power. Later train on nested subsets of the main training learners, compare validation learning curves and calibration, and expand if improvements have not stabilized. Repeated rows from a learner are correlated; report learner-level uncertainty, not an inflated independent sample count based on event rows.

Every learner receives 16 new task IDs per skill, giving **96 practice opportunities**, plus one independent assessment per skill at each of three delays, giving **18 assessment opportunities**. Retries add responses; instruction/help/grade events add rows. Thus:

- Pilot: 19,200 practice + 3,600 assessment opportunities.
- Main: 192,000 practice + 36,000 assessment opportunities.
- Each stress cohort: 19,200 practice + 3,600 assessment opportunities.
- Total: 249,600 practice + 46,800 assessment opportunities, or 296,400 first attempts before retries.

The number of eligible training targets is smaller than practice opportunities because unresolved grades and unknown assistance are excluded. Final measured counts appear in the validation section.

## 3. Python task definitions and why they exist

| Skill | Assessed behavior | Example |
|---|---|---|
| Range boundaries | Determine values included by a bounded range | Printed `list(range(start, stop, step))` |
| Loop tracing | Track repeated updates | Printed value after a fixed-count loop |
| Accumulator initialization | Include the initial accumulator value | Printed sum starting from a nonzero value |
| Conditional evaluation | Select the branch of a condition | Which of two labels is printed |
| Function return values | Evaluate a simple return expression | Printed result of a function call |
| List indexing | Locate a value using an index | Printed element at a positive/negative index |

`catalog.py` authors 432 task records: six skills × three difficulty labels × 24 parameterized variants. Each task asks for the outputs of two snippets assessing the same skill. Each correct output is worth 0.5; both are required for full success. These two criteria contribute to one response, not two independent student attempts.

Per skill and difficulty, variants 0–15 belong to practice and 16–23 to assessment. Each learner samples practice task IDs without replacement, and receives distinct assessment task IDs across delays. However, snippets can be equivalent or repeated after parameterization; both pools share template families. This is **not a held-out-template or conceptual-transfer benchmark**. The simulator does not model memory for a specific repeated snippet.

The task bank provides explicit scope and checkable reference answers. Tests execute only these trusted, locally authored snippets and compare all 864 outputs with their references. No untrusted student code is executed. Task answers do not enter memory features or determine numerical sampled answers by text evaluation.

Difficulty labels are authored estimates (`easy`, `medium`, `hard`). Reference active times are 30, 60, and 90 seconds respectively. These are unvalidated design assumptions, not human timing benchmarks. We preserve that provenance. Reference time and difficulty are deliberately correlated in this initial bank; conclusions about the independent value of these features require further variation.

## 4. Simulator components and responsibilities

All numerical components are in `simulator.py`; their responsibilities are separated through methods and explicit output boundaries:

1. **Learner parameter sampling:** creates correlated learner-level attributes with skill-level variation.
2. **Retention dynamics:** evaluates current hidden strength after elapsed time.
3. **Instruction/practice learning:** updates only the directly practised skill using bounded gains.
4. **Response/assistance simulation:** samples criterion outcomes using strength, difficulty, help, and variability.
5. **Timing simulation:** samples active response durations; elapsed session gaps are separate.
6. **Grading adapter:** turns sampled criterion truth into partial credit/full success or an unresolved result.
7. **Observation process:** emits permitted chronological events and explicit missing values.
8. **Behavior policy/scheduler:** chooses tasks and instruction without reading hidden strength.
9. **Assessment forks:** check independent performance after time gaps without contaminating later assessments.

The two retention mechanisms are alternatives within the simulator, not two trained memory models. Their numerical parameters are not fitted to real learners.

## 5. Sampled learner variation

One random stream samples parameters. Each learner receives base starting strength, learning rate, retention, and speed. Each of their six skill records then varies around that base, so neither all learners nor all skills behave alike.

| Parameter | Sampling rule |
|---|---|
| Base initial strength | Beta(2,2) |
| Skill initial strength | Base + Uniform(-0.2,0.2), clipped to [0.02,0.95] |
| Base learning rate | Uniform(0.08,0.28) |
| Skill learning rate | Base × LogNormal(0,0.25), clipped to [0.04,0.4] |
| Base half-life | Log-uniform from 5 to 60 days |
| Skill half-life | Base × LogNormal(0,0.35), clipped to [2,120] days |
| Base speed multiplier | LogNormal(0,0.25) |
| Skill speed multiplier | Base × LogNormal(0,0.15), clipped to [0.4,2.5] |
| Immediate help responsiveness | Uniform(0.6,1.8) |
| Retained-help fraction | Exactly 0 for a 25% mixture; otherwise Uniform(0.05,0.5) |
| Instructional action doses | Independently Uniform(0.35,1.1) for each learner-skill-action combination |

`LogNormal(0,s)` means the underlying normal has mean 0 and standard deviation s. These are transparent choices to create variation, not fixed labels such as “good” and “bad” students. Action doses vary so one teaching action is not assigned the largest gain for everybody. The simulator still contains strong assumptions and is not sufficient for a claim about effective human teaching.

## 6. Forgetting and learning equations

Let `k0` be hidden strength at its last update, `dt` the elapsed days, and `h` the skill's hidden half-life.

Primary retention:

```text
k(now) = k0 * 2 ** (-dt / h)
```

Changed-mechanism stress test:

```text
k(now) = k0 / (1 + dt / h)
```

Both retain half the starting strength after h days, but differ at longer delays. Neither sets strength to 1 after exposure. Strength is evaluated lazily from an anchored timestamp; querying it does not rewrite the state or apply decay twice. In this version half-life is fixed per learner-skill; practice increases strength, not half-life. We do not claim to model all spacing effects or durable consolidation.

The motivation is the exponential half-life idea in [Settles and Meeder (2016), section 3.3](https://aclanthology.org/P16-1174.pdf). Their recall formula and trained language-learning model are not reproduced here. We apply decay to hidden numerical strength and separately generate task responses. The power-law alternative is a declared robustness mechanism, not a fitted psychological law.

Learning after an activity uses:

```text
k_after = k_before + (1 - k_before) * (1 - exp(-learning_rate * dose))
```

This yields bounded gains that become smaller near strength 1. Instruction uses the sampled action dose. Unaided response practice uses dose 0.45. Assisted response practice uses `0.45 * retained_help_fraction`; delivering help uses `0.25 * retained_help_fraction`. Help can therefore raise immediate success without any retained benefit. Assessments cause no learning.

Only the directly taught skill changes. Prerequisite review for loop tracing targets range boundaries; prerequisite review for accumulator initialization targets loop tracing. For other skills, a selected prerequisite-review action falls back to explanation because no prerequisite is modeled. These are curriculum mappings, not a claim that skills are cognitively independent in real people.

## 7. Response and assistance simulation

Hidden task demand is the difficulty center (0.25, 0.45, 0.65) plus a task-stable Uniform(-0.12,0.12) offset. A shared Normal(0,0.65) response shock correlates the two rubric criteria.

```text
z_j = 5 * (strength - demand) + 0.65 + help_boost + shared_noise + criterion_offset_j
p_j = 0.03 + 0.94 * sigmoid(z_j)
criterion_j ~ Bernoulli(p_j)
criterion_offsets = [+0.15, -0.15]
```

The 0.03 and 0.97 bounds allow slips and guesses. Full success requires both sampled criteria. Criterion outcomes are conditionally independent given the shared shock, and correlated marginally. Task demand, probabilities, shock effects, and hidden strength never enter the model feature export.

Conceptual help adds `1.0 * help_responsiveness`; scaffolding adds `1.6 * help_responsiveness`. Answer reveal forces the current criteria to true but does not create perfect retained knowledge. Assistance resets before each new task. At most one assisted retry is allowed per practice opportunity in this version.

The current hint count is therefore 0, 1, or unknown; this dataset does **not** establish performance with many hints or long retry chains. We record the count field so later versions can extend it without changing the interface.

## 8. Simulated grading and Jev

Resolved grades use `score = (criterion_a + criterion_b)/2` and `full_success = criterion_a AND criterion_b`. Three percent of grade events are marked unresolved; they contain no score, criterion judgments, or success label. Their hidden truth is retained only in administration exports for audit.

Every grade uses `grader = simulated_rubric_v1`, `rubric_id = two_required_outputs_v1`, and null grader confidence. There are no fabricated Jev confidence values. Resolved grades are idealized and error-free relative to the generated criterion truth. Unresolved outcomes are not a complete model of grading errors; missingness is initially random.

A future Jev adapter can emit the same structured result from actual answer text. This requires independent rubric cases and grading validation. That replacement will not prove the memory model transfers without further evaluation. We do not call Jev merely to “grade” an already generated numerical outcome.

## 9. Durations, sessions, scheduling, and missingness

Active response time is sampled before grade availability:

```text
duration = reference_seconds * speed * (1.25 - 0.5 * strength)
           * LogNormal(0,0.35) * (0.75 if assisted else 1)
duration = clip(duration, 3, 600) seconds
```

Duration depends on strength in this simulation, but variation prevents a deterministic speed-to-grade mapping. This is an assumption to test, not evidence that faster students understand better. We store raw active time and derive active/reference time when both are available. The current response's time is never used to predict that same response.

Instruction takes 45 simulated seconds and help takes 15 seconds. Grades become available 0.1, 2, or 30 seconds after a response. The scheduler waits for grading before continuing; overlapping tasks and late asynchronous grade corrections are not modeled in this version.

Each learner completes eight sessions of 12 first-attempt opportunities; retries do not consume additional first-attempt opportunities. Primary session gaps are uniformly chosen from 0.25, 1, or 3 days. Long-gap stress uses 0.25, 1, 3, 14, or 60 days. Final span distributions are reported rather than claiming every learner spans a year.

Half the learners in every partition use blocked practice (all 16 tasks for one skill together), and half use mixed practice (each consecutive block of six tasks covers all skills once in shuffled order). These are behavior policies, not randomized paired evidence of schedule benefit. Instruction precedes a task with probability 0.65; its action is sampled uniformly from four standalone actions. Partial scaffold is available as help after failure. The scheduler never consults hidden learner strength.

After a resolved unsuccessful first response, an assisted retry occurs with probability 0.65. Help is conceptual/scaffold/reveal with probabilities 0.5/0.4/0.1. The retry policy uses an already available grade, not hidden truth. Unresolved grades do not trigger this retry rule.

Missingness rates: active response time 4%; task reference duration 4%; assistance metadata 2%; unresolved grading 3%. These independent missingness draws create imperfect observations, not an empirically fitted missingness process. Unknown assistance is not independent; its response cannot be a primary unaided target. Raw help events remain visible even when a response's assistance snapshot is unknown; eligibility is intentionally conservative.

## 10. What one row means and how data are stored

| Files | Row meaning | Intended use |
|---|---|---|
| `observations/<split>.jsonl.gz` | One practice event, contiguous by learner and chronological within learner | Auditing or causal replay |
| `assessments/<split>.jsonl.gz` | One event from an independent delayed assessment branch | Evaluation/audit; never concatenate branches as one learner history |
| `model_data/<split>.jsonl.gz` | One learner-skill feature sequence with query-prefix pointers and separate labels | Memory training/evaluation via loader |
| `administration/<split>.jsonl.gz` | One learner's hidden parameters, traces, and assessment truth | Audit only; never model input |
| `config.json` | Cohort settings and frozen learner/schedule assignments | Reproduction |
| `manifest.json` | Data and source-code SHA-256 hashes, version, runtime | Integrity/reproduction |
| `example_learner.json` | First events and one skill sequence | Human inspection |
| `audit.json`, `AUDIT.md` | Measured counts, distributions, checks | Validation evidence |

All times are synthetic Unix seconds beginning at 1794009600.0. Every event has learner, skill, course, session, interaction, branch, and event identifiers, plus occurrence and availability times. These organize data; they are excluded from the predictor's numerical features. Raw events have a strict per-type field allowlist.

The feature sequence includes event type, elapsed gap since the prior same-skill available observation, session change, difficulty/type/reference duration, action, attempt number, time measurements, assistance, grade status, partial credit, and full success. Non-applicable/unavailable fields remain null. Features are not standardized yet; a future training pipeline must encode categories, represent missingness, and fit numerical transformations using training learners only.

### Preventing future information from leaking into a prediction

Each query is created at task presentation, before that task event is appended. `history_end` is the exclusive end of its earlier same-skill feature history. Query inputs also contain target-task characteristics and time since the previous same-skill observation/response. The eventual label is attached separately only after a resolved first independent response.

**Never feed a complete stored sequence to an earlier query.** Later events in that sequence are future information. `materialize_query(record, query)` returns only `events[:history_end]` and the allowed query metadata. `loader.load_training_examples()` supplies those inputs and separate binary labels. The test suite deliberately alters a prefix pointer to include future events and checks that the independent audit rejects it even after the file checksum is updated.

Instruction/help with no answer is retained. No target is created for a retry. Missing grades/assistance do not silently become negative labels. Correct assisted retries remain history while the original unaided first response remains that task's prediction target.

## 11. Delayed assessment design

At the end of practice, copy the learner's final hidden state separately for **each skill and each delay: 1, 7, and 30 days**. Each copy receives one new assessment task ID without hints and without further learning. These delay choices are experimental defaults, not measured human forgetting thresholds.

Every probe starts from the end of the whole practice phase. The gap since that skill's last response may be longer, especially for blocked schedules; the real same-skill gap is retained. One probe cannot strengthen another probe's learner state. Probe queries use only the pre-assessment practice evidence, never outcomes from another delay or skill's assessment.

All raw delayed events are preserved for audit. **Training learners' delayed labels are absent from fitting exports.** Validation/test model rows include assessment queries; their observations do not extend the practice feature history. Stress cohorts are test-only. Choose tuning and calibration on main validation, then report main test and predeclared stress tests without using those results to tune the same experiment.

A fork is not an additional independent learner. One task per skill/delay gives sparse individual evidence; cohort-level metrics can be calculated but precise individual mastery claims cannot.

## 12. Determinism and reproduction

Seeds are 2026110701 (pilot), 2026110702 (main), 2026110703 (power-law), and 2026110704 (long-gap). SHA-256 derives stable per-learner streams for parameters, policy, tasks, response outcomes, timings, missingness, and gaps. Assessment forks get separate streams. Split assignment has its own seeded stream and never inspects grades.

Compressed JSONL uses sorted keys and a fixed gzip timestamp. Regeneration under the recorded runtime/source is tested byte-for-byte on a complete smaller cohort. Source code and cohort data hashes are recorded; no existing cohort is silently overwritten. Hashes detect accidental changes, not adversarial authenticity.

This folder is self-contained: it does not import or reuse the old `research/` simulator or old pilot. Standard-library Python is sufficient; the validated runtime is Python 3.12.14 from the existing project environment. No external Python packages are required.

From `code/3/data_Set/v1`:

```bash
../../../2/.venv/bin/python run_tests.py
../../../2/.venv/bin/python build_fixtures.py
../../../2/.venv/bin/python build_dataset.py --cohort all --output dataset_reproduction
../../../2/.venv/bin/python audit_dataset.py dataset_reproduction/main
```

`--cohort` also accepts `pilot`, `main`, `power_law`, and `long_gap` separately. Use a fresh output directory for reproduction. The generated data live under `dataset/` in this delivered build.

Example future training input inspection (does not train anything):

```python
from loader import load_training_examples
features, label = next(load_training_examples("dataset/main"))
# features contains history + target_task + elapsed gaps; label is 0 or 1.
```

Do not train on `administration/`, the entire raw history at once, stress cohorts, or delayed training-assessment files. A future recurrent model can learn with chronological chunks and persist a state per learner-skill, but no such model is fitted here.

## 13. Test plan and what a pass means

Tests are executed by `run_tests.py`, which saves each real test name and status in `reports/test_results.json` and verbose execution output in `reports/test_output.txt`. They are not manually entered pass claims.

| Group | Checks and why they matter |
|---|---|
| Retention/learning | Half-life, zero gap, bounded gains, invalid parameters, read-only queries, partitioned time, separate skill updates, varied learners |
| Evidence/causality | Assisted success, unknown help, unresolved labels, partial credit, future result isolation, missing values, hidden-field rejection, duplicate events, false independence, rubric derivation, timing order, help reset, cross-course reuse, retained time gaps, identifier-independent features |
| Generation/content | All authored reference outputs, three controlled histories, reproducibility, exact skill budgets, balanced splits, unique task IDs, matched-randomness effects of strength/help, no assessment learning, independent forks |
| Disk/data | Full replay audit, safe loader behavior, delayed-label exclusion, held-out loading, overwrite protection, byte-identical reproduction, checksum tampering, future-feature tampering despite a revised checksum |

The three controlled fixtures prescribe outcomes to check data handling: independent success across courses, assisted success followed by independent failure, and return after a gap with another skill in between. They are stored separately in `fixtures/`; they are not mixed into random cohorts or claimed as evidence of human behavior.

After each cohort is written, `audit_dataset.py` reloads all files. It checks hashes, schema, event chronology, grade linkage, assistance consistency, task budgets, schedules, learner separation, assessment isolation, and every exported feature sequence/query against fresh causal replay. This is a complete consistency audit, not a sample of the main dataset. Its validation categories are reported separately from unit-test counts.

Passing establishes the specified mechanics and data-handling contracts for these tests. It does not establish human fidelity, Python reasoning, grader accuracy, cross-subject transfer, prediction quality, or PPO effectiveness. Outcome distributions are reported descriptively; we do not search test seeds until obtaining favorable success rates.

## 14. Known limits and next work

- Synthetic generation assumptions create the patterns a model can learn. More rows alone cannot fix a wrong learner model.
- Half-lives are fixed, skills do not automatically transfer knowledge, and action effectiveness is stylized.
- The behavior policy is declared and mostly random; this is not yet a validated RL environment or a causal teaching-policy comparison.
- One hint/one retry and random missingness are simplifying limits.
- Responses are numerical criteria, not actual student answers; idealized resolved grading omits grader mistakes.
- Task pools share templates and may include equivalent snippets; task-ID separation is not content generalization.
- Timing/reference and difficulty assumptions require calibration before real use.
- Training histories cover their generated spans, not automatically a full year. Long-gap stress results must be evaluated before claiming long-term reliability.
- There is no trained memory model, Jev evaluation, LLM tutor integration, or PPO result in this deliverable.

Next: inspect this documentation and the observed audit distributions, then train constant/statistical/GRU candidates on the main fitting export. Use held-out learners, compare calibration by gap/skill/assistance history, and evaluate both stress cohorts. Dataset size can then be revisited using learning curves rather than a fixed arbitrary target.

## 15. Executed validation results

Completed on 7 November 2026. These numbers are read from the generated audit and test reports; no model has been trained.

**Automated tests: 48/48 passed; 0 failures; 0 errors; 0 skipped.** The suite was rerun after dataset construction.

**Full dataset audits: all four cohorts passed all 17 validation categories each (68 cohort-category passes).** These are distinct from the 48 automated test cases. Global checks also verified disjoint learner IDs across cohorts and that the saved task catalog matches its source.

| Cohort | Learners | Event rows | First-attempt opportunities | Responses including retries |
|---|---:|---:|---:|---:|
| [long_gap](dataset/long_gap/AUDIT.md) | 200 | 97,486 | 22,800 | 28,323 |
| [main](dataset/main/AUDIT.md) | 2,000 | 924,179 | 228,000 | 266,399 |
| [pilot](dataset/pilot/AUDIT.md) | 200 | 91,932 | 22,800 | 26,481 |
| [power_law](dataset/power_law/AUDIT.md) | 200 | 92,372 | 22,800 | 26,658 |

Total: **2,600 learners, 1,205,969 event rows, and 347,861 responses** including retries. These are synthetic observations, not human participants.

### Main memory-model partitions

| Partition | Learners | Eligible practice targets | Eligible delayed targets in model export |
|---|---:|---:|---:|
| train | 1,600 | 146,035 | 0 |
| validation | 200 | 18,284 | 3,424 |
| test | 200 | 18,261 | 3,418 |

The main training loader provides **146,035** causal input/label pairs. Validation/test delayed targets use only earlier practice evidence. Raw training assessments are retained for auditing, not model fitting. Stress-cohort practice targets are evaluation-only despite having the same record shape.

### Main delayed outcomes by gap

| Partition/delay | Eligible responses | Full successes | Success fraction |
|---|---:|---:|---:|
| test/1d | 1,140 | 813 | 0.713 |
| test/30d | 1,143 | 344 | 0.301 |
| test/7d | 1,135 | 642 | 0.566 |
| validation/1d | 1,151 | 816 | 0.709 |
| validation/30d | 1,132 | 315 | 0.278 |
| validation/7d | 1,141 | 656 | 0.575 |

These are generated outcome rates, not model accuracy and not evidence that a teaching policy improved learning. Tasks vary in difficulty and learners share histories across delays.

### Main observation quality

| Measure | Observed count |
|---|---:|
| Partial-credit grades | 63,998 |
| Unresolved grades | 7,990 |
| Responses with unknown assistance | 5,372 |
| Missing active durations | 10,625 |
| Missing reference durations | 9,159 |
| Assisted retries | 38,399 |

### Actual practice time spans

| Cohort | Minimum days | Median days | Maximum days |
|---|---:|---:|---:|
| pilot | 2.61 | 9.62 | 17.16 |
| main | 1.89 | 10.11 | 19.17 |
| power_law | 3.37 | 10.33 | 19.14 |
| long_gap | 7.59 | 107.19 | 269.12 |

The longest observed practice span is about 269 days; adding the last assessment extends it by 30 days. This release does **not** contain a full year of practice for every learner or establish reliable year-long prediction. Longer histories require an explicitly expanded version.

### Individual executed tests

| Test | Result |
|---|---|
| `DiskAuditTests.test_causal_audit_rejects_future_features_even_with_new_checksum` | passed |
| `DiskAuditTests.test_complete_regeneration_is_byte_identical` | passed |
| `DiskAuditTests.test_delayed_labels_absent_from_training_records` | passed |
| `DiskAuditTests.test_evaluation_loader_has_delayed_targets` | passed |
| `DiskAuditTests.test_full_audit_passes` | passed |
| `DiskAuditTests.test_modified_file_is_rejected` | passed |
| `DiskAuditTests.test_overwrite_is_refused` | passed |
| `DiskAuditTests.test_training_loader_only_exposes_permitted_input` | passed |
| `EvidenceTests.test_assisted_success_is_not_unaided_target` | passed |
| `EvidenceTests.test_cross_course_reuses_same_skill_record` | passed |
| `EvidenceTests.test_duplicate_event_rejected` | passed |
| `EvidenceTests.test_false_independence_rejected` | passed |
| `EvidenceTests.test_first_query_has_empty_history` | passed |
| `EvidenceTests.test_future_grade_availability_reordering_rejected` | passed |
| `EvidenceTests.test_future_result_cannot_change_prior_input` | passed |
| `EvidenceTests.test_intervening_skill_does_not_erase_elapsed_gap` | passed |
| `EvidenceTests.test_missing_duration_and_reference_stay_missing` | passed |
| `EvidenceTests.test_model_features_reject_hidden_field` | passed |
| `EvidenceTests.test_new_task_resets_help_and_attempt` | passed |
| `EvidenceTests.test_partial_credit_is_not_full_success` | passed |
| `EvidenceTests.test_query_rejects_identity_feature` | passed |
| `EvidenceTests.test_raw_events_reject_hidden_field` | passed |
| `EvidenceTests.test_reveal_does_not_create_perfect_hidden_knowledge` | passed |
| `EvidenceTests.test_student_id_renaming_does_not_change_features` | passed |
| `EvidenceTests.test_unknown_assistance_excluded` | passed |
| `EvidenceTests.test_unresolved_grade_has_no_target` | passed |
| `EvidenceTests.test_wrong_rubric_score_rejected` | passed |
| `GenerationTests.test_assessment_cannot_modify_training_state` | passed |
| `GenerationTests.test_assessment_history_has_no_other_probe` | passed |
| `GenerationTests.test_assessments_do_not_gain_strength_from_response` | passed |
| `GenerationTests.test_catalog_answers_match_trusted_snippet_execution` | passed |
| `GenerationTests.test_help_improves_current_probability_under_matched_randomness` | passed |
| `GenerationTests.test_higher_strength_increases_probability_under_matched_randomness` | passed |
| `GenerationTests.test_no_repeated_task_ids_per_learner` | passed |
| `GenerationTests.test_schedule_has_exact_budget_per_skill` | passed |
| `GenerationTests.test_seed_reproduces_all_observations` | passed |
| `GenerationTests.test_split_is_disjoint_and_balanced` | passed |
| `GenerationTests.test_three_controlled_histories_validate` | passed |
| `RetentionTests.test_half_life_is_half_strength` | passed |
| `RetentionTests.test_instruction_does_not_update_unrelated_skill` | passed |
| `RetentionTests.test_invalid_config_rejected` | passed |
| `RetentionTests.test_invalid_time_or_half_life_rejected` | passed |
| `RetentionTests.test_learning_is_bounded_and_zero_dose_has_no_gain` | passed |
| `RetentionTests.test_parameters_vary_between_learners_and_skills` | passed |
| `RetentionTests.test_partitioned_time_matches_single_advance` | passed |
| `RetentionTests.test_power_law_agrees_at_half_life_but_differs_later` | passed |
| `RetentionTests.test_repeated_queries_do_not_accelerate_forgetting` | passed |
| `RetentionTests.test_zero_gap_preserves_strength` | passed |

### Evidence files

- [Machine-readable test results](reports/test_results.json)
- [Verbose test execution output](reports/test_output.txt)
- [Top-level dataset manifest and totals](dataset/DATASET_MANIFEST.json)
- [Pilot full audit](dataset/pilot/AUDIT.md)
- [Main full audit](dataset/main/AUDIT.md)
- [Power-law full audit](dataset/power_law/AUDIT.md)
- [Long-gap full audit](dataset/long_gap/AUDIT.md)

All audit JSON files additionally include per-skill target counts, positive outcomes, response-duration distributions, and same-skill gap distributions. Source/data fingerprints are in each cohort manifest.
