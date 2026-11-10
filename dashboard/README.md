# Recall — interactive Python memory dashboard

A local learner-facing dashboard for the frozen v2.1 time-decayed GRU. Answer actual Python questions, receive deterministic rubric grades, request prepared hints, and inspect the predictions produced from your own evidence. No LLM or API key is required.

For the distinction between shared weights and personal states, measured starting probabilities, and difficulty-specific predictions, see [How the memory model works](MEMORY_MODEL_EXPLAINED.md).

## Start

From the project root:

```sh
3/dashboard/start.command
```

Open **http://127.0.0.1:8765**. Create a profile using a name or nickname, select a skill, and start a question. You can also double-click `start.command` in Finder. Keep its Terminal window open while using the dashboard; Ctrl+C stops it. On a fresh machine, set up the existing `3/model/.venv` using the model README first.

The launcher uses the existing model environment and needs no additional packages or frontend build. It uses MPS when available, otherwise CPU for inference. To require the Mac GPU explicitly:

```sh
3/dashboard/start.command --device mps
```

Other options: `--port 8766`, `--db /absolute/path/sessions.sqlite3`, `--max-hints 2`, `--max-retries 1`. The authored bank supports 0–3 hints; the default is three hints and two retries, for three attempts total. The retry setting accepts 0–10. Changing the caps does not make evidence outside the training conditions empirically validated.

## What is included

- Six Python skills, three difficulty settings, and five question formats. The existing v2.1 bank has 2,160 authored variants: 1,440 practice and 720 independent-check items.
- Single-choice and multiple-select controls, output fields, integer blanks, and a Python completion editor.
- Question first; no automatic introductory instruction. Options are shuffled per question, with stable IDs retained for grading.
- Up to three progressively more explicit, reviewed skill-specific hints, refined from the generic dataset hints. Hint text and `dashboard-hints-1` provenance are stored with each delivery.
- Separate, chronological help requests, hint deliveries, responses, grades, retries, and solution reveals.
- Independent-check grades preserve the first unaided result. Asking for help before submitting marks the check incomplete and switches to review. Retrying after a saved independent grade is review and does not overwrite that grade.
- Fresh questions exclude task IDs already opened by the same profile, including skipped items. If a filter is exhausted, choose a different format or difficulty; the app does not silently repeat tasks.
- Skill overview, event-derived performance counts, recent question history, and JSON export of actual interactions.
- A 0–60-day memory explorer using the same fixed reference question at every horizon. Viewing it never changes the saved state or advances real timestamps.
- Saved profiles and interactions survive browser/server restarts. Answer drafts and estimated focused time are retained in browser local storage; submitted evidence is stored in SQLite.

## What the numbers mean

**Grades** come from the deterministic grading adapter. **Predictions** come from the frozen memory model. The model never grades answer text.

A question's pre-answer probability is computed before presentation and saved with that task event. It is hidden in the interface until submission. It predicts full success on the first unaided attempt only; it does not predict the probability of succeeding with hints or on retries.

The skill overview's prediction is the mean over **five fixed medium-difficulty reference tasks**, one per format, using that learner's current same-skill evidence. Those reference tasks are the first assessment-pool variant for each skill/format in the catalogue. All five have equal aggregation weight. This is a task-set summary, not mastery, grader confidence, or a validated cross-skill scale. The card says *Unassessed* until at least one resolved grade is recorded.

The memory explorer uses a fixed medium-difficulty output-prediction reference task. It starts from the current evidence anchor and queries temporary decayed states. Its curve is a model forecast, not a measured human forgetting curve, and is not mathematically constrained to be monotonic. Empty histories use the shared learned prior.

## Grading

| Format | Rule |
|---|---|
| Single-choice | Exact answer-option ID; 0 or 1 credit |
| Multiple-select | Sum weights of selected correct options, subtract weights of selected wrong options, clip to [0,1]. Full success requires the exact set. |
| Output prediction | Compare each answer against its authored output; whitespace around the answer and equivalent integer/list formatting are normalized. Criteria retain their unequal weights. |
| Fill-in-the-blank | Parse an integer literal and compare against the specified missing literal. Boolean values and arbitrary expressions are not accepted as integers. |
| Code completion | Interpret `solve(x)` using the bounded exercise interpreter and compare against the task's 2–4 behavior tests. Use their authored weights. |

Answer keys, solutions, and selection rubric kinds stay on the server until the relevant result/solution is shown. Criterion flags are feedback after submission. Solution reveal closes the question and is recorded separately.

### Completion interpreter boundary

Submitted text is parsed with Python `ast` and interpreted directly. It is **never passed to `exec`, `eval`, a shell, or the host Python execution engine**. This is a deliberately limited exercise interpreter, not a full Python runtime or an OS sandbox.

Supported: a single unannotated `def solve(x)`, assignments to names, return, pass, if/else, for loops, arithmetic (+, −, ×, /, //, %), comparisons, boolean operators, lists/tuples, indexing/slicing, and positional calls to `range`, `list`, `len`, `sum`, `min`, `max`, `abs`. No imports, attributes, file/network access, arbitrary calls, recursion, while loops, comprehensions, or powers. Limits: 8,000 source characters, 1,200 AST nodes, 10,000 interpreted operations per case, 2,000 sequence elements, 256-bit integers.

Unsupported syntax and resource limits produce **unresolved**, with no score and no independent target label. They may still update memory as an attempt with an unknown grade. Syntax errors or supported code that gives a wrong result/runtime error produce resolved failed criteria. Tests establish only the authored cases, not universal program correctness. If unrestricted Python exercises are added later, replace this adapter with a separately isolated runner and recheck the grading contract.

## Files and responsibilities

| Path | Responsibility |
|---|---|
| `server.py` | Loopback HTTP server, JSON routes, origin/token protection, startup configuration |
| `service.py` | Question/assessment rules, chronological events, grading orchestration, profile summaries |
| `catalogue.py` | Read-only bank, authored hints, answer-safe public projection, exact model feature mapping |
| `grading.py` | Real-answer rubric adapter and bounded Python interpreter |
| `memory_service.py` | Load/freeze checkpoint, causal evidence encoding, state caching, prediction API |
| `store.py` | SQLite journal and transactional persistence |
| `static/` | Responsive custom HTML/CSS/JavaScript interface; no remote assets or dependencies |
| `tests/` | Grading, chronology, replay, idempotence, persistence, isolation, and HTTP checks |
| `reports/` | Actual test output, verification and browser screenshots from an isolated QA profile |
| `data/sessions.sqlite3` | Your real profiles/answers/events; created at runtime and ignored by Git |

Custom HTML/CSS was chosen instead of the earlier proposed Streamlit UI to give the requested minimal, polished visual treatment while reusing the Python model directly. There is no change to model architecture or training.

## Model and storage contracts

The checkpoint is `../model/runs/v2.1-mps-main/time_decay_gru/seed_20261111/model.pt`. The service checks its mode/version, records its SHA-256, calls `.eval()`, disables parameter gradients, and performs inference only. The original dataset, model source, and checkpoints are read-only dependencies.

A cache entry is keyed by learner, canonical skill, and model hash. It contains the original EvidenceStream, hidden vector, last evidence anchor, and consumed event cursor. Only newly appended evidence is encoded. A restart reconstructs states from the retained journal; query calls never commit a decayed state. Dataset replay equivalence is tested against the existing full-sequence model implementation, including nonempty histories and delayed queries.

The model receives exactly the existing structured feature allowlist. Question text, answer text, names, IDs, and solution text never enter the neural network. Administrative events retain their gaps but do not step the GRU. Response and grade are combined at grade availability; hints and reveals are separate tokens. Raw timestamps are UTC Unix seconds; UI dates are displayed in the browser's local timezone.

Timing is a **browser-visible-and-focused estimate**, not a measured cognitive response time. It stops while the app is unfocused, hidden, displaying feedback, or processing a request. Server wall duration is stored separately. Missing timing remains `null`; it is not replaced with an invented duration. Closing a browser may lose part of the last half-second of unsaved draft timing. The reference response times and difficulty labels remain authored, unvalidated estimates.

The journal is append-only through the app. Event writes and interaction state changes are one SQLite transaction. Replayed request IDs return the original result; a second distinct submission for an already-graded attempt is rejected. The HTTP server serializes application operations. Profiles are local organizational records, not authenticated accounts; this is a single-machine research tool, bound to `127.0.0.1`, not a production multi-user service.

### SQLite tables and export fields

| Table | Fields |
|---|---|
| `profiles` | `id`, `name`, `created` |
| `interactions` | `id`, `learner`, `skill`, `task`, JSON `payload` |
| `events` | Incrementing `id`, `learner`, `skill`, `interaction`, `at`, `kind`, JSON `payload` |
| `requests` | Unique request `id`, saved JSON `result` for idempotence |

Interaction payload: selected task, ordered options, mode/original mode, creation and attempt-start timestamps, attempt number, delivered hints, immutable results per attempt, original prediction, status, closed/revealed/incomplete flags. The public projection omits hidden predictions and answer keys before submission.

Event payload: `source=real_local_session`, schema version, model hash, structured model `features`, plus event-specific observations. Responses store actual answer text/option IDs, wall duration, and timing provenance. Grades store criterion outcomes, partial score, full success, grading status and grader version. Hints store their exact text/kind/version. Task events store the pre-answer prediction and target features. Export contains the selected profile, its interactions, and its chronological events. Browser drafts are not exported until submitted.

Feature columns: `event_type`, `gap_seconds`, `new_session`, structured `task`, `attempt_number`, `active_response_seconds`, `relative_response_time`, `hint_count`, `hint_type`, `answer_revealed`, `assistance`, `score`, `full_success`, `grading_status`. Unavailable fields stay null. Task features are difficulty, format, reference seconds, snippet/option/criterion counts, and sorted credit/penalty weights. The first event has unknown new-session status; following events currently use false, matching the within-interaction v2.1 projection. No fabricated session ID is learned.

## Verification and limits

Run:

```sh
3/model/.venv/bin/python -m unittest discover -s 3/dashboard/tests -v
```

Run from a normal Terminal; one test opens a temporary loopback HTTP port. See [verification](reports/VERIFICATION.md) and [test output](reports/test_output.txt) for actual results. Browser QA uses a separate temporary database, never your real session database.

This release collects real local evidence but **does not retrain** on it. Synthetic-trained probabilities are not yet validated for real students or other subjects. The 2,160 variants share a small number of task families, so counts do not imply broad curriculum coverage. Hint review and reference-answer tests are engineering checks, not a human educational validation. Jev, LLM tutoring, unrestricted execution, synthetic history-replay UI, and PPO are not part of this dashboard release. The user's later real-answer scope superseded the initial replay/manual-entry proposal.

Primary implementation references: [Python AST](https://docs.python.org/3/library/ast.html), [SQLite transactions](https://docs.python.org/3/library/sqlite3.html). Model definitions remain in the existing model documentation.
