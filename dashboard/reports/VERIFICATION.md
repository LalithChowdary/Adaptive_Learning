# Dashboard verification — 2026-11-10

## Executed checks

- **19/19 automated tests passed**, zero failures/errors/skips. See `test_output.txt` for individual names and the recorded run.
- The catalogue test grades **all 2,160 reference answers** correctly, including all code-completion reference functions through the bounded interpreter.
- Tests cover multi-select penalties; weighted partial credit; output normalization; alternative completion solutions; syntax failures; unsupported imports/attributes/calls/loops and resource limits; hidden answers; first unaided target eligibility; chronological multiple hints/retries; caps; assessment-to-review transitions; preservation of the original independent grade; request idempotence; duplicate-submission rejection; persisted sessions; learner/skill isolation; and exports.
- State-query tests show no evidence changes or double decay. The service matches the existing full-sequence model across every eligible query in one held-out learner-skill history, including nonempty practice prefixes and delayed queries, to six decimal places on CPU.
- The HTTP test confirms static serving, profile creation with the current token, and rejection of writes without the token. This test needs loopback access and was run outside the restricted shell sandbox.
- **82/82 dependency integrity checks passed** against the existing v2.1 dataset release inventory and model run source inventory. See `dependency_integrity.json`. No dataset regeneration or model fitting was performed.
- Selected checkpoint SHA-256: `353ccee669f66f60acf1a6e08f9cae410f2c1ea7744c7c56e2a5523a834866de`.

## Browser QA

Executed against a separate temporary SQLite file at `/tmp/recall-dashboard-qa.sqlite3` on port 8766, with MPS inference. The QA profile and its answers are not in the delivered session database.

1. Created the QA profile and inspected the six-skill landing screen at narrow and desktop browser widths.
2. Started a single-choice range question and verified that the pre-answer probability stayed hidden.
3. Submitted a wrong first answer; the interface showed 0% rubric credit and revealed the saved pre-answer prediction.
4. Requested two successive hints, started a retry, and submitted the correct answer. The interface explicitly labeled the result **assisted attempt**.
5. Started a fresh code-completion exercise, entered a real `solve(x)` implementation, reloaded, and verified that the draft and active task survived.
6. Submitted that function and observed full credit from the behavior tests.
7. Inspected the progress view: three graded attempts, one successful first unaided answer out of two, two hints, and exactly one assessed skill. Other skills remained unassessed.
8. Opened the memory explorer and observed separate predictions at 0, 7, 30, and 60 days from the recorded evidence.
9. Checked captured browser logs: no warning or error messages at the end of QA.
10. Stopped the QA instance, started the final server on port 8765 with `--device mps`, and opened its clean profile-creation page. At handoff, its database had zero profiles and zero events.

Screenshots: `progress-preview.jpg` is from the explicitly named QA profile; `welcome-preview.jpg` is the clean delivered app. Screenshots establish the UI states inspected, not real learning effectiveness.

## Scope of these results

Automated tests establish the engineering contracts above, not that the numerical simulator accurately models people. The hint wording, difficulty labels, timings, and task-family coverage still require real-use review. The completion interpreter supports a documented subset of Python, not arbitrary Python. No claim of real-student retention calibration, cross-subject generalization, or PPO readiness follows from these checks.
