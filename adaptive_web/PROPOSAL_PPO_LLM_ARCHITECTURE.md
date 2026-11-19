# Architecture Proposal: Hard PPO Action Control with Bounded LLM Realization

## 1. Executive Summary & Core Architectural Decision

In an adaptive Python learning system built for a research paper presentation, how should the frozen reinforcement learning policy (`delayed_retention_ppo`) and the cognitive memory model (`monotonic_gru`) interact with a Large Language Model (LLM)?

We compare two competing architectures:
1. **Alternative A — PPO as Background Suggestion to the LLM**: The LLM acts as the primary decision-maker and receives PPO's recommended action merely as a soft prompt hint.
2. **Proposed Architecture — Hard PPO Action Control + Bounded LLM Realization (Gated Option B)**: The 3-seed `delayed_retention_ppo` ensemble retains **hard control** over pedagogical action selection (the 8-action space) and prerequisite/retention gating, while the LLM acts as a **bounded pedagogical realizer** that curates study paths, generates subgoal-labeled course material, and authors fresh tasks within strict JSON/AST schemas.

We recommend and implement **Hard PPO Action Control + Bounded LLM Realization** because it preserves the theoretical guarantees of the trained RL policy, prevents LLM prompt drift from violating pedagogical constraints, and enables transparent scientific auditing via a toggleable **Glass-Box Research Inspector**.

---

## 2. Comparison: Background Suggestion vs. Hard PPO Control

| Dimension | Alternative A: Background Suggestion | Proposed: Hard PPO Action Control |
| :--- | :--- | :--- |
| **Decision Authority** | LLM may ignore or override PPO's action | `delayed_retention_ppo` selects action $a_t \in \{0..7\}$ under Gated Option B |
| **Gating Integrity** | Prompt-based rules can be hallucinated away | Hard boolean `valid_mask` enforces post-lesson follow-up & self-explanation rules |
| **State Representation** | Unstructured chat history | Exact 36-dim observation vector $s_t$ from 47-dim `monotonic_gru` evidence tokens |
| **Attribution & Provenance** | Impossible to separate RL vs. LLM effect | Explicit `policy_source` (`delayed_retention_ppo` vs. `learner_manual`) logged per turn |
| **Offline / Fallback Safety** | Fails if LLM ignores format | Deterministic `v2.2` 2,160-item bank and offline fallback templates |

---

## 3. Cognitive Memory & RL Policy Specifications

- **Memory Model (`monotonic_gru`)**:
  - Input: **47**-dimensional structured `Evidence` tokens + **19**-dimensional task features (zero raw text or code embeddings).
  - Outputs: Instantaneous recall probabilities, **0–60 day** monotonic retention curves across Easy/Medium/Hard difficulties, active strength $k_t$, and permastore floor $b_t$.
- **RL Policy (`delayed_retention_ppo`)**:
  - Observation Space: **36**-dimensional vector $s_t$ capturing memory retention, assistance/leakage $l_t$, Gated Option B state, prerequisite health, remaining budget (total budget 32.0), per-skill 30-day retention, and active topic one-hot encoding.
  - Action Space (**8** actions):
    - `0`: `give_easier_question`
    - `1`: `give_same_difficulty_question`
    - `2`: `give_harder_question`
    - `3`: `explain_concept`
    - `4`: `show_worked_example`
    - `5`: `request_self_explanation`
    - `6`: `review_prerequisite`
    - `7`: `advance_next_topic`
- **Canonical Python Curriculum (6 Skills)**:
  1. `range_boundaries` (root)
  2. `loop_tracing` (prereq: `range_boundaries`)
  3. `accumulator_initialization` (prereq: `range_boundaries`, `loop_tracing`)
  4. `conditional_evaluation` (prereq: `loop_tracing`)
  5. `list_indexing` (prereq: `range_boundaries`)
  6. `function_returns` (prereq: `conditional_evaluation`, `accumulator_initialization`)

---

## 4. Adaptive Learning UX Grounded in Five Research Systems

Our workspace UI (`3/adaptive_web/static/`, served on port `8780` with the `#f7f8f3` / `#293321` / `#c6f36a` editorial palette) synthesizes proven UX patterns from five leading adaptive learning platforms:

1. **Math Academy**: Explicit **prerequisite DAG** visualization, mastery/retention indicators per node, and automatic prerequisite detouring (Action `6: review_prerequisite`) when foundational retention decays.
2. **Execute Program**: Short, interactive **Learn → Code → Remember** micro-cadence where passive explanations are immediately followed by active recall/coding checks (enforced by Gated Option B `awaiting_question`).
3. **Duolingo (Birdbrain + Max)**: Difficulty calibration matched to the learner's zone of proximal development (`give_easier_question` / `give_same_difficulty_question` / `give_harder_question`) paired with contextual LLM-generated explanations.
4. **DataCamp & Codecademy**: In-browser code completion workspace with deterministic **Python AST sandbox** grading (`def solve(x):`) and immediate behavioral test feedback.
5. **OATutor**: Progressive **3-tier hint scaffolding** (`conceptual` → `targeted` → `scaffold`) with explicit hint leakage tracking $l_t \in [0, 1]$ and self-explanation prompts (Action `5`) to consolidate long-term permastore floor $b_t$.

---

## 5. Glass-Box Research Inspector & Provenance

For live paper presentations, the UI includes a toggleable **Glass-Box Research Inspector** panel:
- **Live 36-dim $s_t$ & Action Masking**: Displays the exact 36-dim observation vector, Gated Option B `valid_mask`, mask reasons, 3-seed ensemble probabilities, and critic value $V(s_t)$.
- **47-dim Token Audit**: Proves that `monotonic_gru` receives zero text/code embeddings.
- **Provenance Tracking**: Every turn records `policy_source` (`delayed_retention_ppo` when driven by the RL policy, or `learner_manual` when overridden by the presenter/learner) in SQLite WAL storage and `/api/export`.

---

## 6. Limitations & Disclosure

- **Simulated Calibration**: Both `monotonic_gru` and `delayed_retention_ppo` were trained and calibrated on simulated learner trajectories over the `v2.2` 2,160-item Python bank and remain unvalidated on real-student classroom populations.

---

## 7. API Surface & Data Isolation Guarantees

The `3/adaptive_web/` application runs on port `8780` with an isolated SQLite WAL database (`3/adaptive_web/data/adaptive_web.sqlite3`), leaving `3/dashboard/`, `3/data_Set/v2.2/`, `3/model/`, and `3/ppo/` untouched.

### Endpoints
- `GET /api/state` — Bootstrap state, skill DAG summaries, `ppo` telemetry, and `token_audit`.
- `GET /api/curve` — 0–60 day Easy/Medium/Hard retention forecast and permastore floor $b_t$.
- `GET /api/export` — Full provenance export (`profile`, `model_sha256`, `ppo_sha256`, `interactions`, `events`, `learning_paths`, `course_materials`, `tutor_turns`, `generated_tasks`).
- `POST /api/profile` — Create isolated learner profile.
- `POST /api/select_skill` — Curate personalized `Learn -> Code -> Remember` path and subgoal-labeled course material.
- `POST /api/ppo_step` — Execute `delayed_retention_ppo` action (or `learner_manual` override) under Gated Option B.
- `POST /api/hint` — Deliver progressive 3-tier hints (`conceptual`, `targeted`, `scaffold`) with leakage $l_t$.
- `POST /api/submit` — Deterministic grading across all 5 formats + Python AST sandbox (`def solve(x):`) + self-explanation evaluation.
- `POST /api/retry`, `POST /api/reveal`, `POST /api/finish` — Interaction lifecycle controls.


### Running Locally
```bash
PYTHONPATH=3 3/model/.venv/bin/python -m adaptive_web.server --port 8780
```
Open `http://127.0.0.1:8780` in a browser to explore the learner workspace, 0–60 day retention curves, and the toggleable Glass-Box Research Inspector.
