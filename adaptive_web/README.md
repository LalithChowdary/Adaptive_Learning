# Adaptive Python Tutor (`adaptive_web`)

An interactive adaptive Python learning web application powered by:
- **Frozen 3-seed `delayed_retention_ppo`** reinforcement learning policy (8-action pedagogical control with Gated Option B)
- **Frozen `monotonic_gru`** cognitive memory model (47-dim structured evidence tokens + 19-dim task features, zero text/code embeddings)
- **`v2.2` 2,160-item question bank** across 6 canonical Python skills and 5 task formats
- **Bounded LLM pedagogical realization** (offline deterministic mock by default, or optional OpenAI / Anthropic / Gemini)
- **Toggleable Glass-Box Research Inspector** for live telemetry and token auditing

---

## 1. Prerequisites & Installation

- **Python 3.10+** (standard library HTTP server + SQLite WAL backend)
- **Model dependencies** listed in [`../model/requirements.txt`](../model/requirements.txt):
  - `numpy`
  - `scikit-learn`
  - `torch`

From the repository root (`Adaptive_Learning/`), create a virtual environment and install dependencies:

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r model\requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r model/requirements.txt
```

---

## 2. Running `adaptive_web`

Run the server from the **repository root** (`Adaptive_Learning/`):

```bash
python -m adaptive_web.server --port 8780
```

Then open your browser to:

```text
http://127.0.0.1:8780
```

> **Note:** The server binds to `127.0.0.1` only and enforces local `Host` / `Origin` checks.

### CLI Options
| Flag | Default | Description |
| --- | --- | --- |
| `--port` | `8780` | Port to listen on (`ADAPTIVE_WEB_PORT`) |
| `--db` | `adaptive_web/data/adaptive_web.sqlite3` | SQLite database path (`ADAPTIVE_WEB_DB`) |
| `--device` | `cpu` | PyTorch device: `cpu`, `mps`, or `auto` (`DEVICE`) |

### Optional: Cloud LLM Configuration
By default, `adaptive_web` runs **100% offline** using a deterministic pedagogical test double (`LLM_PROVIDER=mock` or `auto` without API keys). To enable live LLM path curation, course material generation, and fresh question authoring:

1. Copy `adaptive_web/.env.example` to `adaptive_web/.env` (or place `.env` in the repository root).
2. Set your provider, key, and model:

```env
ADAPTIVE_WEB_PORT=8780
ADAPTIVE_WEB_DB=adaptive_web/data/adaptive_web.sqlite3
DEVICE=cpu

# LLM Provider: auto | mock | openai | anthropic | gemini
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

Or export them directly in your shell before starting `python -m adaptive_web.server`.

---

## 3. Navigating the Interface

The web UI is organized into a **Top Bar**, a **Left Sidebar**, a **Center Workspace** (4 stacked cards), and a toggleable **Right Inspector Panel**.

### Top Bar
- **Learner Profile Dropdown (`Select learner profile`)**: Switch between existing learner profiles stored in SQLite.
- **`New learner name...` + `Create Profile`**: Type a learner name and click **Create Profile** (or press Enter) to create a fresh profile with isolated memory and skill states.
- **`Glass-Box Inspector`**: Toggle the right-hand research telemetry panel on or off.

### Left Sidebar: Skill Prerequisite DAG
Displays the **6 canonical Python skills** in prerequisite order:
1. `range_boundaries` (Root)
2. `loop_tracing` (Prereq: `range_boundaries`)
3. `accumulator_initialization` (Prereqs: `range_boundaries`, `loop_tracing`)
4. `conditional_evaluation` (Prereq: `loop_tracing`)
5. `list_indexing` (Prereq: `range_boundaries`)
6. `function_returns` (Prereqs: `conditional_evaluation`, `accumulator_initialization`)

- **Click any skill** to load its personalized learning path, course material, practice workspace, and retention curve.
- Each skill card shows its status (**`Unassessed`** vs. **`Assessed`**), medium recall probability $P(\text{Med})$, prerequisites, and permastore floor $b_t$.

### Center Workspace

#### 1. Personalized Learning Path
- Automatically curates a 3-step **`Learn → Code → Remember`** study sequence tailored to your `monotonic_gru` memory state for the selected skill.
- The top-right badge indicates the active LLM mode (`LLM` / provider status).

#### 2. Course Material & Subgoal Worked Example
- Displays the concept explanation, mental model, common pitfalls, and a step-by-step worked code example annotated with **Subgoal Labels** (`Subgoal 1`, `Subgoal 2`, etc.).

#### 3. Interactive Practice & Assessment
Use the toolbar at the top of this card to drive your learning session:
- **Question Source Dropdown**:
  - `Source: v2.2 Bank (2,160 items)` — draws curated items from the benchmark bank.
  - `Source: Fresh LLM Question` — generates a new schema-validated question on the fly.
- **`Next PPO Step`**: Lets the `delayed_retention_ppo` RL policy choose the optimal next pedagogical action for your current state.
- **`Manual Override...` + `Run Manual`**: Manually trigger any of the 8 PPO actions:
  - `0`: Give Easier Question
  - `1`: Give Same Difficulty
  - `2`: Give Harder Question
  - `3`: Explain Concept *(passive lesson — Gated Option B requires answering a follow-up question before another passive lesson)*
  - `4`: Show Worked Example *(passive lesson — Gated Option B requires answering a follow-up question)*
  - `5`: Request Self-Explanation *(unlocks after a question to consolidate transient fluency into permastore floor $b_t$)*
  - `6`: Review Prerequisite *(detours to a weak or decayed prerequisite skill)*
  - `7`: Advance Next Topic *(unlocks when readiness on the current skill is high)*

When a question is active (supporting `single_choice`, `multiple_select`, `output_prediction`, `fill_in_the_blank`, and `code_completion`):
- **`Submit`**: Grade your answer (code completion is executed in a deterministic Python AST sandbox).
- **`Hint (0/3)`**: Request up to 3 progressive hints (`conceptual`, `targeted`, `scaffold`).
- **`Retry (0/2)`**: Up to 2 retries after an incorrect submission.
- **`Reveal`**: Unlocks after at least one submission to show the solution and explanation.
- **`Finish`**: Completes the current interaction and unlocks the next PPO or manual step.

#### 4. 0–60 Day Retention Forecast (`monotonic_gru`)
- Before your first graded attempt on a skill, displays an **`Unassessed`** guard.
- After grading, renders a live SVG chart plotting **Easy**, **Medium**, and **Hard** recall probabilities across days `0, 1, 3, 7, 14, 30, 45, 60`, along with the dashed **Permastore Floor $b_t$** line.

### Right Panel: Glass-Box Research Inspector
Click **`Glass-Box Inspector`** in the top bar to view live model internals:
- **Token Dim (`47`)** and **Text/Code Embeddings (`ZERO`)** audit
- **PPO Critic Value $V(s_t)$**
- **Recommended Action** and plain-language rationale
- **`Awaiting Question` (`True` / `False`)** Gated Option B state

---

## 4. Further Reading

- [`MANUAL_TESTING_AND_USAGE_GUIDE.md`](MANUAL_TESTING_AND_USAGE_GUIDE.md) — Step-by-step browser walkthrough and `curl` API recipes
- [`PROPOSAL_PPO_LLM_ARCHITECTURE.md`](PROPOSAL_PPO_LLM_ARCHITECTURE.md) — Architecture comparison and Gated Option B design
- [`TEST_REPORT_AND_CATALOG.md`](TEST_REPORT_AND_CATALOG.md) — Automated test suite report and coverage catalog
