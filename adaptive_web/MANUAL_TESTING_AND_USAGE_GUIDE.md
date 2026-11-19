# Manual Testing & Usage Guide — Adaptive Python Tutor (`3/adaptive_web/`)

This guide explains every feature of the `3/adaptive_web/` application and provides step-by-step instructions for testing and presenting the website both in a web browser and via `curl` / CLI.

---

## 1. Complete Feature Overview

### A. Pedagogical & Curriculum Features
1. **6-Skill Prerequisite DAG (Math Academy-inspired)**
   - Ordered in canonical PPO curriculum order:
     1. `range_boundaries` (Root)
     2. `loop_tracing` (Prereq: `range_boundaries`)
     3. `accumulator_initialization` (Prereqs: `range_boundaries`, `loop_tracing`)
     4. `conditional_evaluation` (Prereq: `loop_tracing`)
     5. `list_indexing` (Prereq: `range_boundaries`)
     6. `function_returns` (Prereqs: `conditional_evaluation`, `accumulator_initialization`)
   - Each node displays live **Unassessed** vs. **Assessed** status, medium recall probability $P(\text{Med})$, prerequisites, and permastore floor $b_t$.
2. **Personalized Learning Path Curation (`Learn → Code → Remember`, Execute Program-inspired)**
   - Selecting any skill curates a 3-phase study path tailored to the learner's `monotonic_gru` state (unassessed cold start vs. high fragility $P(\text{Med}) - P(30\text{d})$ vs. mastery consolidation).
3. **Full Course Material & Subgoal-Labeled Worked Examples**
   - Generates concept explanations, mental models, common pitfalls, and step-by-step worked code traces annotated with explicit **Subgoal Labels** (`Subgoal 1`, `Subgoal 2`, etc.).
4. **Hybrid Question Delivery (`v2.2` 2,160-Item Bank + Fresh LLM Question Generation)**
   - Supports all **5 canonical task formats**:
     - `single_choice`
     - `multiple_select` (with partial credit and wrong-selection penalties)
     - `output_prediction`
     - `fill_in_the_blank` (exact integer literal matching)
     - `code_completion` (`def solve(x):` graded inside a deterministic **Python AST Sandbox Interpreter** with behavioral test cases)
   - Every fresh LLM-generated question is schema-validated and projected into the exact 8-key `memory_task` metadata required by `monotonic_gru`.
5. **Progressive 3-Tier Hint Scaffolding with Leakage $l_t \in [0, 1]$ (OATutor-inspired)**
   - Up to **3 progressive hints** per question:
     - Level 1: `conceptual` ($l_t = 0.25$)
     - Level 2: `targeted` ($l_t = 0.50$)
     - Level 3: `scaffold` ($l_t = 0.75$)
   - Using hints marks the pass as **assisted** (`unverified_assisted = True`), prompting PPO to schedule an unaided follow-up verification check.
6. **Retry, Solution Reveal & Assessment Integrity**
   - Up to **2 retries** per question after an incorrect submission.
   - **Solution Reveal** unlocks after at least one attempt.
   - In **Assessment Mode**, requesting a hint before submitting marks `assessment_incomplete = True` and downgrades the interaction to `review` mode so assisted answers never corrupt independent assessment metrics.

### B. RL Policy (`delayed_retention_ppo`) & Cognitive Memory (`monotonic_gru`) Features
7. **Hard PPO Action Control with Gated Option B (8-Action Ensemble)**
   - Queries all 3 frozen seeds (`seed_20261120.pt`, `seed_20261121.pt`, `seed_20261122.pt`) over the **36-dim observation vector $s_t$**:
     - `0`: `give_easier_question`
     - `1`: `give_same_difficulty_question`
     - `2`: `give_harder_question`
     - `3`: `explain_concept` (Passive lesson — Gated Option B forces a follow-up question on the next step)
     - `4`: `show_worked_example` (Passive lesson — Gated Option B forces a follow-up question on the next step)
     - `5`: `request_self_explanation` (Unlocks right after a question to consolidate transient fluency $k_t$ into permastore floor $b_t$, then re-locks)
     - `6`: `review_prerequisite` (Unlocks when a prerequisite skill is weak or decayed)
     - `7`: `advance_next_topic` (Unlocks when current skill readiness is high)
8. **0–60 Day Monotonic Retention Curves**
   - Live SVG visualization plotting **Easy**, **Medium**, and **Hard** retention curves across `0, 1, 3, 7, 14, 30, 45, 60` days alongside the dashed **Permastore Floor $b_t$** line.
   - Displays an explicit **Unassessed** guard before the first graded attempt on a skill.
9. **Toggleable Glass-Box Research Inspector & Provenance Export**
   - Live side-panel for paper presentations showing the 47-dim structured evidence token audit (**zero text/code embeddings**), critic value $V(s_t)$, recommended PPO action, Gated Option B flags, and manual override controls (`policy_source = "delayed_retention_ppo"` vs. `"learner_manual"`).

---

## 2. Starting the Server

From the repository root (`/Users/lalith/snu/sem7/minor_pro/code`):

```bash
PYTHONPATH=3 3/model/.venv/bin/python -m adaptive_web.server --port 8780
```

- **Default URL:** `http://127.0.0.1:8780`
- **Isolated SQLite WAL DB:** `3/adaptive_web/data/adaptive_web.sqlite3` (never touches `3/dashboard/data/sessions.sqlite3`).

### Optional: Enabling a Live Cloud LLM (OpenAI / Anthropic / Gemini)
By default, the server runs offline using a deterministic pedagogical test double (`LLM_PROVIDER=mock`). To connect a live LLM API:

```bash
export LLM_PROVIDER=openai          # openai | anthropic | gemini
export LLM_API_KEY="sk-..."         # Your API key
export LLM_MODEL="gpt-4o-mini"      # Model identifier
PYTHONPATH=3 3/model/.venv/bin/python -m adaptive_web.server --port 8780
```

---

## 3. Step-by-Step Browser Walkthrough (Manual UI Testing)

### Step 1: Create a Learner Profile
1. Open `http://127.0.0.1:8780` in your browser.
2. In the top bar, type a name (e.g., `Demo Student`) into **"New learner name..."** and click **Create Profile**.
3. Notice the left sidebar (**Skill Prerequisite DAG**) shows all 6 canonical skills labeled `Unassessed` with `floor b_t: 0.00`.

### Step 2: Select a Skill & Inspect the Curated Path
1. Click **`Range boundaries`** (or **`Loop tracing`**) in the left Skill Prerequisite DAG.
2. Observe the **Personalized Learning Path** card populate with the 3-step `learn → code → remember` sequence.
3. Observe the **Course Material & Subgoal Worked Example** card populate with the conceptual explanation, mental model, code block, and green-bordered **Subgoal** steps.

### Step 3: Test Gated Option B (Passive Lesson → Forced Question)
1. Click the **Glass-Box Inspector** button in the top bar to open the right-hand telemetry panel.
2. In the **Interactive Practice & Assessment** card, select **`3: Explain Concept`** from the `Manual Override...` dropdown and click **Run Manual**.
3. Look at the Glass-Box Inspector: **`Awaiting Question`** is now **`True`**.
4. Try running **`3: Explain Concept`** or **`4: Show Worked Example`** again — the backend enforces Gated Option B and rejects consecutive passive lessons until a follow-up question is answered.

### Step 4: Answer a Question, Request Progressive Hints, Retry & Finish
1. Click **Next PPO Step** to receive an adaptive question.
2. Click **Hint (0/3)** up to 3 times to inspect the 3-tier progressive scaffolding (`Hint 1 (conceptual)`, `Hint 2 (targeted)`, `Hint 3 (scaffold)`).
3. Select/enter a wrong answer first and click **Submit** — observe the feedback banner (`Needs Review`).
4. Click **Retry**, select/enter the correct answer, and click **Submit** (`Passed`).
5. Click **Finish** to close the question and unlock the next turn.
6. Look at the **0–60 Day Retention Forecast** card: the `Unassessed` guard is replaced by the live **Easy / Medium / Hard** SVG decay curves and dashed purple **Permastore Floor $b_t$** line.

### Step 5: Consolidate Permastore Floor $b_t$ via Self-Explanation (Action 5)
1. Right after finishing a question, select **`5: Request Self-Explanation`** in `Manual Override...` and click **Run Manual**.
2. Type a 1-sentence explanation of the rule (e.g., for `range_boundaries`: *"Python range(start, stop) includes start and stops strictly before stop."*) and click **Submit**, then **Finish**.
3. Check the skill badge and retention curve header: **`floor b_t`** and 30-day retention increase significantly.

### Step 6: Test Prerequisite Detour (Action 6)
1. Click **`Function returns`** in the left Skill DAG (its prerequisites `conditional_evaluation` and `accumulator_initialization` are still unassessed/weak).
2. Select **`6: Review Prerequisite`** and click **Run Manual**.
3. Observe that PPO automatically detours to a weak prerequisite skill (`conditional_evaluation` or `accumulator_initialization`) to unblock the bottleneck.

---

## 4. Manual CLI / `curl` API Testing Recipes

Every POST endpoint requires the `X-CSRF-Token` header obtained from `GET /api/state`. Run the commands below in a terminal while the server is running on port `8780`:

```bash
# 1. Fetch bootstrap state & extract CSRF token
TOKEN=$(curl -s http://127.0.0.1:8780/api/state | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('csrf_token') or d.get('token'))")
echo "CSRF Token: $TOKEN"

# 2. Create a new learner profile
LEARNER=$(curl -s -X POST http://127.0.0.1:8780/api/profile \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $TOKEN" \
  -d '{"name": "CLI Tester", "request_id": "req-profile-0001"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('learner') or d.get('learner_id') or d['profile']['id'])")
echo "Learner ID: $LEARNER"

# 3. Select a skill (curates learning path + course material + PPO state)
curl -s -X POST http://127.0.0.1:8780/api/select_skill \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $TOKEN" \
  -d "{\"learner\": \"$LEARNER\", \"skill\": \"range_boundaries\", \"request_id\": \"req-select-0001\"}" | python3 -m json.tool | head -n 35

# 4. Trigger a PPO question step (e.g. code_completion or single_choice)
STEP=$(curl -s -X POST http://127.0.0.1:8780/api/ppo_step \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $TOKEN" \
  -d "{\"learner\": \"$LEARNER\", \"skill\": \"range_boundaries\", \"action_id\": 1, \"format\": \"code_completion\", \"difficulty\": \"easy\", \"request_id\": \"req-step-0001\"}")
echo "$STEP" | python3 -m json.tool | head -n 35

# 5. Query 0-60d retention curve
curl -s "http://127.0.0.1:8780/api/curve?learner=$LEARNER&skill=range_boundaries&format=output_prediction" | python3 -m json.tool | head -n 30

# 6. Export full research provenance bundle (interactions, events, tutor_turns, token_audit)
curl -s "http://127.0.0.1:8780/api/export?learner=$LEARNER" | python3 -m json.tool | head -n 40
```
