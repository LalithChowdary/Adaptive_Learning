# `srinivas_edit.md` — Detailed Changelog & Feature Documentation

This document records all changes, UI/UX redesigns, backend enhancements, bug fixes, and documentation updates made to **`adaptive_web`** (`Adaptive_Learning` repository).

---

## 1. Summary of Modified & Created Files

| File | Branch | Summary of Changes |
| --- | --- | --- |
| [`README.md`](README.md) | `main` & `ui-improvements` | Added Quick Start guide for `adaptive_web`, updated repository structure (`adaptive_web/`, `dashboard/`, `data_Set/`, `model/`, `ppo/`), and linked to `adaptive_web/README.md`. |
| [`adaptive_web/README.md`](adaptive_web/README.md) | `main` & `ui-improvements` | Created a complete external-user setup, installation, CLI/environment configuration, and UI navigation guide; fixed broken local paths. |
| [`adaptive_web/static/index.html`](adaptive_web/static/index.html) | `ui-improvements` | Redesigned the center workspace from a 4-card vertical scroll stack into a **Skill Header + 3-Tab layout (`Learn` · `Practice` · `Progress`)**, collapsible **Reference Lesson & Worked Example** drawer, interactive **Ask the Tutor** doubt-clarification chat box, and collapsible **Advanced / Instructor Controls**. |
| [`adaptive_web/static/style.css`](adaptive_web/static/style.css) | `ui-improvements` | Added styles for sticky left sidebar, skill header bar, workspace tabs, collapsible reference drawer, tutor chat thread/bubbles/form, color-coded pass/fail feedback banners, retention curve legend, and collapsible advanced controls. |
| [`adaptive_web/static/app.js`](adaptive_web/static/app.js) | `ui-improvements` | Implemented tab navigation, 1-click question progression (`Next →`), inline PPO lesson/worked-example delivery in the `Practice` tab, interactive tutor chat client (`/api/chat`), cross-skill PPO routing updates, and fixed the `data.active` vs. `data.active_interaction` state key mismatch. |
| [`adaptive_web/llm_pedagogy.py`](adaptive_web/llm_pedagogy.py) | `ui-improvements` | Added `PedagogyRealizer.answer_learner_doubt()` to answer student questions grounded in the active topic's reference lesson and worked example using the live cloud LLM (`self.llm.complete_json`) with a deterministic offline fallback. |
| [`adaptive_web/app.py`](adaptive_web/app.py) | `ui-improvements` | Added `AdaptiveLearningApp.chat_doubt()` with input validation, idempotency support (`request_id`), and automatic lookup of the learner's latest course material. |
| [`adaptive_web/server.py`](adaptive_web/server.py) | `ui-improvements` | Added `POST /api/chat` HTTP endpoint wired to `app.chat_doubt()` with CSRF validation and error handling. |

---

## 2. Detailed Breakdown of Every Feature & Change

### A. External User Documentation (`README.md` & `adaptive_web/README.md`)
- **Problem:** The original `adaptive_web/README.md` contained hardcoded paths from an older directory layout (`PYTHONPATH=3 3/model/.venv/bin/python ...` and `file:///Users/lalith/...`) and had no guide explaining how to navigate the web interface.
- **Changes Made:**
  1. **Prerequisites & Virtual Environment Setup:** Added step-by-step instructions for creating `.venv` and installing `model/requirements.txt` (`numpy==2.3.5`, `scikit-learn==1.9.1`, `torch==2.14.1`) on both Windows PowerShell and macOS/Linux.
  2. **Server Launch & CLI Flags:** Documented `python -m adaptive_web.server --port 8780` from the repository root, along with `--port`, `--db`, and `--device` flags.
  3. **Cloud LLM & Offline Mode Setup:** Documented `.env` configuration for `LLM_PROVIDER` (`auto`, `mock`, `openai`, `anthropic`, `gemini`), `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL`.
  4. **Interface Navigation Guide:** Documented every section of the UI (Top Bar, Left Sidebar 6-Skill DAG, Center Workspace cards & controls, 5 question formats, progressive hints, retries, solution reveal, 0–60 day retention chart, and Glass-Box Research Inspector).

---

### B. 3-Tab Center Workspace & Skill Header (Eliminating Vertical Scrolling)
- **Problem:** Previously, the center column stacked 4 large cards vertically (*Personalized Learning Path* → *Course Material & Subgoal Worked Example* → *Interactive Practice & Assessment* → *0–60 Day Retention Forecast*). Clicking a skill in the left sidebar loaded a long wall of text at the top, forcing the user to scroll far down to find the practice questions or retention chart.
- **Changes Made:**
  1. **Skill Header Card (`.skill-header-card`):**
     - Added at the top of the center column to display the human-readable topic title (e.g., `Range boundaries`), dynamic subtitle guidance, readiness badge (`Not started` vs. `% Ready`), and curriculum mode badge (`Offline Curriculum` vs. `AI Tutor (<model>)`).
  2. **3 Navigation Tabs (`.workspace-tabs`):**
     - **`Learn` (`#tab-panel-learn`):** Shows the 3-step `Learn → Code → Remember` **Study Plan** card with a prominent **`Start Practice →`** button, followed by the collapsible **Reference Lesson & Worked Example** drawer and the **Ask the Tutor** chat section.
     - **`Practice` (`#tab-panel-practice`):** Dedicated, distraction-free interactive question, code editor, hints, and feedback workspace.
     - **`Progress` (`#tab-panel-progress`):** Displays the `0–60 Day Retention Forecast` SVG curves with a newly added color legend (`Easy`, `Medium`, `Hard`, `Long-term floor`).
  3. **Smart Sidebar Topic Selection:**
     - Clicking any of the 6 topics in the left **Python Topics** sidebar immediately updates the Skill Header and automatically opens the **`Practice`** tab if an active question is already in progress on that skill, or the **`Learn`** tab otherwise.
     - Made the left sidebar sticky (`.sidebar-card { position: sticky; top: 16px; }`) on desktop viewports so topic navigation remains visible at all times.

---

### C. Compact `Learn` Tab & Collapsible Reference Drawer
- **Problem:** When selecting a profile or topic, displaying both the **Study Plan** box and the full **Lesson & Worked Example** expanded on the same page caused users to immediately click `Start Practice →` without scrolling down, making the long page layout cluttered.
- **Changes Made:**
  1. **Compact Default View:** Only the **Study Plan** card (`Learn → Code → Remember` + **`Start Practice →`**) is expanded by default on the `Learn` tab, fitting cleanly above the fold without scrolling.
  2. **Collapsible Reference Drawer (`<details id="course-material-card" class="card reference-details">`):**
     - Placed the full **Reference Lesson & Worked Example** inside a collapsible `<details>` section labeled *"Optional — read & clarify doubts before practicing"*.
     - Learners who want to study the reference material beforehand can expand it anytime, while learners who click **`Start Practice →`** immediately receive lessons/examples adaptively when PPO prescribes them.

---

### D. PPO-Driven Inline Lesson & Worked Example Delivery in `Practice`
- **Problem:** Under `delayed_retention_ppo` (Gated Option B), actions **`3: explain_concept`** and **`4: show_worked_example`** are passive priming lessons that force a follow-up question on the next step (`awaiting_question = True`). Previously, triggering action `3` or `4` only showed a small notice telling the user to scroll up to the course material card.
- **Changes Made:**
  1. **Inline Rendering in `Practice` (`state.activeLesson` in `app.js`):**
     - Whenever `/api/ppo_step` returns `type: "lesson"`, `app.js` stores the returned lesson payload and `primed_mode` (`"concept"` or `"example"`).
     - Inside the **`Practice`** tab, it immediately renders:
       - **For `3: explain_concept` (`Concept Lesson` badge):** The concept explanation, 3-part theory grid (`theory_sections`), mental model, and common pitfalls to watch out for.
       - **For `4: show_worked_example` (`Worked Example` badge):** The worked example title, dark-themed Python code block, and step-by-step green-bordered subgoals (`Subgoal 1`, `Subgoal 2`, etc.).
  2. **1-Click Transition to Follow-Up Question:**
     - Directly below the inline lesson/worked example, added a **`Got it, next question →`** primary button that immediately calls `/api/ppo_step` to fetch the Gated Option B follow-up question, plus an **`Open Reference`** button that opens the full reference drawer on the `Learn` tab.

---

### E. Streamlined 1-Click Question Flow & Collapsible Advanced Controls
- **Problem:** In the original UI:
  - The primary button was labeled **`Next PPO Step`** (RL jargon unfamiliar to students).
  - After submitting an answer, the user had to click **`Finish`** (which emptied the workspace back to `"Idle / Ready"`) and then click **`Next PPO Step`** a second time to get another question.
  - Research controls (`Manual Override 0..7`, `Run Manual`, `Source: v2.2 Bank / Fresh LLM`, `Format: Auto / ...`) were mixed directly into the main student toolbar.
- **Changes Made:**
  1. **Learner-Friendly Primary Flow (`requestNextStep` in `app.js`):**
     - When no question is active in `Practice`, displays a centered **`Start Practice →`** button (and a **`Review Lesson`** button).
     - When a question is active, displays contextual action buttons:
       - **`Submit`** (enabled while an attempt is available)
       - **`Hint (0/3)`** (progressive 3-tier hints, hidden once passed or after 3 hints)
       - **`Retry (0/2)`** (shown after an incorrect submission when retries remain)
       - **`Reveal Solution`** (shown after an incorrect submission to inspect the answer and explanation)
       - **`Next →`** (shown immediately after submitting an answer; clicking **`Next →`** automatically calls `/api/finish` on the current interaction **and** `/api/ppo_step` in a single sequence so the next question loads in **one click**).
       - **`Stop`** (finishes the current question and returns to the ready state if the student wants to pause).
  2. **Color-Coded Feedback Banners (`.feedback-banner`):**
     - Added distinct green (`.feedback-pass`) and amber (`.feedback-fail`) feedback banners with clear next-step guidance.
  3. **Collapsible Advanced / Instructor Controls (`.advanced-controls`):**
     - Moved `Question Source`, `Question Format`, `Next PPO Step`, `Manual Override (0–7)`, and `Run Manual` into a collapsed `<details class="advanced-controls">` drawer at the bottom of the `Practice` card so researchers can still test manual PPO overrides without cluttering the student view.

---

### F. Interactive "Ask the Tutor — Clarify Your Doubts" Chat (`Learn` Tab + `/api/chat`)
- **Feature Goal:** Allow students reading the Reference Lesson & Worked Example on the `Learn` tab to ask questions and clarify doubts about the material before proceeding to practice.
- **Backend Changes:**
  1. **`PedagogyRealizer.answer_learner_doubt()` ([`adaptive_web/llm_pedagogy.py`](adaptive_web/llm_pedagogy.py)):**
     - Accepts `skill`, `question`, optional chat `history` (last 6 turns), and the learner's `course_material`.
     - When `self.llm.is_live()` is `True`, queries the configured cloud LLM with the current skill's `concept_explanation`, `mental_model`, `pitfalls`, and `worked_example`, returning structured JSON `{ "reply": "...", "code_example": "...", "source": "llm" }`.
     - When offline or if an `LLMClientError` occurs, falls back deterministically to curriculum-grounded explanations (`_RICH_COURSE_CURRICULUM` and `DEFAULT_LESSONS`) tailored to whether the student asked about examples/traces, pitfalls/mistakes, or core concepts.
  2. **`AdaptiveLearningApp.chat_doubt()` ([`adaptive_web/app.py`](adaptive_web/app.py)):**
     - Validates `skill` and `message` (non-empty, max 1,000 characters), supports `request_id` idempotency caching, looks up the learner's latest `course_material` from SQLite, and returns the tutor's response.
  3. **`POST /api/chat` Route ([`adaptive_web/server.py`](adaptive_web/server.py)):**
     - Exposes `/api/chat` with CSRF protection and JSON request/response handling.
- **Frontend Changes:**
  1. **Chat UI (`#tutor-chat-section` in [`adaptive_web/static/index.html`](adaptive_web/static/index.html) & [`style.css`](adaptive_web/static/style.css)):**
     - Added at the bottom of the **Reference Lesson & Worked Example** drawer.
     - Displays a scrollable message thread (`#tutor-chat-messages`) with distinct user bubbles (`.chat-msg-user`) and AI Tutor bubbles (`.chat-msg-assistant`), including syntax-styled `<pre class="code-block">` blocks whenever the tutor provides a Python code snippet.
     - Maintains per-skill conversation history (`state.chatBySkill[state.skill]`) during the session.
     - Supports sending via the **`Ask Tutor`** button or pressing `Enter` in `#tutor-chat-input`.
     - Includes a **`Doubts cleared? Start Practice →`** button right below the chat that switches to `Practice` and loads the first adaptive step.

---

### G. Bug Fixes & Environment / Database Resolution
1. **Fixed `/api/state` Active Question Key Mismatch (`data.active` vs. `data.active_interaction`):**
   - **Bug:** The backend `AdaptiveLearningApp.get_state()` ([`adaptive_web/app.py`](adaptive_web/app.py)) returns the open interaction under the JSON key `"active"`, whereas `adaptive_web/static/app.js` was looking up `d.active_interaction`. As a result, `renderWorkspace()` always received `undefined` after `refreshState()`, causing the `Practice` tab to fall back to the `"Start Practice →"` screen instead of displaying the active question.
   - **Fix:** Added `getActiveInteraction(data)` in [`adaptive_web/static/app.js`](adaptive_web/static/app.js) returning `(data && (data.active || data.active_interaction)) || null` and wired it across workspace rendering, skill switching, profile switching, and `Start Practice`.
2. **Fixed Cross-Skill PPO Action Routing (`review_prerequisite` & `advance_next_topic`):**
   - **Bug:** When PPO chose action `6` (`review_prerequisite`) or `7` (`advance_next_topic`), the backend created the new question under `res.target_skill`, while the frontend continued querying `/api/state` for the previous `state.skill`.
   - **Fix:** Updated `requestNextStep()` in [`adaptive_web/static/app.js`](adaptive_web/static/app.js) to check `if (res && res.target_skill && res.target_skill !== state.skill)`, update `state.skill = res.target_skill`, and initialize that skill before refreshing state.
3. **Resolved `.env` Database Path Override (`ADAPTIVE_WEB_DB`):**
   - Identified that `ADAPTIVE_WEB_DB=3/adaptive_web/data/adaptive_web.sqlite3` in `.env` was pointing relative to the parent `Desktop` directory (`C:\Users\srini\Desktop\3\...`), hiding existing learner profiles in `adaptive_web/data/adaptive_web.sqlite3`. Removed `C:\Users\srini\Desktop\3` after updating `.env` and restored all existing learner profiles with live OpenAI (`openai/gpt-oss-120b`) integration active.
