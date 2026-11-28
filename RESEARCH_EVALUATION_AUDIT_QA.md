# Research Evaluation Plan — Round 1 Audit Q&A & Resolution Log

**Audited Plan:** `3/RESEARCH_EVALUATION_PLAN.md`  
**Audit Report Answered:** `review_r1.md` (`ISSUE-01` through `ISSUE-09`: 1 Critical, 4 High, 3 Medium, 1 Low)  
**Resolution Status:** All 9 issues verified directly against the workspace codebase and resolved in `3/RESEARCH_EVALUATION_PLAN.md`. Zero source code (`.py`, `.js`, `.html`, `.css`), datasets, checkpoints, or configs were modified, and zero experiments were executed.

---

## Summary Resolution Matrix

| Issue ID | Severity | Title | Verification Status | Exact Sections Updated in `3/RESEARCH_EVALUATION_PLAN.md` |
|---|---|---|---|---|
| `ISSUE-01` | CRITICAL | Ablation 3 Mask & Routing Leak + `obs_dim` / `clock` ambiguity | Confirmed in `3/ppo/env.py`, `agent.py` | §7.3.1, §7.3.2, §9 (`#4`) |
| `ISSUE-02` | HIGH | Silent `PedagogyRealizer` internal fallbacks despite `LLM_FALLBACK_PROVIDERS=""` | Confirmed in `3/adaptive_web/llm_pedagogy.py`, `llm_client.py` | §0.3 (`#4`), §1.3, §4.1, §4.2, §4.3, §7.1, §7.2, §8.1 (`3–5`), §8.2 (`T-A1`, `T3a`, `T3b`, `T3c`), §9 (`#3`) |
| `ISSUE-03` | HIGH | Tier 2 4-Arm `rerank_boost` scale, `0.0`-score fallback, `Dense` text, & `rewrite_query()` asymmetries | Confirmed in `3/adaptive_web/rag.py` | §3.2, §3.5, §7.1, §8.1 (`2`), §8.2 (`T2`), §9 (`#2`) |
| `ISSUE-04` | HIGH | Tier 3c non-circular sourcing, prompt bank, `deterministic_score` leakage, & `unresolved` handling | Confirmed in `3/adaptive_web/grader.py`, `llm_pedagogy.py`, `app.py` | §4.3, §8.1 (`5`), §8.2 (`T3c`), §9 (`#3`) |
| `ISSUE-05` | HIGH | Ablation 2 `active_preview` RAG distortion & `chat_doubt` stateful contamination | Confirmed in `3/adaptive_web/llm_pedagogy.py`, `app.py` | §4.1, §7.2, §9 (`#3`) |
| `ISSUE-06` | MEDIUM | Tier 2 `qgen_curriculum` (`k=2`) pool saturation & Ablation 1 downstream protocol | Confirmed in `3/adaptive_web/llm_pedagogy.py` | §3.3, §3.4, §3.5, §7.1, §8.1 (`2`), §8.2 (`T-A1`) |
| `ISSUE-07` | MEDIUM | Tier 1 delayed aggregate scope, empty ECE bins, T1-B source, & missing R3 tables | Confirmed in `3/model/memory/evaluation.py`, `run_v22.py`, `tabular.py`, `EVALUATION_METRICS_EXPLAINED.md` | §2.2, §2.4, §2.5, §8.1 (`1`), §8.2 (`T1-A`–`T1-E`), §9 (`#1`) |
| `ISSUE-08` | MEDIUM | Missing R3 tables for Budget Curves, 8-Action Shares, 5-Check Suite, Ablation 1 Downstream, & `env.py` `total_budget` | Confirmed in `3/ppo/runs/v2.2-ppo-main/REPORT.md`, `3/ppo/env.py` | §5.2, §7.1, §8.2 (`T-A1`, `T4-A`–`T4-E`), §9 (`#4`) |
| `ISSUE-09` | LOW | Tier 3a/3b/3c minor schema & definition clarifications | Confirmed in `3/adaptive_web/llm_pedagogy.py` | §4.1, §4.2, §8.1 (`3–5`), §8.2 (`T3b`) |

---

## Point-by-Point Codebase Verification & Answers

### `ISSUE-01` [CRITICAL]: Ablation 3 (`ppo_no_longterm_state`) Mask & Routing Leak + `obs_dim` / `clock` Ambiguity
- **Codebase Verification:**
  - `3/ppo/env.py:346-356`: `_weak_prereqs_or_decayed(self, skill)` explicitly inspects `self.metrics_cache[p]["p_30d"] < 0.56 or self.metrics_cache[p]["floor_b"] < 0.40` (for prerequisites) and `self.metrics_cache[s]["p_30d"] < 0.52 or self.metrics_cache[s]["floor_b"] < 0.36` (for visited topics).
  - `3/ppo/env.py:384-385`: `valid_action_mask()` sets `mask[6] = True` iff `_weak_prereqs_or_decayed(skill)` is non-empty.
  - `3/ppo/agent.py:40-44`: `MaskedActorCritic.forward(obs, mask)` receives `mask` directly (`logits.masked_fill(~mask, -1e9)`), and `agent.py:26, 33` initializes orthogonal layers sized to `obs_dim`.
  - `3/ppo/env.py:772, 789`: When `action == 6` (`review_prerequisite`) or `action == 7` (`advance_next_topic` after all skills are visited), `step()` routes to `min(..., key=lambda s: self.metrics_cache[s]["p_30d"])`.
  - `3/ppo/env.py:172-181, 802`: `query_metrics()` applies `elapsed_days` since the last event on that skill (`gaps = self.ref_gaps_tensor + elapsed_days`, `_at_time(state, elapsed_days)`) and `_refresh_all_metrics()` runs at session boundaries (`env.py:802`), meaning `obs[0..2]` (`p_easy`, `p_med`, `p_hard`) and `obs[7]` (`active_k`) are instantaneous (`query_gap = 0`) evaluations at current `self.clock`.
- **Answer & Resolution:**
  1. **Removed all `p_30d` / `floor_b` leakage from `valid_action_mask()` and `step()` routing:** In the `ppo_no_longterm_state` ablated environment, `_weak_prereqs_or_decayed` in `valid_action_mask()` (`mask[6]`) is replaced with short-term `p_med`-only criteria (`p_med < 0.60` for prerequisites; `p_med < 0.55` for visited topics), and `Action 6` (`env.py:772`) and `Action 7` (`env.py:789`) target selection `min(..., key=p_30d)` is replaced with `min(..., key=lambda s: self.metrics_cache[s]["p_med"])`. Step shaping and terminal reward (`env.py:804-825`) remain identical.
  2. **Pinned unique network parameterization:** Removed `"either obs_dim=23 ... or obs_dim=36 ..."` and uniquely pinned **`obs_dim = 23`** slicing `obs[0..2, 7..19, 21, 30..35]` into `MaskedActorCritic(obs_dim=23, act_dim=8, hidden_dim=64)`.
  3. **Clarified clock semantics:** Updated §7.3.1 and §7.3.2 to state accurately that `obs[0..2]` and `obs[7]` are `query_gap = 0` evaluations at current `self.clock` (including elapsed inter-session decay, with zero future 7d/30d or `floor_b` exposure).
  4. **Recorded in Pre-Execution Harness Checklist (§9 item `#4`):** Documented the exact `env.py` mask/routing/obs changes without modifying `.py` files now.
- **Sections Updated:** §7.3.1, §7.3.2, §9 (`#4`).

---

### `ISSUE-02` [HIGH]: Silent `PedagogyRealizer` Fallbacks Despite `LLM_FALLBACK_PROVIDERS=""`
- **Codebase Verification:**
  - `3/adaptive_web/llm_pedagogy.py:963-1014` (`generate_question`): catches `except LLMClientError: pass` and falls back silently to `sample_v22_task` (`llm.clone.*`, `difficulty_source="llm_pedagogy_fallback"`) or deterministic `synth.*` (`version="synth-v1"`). Note also that on success (`lines 956-962`), `generate_question` does not inject a `"source": "llm"` field, so live-LLM origin must be confirmed via `llm_client.last_call_info()` and absence of fallback markers.
  - `3/adaptive_web/llm_pedagogy.py:1146-1153` (`evaluate_self_explanation`): catches `except LLMClientError: pass` and silently returns `det_grade` with `"source": "deterministic_double"`.
  - `3/adaptive_web/llm_pedagogy.py:1472-1494` (`answer_learner_doubt`): catches `except LLMClientError: pass` AND falls through if `reply` is empty (`if reply:` at line `1474`), returning `"source": "fallback"`.
  - `3/adaptive_web/llm_client.py:147-153`: `LLMClient.last_call_info()` exposes `provider`, `model`, `failover_used`, and `attempts`.
- **Answer & Resolution:**
  1. **Explicit Harness Enforcement:** Updated §0.3 (`#4`), §1.3, §4.1, §4.2, §4.3, §7.1, §7.2, and §9 (`#3`) so the evaluation harness checks **`res.get("source") == "llm"`** (or non-fallback metadata in `generate_question`) **and** **`llm_client.last_call_info()`** (`failover_used == False`) on every call.
  2. **Strict Non-Scoring of Fallbacks:** `"fallback"` (Tier 3a / Ablations 1–2) and `"deterministic_double"` (Tier 3c) outputs are never scored as live-LLM outputs; transient errors allow at most `max_retries = 2` with backoff, and any remaining failures are logged explicitly as `api_error` and excluded from live-LLM numerator/denominators (with `live_llm_n` and `api_error_rate` reported separately).
  3. **Tier 3b Denominator & Error Attribution:** Distinguished schema-validation failures from `api_error` (`LLMClientError` / empty JSON) via `source`, `api_error`, and `rejection_reason`, while keeping **both** in the `N = 180` denominator for `First-Pass Valid`, `Post-Repair Valid`, and `Usable Rate` (and reporting `api_error_rate` separately).
  4. **Updated JSONL Schemas:** Added `source`, `api_error`, and `failover_used` to `tier3a_eval.jsonl`, `tier3b_eval.jsonl`, and `tier3c_eval.jsonl` in §8.1.
- **Sections Updated:** §0.3 (`#4`), §1.3, §4.1, §4.2, §4.3, §7.1, §7.2, §8.1 (`3–5`), §8.2 (`T-A1`, `T3a`, `T3b`, `T3c`), §9 (`#3`).

---

### `ISSUE-03` [HIGH]: Tier 2 4-Arm Reranking, `0.0`-Score Fallback, `Dense` Indexed Text, & `rewrite_query()` Asymmetries
- **Codebase Verification:**
  - `3/adaptive_web/rag.py:58-61`: `KnowledgeChunk.searchable_text` joins `skill, difficulty, task_type, title, section, content`.
  - `3/adaptive_web/rag.py:589-619`: `rewrite_query()` appends `SKILL_META`, `difficulty`, `task_type`, `_PYTHON_SYNONYMS`, and truncated `extra_context`.
  - `3/adaptive_web/rag.py:697-765`: `FTS5` BM25 matches `_build_safe_fts_query(rewritten)` up to `LIMIT max(48, top_k * 12)` without `0.0`-score `curriculum` retention.
  - `3/adaptive_web/rag.py:795`: Vector TF-IDF retains `score > 0.0 or (ch.source_type == "curriculum" and ch.skill == skill)`.
  - `3/adaptive_web/rag.py:862`: Dynamic `episodic` chunks get `+0.015` (and `inst.*` `+0.01`) cosine boost before ranking.
  - `3/adaptive_web/rag.py:892-926`: `final_score = rrf + rerank_boost`, where 2-source `rrf <= 2/61 ≈ 0.0328` while `rerank_boost` can reach `+0.075`.
- **Answer & Resolution:**
  1. **Equal-Scale Reranking + Un-Reranked Diagnostic:** All 4 arms apply the exact same `rerank_boost` (`rag.py:898-921`) over an **equal-scale** base score: two-source arms (`Arm 2`, `Arm 4`) use `1/(60+r1) + 1/(60+r2)` (max `2/61`), while single-source arms (`Arm 1`, `Arm 3`) use matched-scale `2.0 / (60 + rank)` (max `2/61`) so `rerank_boost` never carries double relative weight in single-stage arms. In addition, **un-reranked** `nDCG@k` and `Recall@k` are logged and reported for all 4 arms as a pure-retriever diagnostic.
  2. **Matched `Dense` Text & Vector Rules:** `Dense` encodes `KnowledgeChunk.searchable_text` (`rag.py:59-61`), and all vector-scored arms (`Arm 2`, `Arm 3`, `Arm 4`) apply identical `episodic +0.015` (`rag.py:862`) and matching-skill `curriculum` retention (`rag.py:795`).
  3. **Explicit `Arm 1` `< top_k` Behavior:** `Arm 1` (`BM25 alone`) runs strict FTS5 BM25 matching (`rag.py:697-765`) without `0.0`-score vector `curriculum` fallback, logging `bm25_underfilled` (`len(results) < top_k`) as an operational diagnostic.
  4. **Dev-Selected `Dense` Query Input Policy:** On the **Dev** split (`N_dev = 66`), `Dense` is evaluated with both `(a) raw query + clean metadata` and `(b) rewrite_query()`, and the winning query-input rule on Dev `nDCG@k` is frozen for both `Arm 3` and the `Dense` branch of `Arm 4` before evaluating Held-Out Test (`N_test = 132`), while `BM25` always uses `rewrite_query()`.
- **Sections Updated:** §3.2, §3.5, §7.1, §8.1 (`2`), §8.2 (`T2`), §9 (`#2`).

---

### `ISSUE-04` [HIGH]: Tier 3c Non-Circular Sourcing, Prompt Bank, `deterministic_score` Leakage, & `unresolved` Bug
- **Codebase Verification:**
  - `3/adaptive_web/grader.py:110-163, 200-313`: `grade_self_explanation()` uses hardcoded `_CONCEPT_ALIASES`, `_HEDGING_PATTERNS`, `_CAUSAL_CONNECTORS`, and `_infer_concepts_from_prompt(prompt)`, returning `grading_status="unresolved"`, `score=None`, `full_success=None` at `lines 263-274` when `not key_concepts` and `confidence < 0.35`.
  - `3/adaptive_web/llm_pedagogy.py:1113, 1126-1138`: `evaluate_self_explanation()` calls `grade_self_explanation(prompt, learner_explanation)` (`rubric=None`), passes `"deterministic_score": det_grade["score"]` and `"deterministic_full_success": det_grade["full_success"]` to the live LLM (`lines 1126-1127`), and hardcodes `"grading_status": "resolved"` at `line 1138`.
  - `3/adaptive_web/app.py:423-454`: Defines the production 18-prompt `se_prompt_bank` (`3` prompts × `6` skills).
- **Answer & Resolution:**
  1. **Pinned Prompt Pool & Rubric:** Pinned `app.py:423-454` `se_prompt_bank` (`18` prompts, `3`/skill, `6–7` responses/prompt) with `rubric=None`, matching production `evaluate_self_explanation()`.
  2. **Strictly Blinded Non-Circular Response Sourcing:** Required the `N = 120` responses to be authored/collected from independent Python-competent contributors (or anonymized pilot logs) **strictly blinded to `_CONCEPT_ALIASES`, `_HEDGING_PATTERNS`, `_CAUSAL_CONNECTORS`, and `_infer_concepts_from_prompt()`**, with human reviewers likewise blinded to `det_score_0_1`, `det_confidence`, and `llm_confidence`.
  3. **Explicit Disclosure of Conditional `llm_confidence`:** Clarified in §4.3 and Table T3c that `evaluate_self_explanation()` sends `deterministic_score` to the LLM (`llm_pedagogy.py:1126`), so `llm_confidence` is conditioned on `det_score_0_1` rather than an independent LLM score (and scored strictly on `source == "llm"` calls).
  4. **`resolved_coverage` & `unresolved` Handling:** Defined `resolved_coverage = n_resolved / 120`. MAE, RMSE, and QWK are computed on resolved items (`det_score_0_1 is not None`), while any `unresolved` item (`score is None`) counts as `full_success = False` in Full-Success P/R/F1 across all `N = 120` items. Logged the `llm_pedagogy.py:1138` `"grading_status": det_grade["grading_status"]` fix in §9 (`#3`).
- **Sections Updated:** §4.3, §8.1 (`5`), §8.2 (`T3c`), §9 (`#3`).

---

### `ISSUE-05` [HIGH]: Ablation 2 `active_preview` RAG Distortion & `chat_doubt` Stateful Contamination
- **Codebase Verification:**
  - `3/adaptive_web/llm_pedagogy.py:1395-1415`: `answer_learner_doubt` derives `active_preview` from `active_question` and passes `extra_context=active_preview` to `self.rag.retrieve()`.
  - `3/adaptive_web/llm_pedagogy.py:1429-1436, 1475-1483`: `has_open_unpassed` controls both `eduguard_rule` in `system_prompt` and `_enforce_eduguard()`.
  - `3/adaptive_web/app.py:1186-1246`: `chat_doubt` runs cross-skill `_is_pasted_active_question()` detection (`lines 1189-1196`) and persists `chat_messages` plus `self.rag.record_episodic(..., event_kind="doubt")` (`lines 1219-1246`).
- **Answer & Resolution:**
  1. **Isolated Per-Prompt DB Snapshot via `chat_doubt()`:** Tier 3a and Ablation 2 prompts run through `AdaptiveLearningApp.chat_doubt()` (exercising cross-skill pasted-question detection at `app.py:1189-1196`) on an **isolated per-prompt SQLite DB snapshot** cloned from the seeded learner profile + open interaction, guaranteeing zero cross-prompt `chat_messages` or `episodic` doubt leakage.
  2. **Zero-Distortion `EduGuard OFF` Specification:** `EduGuard OFF` explicitly passes the **exact same `active_question`** (keeping `active_preview` in `rag.retrieve()` at `llm_pedagogy.py:1399, 1413` identical to `EduGuard ON`), disabling **only** `_is_pasted_active_question` blocking, the `eduguard_rule` prompt clause (using the neutral non-blocking string), and `_enforce_eduguard()` (`guardrail_status = "disabled"`). Recorded `eduguard_enabled: bool = True` in §9 (`#3`).
- **Sections Updated:** §4.1, §7.2, §9 (`#3`).

---

### `ISSUE-06` [MEDIUM]: Tier 2 `qgen_curriculum` (`k=2`) Pool Saturation & Ablation 1 Downstream Protocol
- **Codebase Verification:**
  - `3/adaptive_web/llm_pedagogy.py:836-856`: `retrieve_exemplars` passes `query=f"{skill} {difficulty} {task_format} {' '.join(recent_mistakes or [])}"`; `question_curriculum` passes `f"{skill} {difficulty} pitfalls edge cases"` with `source_types=("curriculum", "episodic")`, `top_k=2`. Without seeded `episodic` rows, `curriculum` has only `4` chunks/skill.
- **Answer & Resolution:**
  1. Required every Dev (`30`) and Test (`60`) q-gen query to use a seeded learner profile with **non-empty `episodic` chunks (`>= 3–5` target-skill rows)** plus **varied `recent_mistakes` and `exclude_task_ids`** across queries (`§3.3`, `§3.4`, `§8.1`).
  2. Specified explicitly in `§3.5` and `§7.1` (and Table `T-A1`) that the Ablation 1 fixed-LLM downstream check (`72` chat + `60` q-gen) uses the **exact Tier 3a & Tier 3b 2-reviewer blinded adjudication protocol** with `source == "llm"` verified.
- **Sections Updated:** §3.3, §3.4, §3.5, §7.1, §8.1 (`2`), §8.2 (`T-A1`).

---

### `ISSUE-07` [MEDIUM]: Tier 1 Delayed Aggregates, Empty ECE Bins, T1-B Source, & Missing R3 Tables
- **Codebase Verification:**
  - `3/model/memory/evaluation.py:25, 68-76, 87`: `masks["assessment"]` pools all `delay != 0` queries; `probability_report` skips empty bins (`if not count: continue`) and omits `roc_auc`; `paired_intervals` defaults to `draws=1000`.
  - `3/model/run_v22.py:26, 40-41, 136-147`: `SEEDS` is hardcoded; `summarize()` lacks `min/max`; `delayed_equal_horizon_log_loss` averages `day_1`, `day_7`, `day_30` `log_loss`, while `EVALUATION_METRICS_EXPLAINED.md:115-127` reports pooled `assessment` for Delayed Brier/ECE/Acc.
  - `3/model/memory/tabular.py:19-22`: `HistGradientBoostingClassifier` subsamples `200,000` rows out of `219,284` using `random_state=seed`.
- **Answer & Resolution:**
  1. **Pinned Delayed Scope:** `Del. LL` in Table `T1-A` is `delayed_equal_horizon_log_loss` (equal-weight `day_1`, `day_7`, `day_30`); `Del. AUC`, `Del. Brier`, `Del. ECE-10` (and `Del. Acc`) are computed on pooled `assessment` (`delay != 0`) and averaged across seeds (`Mean ± SD`).
  2. **Pinned Table `T1-B` & Empty-Bin Padding:** Pinned `T1-B` to seed-averaged probabilities across the 5 pre-registered seeds on `main_test` (`ordinary` and pooled `assessment`), with all 10 uniform bins emitted (empty bins padded `queries: 0, mean_prediction: null, observed_rate: null`).
  3. **Pre-Run Harness Updates & `history_boosting` Multi-Seed:** Documented `--seeds`, `min/max` in `summarize()`, `roc_auc` + 10-bin padded `calibration` in `evaluation.py`, `draws=2000` in `paired_intervals()`, and 5-seed `history_boosting` in §2.5 & §9 (`#1`).
  4. **Added Missing R3 Tables:** Added **Table T1-C** (`Follow-up` & `1d/7d/30d` Log Loss), **Table T1-D** (`8` Paired Bootstrap Contrasts), and **Table T1-E** (6-Check Behavioral Gate).
- **Sections Updated:** §2.2, §2.4, §2.5, §8.1 (`1`), §8.2 (`T1-A`–`T1-E`), §9 (`#1`).

---

### `ISSUE-08` [MEDIUM]: Missing Tier 4 & Ablation 1 R3 Diagnostic Tables + `env.py` `total_budget`
- **Codebase Verification:**
  - `3/ppo/runs/v2.2-ppo-main/REPORT.md:15-31`: Reports 8-Action Distribution and 5-Check Behavioral Sanity Suite per cohort.
  - `3/ppo/env.py:78, 423, 815`: Hardcodes `TOTAL_BUDGET = 32.0` inside `_observation()` (`line 423`) and `step()` (`line 815`).
- **Answer & Resolution:**
  1. **Added Missing R3 Tables in §8.2:** Added **Table T-A1** (Ablation 1 Fixed-LLM Downstream Check), **Table T4-D** (Tier 4 Matched Budget Curves `16.0/24.0/32.0/40.0`), and **Table T4-E** (8-Action Distribution + 5-Check Behavioral Sanity Suite).
  2. **Expanded Cohort Coverage in T4-A/T4-B/T4-C:** Added explicit `power_law_test` and `long_gap_test` blocks/rows in Tables `T4-A`, `T4-B`, and `T4-C`.
  3. **Budget Parameterization Note:** Documented `AdaptiveTutorEnv(..., total_budget=...)` (`env.py:423, 815`) in §5.2 and §9 (`#4`).
- **Sections Updated:** §5.2, §7.1, §8.2 (`T-A1`, `T4-A`–`T4-E`), §9 (`#4`).

---

### `ISSUE-09` [LOW]: Tier 3a/3b/3c Minor Schema & Definition Clarifications
- **Codebase Verification:** `3/adaptive_web/llm_pedagogy.py:801-808, 910` confirms `temperature=0.2` and anti-duplicate `0.85` check against only the `3` retrieved exemplars.
- **Answer & Resolution:**
  1. Added `claim_support_fraction` to `tier3a_eval.jsonl` (§8.1) and clarified structured consensus or third-party adjudication in §4.1.
  2. Stated explicitly in §4.2 and Table `T3b` that `Post-Repair Valid` is cumulative `((first_pass_valid + repaired_valid) / 180)` with conditional `repair_yield = repaired_valid / repair_attempted` reported separately, and specified per-cell variation (`recent_mistakes` + `exclude_task_ids`) across the `2` requests/cell.
  3. Added `prompt_id` to `tier3c_eval.jsonl` (§8.1).
- **Sections Updated:** §4.1, §4.2, §8.1 (`3–5`), §8.2 (`T3b`).
