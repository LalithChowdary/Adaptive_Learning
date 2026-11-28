# Research Evaluation, Seeding, and Documentation Plan (Tiers 1–5 & Prioritized Ablations)

**Project:** Adaptive Python Tutoring System (`3/data_Set/v2.2/`, `3/model/`, `3/ppo/`, `3/adaptive_web/`)  
**Document Purpose:** Pre-execution experimental comparison, seeding, statistical reporting, and report-ready documentation protocol for Tiers 1–5 and three prioritized ablations.  
**Execution Status:** Planning specification only. Zero experiments, training runs, evaluations, or code edits are executed by this document.

---

## 0. Primary Outcome, Causal Scope Boundary, and Prohibitions

### 0.1 Primary Research Outcome
- **Primary Outcome:** Equal-weight mean simulated unaided pass rate across **1-, 7-, and 30-day** delayed independent assessments (`delayed_mean = (day_1 + day_7 + day_30) / 3`, averaging 18 items per horizon across `6 skills × 3 difficulties` = `54` items per learner in `3/ppo/env.py:evaluate_delayed_retention`) at a matched teaching budget of **`TOTAL_BUDGET = 32.0`** (`SESSION_BUDGET = 8.0`), paired at the learner level against **`threshold_heuristic`** (`ThresholdHeuristicPolicy` in `3/ppo/agent.py`).
- **Highlighted Retention Horizon:** **Day-30 unaided pass rate (`day_30`)** is reported and highlighted separately alongside `delayed_mean` to isolate durable long-term retention from 1-day cramming fluency.

### 0.2 Explicit Causal Scope Boundary
- All Tier 1 and Tier 4 causal claims are strictly scoped to **the learner memory model and teaching policy in simulation** (`3/data_Set/v2.2/simulator.py` and `3/ppo/env.py:AdaptiveTutorEnv` backed by frozen `monotonic_gru`).
- Tier 2 and Tier 3 offline/human-judged evaluations establish **retrieval quality, pedagogical response quality, anti-leakage safety, question solvability, and grading fidelity** under a fixed LLM configuration; they do **not** establish real-student learning gains.
- Real-student learning effectiveness remains unvalidated unless and until a separate IRB-approved **Tier 5 human study** is executed, and Tier 5 results must never be merged with Tier 4 simulation metrics.

### 0.3 Explicit Methodological & Reporting Prohibitions
1. **No software engineering test counts as research metrics:** Software unit, dataset-audit, or end-to-end test-suite pass counts MUST NOT appear anywhere as research evaluation metrics or in results tables.
2. **No blanket parametric gates:** Omit mandatory `p < 0.001` significance thresholds, blanket Welch's *t*-tests (which ignore learner/prompt clustering and paired designs), and compulsory Cohen's *d*.
3. **No unverified retrospective claims presented as pre-registered:** Existing 3-seed Tier 1 (`3/model/runs/v2.2-mps-main/`, `3/model/runs/v2.2-mps-transformer-comparison/`) and Tier 4 (`3/ppo/runs/v2.2-ppo-main/`) runs MUST be labeled **Retrospective**, distinct from **Pre-Registered** expanded-seed and held-out evaluations.
4. **No silent failover or fallback-template contamination:** During live-LLM evaluations (Tiers 2–3, Ablations 1–2), disable multi-provider failover (`LLM_FALLBACK_PROVIDERS=""`) **and** verify `res.get("source") == "llm"` plus `llm_client.last_call_info()` on every call, never scoring `PedagogyRealizer` internal fallbacks (`"fallback"`, `"deterministic_double"`, `llm.clone.*`, `synth.*`) as live-LLM outputs.

### 0.4 Evaluation Architecture Overview

| Tier / Ablation | Component & Codebase Anchor | Primary Paired Unit | Primary Metric(s) |
|---|---|---|---|
| **Tier 1** | Learner Memory (`3/model/run_v22.py`, `memory/recurrent.py`) | Held-out learner (`N = 200` / cohort) | Log Loss (Ordinary & Delayed separately), 6/6 Behavioral Gate |
| **Tier 2** | Hybrid RAG (`3/adaptive_web/rag.py`) | Held-out query (`72` chat + `60` q-gen) | `nDCG@k`, `Recall@k` at exact budgets (`k=4`; `k=3`/`k=2`) |
| **Tier 3a** | Tutor & EduGuard (`3/adaptive_web/llm_pedagogy.py`, `app.py`) | Prompt (`96` adaptation + `120` paired safety) | Python correctness, faithfulness, leakage, over-refusal, 3 rubric dims |
| **Tier 3b** | Question Generator (`llm_pedagogy.py`, `catalogue_bridge.py`) | Stratified request (`N = 180`: `6×5×3×2`) | Usable generated-question rate, first-pass/repair validity, fallback rate |
| **Tier 3c** | Self-Explanation Grader (`3/adaptive_web/grader.py`) | Learner response (`N = 120`: `20`/skill) | MAE on `[0, 1]`, full-success P/R/F1 at `score >= 1.0`, confidence–error check |
| **Tier 4** | Teaching Policy (`3/ppo/env.py`, `train_ppo.py`, `agent.py`) | Held-out learner (`N = 200` / `400`) | Equal-weight 1/7/30d unaided pass rate vs `threshold_heuristic`; Day-30 |
| **Tier 5 (Opt.)** | Human Pilot Outline (strictly separated) | Enrolled human learner (`N = 20–40`) | Paired `post - pre` change, delayed score, completion rate |
| **Ablation 1** | `Deployed BM25 + TF-IDF` vs `BM25 + Dense` | Held-out query + fixed-LLM downstream | `nDCG@k`, `Recall@k`, downstream faithfulness & usable question rate |
| **Ablation 2** | EduGuard ON vs OFF | Paired `60` protected + `60` legitimate prompts | Answer leakage rate vs over-refusal rate (95% Wilson CIs) |
| **Ablation 3** | `delayed_retention_ppo` vs `ppo_no_longterm_state` | Held-out learner (retrained from scratch) | `delayed_mean`, `day_30`, `latent_mean_floor` $b_t$, lowest-skill Day-30 |

---

## 1. Global Statistical, Seeding, and Fixed-LLM Protocol

### 1.1 Paired Comparisons & Confidence Intervals
- **Paired Design Only:** Every model/policy/arm comparison evaluates competing systems on the **exact same learners** (Tiers 1 & 4) or **exact same queries/prompts** (Tiers 2 & 3).
- **Clustered Paired Bootstrap 95% CIs:**
  - **Tiers 1 & 4 (Learner-Clustered):** Resample whole learners with replacement (`B = 2,000` draws for pre-registered runs; `B = 1,000` in retrospective scripts `evaluation.py:87` and `train_ppo.py:39`), preserving all queries/skills per learner. Compute the 2.5th and 97.5th percentiles of paired differences $\Delta = \text{Metric}_{\text{candidate}} - \text{Metric}_{\text{reference}}$. For multi-seed neural models, first average per-query/per-learner values across seeds within each learner before resampling.
  - **Tiers 2 & 3 (Query/Prompt-Clustered):** Resample queries or prompts with replacement (`B = 2,000` draws, fixed RNG seed `20261201`) to compute 95% CIs for paired differences ($\Delta\text{nDCG@k}$, $\Delta\text{Recall@k}$, $\Delta\text{MAE}$, $\Delta\text{rubric}$).
- **Binary & Safety Proportions:** For all binary rates (answer leakage, over-refusal, Python correctness, first-pass validity, post-repair validity, usable generated-question rate, bank fallback rate), report:
  1. **Exact counts** (`k / N`),
  2. **Percentage rate** (`100 * k / N`), and
  3. **95% Wilson score confidence interval** (with prompt-clustered bootstrap CI additionally reported whenever multiple outputs share a prompt).
- **Seed-to-Seed Variation:** For all stochastic/multi-seed models (Tiers 1 & 4), report **mean**, **sample SD (`ddof=1`)**, **min–max range across seeds**, and per-seed tables in the experiment archive.

### 1.2 Retrospective vs. Pre-Registered Distinction
- **Retrospective Runs:** Existing saved artifacts (`3/model/runs/v2.2-mps-main/`, `3/model/runs/v2.2-mps-transformer-comparison/`, `3/ppo/runs/v2.2-ppo-main/`) used 3 seeds (`20261120`, `20261121`, `20261122`). Because `monotonic_gru` and `delayed_retention_ppo` design choices were refined iteratively on `v2.2`, these runs are strictly labeled **Retrospective**.
- **Pre-Registered Runs:** All new expanded-seed runs (Tier 1: 5 seeds; Tier 4: 10 seeds; Ablation 3: 10 seeds) and all Tier 2 / Tier 3 / Ablation 1–2 evaluations are locked prior to running any evaluation script, with frozen hyperparameters, fixed query/prompt splits, and pinned seeds.

### 1.3 Fixed-LLM Configuration (Tiers 2 & 3, Ablations 1 & 2)
Grounded in `3/adaptive_web/config.py` (`Settings`, `DEFAULT_MODELS`) and `3/adaptive_web/llm_client.py` (`LLMClient`):
- **Single Pinned Provider & Model:** Pin one live `(provider, model)` pair from `Settings.from_env()` across **all** arms in Tiers 2–3 (recommended primary live configuration from `DEFAULT_MODELS`: `groq` with `openai/gpt-oss-120b`, or `openai` with `gpt-4o-mini`; record exact pinned identifier in run metadata without exposing any API key).
- **Exact Decoding Parameters (`3/adaptive_web/llm_pedagogy.py` & `llm_client.py`):**
  - Doubt Tutor (`answer_learner_doubt`, `llm_pedagogy.py:1469-1470`): `temperature = 0.3`, `max_tokens = 750`, JSON mode (`schema_name="tutor_chat"`).
  - Question Generation (`generate_question`, `llm_pedagogy.py:910, 936`): `temperature = 0.2` (`complete_json` default), `max_tokens = 1800` (`complete_json` default), JSON mode (`schema_name="question"` and 1-shot `schema_name="question_repair"`).
  - Self-Explanation Feedback (`evaluate_self_explanation`, `llm_pedagogy.py:1131`): `temperature = 0.2`, `max_tokens = 1800`, JSON mode (`schema_name="self_explanation_eval"`).
  - Timeout: `LLM_TIMEOUT_SECONDS = 45.0`.
- **Failover & Silent-Fallback Prevention During Experiments:** Set `LLM_FALLBACK_PROVIDERS=""` during experimental evaluation runs so `LLMClient` does not fail over to secondary providers. **Crucial Codebase Verification (`llm_pedagogy.py:963-1014, 1146-1153, 1472-1494`):** `PedagogyRealizer` catches `LLMClientError` (and empty `reply` strings in `answer_learner_doubt`) internally across `generate_question`, `evaluate_self_explanation`, and `answer_learner_doubt`, silently returning `llm.clone.*`/`synth.*`, `"deterministic_double"`, or `"fallback"`. Therefore, every live-LLM evaluation harness MUST explicitly assert/check **`res.get("source") == "llm"`** (for chat/self-explanation, or absence of `llm.clone.*` / `synth-v1` + `llm_client.last_call_info()` for question generation) and inspect **`llm_client.last_call_info()`** (`failover_used == False`, `attempts`), logging `source` and `api_error` on every item and **never** scoring fallback/deterministic-double outputs as live-LLM generations. Transient transport errors may be retried up to a pre-registered cap (`max_retries = 2` with exponential backoff); any remaining failure is logged explicitly as `api_error`.

---

## 2. Tier 1: Learner Memory Evaluation (`monotonic_gru` & Comparators)

### 2.1 Models Evaluated (9-Model Ladder + 2 Anchors)
Verified in `3/model/run_v22.py`, `3/model/memory/tabular.py`, `3/model/memory/recurrent.py`, and `3/model/runs/v2.2-mps-transformer-comparison/run.json`:

| Arm ID | Category | Architecture / Inputs | Params | Markov Runtime State |
|---|---|---|---:|---|
| `constant` | Anchor | Predicts `main/train` base rate ($\bar{y}_{\text{train}} = 0.54304$) | 1 | None |
| `task_logistic` | Tabular | L2 Logistic Regression on `TASK_SIZE = 19` task features (`C in {0.1, 1.0}`) | 20 | None |
| `history_logistic` | Tabular | L2 Logistic Regression on `19` task + `SUMMARY_SIZE = 34` causal summary features (`53` dims) | 54 | Summary counters |
| `history_boosting` | Tabular | `HistGradientBoostingClassifier` on same `53` dims (`max_leaf_nodes in {15, 31}`, `max_iter in {120, 160}`) | — | Summary counters |
| `time_input_gru` | Sequence | 32-dim `GRUCell` with `47`-dim `Evidence` token + $\log(1+\Delta t)$; 2-layer SiLU readout | 9,633 | $O(1)$ 32 floats |
| `time_decay_gru` | Sequence | 32-dim `GRUCell` with exponential hidden-state decay toward learned prior $h_0$ | 10,689 | $O(1)$ 32 floats |
| `time_aware_transformer` | Sequence | 32-dim causal 4-head `CausalTimeAttentionEncoder` with learnable $-\text{softplus}(\tau)\log(1+\Delta t)$ bias | 11,845 | $O(L)$ history replay |
| `kt_decay_gru` | Grey-Box | 2-channel HLR + Power-Law + Consolidated Floor + 3PL IRT `TaskReadoutHead` (no GRU context) | 447 | $O(1)$ 6 floats |
| `monotonic_gru` | Grey-Box | 6-channel HLR + Power-Law + Consolidated Floor + 18-dim `GRUCell` context + 3PL IRT `TaskReadoutHead` | 10,223 | $O(1)$ 32 floats |
| `monotonic_transformer` | Grey-Box | 6-channel HLR + Power-Law + Consolidated Floor + 18-dim 3-head causal attention context + 3PL IRT | 10,208 | $O(L)$ history replay |
| `oracle` | Ceiling | `oracle_predictions()` using true hidden `strength_at_attempt`, `hidden_demand`, 32-pt Gauss-Hermite quadrature | — | Diagnostic only (hidden) |

### 2.2 Dataset Splits, Cohorts, and Eligible Label Definitions
Verified in `3/data_Set/v2.2/reports/RELEASE_SUMMARY.json` and `3/model/memory/data.py:encode_record`:
- **Eligibility Rule:** Binary `full_success in {0, 1}` on **resolved** (`grading_status == "resolved"`, `target is not None`), **first unaided attempt** (`attempt_number == 1`, `assistance == "independent"`, `hint_count == 0`). Unresolved/assisted/retry attempts remain in the input `EvidenceStream` as causal history but are never scored as prediction targets.
- **Disjoint Partitioning:** `data.py:load_partition` and `run_v22.py:265,288` enforce zero learner overlap across `train`, `validation`, `main_test`, `power_law_test`, and `long_gap_test`.
- **Reported Slices (Ordinary Practice vs. Delayed Assessments Kept Strictly Separate):**
  - `ordinary`: `delay == 0 & purpose == "ordinary"` (regular practice first attempts)
  - `followup`: `delay == 0 & purpose == "followup"` (fresh unaided follow-up after assisted practice)
  - `day_1`, `day_7`, `day_30`: frozen practice-prefix delayed assessments at 1, 7, and 30 days
  - `assessment` (pooled delayed): all `delay != 0` queries (`evaluation.py:25`, combining `day_1`, `day_7`, and `day_30`)
  - `delayed_mean` (`delayed_equal_horizon_log_loss`): unweighted mean of per-horizon Log Loss across `day_1`, `day_7`, `day_30` (`run_v22.py:144-146`).

| Cohort / Split | Learners | Ordinary | Follow-up | 1-Day | 7-Day | 30-Day | Total Eligible | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `main/train` | 1,600 | — | — | — | — | — | 219,284 (`192,779` practice + `26,505` delayed) | Model fitting |
| `main_validation` | 200 | 17,687 | 6,399 | 1,116 | 1,113 | 1,098 | 27,413 (`24,086` practice + `3,327` delayed) | Hyperparam & checkpoint selection |
| `main_test` | 200 | 17,664 | 6,258 | 1,110 | 1,106 | 1,097 | 27,235 (`23,922` practice + `3,313` delayed) | In-distribution held-out test |
| `power_law_test` | 200 | 17,720 | 6,333 | 1,105 | 1,102 | 1,116 | 27,376 (`24,053` practice + `3,323` delayed) | Stress: power-law $(1+\Delta t/h)^{-1}$ decay |
| `long_gap_test` | 200 | 17,722 | 6,846 | 1,096 | 1,101 | 1,117 | 27,882 (`24,568` practice + `3,314` delayed) | Stress: extended inter-session gaps |

### 2.3 Validation Selection Rule & Behavioral Sanity Gate
- **Selection Objective (`evaluation.py:validation_losses(..., balanced=True)`):**
  $$\text{ValidationSelectionLogLoss} = 0.5 \cdot \text{PracticeLogLoss}_{\text{val}} + 0.5 \cdot \frac{\text{LogLoss}_{1\text{d,val}} + \text{LogLoss}_{7\text{d,val}} + \text{LogLoss}_{30\text{d,val}}}{3}$$
- **Hard Behavioral Gate (`memory/behavioral.py:evaluate_behavioral_suite`, 6 invariants):** A sequence architecture is eligible for selection only if **all seeds pass all 6/6 checks**:
  1. `zero_score_decreases_all_speeds` (`5s`, `6.426s`, `60s`, `180s`, `None`),
  2. `full_success_increases_all_speeds` (`5s`, `60s`, `180s`, `None`),
  3. `assistance_and_outcome_ordering` (`Unaided 100% > Assisted 3-hint 100% > Unaided 0%`),
  4. `difficulty_monotonicity_all_formats` (`Easy > Medium > Hard` across all 5 formats),
  5. `non_increasing_forgetting_after_pass_and_fail` (0–60d never increases; 2-pass retains $\ge 60\%$ of day-0 at day 30 and stays $>$ cold start through day 60),
  6. `user_conditionals_trace_behaves_sensibly` (recorded SQLite trace drops after attempt 1 and drops further after retry).
- **Checkpoint Choice (`run_v22.py:select_model`):** Lowest mean `ValidationSelectionLogLoss` among passing models; median-validation seed chosen as deployed checkpoint.

### 2.4 Metrics, Delayed Aggregate Scope & Codebase Verification Note
- **Primary:** Log Loss (`loss_per_query`, clipped `[1e-7, 1 - 1e-7]`). On delayed assessments, `Del. LL` in Table T1-A reports **`delayed_equal_horizon_log_loss`** (equal-weight mean across `day_1`, `day_7`, and `day_30`), with individual `followup`, `day_1`, `day_7`, and `day_30` log losses reported in Table T1-C.
- **Supporting (Delayed Aggregates Evaluated on Pooled `assessment` Slice):**
  - **ROC-AUC (`Ord. AUC`, `Del. AUC`):** Computed on `ordinary` and pooled `assessment` (`delay != 0`) slices, averaged across seeds (`Mean ± SD`). *(Codebase note: `evaluation.py:probability_report` currently omits `roc_auc`. For retrospective runs, ROC-AUC is computed directly from saved `*_predictions.csv.gz` files; for pre-registered runs, `roc_auc` will be added to `probability_report()` — see §9.)*
  - **Brier Score (`Ord. Brier`, `Del. Brier`):** $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$ on `ordinary` and pooled `assessment` (`delay != 0`), matching `EVALUATION_METRICS_EXPLAINED.md:115-127`.
  - **10-Bin ECE (`ece_10`), Bin Counts, & Reliability Diagram (`Ord. ECE-10`, `Del. ECE-10`):** 10 uniform bins $[0.0, 0.1), \dots, [0.9, 1.0]$ on `ordinary` and pooled `assessment` (`delay != 0`), always emitting all 10 bins (padding empty bins with `queries: 0`, `mean_prediction: null`, `observed_rate: null`; see §9 vs `evaluation.py:68-69`). Table T1-B populates bin values from **seed-averaged probabilities** across the 5 pre-registered seeds on `main_test` (with `selected_checkpoint` values additionally archived for retrospective runs).
  - **Accuracy (`accuracy_05`):** Secondary sanity metric at threshold `0.5` on `ordinary` and pooled `assessment`.
- **Diagnostic:** Simulator Oracle Signal %:
  $$\text{SignalFraction}(m) = \frac{\text{LogLoss}_{\text{constant}} - \text{LogLoss}_m}{\text{LogLoss}_{\text{constant}} - \text{LogLoss}_{\text{oracle}}}$$
- **Paired Contrasts:** Learner-clustered paired bootstrap 95% CIs on `ordinary`, `followup`, `day_1`, `day_7`, `day_30` for the 8 contrasts in `run_v22.py:177-186` (including `monotonic_gru` vs `time_aware_transformer`, `time_decay_gru`, `kt_decay_gru`, `monotonic_transformer`, and best tabular model), tabulated in Table T1-D.

### 2.5 Retrospective vs. Pre-Registered Seeding Plan (Tier 1)
- **Retrospective Runs (`3/model/runs/v2.2-mps-main/` & `v2.2-mps-transformer-comparison/`):**
  - Seeds: **`20261120`, `20261121`, `20261122`** (3 seeds per neural model; seed `20261120` for deterministic tabular models), `35` epochs, `batch_size=64`, `OneCycleLR(max_lr=0.003)`, `mps:0`. Selected checkpoint: `monotonic_gru/seed_20261122/model.pt`.
- **Pre-Registered Expanded-Seed Re-Run:**
  - **Seeds:** **5 fresh deterministic seeds (`20261201`, `20261202`, `20261203`, `20261204`, `20261205`)** for all 6 sequence models (`time_input_gru`, `time_decay_gru`, `time_aware_transformer`, `kt_decay_gru`, `monotonic_gru`, `monotonic_transformer`) trained from scratch without `--reuse-from` on `v2.2` `main/train`, selected on `main_validation`, and evaluated once on `main_test`, `power_law_test`, and `long_gap_test`. Because `HistGradientBoostingClassifier` (`3/model/memory/tabular.py:19-22`) subsamples `200,000` rows out of `219,284` via `random_state=seed`, `history_boosting` is also fit across all 5 seeds (`20261201..20261205`) in pre-registered runs (while `task_logistic` and `history_logistic` remain single-seed deterministic).
  - Pre-execution harness updates (`run_v22.py` `--seeds`, `min/max` in `summarize()`, `roc_auc` + 10-bin padded `calibration` in `evaluation.py`, and `draws=2000` in `paired_intervals()`) are specified in §9. Report seed-level mean, SD (`ddof=1`), min–max range, full `roc_auc`, 10-bin calibration tables + reliability diagrams, and `B=2,000` learner-bootstrap CIs.

---

## 3. Tier 2: Retrieval Evaluation (`3/adaptive_web/rag.py`)

### 3.1 Exact Corpus Inventory (Verified in `3/adaptive_web/rag.py` & `knowledge/`)
1. **`curriculum` (`24` static chunks):** 6 Markdown guides (`3/adaptive_web/knowledge/<skill>.md`, 1-to-1 with `PPO_SKILLS_ORDER`), each parsed into 4 `#`/`##`/`###` sections (`max_words=260`, `overlap=35`). Dynamic `inst.*` instructor-ingested chunks are excluded from the benchmark corpus unless explicitly tested in an ingestion isolation sub-check.
2. **`v22_exemplar` (`2,160` items):** Full `3/data_Set/v2.2/dataset/task_catalog.json` (`6 skills × 3 difficulties × 5 formats × 24 items` = `1,440` `practice` + `720` `assessment`), indexed as `ex.<task_id>` with strict SQL/Python metadata pre-filtering on `skill`, `difficulty`, and `task_type`.
3. **`episodic` (dynamic per-learner):** Learner-scoped `"mistake"`, `"assisted_attempt"`, `"reveal"`, and `"doubt"` chunks (`ep.<learner>.<skill>.<hash>`), strictly isolated to `(learner, skill)` via SQL `WHERE` and `_passes_metadata_filter`.

### 3.2 Exact 4-Arm Matched Comparison & Scoring Symmetry
Verified against `3/adaptive_web/rag.py:58-61, 589-619, 697-765, 795, 862, 892-926`:
All 4 arms share identical metadata pre-filtering (`source_types`, `allowed_skills`, `difficulty`, `task_type`, `learner`, `exclude_task_ids`) and identical `top_k` context budgets, with the following **exact matched scoring, query-input, and fallback rules**:

1. **Matched Reranking & Pure-Retriever Diagnostic (`rag.py:892-926`):**
   - **Primary (Deployed-Matched Reranked Retrieval):** Because `rerank_boost` (`<= 0.075`) exceeds the 2-source `rrf` ceiling (`2 / 61 ≈ 0.0328`), all 4 arms apply the **identical** `rerank_boost` (`rag.py:898-921`) over an **equal-scale** rank base score: two-source arms (`Arm 2`, `Arm 4`) use `rrf = 1/(60+r1) + 1/(60+r2)` (max `2/61`), and single-source arms (`Arm 1`, `Arm 3`) use matched-scale `2.0 / (60 + rank)` (max `2/61`), preventing `rerank_boost` from having double relative weight in single-stage arms.
   - **Un-Reranked Pure-Retriever Diagnostic:** Alongside reranked metrics, report **un-reranked** `nDCG@k` and `Recall@k` for every arm (raw `BM25` rank, raw `BM25 + TF-IDF` RRF without `rerank_boost`, raw `Dense` cosine without `rerank_boost`, raw `BM25 + Dense` RRF without `rerank_boost`) to isolate pure lexical/dense matching from pedagogical metadata reranking.
2. **Exact `Dense` Indexed Text & Vector Scoring Parity (`rag.py:59-61, 795, 862`):**
   - `Dense` encodes the exact `KnowledgeChunk.searchable_text` string (`f"{skill} {difficulty} {task_type} {title} {section} {content}"`, matching `TF-IDF` in `rag.py:59-61, 770-772, 838-840`).
   - All vector-scored arms (`Arm 2`, `Arm 3`, `Arm 4`) apply identical dynamic `episodic` (`+0.015`) / custom `inst.*` (`+0.01`) similarity offsets (`rag.py:862`) and identical matching-skill `curriculum` retention (`score > 0.0 or (ch.source_type == "curriculum" and ch.skill == skill)` at `rag.py:795`).
3. **`Arm 1` (`BM25 alone`) `< top_k` Behavior:**
   - `Arm 1` uses strict FTS5 BM25 matching (`rag.py:697-765`) **without** `0.0`-score vector `curriculum` fallback, logging `bm25_underfilled_queries` (`len(results) < top_k`) as an operational diagnostic (and optionally reporting a matched `Arm 1 + curriculum fallback` sensitivity line if any query returns `< top_k` rows).
4. **Dev-Selected Query Input Policy for `Dense` vs. `BM25` (`rag.py:589-619`):**
   - Lexical `BM25` (`Arm 1`, `Arm 2`, `Arm 4`) and `TF-IDF` (`Arm 2`) always receive `rewrite_query()` (`_PYTHON_SYNONYMS` + `SKILL_META`).
   - Because keyword-stuffed `rewrite_query()` strings can distort pretrained bi-encoders, evaluate `Dense` on the **Dev split (`N_dev = 66`)** under **both** `(a) raw query + clean metadata context` and `(b) rewrite_query()`, select the better `Dense` query input rule strictly on Dev `nDCG@k`, and freeze that exact rule (for both `Arm 3` and the `Dense` branch of `Arm 4`) before touching the Held-Out Test split (`N_test = 132`).

| Arm | Retrieval Mechanism | Query Input & Scoring Rule | Hyperparameter / Selection Rule |
|---|---|---|---|
| **Arm 1: `BM25 alone`** | SQLite `FTS5` `bm25(rag_chunks_fts, ..., 2.0, 1.5, 1.0)` (`porter unicode61`), strict FTS5 match (no `0.0`-score fallback) | `rewrite_query()`; `2/(60+rank) + rerank_boost` (plus raw un-reranked diagnostic) | Fixed (`rag.py:697-765`) |
| **Arm 2: `Deployed Hybrid (BM25 + TF-IDF)`** | Exact `HybridRAGEngine.retrieve()` (`rag.py:664-991`): `FTS5` BM25 + `TfidfVectorizer(ngram_range=(1,2), max_features=8192, sublinear_tf=True)` cosine + `RRF (k=60)` + `episodic +0.015` + `rag.py:795` fallback | `rewrite_query()`; `rrf + rerank_boost` (plus un-reranked `rrf` diagnostic) | Fixed as deployed |
| **Arm 3: `Dense`** | Frozen pre-trained dense bi-encoder (`BAAI/bge-small-en-v1.5` vs `all-MiniLM-L6-v2`) over `KnowledgeChunk.searchable_text` + `episodic +0.015` + `rag.py:795` fallback | Dev-selected query input (`raw+meta` vs `rewrite_query`); `2/(60+rank) + rerank_boost` (plus un-reranked cosine diagnostic) | **Encoder & query-input selected strictly on Dev split**, evaluated **once untouched** on Test split |
| **Arm 4: `BM25 + Dense`** | `FTS5` BM25 + Dev-selected `Dense` bi-encoder fused via `RRF (k=60)` | `BM25` on `rewrite_query()`, `Dense` on Dev-selected input; `rrf + rerank_boost` (plus un-reranked `rrf` diagnostic) | Fixed from Dev selection, evaluated **once untouched** on Test split |

### 3.3 Exact Context Budgets & Separate Scoring
Verified in `3/adaptive_web/llm_pedagogy.py`:
1. **Tutor Chat (`answer_learner_doubt`, `llm_pedagogy.py:1414`):** `top_k = 4` over `source_types=("curriculum", "episodic")` with `include_prerequisites=(mastery_tier in ("unassessed", "lacking"))`.
2. **Question Generation (`generate_question`, `llm_pedagogy.py:842, 854`):** Scored **separately** for each retrieval call:
   - **Exemplar retrieval (`retrieve_exemplars`, `top_k = 3`):** `source_types=("v22_exemplar",)` strictly pre-filtered on `(skill, difficulty, task_format)` with `exclude_task_ids` and `recent_mistakes` varied across queries (`llm_pedagogy.py:840`).
   - **Curriculum/Episodic retrieval (`purpose="question_curriculum"`, `top_k = 2`):** `source_types=("curriculum", "episodic")`. Every Dev and Test q-gen query uses a seeded learner profile with **non-empty `episodic` chunks (`>= 3–5` target-skill rows)** alongside the `4` static curriculum chunks so the candidate pool (`>= 7–9` chunks) avoids trivial `k=2` saturation.

### 3.4 Dev vs. Held-Out Test Query Splits & Relevance Annotation
- **Query Benchmark Construction:** Stratified across all 6 skills, 4 `mastery_tier` states (`unassessed`, `lacking`, `fragile`, `mastered`), and query types (conceptual question, error/trace debugging, synonym/vague phrasing, episodic history recall, and `difficulty × format` question-gen requests). Every Dev (`30`) and Test (`60`) q-gen query—and episodic chat queries—uses realistic seeded learner profiles containing **non-empty `episodic` mistake/reveal/doubt records** (`>= 3–5` per target skill) plus **distinct `recent_mistakes` strings and `exclude_task_ids`** (plus distractor learners with overlapping skills to verify zero cross-learner leakage).
- **Disjoint Splits:**
  - **Dev Split (`N_dev = 66`):** `36` chat queries (`6 skills × 6`) + `30` question-gen queries (`6 skills × 5 formats`, balanced difficulties). Used solely to select the `Dense` encoder and the `Dense` query-input policy (`raw+meta` vs `rewrite_query()`).
  - **Held-Out Test Split (`N_test = 132`):** `72` chat queries (`6 skills × 4 tiers × 3 styles`) + `60` question-gen queries (`6 skills × 5 formats × 2 difficulties`). Evaluated **once untouched**.
- **Human Relevance Labeling:** 2 blinded Python-competent annotators independently grade candidate `(query, chunk)` pairs pooled across all 4 arms on a 3-point scale `{0: irrelevant, 1: partially relevant / helpful prerequisite, 2: directly relevant}` with third-party/consensus adjudication for disagreements; report inter-annotator weighted $\kappa$.

### 3.5 Metrics & OOS Abstention Verification
- **Primary:** Reranked `nDCG@k` and `Recall@k` (using binary relevance $\ge 1$, plus graded $\ge 2$ sensitivity check) at exact budgets (`k=4` chat; `k=3` exemplars and `k=2` curriculum/episodic scored separately), alongside **un-reranked** `nDCG@k` and `Recall@k` pure-retriever diagnostics, all with query-clustered paired bootstrap 95% CIs (`B=2,000`).
- **Operational & Safety:**
  - `p95` retrieval latency (ms) per call,
  - Exemplar metadata match rate (`skill == req_skill & difficulty == req_diff & task_type == req_fmt`),
  - `bm25_underfilled_queries` (`len(results) < top_k` diagnostic for `Arm 1`),
  - Cross-learner episodic retrieval violations (**must be `0`** across multi-learner probe queries where Learner A queries while Learner B has matching episodic mistakes),
  - **Fixed-LLM Downstream Check (Exact Tier 3a/3b Blinded Protocol):** Using the pinned live LLM config (§1.3) with `res.get("source") == "llm"` and `llm_client.last_call_info()` verified, compare `Deployed Hybrid (BM25 + TF-IDF)` vs `BM25 + Dense` on downstream Tutor Responses (`72` test chat queries) and Question Generation (`60` test q-gen requests), scored via the **exact Tier 3a (`python_correct`, `claim_faithfulness`, 3 rubric dims) and Tier 3b (`usable_rate`, `first_pass_valid`, `post_repair_valid`, `bank_fallback`) 2-reviewer blinded adjudication protocol** (arm identities hidden and randomized; results tabulated in Table T-A1).
- **OOS Abstention Status (Verified Against Codebase):**
  - **Finding:** Out-of-scope (OOS) abstention is **NOT currently implemented** in `3/adaptive_web/rag.py` or `llm_pedagogy.py`: `rag.py:795` unconditionally includes matching-skill `curriculum` chunks even when cosine similarity is `0.0` (`score > 0.0 or (ch.source_type == "curriculum" and ch.skill == skill)`), `retrieve()` returns `top_k` results with no minimum similarity threshold or abstention flag, and `answer_learner_doubt()` accepts only `skill in PPO_SKILLS_ORDER` with `guardrail_status in {"grounded_open", "grounded_protected", "socratic_redirect"}`.
  - **Plan Rule:** Under the current codebase, OOS abstention is marked **Not Supported / Deferred** and is NOT fabricated. If an explicit OOS similarity/classifier gate is added prior to execution as a pre-registered extension, it must be evaluated separately on an explicit OOS query split (reporting OOS false-accept rate and in-scope false-abstention rate).

---

## 4. Tier 3: LLM Pedagogical Realization, EduGuard & Grading

### 4.1 Tier 3a: Tutor Responses & EduGuard (`chat_doubt`, `answer_learner_doubt`, `_enforce_eduguard`)
- **Verified Codebase State & Guardrail Labels (`3/adaptive_web/llm_pedagogy.py:110-134, 1300-1357`, `app.py:1186-1246`):**
  - `mastery_tier`: `"unassessed"` (`not assessed`), `"lacking"` (`p_med < 0.70 or observed_pass_rate < 0.70`), `"fragile"` (`fragility > 0.15`), `"mastered"` (otherwise).
  - `guardrail_status`: `"grounded_open"`, `"grounded_protected"`, `"socratic_redirect"`, (plus `"disabled"` in Ablation 2 `EduGuard OFF`).
- **Isolated Per-Prompt Execution Harness (Zero Contamination):**
  - Execute every prompt through **`AdaptiveLearningApp.chat_doubt()`** (`app.py:1186-1246`) so cross-skill `_is_pasted_active_question` detection (`app.py:1189-1196`) is exercised alongside `answer_learner_doubt()`.
  - To prevent sequential `chat_messages` history and `episodic` `"doubt"` rows (`app.py:1219-1246`) from contaminating later prompts, run each prompt against an **isolated per-prompt SQLite DB snapshot** cloned from the seeded learner profile + open interaction (zero cross-prompt `chat_messages` or `episodic` doubt leakage).
  - Verify **`res.get("source") == "llm"`** and **`llm_client.last_call_info()`** (`failover_used == False`) on every prompt; never score `"fallback"` template outputs (`llm_pedagogy.py:1496-1572`) as live-LLM responses (log `source` and `api_error` explicitly, retrying transient transport errors up to `max_retries = 2` and reporting any remaining `api_error_count` separately).
- **Stratified Evaluation Sets (Justified by `6 skills × 4 mastery tiers × 5 formats`):**
  1. **Pedagogical Adaptation & Quality Set (`N = 96` prompts):** `6 skills × 4 mastery tiers (unassessed, lacking, fragile, mastered) × 4 doubt archetypes` (conceptual rule question, code-trace request, misconception/bug diagnosis, edge-case/transfer query), with seeded learner histories matching each `mastery_tier`.
  2. **EduGuard Paired Safety Set (`N = 120` prompts):**
     - `60` **protected answer-fishing / leaked-question prompts** (`6 skills × 5 formats × 2 attack styles`: explicit answer-fishing + verbatim/cross-skill pasted unpassed open question with an open unpassed interaction).
     - `60` **legitimate conceptual/debugging doubts while an unpassed question is open** (`6 skills × 5 formats × 2 legitimate doubt styles`: general concept rule question, distinct toy example trace).
- **Metrics (Counts, Rates, 95% Wilson CIs, and Rubric Means with Prompt-Clustered Bootstrap CIs):**
  - **Python Correctness (`0/1`):** Factual correctness of Python semantics, execution traces, and any `code_example` (scored **separately** from faithfulness).
  - **Claim-Level Faithfulness (`0/1` and `claim_support_fraction`):** Whether all curriculum/history claims are grounded in retrieved `rag_context` / learner state without fabricated rules or hallucinated episodic history (scored **separately** from Python correctness; both binary faithfulness and per-prompt `claim_support_fraction = supported_claims / total_claims` logged in `tier3a_eval.jsonl`).
  - **Answer Leakage Rate (on `60` protected prompts):** Count, rate, 95% Wilson CI — measuring **both** automated EduGuard status and human-verified semantic answer/blank/option leakage.
  - **Over-Refusal Rate (on `60` legitimate doubts):** Count, rate, 95% Wilson CI — triggering `socratic_redirect` or withholding helpful conceptual explanation on a legitimate non-fishing doubt.
  - **3 Pedagogical Rubric Dimensions (`0–2` ordinal scale per dimension):**
    1. *Addresses learner's specific difficulty/doubt*,
    2. *Provides an actionable next step*,
    3. *Adapts depth, scaffolding, and tone to `mastery_tier`* (`unassessed`, `lacking`, `fragile`, `mastered`).
- **Blinded Human Review & Agreement-Gated LLM Judge Protocol:**
  - **2 blinded Python-competent human reviewers** independently score the entire shared set (`96` adaptation + `120` safety prompts, arms/tiers blinded and randomized), followed by **structured consensus or third-party adjudication** of every disagreement (recording `adjudicated_by` in `tier3a_eval.jsonl`).
  - Measure inter-rater reliability on the shared human sample via **Cohen's $\kappa$** (binary correctness, leakage, over-refusal) and **weighted Cohen's $\kappa$** (ordinal rubric dimensions).
  - **External LLM Judge Gate:** An external LLM judge (using a separate model family/checkpoint from the generator where configured, blinded to arm) may extend coverage to larger prompt sets **ONLY IF**: (a) human–human agreement $\kappa \ge 0.70$ across scored dimensions, AND (b) judge-vs-adjudicated-human agreement $\kappa \ge 0.75$ with leakage/over-refusal discrepancy $\le 2.0$ percentage points on the shared calibration set. If the gate fails, report human-adjudicated scores only.

### 4.2 Tier 3b: Generated Questions (`generate_question`, `validate_task_schema`)
- **Verified Formats & AST Verification (`3/adaptive_web/catalogue_bridge.py`, `3/dashboard/grading.py`):**
  - All 5 formats: `single_choice`, `multiple_select`, `output_prediction`, `fill_in_the_blank`, `code_completion`.
  - `code_completion` is verified by `validate_task_schema(..., verify_code_tests=True)` using the bounded AST `Interpreter` (`def solve(x):`, max 8,000 chars, 1,200 AST nodes, 10,000 steps, 2,000 sequence items; allowed built-ins: `range, list, len, sum, min, max, abs`).
- **Stratified Sample Size & Per-Cell Request Variation:** **`N = 180` generation requests** = `6 skills × 5 formats × 3 difficulties × 2 requests/cell` (`30`/skill, `36`/format, `60`/difficulty), using the pinned live LLM (§1.3). To prevent near-duplicate prompts at `temperature = 0.2` (`llm_pedagogy.py:910`), the `2` requests inside each `(skill, difficulty, format)` cell vary `recent_mistakes` (distinct misconception focus from seeded `episodic` rows) and `exclude_task_ids` (distinct held-out exclusion lists), with `max_text_similarity <= 0.85` enforced against retrieved `v2.2` exemplars plus an across-all-180 pairwise similarity diagnostic.
- **Live-LLM & API-Failure Accounting (`llm_pedagogy.py:956-1014`):** Inspect `llm_client.last_call_info()` and output metadata (`task_id` not starting with `llm.clone.` or `synth.`, `difficulty_source != "llm_pedagogy_fallback"`, `version != "synth-v1"`) to record `source` (`"llm"|"v22_clone"|"synth"`) and `api_error` (`str|null`) in `tier3b_eval.jsonl`. Distinguish schema-validation failures from `api_error` (`LLMClientError` / empty JSON) via `rejection_reason` while **retaining both in the `N = 180` denominator** for `First-Pass Valid`, `Post-Repair Valid`, and `Usable Rate`, and reporting `api_error_rate` separately.
- **Metrics (All with Counts, Rates, 95% Wilson CIs, Reported Aggregate + Marginals by Skill, Format, and Difficulty):**
  1. **First-pass validity rate:** Fresh LLM item (`source == "llm"`) passes `_validate_with_guardrails(..., verify_code_tests=True)` on the initial JSON call (`/ 180`).
  2. **Post-repair validity rate (Cumulative):** Fresh LLM item passes `_validate_with_guardrails(..., verify_code_tests=True)` either on first pass or after 1-shot `question_repair`, i.e., `((first_pass_valid + repaired_valid) / 180)` (with conditional repair yield `repaired_valid / repair_attempted` reported separately as a supporting metric).
  3. **Bank fallback rate (counted separately):** Requests falling back to `sample_v22_task` cloning (`llm.clone.*`) or `synth.*` template (`version == "synth-v1"`). Fallback items are **never** counted as fresh valid/usable generations.
  4. **API error rate (diagnostic):** Fraction of `N = 180` requests experiencing `LLMClientError` / empty JSON before fallback.
  5. **Usable generated-question rate (Primary Tier 3b Metric):**
     $$\text{Usable Rate} = \frac{\text{Fresh, human-verified correct \& solvable items matching requested skill, format, and difficulty}}{\text{All } N = 180 \text{ generation requests}}$$
     Verified via 2 blinded Python-competent reviewers auditing answer keys/rubrics/solvability/difficulty alignment across all 5 formats + executable AST `Interpreter` runs (`code_completion`) and reference code execution (`output_prediction`, `fill_in_the_blank`).
  6. **Novelty diagnostic:** Distribution of `max_text_similarity` against retrieved `v2.2` exemplars (`<= 0.85` enforced) and cross-cell similarity across the 180 items.

### 4.3 Tier 3c: Self-Explanation Grader (`3/adaptive_web/grader.py`, `llm_pedagogy.py`)
- **Verified Codebase Behavior (`llm_pedagogy.py:1104-1153`, `grader.py:110-163, 200-313`, `app.py:423-454`):**
  - `PedagogyRealizer.evaluate_self_explanation()` always delegates authoritative `score` and `full_success` to `grade_self_explanation(prompt, learner_explanation, rubric=None)`.
  - `grade_self_explanation()` computes `score = round(len(hit) / len(key_concepts), 4)` on `[0, 1]`, sets **`full_success = (len(missing) == 0 and score >= 1.0)`** (exact fixed threshold: **`score >= 1.0`**), and computes `grader_confidence` separately on `[0.05, 0.99]` without multiplying `score`. When `not key_concepts` and `confidence < 0.35`, `grading_status = "unresolved"` (`score = None`, `full_success = None`).
  - **Non-Independent `llm_confidence` Disclosure (`llm_pedagogy.py:1126-1127`):** `evaluate_self_explanation()` passes `deterministic_score` (`det_grade["score"]`) and `deterministic_full_success` inside `context_payload` to the live LLM. Therefore, `llm_confidence` is a **confidence check conditioned on `det_score_0_1`**, NOT an independent LLM score. Additionally, `llm_pedagogy.py:1138` currently hardcodes `"grading_status": "resolved"` even if `det_grade["score"] is None` (see §9 pre-execution fix note).
- **Pinned Prompt Pool & Strictly Blinded Non-Circular Corpus (`N = 120` responses):**
  - **Prompt Pool (`app.py:423-454` `se_prompt_bank`, `rubric=None`):** Use all `18` production prompts (`3` per skill across the `6` canonical skills) with `rubric=None`, matching `evaluate_self_explanation()` (`6–7` responses per prompt; `20` responses per skill = `120` total).
  - **Non-Circular Response Sourcing:** The `120` responses (`~15%` empty/nonsense/code-only, `~25%` incorrect/misconception, `~30%` partial coverage, `~30%` complete causal explanations, including natural paraphrases and hedged/keyword-stuffed edge cases) MUST be authored/collected from **independent Python-competent contributors (or anonymized pilot logs) strictly blinded to `grader.py`'s `_CONCEPT_ALIASES`, `_HEDGING_PATTERNS`, `_CAUSAL_CONNECTORS`, and `_infer_concepts_from_prompt()` logic**—never generated with visibility into `grader.py`.
- **Blinded Human Reference:** 2 blinded Python-competent reviewers (blinded to `det_score_0_1`, `det_confidence`, and `llm_confidence`) score each response continuously on `[0.0, 1.0]` and assign a predefined 3-level ordered category `{0: Incorrect/Absent, 1: Partial, 2: Full Conceptual Mastery}`, followed by structured adjudication.
- **Metrics & `unresolved` / Fallback Handling:**
  1. **`resolved_coverage`:** `n_resolved / 120` (`det_score_0_1 is not None`), reported alongside `api_error_rate` for the live-LLM confidence check.
  2. **MAE** (and RMSE supporting) of `det_score_0_1` vs human-adjudicated `[0, 1]` score on resolved items (`det_score_0_1 is not None`, with prompt-clustered 95% bootstrap CI).
  3. **Full-success Precision, Recall, and F1** over all `N = 120` items at the fixed rubric threshold (`score >= 1.0` vs human `Full Conceptual Mastery`), treating any `unresolved` item (`score is None`) as `full_success = False` so unresolved items cannot be silently excluded from recall.
  4. **Confidence–Error Check:** Spearman $\rho$ (and Pearson $r$ + mean `|score - human_score|` stratified by confidence tercile) between `grader_confidence` and `|score - human_score|` on resolved items—evaluated separately for deterministic `det_confidence` (`N_resolved`) and **strictly live-LLM (`source == "llm"`)** `llm_confidence` (never mixing `"deterministic_double"` fallbacks into live `llm_confidence`, verified via `res.get("source") == "llm"` + `llm_client.last_call_info()`).
  5. **Quadratic Weighted Kappa (QWK):** Reported **only** on resolved items under the predefined ordered 3-category mapping (`0` if `score == 0`; `1` if `0 < score < 1.0`; `2` if `score >= 1.0`) against human-adjudicated 3-level categories.

---

## 5. Tier 4: Teaching Policy Evaluation (`delayed_retention_ppo` & Baselines)

### 5.1 Exact 5-Policy Comparison & Environment Contract
Verified in `3/ppo/env.py`, `3/ppo/agent.py`, and `3/ppo/train_ppo.py`:

| Policy ID | Class / Mode | Decision Logic & Reward Horizon |
|---|---|---|
| `delayed_retention_ppo` | `MaskedActorCritic` (`64`-unit SiLU MLP actor/critic) | PPO ($\gamma = 0.99, \lambda = 0.95, \epsilon = 0.20$), `36`-dim $s_t$, Gated Option B mask, step shaping (`p_30d`, `min_p30`, `floor_b`, unverified clearance, premature-advance penalty) + terminal `10.0 * delayed_mean` |
| `threshold_heuristic` | `ThresholdHeuristicPolicy` (**Primary Paired Baseline**) | Deterministic rule on `obs[1] = P(Med)` (`>=0.65` advance, `<0.42` explain, `>=0.52` harder, `<0.30` easier) & `obs[13] = awaiting_question` |
| `myopic_rl_bandit` | `MaskedActorCritic` | Bandit PPO ($\gamma = 0.0, \lambda = 0.0$) maximizing immediate turn score (`immediate_score` or `0.35 * max(0, next_p_med - prev_p_med)`) |
| `fixed_curriculum` | `FixedCurriculumPolicy` | Deterministic per-topic cycle: `explain_concept -> Easy -> Med -> Hard -> advance_next_topic` |
| `random_valid` | `RandomValidPolicy` | Uniform random selection over `valid_action_mask()` |

- **Matched Teaching Budget & Spacing:** `TOTAL_BUDGET = 32.0`, `SESSION_BUDGET = 8.0` (`1.5`-day inter-session gap in `main_test` and `power_law_test`; `4.0`-day inter-session gap in `long_gap_test`).
- **Frozen Memory Tracker:** `FrozenMemoryTracker` wraps the frozen `v2.2` `monotonic_gru` checkpoint (`3/model/runs/v2.2-mps-main/monotonic_gru/seed_20261122/model.pt`). Weights remain completely frozen during PPO training and evaluation.

### 5.2 Metrics, Per-Skill Floor, and Diagnostics
- **Primary:** `delayed_mean` (equal-weight mean simulated unaided pass rate across 1-, 7-, and 30-day assessments) and **paired percentage-point (pp) difference vs `threshold_heuristic`** at budget `32.0`, with **learner-level paired bootstrap 95% CI**.
- **Highlighted Horizon:** **Day-30 pass rate (`day_30`)** reported separately alongside `day_1`, `day_7`, and `delayed_expected_mean` (analytic probability diagnostic).
- **Generalization Stress Cohorts:** `power_law_test` (power-law forgetting $(1+\Delta t/h)^{-1}$) and `long_gap_test` (`4.0`-day inter-session breaks) evaluated zero-shot without retraining.
- **Per-Skill Day-30 Pass Rates (Pre-Registered Extension):** Report Day-30 unaided pass rates for **all 6 canonical skills** (`range_boundaries`, `loop_tracing`, `accumulator_initialization`, `conditional_evaluation`, `list_indexing`, `function_returns`) and explicitly label the **minimum (lowest-skill) Day-30 retention**. *(Codebase audit note: `3/ppo/env.py:evaluate_delayed_retention` currently averages across all 6 skills without returning per-skill keys; pre-registered runs will log per-skill Day-30 arrays directly from `evaluate_delayed_retention`.)*
- **Diagnostics:**
  - `latent_mean_floor` (durable permastore floor $b_t$) and `latent_mean_strength` ($k_t$),
  - **Matched Budget Curves:** Evaluate converged policies across budgets `{16.0, 24.0, 32.0, 40.0}` (with `SESSION_BUDGET = 8.0`). *(Codebase note: `3/ppo/env.py:78, 423, 815` currently hardcodes `TOTAL_BUDGET = 32.0`; pre-registered budget curves parameterize `AdaptiveTutorEnv(..., total_budget=...)` so `obs[21]` and `self.done` scale cleanly at each budget — see §9.)*
  - Full 8-action distribution shares (Table T4-E),
  - **5-Check PPO Behavioral Sanity Suite** (`train_ppo.py:219-250`, Table T4-E): `question_turns >= 40%`, `no_action_spam <= 32%`, `advance_next_topic in [4.5%, 25%]`, `floor_b > threshold_heuristic`, `delayed_expected_mean > all 4 baselines`.

### 5.3 Retrospective vs. Pre-Registered Seeding Plan (Tier 4)
- **Retrospective Run (`3/ppo/runs/v2.2-ppo-main/`):**
  - Seeds: **`20261120`, `20261121`, `20261122`** (controlling PyTorch init, rollout sampling, minibatch shuffling, and `TRAIN_SEED_START = 10_000..19_999` draws; `100` updates × `24` episodes/update; `50` val learners `20_000..20_049`; `200` test learners `30_000..30_199`; deterministic evaluation). Memory checkpoint remained frozen (`monotonic_gru/seed_20261122/model.pt`).
- **Pre-Registered Expanded-Seed Re-Run:**
  - **PPO Training Seeds:** **10 fresh deterministic seeds (`20261201..20261210`)** for `delayed_retention_ppo` and `myopic_rl_bandit`, keeping the frozen `monotonic_gru` checkpoint fixed (with an optional sensitivity check across 3 memory seeds if desired, logged separately).
  - **Fresh Disjoint Held-Out Test Cohort:** `400` fresh test learners per cohort (`TEST_SEED_START_PREREG = 40_000..40_399`, strictly disjoint from train `10_000..19_999`, val `20_000..20_049`, and retrospective test `30_000..30_199`) across `main_test`, `power_law_test`, and `long_gap_test`.
  - Report seed-to-seed mean ± SD, min–max range, `B=2,000` learner-clustered bootstrap 95% CIs, all 6 per-skill Day-30 pass rates + minimum skill retention, and budget curves.

---

## 6. Tier 5: Optional Human Pilot Protocol Outline

- **Strict Separation Rule:** Tier 5 is an **optional** real-learner pilot outline; it is never required for simulation claims and must be reported in a separate section with zero mixing into Tier 1–4 tables.
- **Ethics & Prerequisites:** Requires institutional ethics / IRB review (or formal exemption), informed consent, non-coercive recruitment, and anonymized IDs prior to any enrollment.
- **Outline (`N = 20–40` novice/intermediate Python learners):**
  1. **Unaided Pretest:** Balanced 12-item unaided diagnostic (`2` items per skill across the 6 canonical skills, no hints).
  2. **Adaptive Study Session:** Bounded interaction in `3/adaptive_web/` (pinned `delayed_retention_ppo` + frozen `monotonic_gru` + pinned live LLM + Hybrid RAG + EduGuard; capped at `TOTAL_BUDGET = 32.0` / ~45 minutes).
  3. **Delayed Unaided Posttest:** Parallel-form 12-item unaided assessment administered after a fixed delay (7 days primary, with optional 14- or 30-day follow-up).
- **Metrics:** Unaided pretest score, delayed posttest score, paired within-learner change (`post - pre` with learner-bootstrap 95% CI), and **follow-up completion / attrition rate** (`completed / enrolled`, 95% Wilson CI).

---

## 7. Prioritized Ablation Studies

### 7.1 Ablation 1: Deployed Hybrid (`BM25 + TF-IDF`) vs. `BM25 + Dense` (+ Fixed-LLM Downstream Check)
- **Design:** Paired comparison on the exact Tier 2 Held-Out Test split (`72` chat queries at `k=4`; `60` question-gen requests scored separately at `k=3` exemplars and `k=2` curriculum/episodic with non-empty seeded `episodic` chunks) using the `Dense` encoder and query-input rule selected on the Dev split (§3.2).
- **Metrics:** Reranked and un-reranked `nDCG@k`, `Recall@k` (query-clustered paired bootstrap 95% CIs for $\Delta$), `p95` latency (ms), cross-learner episodic violations (`0`), and **fixed-LLM downstream check** (`source == "llm"` + `llm_client.last_call_info()` verified; scored via the **exact Tier 3a & Tier 3b 2-reviewer blinded adjudication protocol** for `python_correct`, `claim_faithfulness`, 3 rubric dims, `usable_rate`, `first_pass_valid`, `post_repair_valid`, and `bank_fallback`; Table T-A1).

### 7.2 Ablation 2: EduGuard ON vs. OFF on Paired Protected & Legitimate Prompts
- **Design:** Paired evaluation on the `N = 120` Tier 3a safety suite (`60` protected answer-fishing / pasted-question prompts + `60` legitimate conceptual/debugging doubts while an unpassed question is open) executed via **`AdaptiveLearningApp.chat_doubt()`** on **isolated per-prompt DB snapshots** (§4.1) under the pinned live LLM (§1.3, verifying `res.get("source") == "llm"` and `llm_client.last_call_info()`):
  - **Arm A (`EduGuard ON`):** Full prompt policy (`eduguard_rule`) + cross-skill `_is_pasted_active_question()` escalation + post-generation `_enforce_eduguard()`.
  - **Arm B (`EduGuard OFF`):** Passes the **exact same `active_question`** (keeping `active_preview` in `rag.retrieve()` at `llm_pedagogy.py:1399, 1413` **100% identical** to `EduGuard ON`, never passing `active_question=None`), identical `rag_chunks`, and identical `mastery_tier` adaptation, while disabling **ONLY**: (1) `_is_pasted_active_question` refusal override (logging any match cleanly without blocking), (2) the `eduguard_rule` prompt clause (using the neutral non-blocking string `"No unpassed exercise is blocking explanation; answer clearly using concrete examples."`), and (3) `_enforce_eduguard()` post-check (`guardrail_status = "disabled"`).
- **Metrics:** Exact counts, rates, and 95% Wilson CIs (plus prompt-paired bootstrap/McNemar diff CI) for **answer leakage on protected prompts** (`60`) and **over-refusal on legitimate doubts** (`60`), plus Python correctness, claim faithfulness, and 3 rubric dims on legitimate doubts (logged in `tier3a_eval.jsonl` with `source` and `api_error`).

### 7.3 Ablation 3: Clean Retrained PPO State Ablation (`ppo_no_longterm_state`) with Complete 36-Dim Obs, Mask & Routing Audit

#### 7.3.1 Complete 36-Dim Audit of `s_t` (`3/ppo/env.py:403-428` & `ppo_engine.py:117-150`)
Dropping a single feature (e.g., only `obs[5] = fragility`, only `obs[4] = p_30d`, or only `obs[6] = floor_b`) is **methodologically invalid** because the dropped feature can be trivially reconstructed from retained coordinates in `s_t`:
- `fragility` (`obs[5]`) $= \max(0, \text{obs}[1] - \text{obs}[4])$;
- `p_30d` (`obs[4]`) is reconstructible from `obs[1] - obs[5]`, from `dot(obs[24:30], obs[30:36])`, and bounded by `obs[22:24]`;
- `floor_b` (`obs[6]`) correlates strongly with `p_30d` (`obs[4]`) and `p30_all` (`obs[24:30]`), and prerequisite `floor_b` is explicitly exposed in `obs[20]`.

**Verified Clock Semantics (`3/ppo/env.py:172-181, 802`):** `obs[0..2]` (`p_easy`, `p_med`, `p_hard`) and `obs[7]` (`active_k`) are **instantaneous (`query_gap = 0`) evaluations at current `self.clock`** (i.e., `elapsed_days` since the last event on that skill is already applied via `_at_time(state, elapsed_days)` and `ref_gaps_tensor + elapsed_days`, refreshed at session boundaries at `env.py:802`). They do NOT expose future 7d/30d horizons or `floor_b`.

Therefore, **Ablation 3 removes ALL 13 long-term retention & consolidation dimensions** (`p_7d`, `p_30d`, `fragility`, `floor_b`, `pre_floor`, `min_p30_all`, `mean_p30_all`, and `p30_all[0..5]`), slicing `obs` down to **23 short-term / turn-context / gating / budget / topic dimensions (`obs[0..2, 7..19, 21, 30..35]`)**:

| Index | Feature Key | Description | Ablation 3 Status | Reconstruction Check |
|---:|---|---|---|---|
| `0` | `p_easy` | `P(Easy, query_gap=0)` at current `self.clock` | **Retain** | Instantaneous (`query_gap=0`) prediction only |
| `1` | `p_med` | `P(Medium, query_gap=0)` at current `self.clock` | **Retain** | Instantaneous (`query_gap=0`) prediction only |
| `2` | `p_hard` | `P(Hard, query_gap=0)` at current `self.clock` | **Retain** | Instantaneous (`query_gap=0`) prediction only |
| `3` | `p_7d` | `P(Medium, query_gap=7d)` | **DROP (Slice Out)** | Removed; cannot be computed without $h_t, b_t$ |
| `4` | `p_30d` | `P(Medium, query_gap=30d)` | **DROP (Slice Out)** | Removed; cannot be computed without $h_t, b_t$ |
| `5` | `fragility` | `max(0, P_med - P_30d)` | **DROP (Slice Out)** | Removed along with `p_30d` |
| `6` | `floor_b` | Permastore floor $b_t$ | **DROP (Slice Out)** | Removed; `active_k` (`obs[7]`) cannot separate $b_t$ from transient $k_t - b_t$ |
| `7` | `active_k` | Active skill channel $k_t$ (`query_gap=0` at `self.clock`) | **Retain** | Scalar mean at current `clock`; does not expose $b_t$, $h_t$, $n_+$, or $n_-$ |
| `8` | `unverified_assisted` | Assisted pass awaiting unaided check | **Retain** | Turn-level verification flag |
| `9` | `last_score` | Last resolved score | **Retain** | Single-turn outcome |
| `10` | `last_hints_norm` | Last hints used / 3 | **Retain** | Single-turn assistance |
| `11` | `last_leakage` | Last hint leakage $l_t$ | **Retain** | Single-turn leakage |
| `12` | `attempts_on_topic_norm` | Topic attempts / 10 | **Retain** | Scalar count only |
| `13–18` | `awaiting_question`, `primed_*`, `diff_*` | Gated Option B & difficulty one-hot | **Retain** | Environment control state |
| `19` | `prereq_min_p_med` | Min prerequisite `P(Med, query_gap=0)` at `self.clock` | **Retain** | Short-term prerequisite check only |
| `20` | `prereq_min_floor_b` | Min prerequisite `floor_b` | **DROP (Slice Out)** | Removed so prerequisite permastore floor is hidden |
| `21` | `budget_remaining_ratio` | Remaining teaching budget ratio | **Retain** | Budget tracking |
| `22` | `min_p30_all` | Min `P(30d)` across 6 skills | **DROP (Slice Out)** | Removed (cross-skill 30d leak) |
| `23` | `mean_p30_all` | Mean `P(30d)` across 6 skills | **DROP (Slice Out)** | Removed (cross-skill 30d leak) |
| `24–29` | `p30_all[0..5]` | Per-skill `P(30d)` across 6 skills | **DROP (Slice Out)** | Removed (would reconstruct `obs[4]`, `obs[22]`, `obs[23]`) |
| `30–35` | `topic_one_hot[0..5]` | Active skill one-hot | **Retain** | Curriculum location |

#### 7.3.2 Algebraic Non-Reconstructibility, Mask/Routing De-Leaking & Clean Retraining Protocol
- **Proof of Non-Reconstructibility:** In `monotonic_gru` (`3/model/memory/recurrent.py:233-286`), $P(30\text{d})$ and `floor_b` depend on the hidden permastore channels $\mathbf{b}_t \in (0,1)^6$, retrieval counters $(n_+, n_-)$, half-lives $\mathbf{h}_t \in [5, 365]$, and future `query_gap = 30d`. None of $\mathbf{b}_t$, $n_+$, $n_-$, $\mathbf{h}_t$, $\mathbf{c}_t$, or future query horizons appear in the retained 23 dimensions (`obs[0..2, 7..19, 21, 30..35]`), and `obs[0..2]` (`P(Easy, Med, Hard)` at `query_gap=0` from current `self.clock`) depend strictly on instantaneous $\mathbf{k}_t$, which is an unidentifiable sum of transient fluency $(\mathbf{k}_t - \tilde{\mathbf{b}}_t)$ and durable floor $\tilde{\mathbf{b}}_t$. Thus neither `p_30d`, `fragility`, nor `floor_b` can be reconstructed from the retained 23 inputs.
- **Mask & Target-Routing De-Leaking (`env.py:346-356, 384-385, 772, 789`, `agent.py:40-44`):**
  - In the unmodified `AdaptiveTutorEnv`, `valid_action_mask()` sets `mask[6] = True` via `_weak_prereqs_or_decayed(skill)` using `p_30d < 0.56 or floor_b < 0.40` (prerequisites) and `p_30d < 0.52 or floor_b < 0.36` (visited topics), and `step()` selects Action `6` (`env.py:772`) and Action `7` (`env.py:789`) targets via `min(..., key=lambda s: self.metrics_cache[s]["p_30d"])`. Because `MaskedActorCritic.forward(obs, mask)` (`agent.py:40-44`) receives `mask` directly, leaving `mask[6]` unmodified would leak a 1-bit `p_30d`/`floor_b` threshold oracle into the ablated policy!
  - Therefore, in the `ppo_no_longterm_state` ablated environment:
    1. Replace `_weak_prereqs_or_decayed` in `valid_action_mask()` (`mask[6]`) with short-term `p_med`-only criteria (`p_med < 0.60` for prerequisites; `p_med < 0.55` for visited topics);
    2. Replace `Action 6` (`env.py:772`) and `Action 7` (`env.py:789`) target selection `min(..., key=p_30d)` with short-term `min(..., key=lambda s: self.metrics_cache[s]["p_med"])`;
    3. Keep the step-shaping and terminal delayed-retention reward (`env.py:804-825`) identical so the ablated policy still optimizes the same objective without long-term features, masks, or routing. *(See §9 Pre-Execution Harness Requirements.)*
- **Pinned Network Parameterization & Clean Retraining Protocol:**
  - Pin **`obs_dim = 23`** uniquely (slicing `obs[0..2, 7..19, 21, 30..35]`) in `MaskedActorCritic(obs_dim=23, act_dim=8, hidden_dim=64)` — never `"either 23 or 36"`, because orthogonal initialization (`agent.py:26, 33`) depends on `obs_dim`.
  - **Retrain from scratch** across the **exact same 10 pre-registered seeds (`20261201..20261210`)**, `100` updates × `24` episodes/update, and evaluate paired on the exact same `400` held-out test learners across `main_test`, `power_law_test`, and `long_gap_test`.

---

## 8. Report-Ready Experiment Documentation Protocol (R3)

When experiments are executed, every tier and ablation must write both **machine-readable JSON/JSONL logs** and **populated Markdown tables** matching the schemas below.

### 8.1 Machine-Readable Logging Schemas

1. **Tier 1 (`tier1_run.json` — extending `3/model/run_v22.py` schema):**
   - Top-level: `status`, `run_classification` (`"retrospective"` | `"pre_registered"`), `dataset_version`, `dataset_manifests`, `implementation_sha256`, `seeds`, `epochs`, `batch_size`, `recurrent_device`, `selected_model`, `selected_checkpoint`.
   - `models.<model_name>[seed_idx]`: `seed`, `best_epoch`, `validation_practice_log_loss`, `validation_delayed_log_loss`, `validation_selection_log_loss`, `behavioral` (`checks_passed`, `checks_total`, `all_passed`, 6 check booleans, `user_conditionals_replay`).
   - `evaluation.<cohort_split>.summary.<model_name>.<slice>` (with `slice` covering `ordinary`, `followup`, `assessment`, `day_1`, `day_7`, `day_30`, and `delayed_equal_horizon_log_loss`): `queries`, `learners`, `log_loss`, `roc_auc`, `brier`, `ece_10`, `accuracy_05` (each `{mean, sd, min, max}`), plus `calibration` (**always 10 padded bins** `{lower, upper, queries, mean_prediction, observed_rate}` with `queries: 0` and `null` rates for empty bins; computed on `ordinary` and pooled `assessment` slices).
   - `evaluation.<cohort_split>.paired_intervals.<slice>.<contrast>`: `{delta_log_loss, lower_95, upper_95, bootstrap_draws, bootstrap_seed}`.
   - `evaluation.<cohort_split>.constant_to_oracle_log_loss_fraction`.

2. **Tier 2 (`tier2_eval.jsonl` + `tier2_summary.json`):**
   - Per-query JSONL record: `{query_id, split ("dev"|"test"), task_context ("chat"|"qgen_exemplar"|"qgen_curriculum"), arm, query_input_mode ("rewrite_query"|"raw_meta"), skill, difficulty, task_type, mastery_tier, seeded_episodic_count, recent_mistakes, exclude_task_ids, top_k, retrieved_doc_ids, relevance_grades_adjudicated, ndcg_at_k, recall_at_k, unreranked_ndcg_at_k, unreranked_recall_at_k, bm25_underfilled (bool), metadata_match_rate, cross_learner_violation, latency_ms}`.
   - `tier2_summary.json`: `dense_selection_dev_metrics` (including encoder & query-input selection), `heldout_test_summary_by_arm` (both reranked primary and un-reranked diagnostic), `paired_bootstrap_95ci_by_query`, `fixed_llm_downstream_check`.

3. **Tier 3a (`tier3a_eval.jsonl` + `tier3a_summary.json`):**
   - Per-prompt JSONL record: `{prompt_id, suite ("adaptation"|"eduguard_protected"|"eduguard_legit"), arm ("eduguard_on"|"eduguard_off"), skill, mastery_tier, format, llm_provider, llm_model, temperature, max_tokens, source ("llm"|"fallback"), api_error (str|null), failover_used (bool), guardrail_status, citations_count, python_correct (0|1|null), claim_faithfulness (0|1|null), claim_support_fraction (float|null), human_answer_leak (0|1|null), human_over_refusal (0|1|null), rubric_difficulty_addressed (0|1|2|null), rubric_actionable_next (0|1|2|null), rubric_tier_adaptation (0|1|2|null), reviewer_agreement_raw, adjudicated_by}`.
   - `tier3a_summary.json`: `live_llm_n`, `api_error_count`, `api_error_rate`, `human_kappa_agreement`, `llm_judge_gate_passed` (bool + calibration $\kappa$), `counts_rates_wilson_95ci`, `rubric_means_bootstrap_95ci`.

4. **Tier 3b (`tier3b_eval.jsonl` + `tier3b_summary.json`):**
   - Per-request JSONL record: `{request_id, skill, format, difficulty, recent_mistakes, exclude_task_ids, llm_provider, llm_model, source ("llm"|"v22_clone"|"synth"), api_error (str|null), failover_used (bool), first_pass_valid (bool), repair_attempted (bool), post_repair_valid (bool), fell_back_to_bank (bool), fallback_kind ("none"|"v22_clone"|"synth"), ast_tests_passed (bool|null), max_exemplar_similarity (float), human_verified_usable (bool), rejection_reason}`.
   - `tier3b_summary.json`: aggregate + marginals by `skill`, `format`, `difficulty` with `count / N`, rate, 95% Wilson CI, `api_error_rate`, cumulative `post_repair_valid` (`(first_pass_valid + repaired_valid) / N`), and conditional `repair_yield` (`repaired_valid / repair_attempted`).

5. **Tier 3c (`tier3c_eval.jsonl` + `tier3c_summary.json`):**
   - Per-response JSONL record: `{item_id, prompt_id, skill, quality_band, human_score_0_1, human_category_0_1_2, det_score_0_1 (float|null), det_full_success (bool|null), det_confidence, llm_confidence (float|null), source ("llm"|"deterministic_double"), api_error (str|null), failover_used (bool), grading_status ("resolved"|"unresolved"), abs_error (float|null)}`.
   - `tier3c_summary.json`: `resolved_coverage`, `live_llm_confidence_n`, `api_error_rate`, `mae`, `rmse`, `full_success_precision`, `full_success_recall`, `full_success_f1`, `confidence_error_spearman_rho`, `confidence_tercile_mae`, `qwk_3class`.

6. **Tier 4 & Ablation 3 (`tier4_run.json` — extending `3/ppo/train_ppo.py` schema):**
   - Top-level: `run_classification` (`"retrospective"` | `"pre_registered"`), `seeds`, `frozen_memory_checkpoint`, `budget`, `test_learners`, `test_learner_seed_range`.
   - `evaluation.<cohort>.summary.<policy>`: `day_1`, `day_7`, `day_30`, `delayed_mean`, `day_1_expected`, `day_7_expected`, `day_30_expected`, `delayed_expected_mean`, `latent_mean_floor`, `latent_mean_strength`, `per_skill_day_30` (dict of all 6 skills), `min_skill_day_30` (`{skill, rate}`), `action_shares`, `budget_curve` (`{16.0, 24.0, 32.0, 40.0}`).
   - `evaluation.<cohort>.paired_intervals`, `behavioral_by_seed`.

7. **Tier 5 (`tier5_pilot_summary.json` — optional, isolated):**
   - `{irb_protocol_id, enrolled_n, completed_n, attrition_rate_wilson_ci, pretest_mean, delayed_posttest_mean, paired_change_mean_bootstrap_95ci, delay_days}`.

---

### 8.2 Ready-to-Populate Result & Diagnostic Table Templates

#### Table T1-A: Tier 1 Held-Out Memory Performance (Ordinary vs. Delayed Reported Separately)
*Populate separately for `main_test`, `power_law_test`, and `long_gap_test`; label `Retrospective (3 seeds)` vs `Pre-Registered (5 seeds)`. `Del. LL` = equal-weight 3-horizon mean (`day_1`, `day_7`, `day_30`); `Del. AUC`, `Del. Brier`, `Del. ECE-10` = pooled `assessment` (`delay != 0`), seed-averaged (`Mean ± SD`).*

| Model | Behav. Gate | Ord. LL (Mean±SD) | Ord. AUC | Ord. Brier | Ord. ECE-10 | Ord. Signal% | Del. LL (Mean±SD) | Del. AUC | Del. Brier | Del. ECE-10 | Del. Signal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant` | — | | | | | 0.0% | | | | | 0.0% |
| `task_logistic` | Tabular | | | | | | | | | | |
| `history_logistic` | Tabular | | | | | | | | | | |
| `history_boosting` | Tabular | | | | | | | | | | |
| `time_input_gru` | / 6 | | | | | | | | | | |
| `time_decay_gru` | / 6 | | | | | | | | | | |
| `time_aware_transformer` | / 6 | | | | | | | | | | |
| `kt_decay_gru` | / 6 | | | | | | | | | | |
| `monotonic_gru` | / 6 | | | | | | | | | | |
| `monotonic_transformer` | / 6 | | | | | | | | | | |
| `oracle` *(diagnostic)* | — | | | | | 100.0% | | | | | 100.0% |

#### Table T1-B: Tier 1 10-Bin Calibration Table & Reliability Diagram Data (`monotonic_gru`, `main_test`)
*All 10 uniform bins emitted (empty bins padded with `queries = 0`). Computed from seed-averaged probabilities across the 5 pre-registered seeds on `ordinary` and pooled `assessment` (`delay != 0`); retrospective `selected_checkpoint` archived separately.*

| Bin Range | Ordinary Queries ($n_b$) | Ord. Mean Pred | Ord. Observed | Delayed (`assessment`) Queries ($n_b$) | Del. Mean Pred | Del. Observed |
|---|---:|---:|---:|---:|---:|---:|
| `[0.0, 0.1)` | | | | | | |
| `[0.1, 0.2)` | | | | | | |
| `[0.2, 0.3)` | | | | | | |
| `[0.3, 0.4)` | | | | | | |
| `[0.4, 0.5)` | | | | | | |
| `[0.5, 0.6)` | | | | | | |
| `[0.6, 0.7)` | | | | | | |
| `[0.7, 0.8)` | | | | | | |
| `[0.8, 0.9)` | | | | | | |
| `[0.9, 1.0]` | | | | | | |

#### Table T1-C: Tier 1 `Follow-up` & Per-Horizon (`1-Day`, `7-Day`, `30-Day`) Log Loss
*Populate separately for `main_test`, `power_law_test`, and `long_gap_test` (`Mean ± SD` and `Min–Max`).*

| Model | `followup` LL | `day_1` LL | `day_7` LL | `day_30` LL | `delayed_mean` LL |
|---|---:|---:|---:|---:|---:|
| `constant` | | | | | |
| `task_logistic` | | | | | |
| `history_logistic` | | | | | |
| `history_boosting` | | | | | |
| `time_input_gru` | | | | | |
| `time_decay_gru` | | | | | |
| `time_aware_transformer` | | | | | |
| `kt_decay_gru` | | | | | |
| `monotonic_gru` | | | | | |
| `monotonic_transformer` | | | | | |
| `oracle` *(diagnostic)* | | | | | |

#### Table T1-D: Tier 1 Learner-Clustered Paired Bootstrap 95% CIs ($\Delta\text{LogLoss} = \text{Candidate} - \text{Reference}$, `B = 2,000`)
*Populate separately for `main_test`, `power_law_test`, and `long_gap_test` (negative $\Delta$ favors Candidate).*

| Contrast (`Candidate minus Reference`) | `ordinary` $\Delta$ (95% CI) | `followup` $\Delta$ (95% CI) | `day_1` $\Delta$ (95% CI) | `day_7` $\Delta$ (95% CI) | `day_30` $\Delta$ (95% CI) |
|---|---|---|---|---|---|
| `history_logistic` minus `task_logistic` | | | | | |
| `history_boosting` minus `history_logistic` | | | | | |
| `monotonic_gru` minus `kt_decay_gru` | | | | | |
| `monotonic_gru` minus `time_decay_gru` | | | | | |
| `monotonic_gru` minus `time_aware_transformer` | | | | | |
| `monotonic_transformer` minus `time_aware_transformer` | | | | | |
| `monotonic_transformer` minus `monotonic_gru` | | | | | |
| `monotonic_gru` minus `best_tabular` | | | | | |

#### Table T1-E: Tier 1 6-Check Behavioral Sanity Gate & Recorded Trace Replay
| Sequence Model | Checks Passed (All Seeds?) | Fast `0%` Wrong (`6.4s`) $\Delta$ | After 3 Hints + 2nd `0%` Wrong | Non-Increasing Forgetting (`0–60d`) | Gate Status |
|---|---:|---:|---:|---|---|
| `time_input_gru` | | | | | |
| `time_decay_gru` | | | | | |
| `time_aware_transformer` | | | | | |
| `kt_decay_gru` | | | | | |
| `monotonic_gru` | | | | | |
| `monotonic_transformer` | | | | | |

#### Table T2: Tier 2 Retrieval Comparison (Held-Out Test Split, Untouched)
*Report primary reranked metrics (`2/(60+rank)` / `rrf + rerank_boost`) and un-reranked pure-retriever diagnostics in parentheses `(raw)`.*

| Arm | Query Input (`BM25` / `Dense`) | Chat (`k=4`) nDCG@4 (Raw) | Chat (`k=4`) Recall@4 (Raw) | Q-Gen Exemplar (`k=3`) nDCG@3 / Recall@3 | Q-Gen Curric. (`k=2`) nDCG@2 / Recall@2 | Exemplar Meta Match% | Cross-Learner Violations | p95 Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `BM25 alone` | `rewrite` / — | | | | | | | |
| `Deployed BM25 + TF-IDF` | `rewrite` / `rewrite` | | | | | | | |
| `Dense` *(Dev-selected)* | — / Dev-sel. | | | | | | | |
| `BM25 + Dense` | `rewrite` / Dev-sel. | | | | | | | |

#### Table T-A1: Ablation 1 Fixed-LLM Downstream Check (`Deployed BM25 + TF-IDF` vs. `BM25 + Dense`)
*Scored via the exact Tier 3a (`72` test chat prompts) and Tier 3b (`60` test q-gen requests) 2-reviewer blinded adjudication protocol with `source == "llm"` verified.*

| Retrieval Arm | Chat `live_n` / `72` | Chat Python Correct (`k/N`, %, CI) | Chat Claim Faithfulness (`k/N`, %, CI) | Chat Mean Rubric (0–2) | Q-Gen First-Pass Valid (`k/60`, %) | Q-Gen Post-Repair Valid (`k/60`, %) | Q-Gen Bank Fallback (`k/60`, %) | Q-Gen Usable Rate (`k/60`, %, CI) |
|---|---:|---|---|---|---|---|---|---|
| `Deployed BM25 + TF-IDF` | | | | | | | | |
| `BM25 + Dense` *(Dev-sel.)* | | | | | | | | |
| **Paired Diff (`BM25+Dense - Deployed`)** | — | | | | | | | |

#### Table T3a: Tier 3a Tutor Quality & EduGuard Safety (Blinded Adjudicated Human Scores, Isolated `chat_doubt` Snapshots)
| Condition | `live_n` / `api_errors` | Python Correct (`k/N`, %, 95% CI) | Claim Faithfulness (`k/N`, %, 95% CI) | Protected Leakage (`k/60`, %, 95% CI) | Legit Over-Refusal (`k/60`, %, 95% CI) | Diff. Addressed (0–2) | Actionable Step (0–2) | Tier Adaptation (0–2) |
|---|---|---:|---|---|---|---|---:|---:|---:|
| `EduGuard ON` (Pinned LLM) | | | | | | | | |
| `EduGuard OFF` (Ablation 2) | | | | | | | | |

#### Table T3b: Tier 3b Generated Question Quality (Aggregate & Marginals, `N = 180`)
*`Post-Repair Valid` is cumulative `(first_pass_valid + repaired_valid) / N`; conditional `repair_yield = repaired_valid / repair_attempted` and `api_error_rate` reported alongside.*

| Slice | Requests ($N$) | `api_error` (`k/N`, %) | First-Pass Valid (`k/N`, %, CI) | Post-Repair Valid (`k/N`, %, CI) | Repair Yield (`rep/att`) | Bank Fallback (`k/N`, %, CI) | Usable Generated Rate (`k/N`, %, CI) |
|---|---:|---|---|---|---|---|---|
| **Overall Aggregate** | 180 | | | | | | |
| By Skill (`6` rows) | 30 each | | | | | | |
| By Format (`5` rows, AST on `code_completion`) | 36 each | | | | | | |
| By Difficulty (`3` rows) | 60 each | | | | | | |

#### Table T3c: Tier 3c Self-Explanation Grader (`N = 120`, `se_prompt_bank`, `rubric=None`, Threshold `score >= 1.0`)
*MAE and QWK computed on resolved items (`resolved_coverage = n_resolved / 120`); Full-Success P/R/F1 computed across all `N = 120` items (`unresolved` counted as `full_success = False`). Note: Live LLM Confidence is conditioned on `det_score_0_1` (`llm_pedagogy.py:1126`).*

| Grader Mode | Resolved Cov. (`n/120`) | MAE (`[0, 1]`, 95% CI) | Full-Success Precision | Full-Success Recall | Full-Success F1 | Conf.–Error Spearman $\rho$ | QWK (3-Cat.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic Rubric (`grade_self_explanation`) | | | | | | | |
| Live LLM Confidence Check (`source == "llm"`, conditioned on `det_score`) | — (`live_n` / `api_err`) | — *(score fixed)* | — | — | — | | — |

#### Table T4-A: Tier 4 Policy Benchmark at Matched Budget `32.0` (`main_test`, `power_law_test`, `long_gap_test`)
*Label clearly as `Retrospective (3 seeds, N=200)` or `Pre-Registered (10 seeds, N=400)`.*

| Cohort | Policy | 1-Day Pass | 7-Day Pass | **30-Day Pass** | **Delayed Mean (Obs)** | Paired Diff vs `threshold_heuristic` (95% CI) | Delayed Expected | Floor $b_t$ | Seed Range (Min–Max) |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `main_test` | `delayed_retention_ppo` | | | | | | | | |
| `main_test` | `threshold_heuristic` | | | | | `0.00 pp` (Ref) | | | — |
| `main_test` | `myopic_rl_bandit` | | | | | | | | |
| `main_test` | `fixed_curriculum` | | | | | | | | — |
| `main_test` | `random_valid` | | | | | | | | — |
| `power_law_test` | `delayed_retention_ppo` | | | | | | | | |
| `power_law_test` | `threshold_heuristic` | | | | | `0.00 pp` (Ref) | | | — |
| `power_law_test` | `myopic_rl_bandit` | | | | | | | | |
| `power_law_test` | `fixed_curriculum` | | | | | | | | — |
| `power_law_test` | `random_valid` | | | | | | | | — |
| `long_gap_test` | `delayed_retention_ppo` | | | | | | | | |
| `long_gap_test` | `threshold_heuristic` | | | | | `0.00 pp` (Ref) | | | — |
| `long_gap_test` | `myopic_rl_bandit` | | | | | | | | |
| `long_gap_test` | `fixed_curriculum` | | | | | | | | — |
| `long_gap_test` | `random_valid` | | | | | | | | — |

#### Table T4-B: Tier 4 All 6 Day-30 Per-Skill Pass Rates & Lowest-Skill Retention (`main_test`, `power_law_test`, `long_gap_test`)
*Populate one 5-policy block per cohort (`main_test`, `power_law_test`, `long_gap_test`).*

| Cohort | Policy | `range_boundaries` | `loop_tracing` | `accumulator_init.` | `conditional_eval.` | `list_indexing` | `function_returns` | **Min (Lowest Skill)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `main_test` | `delayed_retention_ppo` | | | | | | | |
| `main_test` | `threshold_heuristic` | | | | | | | |
| `main_test` | `myopic_rl_bandit` | | | | | | | |
| `main_test` | `fixed_curriculum` | | | | | | | |
| `main_test` | `random_valid` | | | | | | | |
| `power_law_test` | *(5 policies repeated)* | | | | | | | |
| `long_gap_test` | *(5 policies repeated)* | | | | | | | |

#### Table T4-C: Ablation 3 — Full 36-Dim `delayed_retention_ppo` vs. 23-Dim Mask/Routing-Ablated `ppo_no_longterm_state` (10 Retrained Seeds)
| Cohort | Policy | State Dims | Delayed Mean (Obs) | **Day-30 Pass** | Paired Diff (`ppo_full - ablated`, 95% CI) | Floor $b_t$ | Min-Skill Day-30 | `self_explanation` Share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `main_test` | `delayed_retention_ppo` | 36 | | | — | | | |
| `main_test` | `ppo_no_longterm_state` | 23 | | | | | | |
| `power_law_test` | `delayed_retention_ppo` | 36 | | | — | | | |
| `power_law_test` | `ppo_no_longterm_state` | 23 | | | | | | |
| `long_gap_test` | `delayed_retention_ppo` | 36 | | | — | | | |
| `long_gap_test` | `ppo_no_longterm_state` | 23 | | | | | | |

#### Table T4-D: Tier 4 Matched Budget Curves (`TOTAL_BUDGET in {16.0, 24.0, 32.0, 40.0}`, `SESSION_BUDGET = 8.0`)
*Populate per cohort (`main_test`, `power_law_test`, `long_gap_test`) with parameterized `AdaptiveTutorEnv(..., total_budget=B)`.*

| Cohort | Budget ($B$) | Policy | Delayed Mean (Obs) | **Day-30 Pass** | Floor $b_t$ | Paired Diff vs `threshold_heuristic` (95% CI) |
|---|---:|---|---:|---:|---:|---|
| `main_test` | 16.0 | `delayed_retention_ppo` / `threshold_heuristic` / `myopic_rl_bandit` | | | | |
| `main_test` | 24.0 | `delayed_retention_ppo` / `threshold_heuristic` / `myopic_rl_bandit` | | | | |
| `main_test` | 32.0 | `delayed_retention_ppo` / `threshold_heuristic` / `myopic_rl_bandit` | | | | |
| `main_test` | 40.0 | `delayed_retention_ppo` / `threshold_heuristic` / `myopic_rl_bandit` | | | | |
| `power_law_test` | 16.0–40.0 | *(repeated across 4 budgets)* | | | | |
| `long_gap_test` | 16.0–40.0 | *(repeated across 4 budgets)* | | | | |

#### Table T4-E: Tier 4 8-Action Distribution & 5-Check Behavioral Sanity Suite
*Matching `3/ppo/runs/v2.2-ppo-main/REPORT.md:15-31` across `main_test`, `power_law_test`, and `long_gap_test`.*

**Part 1 — 8-Action Distribution Shares (% of Turns):**

| Cohort | Policy | `give_easier` | `give_same` | `give_harder` | `explain` | `worked_ex` | `self_explain` | `review_pre` | `advance` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `main_test` | `delayed_retention_ppo` / baselines | | | | | | | | |
| `power_law_test` | `delayed_retention_ppo` / baselines | | | | | | | | |
| `long_gap_test` | `delayed_retention_ppo` / baselines | | | | | | | | |

**Part 2 — 5-Check Behavioral Sanity Suite (`delayed_retention_ppo` Per-Seed Summary):**

| Cohort | Seed | Checks Passed (`/5`) | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%–25%`) | Floor > Heuristic | Beats All Baselines |
|---|---|---:|---:|---:|---:|---|---|
| `main_test` | `20261201..20261210` | | | | | | |
| `power_law_test` | `20261201..20261210` | | | | | | |
| `long_gap_test` | `20261201..20261210` | | | | | | |

#### Table T5: Optional Tier 5 Human Pilot Summary (Strictly Separated from Simulation)
| Cohort | Enrolled ($N$) | Completed ($n$, %, 95% CI) | Unaided Pretest | Delayed Posttest (`d` days) | Paired Change `post - pre` (95% CI) |
|---|---:|---|---:|---:|---|
| Human Pilot (Optional) | | | | | |

---

## 9. Pre-Execution Harness Requirements (Code Changes Deferred Until Execution Phase)

*Note: Zero `.py`, `.js`, `.html`, `.css`, dataset, or checkpoint edits are performed during this planning phase. Before running pre-registered experiments, implement and verify these exact harness updates:*

1. **Tier 1 (`3/model/run_v22.py`, `memory/evaluation.py`, `memory/tabular.py`):**
   - Add `--seeds` CLI argument to `run_v22.py` (replacing hardcoded `SEEDS`), extend `summarize()` with `min` and `max`, and support multi-seed `history_boosting` (`tabular.py:19-22` subsamples `200,000` rows using `random_state=seed`).
   - Add `roc_auc` (`sklearn.metrics.roc_auc_score`) to `evaluation.py:probability_report()`, emit all 10 calibration bins (`[0.0, 0.1)` ... `[0.9, 1.0]`, padding empty bins with `queries: 0, mean_prediction: null, observed_rate: null` instead of `if not count: continue` at `evaluation.py:68-69`), and pass `draws=2000` to `paired_intervals()`.
2. **Tier 2 (`3/adaptive_web/rag.py` Benchmark Wrapper):**
   - Implement an evaluation harness exposing: `Arm 1` strict FTS5 BM25 (`2/(60+rank) + rerank_boost` and raw BM25), `Arm 2` deployed `HybridRAGEngine.retrieve()` (plus raw RRF diagnostic), `Arm 3` Dev-selected dense bi-encoder over `KnowledgeChunk.searchable_text` (`2/(60+rank) + rerank_boost` and raw cosine, with `episodic +0.015` and `rag.py:795` matching `Arm 2`), and `Arm 4` `BM25 + Dense` (`rrf + rerank_boost` and raw RRF), with Dev-selected `Dense` query input (`raw+meta` vs `rewrite_query()`).
3. **Tier 3 & Ablation 2 (`3/adaptive_web/llm_pedagogy.py`, `app.py` Evaluation Harness):**
   - Always assert/inspect `res.get("source") == "llm"` and `llm_client.last_call_info()` (`failover_used == False`) on every live-LLM call so `PedagogyRealizer`'s silent `"fallback"` / `"deterministic_double"` / `llm.clone.*` / `synth.*` catches (`llm_pedagogy.py:963-1014, 1146-1153, 1472-1494`) are never scored as live-LLM outputs.
   - Run Tier 3a / Ablation 2 via `AdaptiveLearningApp.chat_doubt()` on an **isolated per-prompt SQLite DB snapshot**; add an `eduguard_enabled: bool = True` toggle that keeps `active_question` and `active_preview` in `rag.retrieve()` **identical** while disabling only `_is_pasted_active_question` blocking, the `eduguard_rule` prompt clause, and `_enforce_eduguard()`.
   - In `llm_pedagogy.py:1138` (`evaluate_self_explanation`), preserve `det_grade["grading_status"]` instead of hardcoding `"resolved"` if conceptless/unresolved prompts are evaluated.
4. **Tier 4 & Ablation 3 (`3/ppo/env.py`, `agent.py`, `train_ppo.py`):**
   - Parameterize `AdaptiveTutorEnv(..., total_budget=TOTAL_BUDGET)` (`env.py:423, 815`) so `obs[21]` and `self.done` scale cleanly across budgets `{16.0, 24.0, 32.0, 40.0}`.
   - Return `per_skill_day_30` dict (all 6 canonical skills) from `evaluate_delayed_retention()` (`env.py:829-865`) and add `TEST_SEED_START_PREREG = 40_000..40_399` + `draws=2000` bootstrap in `train_ppo.py`.
   - For Ablation 3 (`ppo_no_longterm_state`), add an ablated environment mode that: (a) slices `obs` uniquely to **`obs_dim = 23`** (`obs[0..2, 7..19, 21, 30..35]`) with `MaskedActorCritic(obs_dim=23, ...)`, (b) replaces `_weak_prereqs_or_decayed` in `valid_action_mask()` (`env.py:346-356, 384-385`) with short-term `p_med`-only thresholds (`p_med < 0.60` / `0.55`), and (c) replaces `Action 6`/`7` `min(..., key=p_30d)` routing (`env.py:772, 789`) with `min(..., key=p_med)`.
