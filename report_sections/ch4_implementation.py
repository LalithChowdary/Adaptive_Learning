"""Chapter 4 — Implementation."""
from report_sections.references import cite, cite_many
from report_sections.style import (
    add_bullet,
    add_code_block,
    add_h1,
    add_h2,
    add_h3,
    add_p,
    add_table,
)


def build(doc, ctx=None):
    """Append Chapter 4 — Implementation to `doc`."""
    add_h1(doc, "Chapter 4 — Implementation", page_break=True)

    add_p(
        doc,
        "While Chapter 3 explained the ideas and mathematics behind our dataset, memory models, "
        "and teaching policy, this chapter explains how the actual software is built and how data "
        "flows from one module to the next. We trace the codebase chronologically across all three "
        "project stages under directories 1/, 2/, and 3/, ending with the two interactive web "
        "applications (Recall on port 8765 and the Adaptive Python Tutor on port 8780). Every "
        "technical term is explained in plain English so that a reader opening the repository for "
        "the first time can understand what each file does and why it is designed that way.",
    )

    # ------------------------------------------------------------------
    # 4.1 Historical Prototypes (1/ LoRA Scripts and 2/ ASSISTments Pipeline)
    # ------------------------------------------------------------------
    add_h2(doc, "4.1 Historical Prototypes (1/ LoRA Scripts and 2/ ASSISTments Pipeline)")

    add_p(
        doc,
        "Before building the Python tutoring platform in 3/, we ran two exploratory prototypes to "
        "test two different parts of the problem: (1) whether a fine-tuned language model could "
        "adjust its teaching tone, and (2) whether recurrent or attention-based memory models work "
        "better on real student logs. Both prototypes are kept intact as historical baselines.",
    )

    add_h3(doc, "Stage 1: Local LoRA Style Adaptation (1/)")
    add_p(
        doc,
        f"In directory 1/, we tested Low-Rank Adaptation (LoRA) {cite('hu2022lora')}—a lightweight "
        "technique that tunes a small set of extra weights inside a large language model instead of "
        "retraining billions of parameters. Using Apple's MLX framework (mlx_lm), we fine-tuned a "
        "4-bit quantized Meta-Llama-3.1-8B-Instruct model into two separate persona adapters:",
    )
    add_bullet(
        doc,
        "1/generate_data.py and 1/adapters/ (Beginner / Struggling Persona): ",
        "Uses random.seed(42) to format 75 synthetic chat examples across five student states "
        "(39 frustration, 15 success, 8 curiosity, 8 tips, and 5 boredom) using an 85/15 split "
        "(63 training and 12 validation conversations in 1/data/). Every assistant reply validates "
        "the student's feeling, steps away from code, and offers a real-world analogy.",
    )
    add_bullet(
        doc,
        "1/generate_data_2.py and 1/adapters-2/ (Advanced / Fast Persona): ",
        "Uses random.seed(99) to create 43 synthetic examples across five fast-learner patterns "
        "(15 quick_understanding, 8 impatience, 8 deep_dive, 4 challenging, and 8 rapid_fire; "
        "36 training and 7 validation examples in 1/data-2/). Replies are short, dense, and "
        "technical with zero analogies.",
    )
    add_bullet(
        doc,
        "Shared LoRA Configuration & Comparison Scripts: ",
        "Both adapters share identical saved hyperparameters in adapter_config.json (rank = 8, "
        "scale = 20.0, 4 adapted layers, batch_size = 1, max_seq_length = 2048, Adam optimizer, "
        "learning rate 1e-5, seed = 0, and 100 iterations; 1/adapters/ also retains 200- and "
        "300-iteration checkpoints from an earlier run). Script 1/test_comparison.py runs "
        "side-by-side generation (temperature 0.7, max_tokens 300) on three prompts including a "
        "novel recursion frustration prompt, and 1/make_ppt.py builds a slide deck comparing No "
        "Adapter, Adapter 1, and Adapter 2.",
    )
    add_p(
        doc,
        "Why Stage 1 was superseded: Stage 1 was purely qualitative—it had no held-out grading "
        "metric, no delayed-retention test, and no teaching policy. It showed that changing tone "
        "alone does not tell a tutor what skill is weak or when to review it. This insight pushed "
        "us toward separating a shared cognitive memory model and a reinforcement learning (RL) "
        "policy from the language generator.",
    )

    add_h3(doc, "Stage 2: ASSISTments 2012-2013 Memory Benchmark (2/)")
    add_p(
        doc,
        f"In directory 2/, we tested real student interaction logs from the ASSISTments 2012-2013 "
        f"dataset {cite('assistments2012')} using PyTorch 2.14.1 on Apple Metal Performance Shaders "
        "(mps). Script 2/compare_memory_models.py streams the raw CSV file "
        "(2012-2013-data-with-predictions-4-final.csv, 6,123,270 rows), removes 2,272 non-tutor "
        "rows and 17,642 auto-graded open_response rows, and retains 6,103,356 interactions.",
    )
    add_p(
        doc,
        "Because 55.71% of the rows lacked a skill_id tag, each item had to be mapped to a token "
        "s:<skill_id> when present or p:<problem_id> otherwise, producing a 118,232-token "
        "vocabulary. Each past step also carried four log-scaled features: hint count, attempt "
        "count, first-response time in milliseconds, and an original-problem flag. Students were "
        "split deterministically into 80% training (37,358 students, 62,469 chunks), 10% validation "
        "(4,577 students, 7,541 chunks, 582,244 valid targets), and 10% test (4,708 students, "
        "7,795 chunks, 606,086 valid targets) using a SHA-1 hash of user_id (seed = 42, "
        "max_len = 128, stride = 127).",
    )
    add_p(
        doc,
        f"We trained two 64-dimensional sequence models (batch_size = 64, lr = 1e-3, 5 epochs, "
        f"BCEWithLogitsLoss): LSTMDKT {cite('piech2015dkt')} (a 2-layer LSTM reading past "
        f"interactions and querying a target item) and SAKT {cite('pandey2019sakt')} (a 4-head "
        "causal self-attention Transformer with positional embeddings). On the 606,086 test "
        "targets, LSTMDKT achieved 0.50280 log loss, 0.7852 ROC-AUC, and 75.58% accuracy, clearly "
        "outperforming SAKT (0.52305 log loss, 0.7698 ROC-AUC, 74.41% accuracy). While this "
        "confirmed that recurrent state summarizes sequential learning well, the 118,232-ID "
        "vocabulary could not generalize to new Python topics, and the logs lacked elapsed time "
        "gaps (Delta t) and 1/7/30-day unaided delayed exams—leading directly to directory 3/.",
    )

    # ------------------------------------------------------------------
    # 4.2 Dataset Generation and Learner-Memory Training Pipeline (3/data_Set/ and 3/model/)
    # ------------------------------------------------------------------
    add_h2(doc, "4.2 Dataset Generation and Learner-Memory Training Pipeline (3/data_Set/ and 3/model/)")

    add_h3(doc, "Custom Python Dataset Pipeline (3/data_Set/v2.2/)")
    add_p(
        doc,
        "The current dataset release (v2.2.0) under 3/data_Set/v2.2/ is built by four core "
        "Python modules. Earlier versions (v1, v2, and v2.1) remain archived untouched, and "
        "3/data_Set/v2.2/RELEASE_MANIFEST.json records SHA-256 cryptographic hashes for all 226 "
        "historical files plus every generated v2.2 artifact to guarantee reproducibility:",
    )
    add_bullet(
        doc,
        "catalog.py (Skill DAG & 2,160-Item Bank): ",
        "Defines the 6 canonical introductory Python skills (range_boundaries, loop_tracing, "
        "accumulator_initialization, conditional_evaluation, list_indexing, function_returns), "
        "5 task formats (single_choice, multiple_select, output_prediction, fill_in_the_blank, "
        "code_completion), and 3 difficulty levels (easy, medium, hard). Combining 6 skills x "
        "3 difficulties x 5 formats x 24 numeric variants produces 2,160 authored items: 1,440 "
        "for practice (variants 0-15) and 720 for independent assessment (variants 16-23).",
    )
    add_bullet(
        doc,
        "simulator.py (Deterministic Cognitive Student Simulator): ",
        f"Models each simulated student using hidden skill strength k_t, consolidated permastore "
        f"floor b_t {cite('bahrick1984permastore')}, and Duolingo Half-Life Regression (HLR) "
        f"half-life growth h_t {cite('settles2016hlr')}. Crucially, v2.2 enforces three fixes "
        "over v2.1: (1) unaided attempts with score = 0 give 0.0 learning dose so failures never "
        "artificially raise skill; (2) failed attempts draw response time from a two-part mixture "
        f"(35% rapid guesses at 0.05-0.25x reference time {cite('wise2005rte')} and 65% engaged "
        "effort); and (3) retention decays only the transient gap (k_t - b_t) toward the durable "
        "floor b_t using exponential decay (main, pilot, long_gap) or power-law decay (power_law).",
    )
    add_bullet(
        doc,
        "build_dataset.py & replay.py (Cohort Assembly & Causal Projection): ",
        "build_dataset.py enforces a strict pilot gate—the 200-learner pilot cohort must pass "
        "full auditing before main (2,000 learners: 1,600 train / 200 validation / 200 test), "
        "power_law (200 test-only), and long_gap (200 test-only) cohorts can be generated. "
        "Each learner receives 96 practice items plus 18 cloned independent delayed exams "
        "(6 skills at 1, 7, and 30 days). Across all 2,600 learners, v2.2 stores 1,813,029 events, "
        "498,383 responses, and 219,284 main training targets (192,779 practice + 26,505 delayed).",
    )

    add_h3(doc, "Learner-Memory Feature & Training Pipeline (3/model/memory/ and 3/model/run_v22.py)")
    add_p(
        doc,
        "The memory package under 3/model/memory/ converts raw chronological logs into tensors "
        "and trains both tabular baselines and neural sequence models on Apple GPU (mps:0):",
    )
    add_bullet(
        doc,
        "features.py (Subject-Independent Feature Contract): ",
        "EvidenceStream walks a single (learner, skill) history in availability order, pairing "
        "each response and grade into an attempt token while preserving hint, solution_reveal, "
        "and instruction events. Each target question becomes a 19-number vector (TASK_SIZE = 19: "
        "5 format one-hots, 3 difficulty one-hots, optional log reference time, snippet/option/"
        "criterion counts, and credit/penalty weight summaries). Each historical event becomes a "
        "47-number token (TOKEN_SIZE = 47) encoding event type, task features, hint type, "
        "assistance type, grading status, and optional value+presence pairs. Text, code, and "
        "learner/skill IDs are strictly excluded so weights transfer across skills.",
    )
    add_bullet(
        doc,
        "recurrent.py (Neural Memory Architectures & monotonic_gru): ",
        f"Implements LearnerGRU {cite('cho2014gru')} across six variants: time_input_gru, "
        f"time_decay_gru, time_aware_transformer {cite('vaswani2017attention')}, kt_decay_gru, "
        "monotonic_gru, and monotonic_transformer. In monotonic_gru (10,223 parameters), the "
        "32-float state vector stores 6 active skill channels k_t, 6 consolidated permastore "
        "floors b_t, cumulative success/failure counters (n_+, n_-), and an 18-float auxiliary "
        "GRU context vector c_t. Before updating, _clean_token zeroes response-speed fields "
        "whenever full_success is false so fast guessing cannot fake competence. Elapsed time "
        f"uses HLR half-life growth h_t in [5, 365] days, FSRS/Wickelgren power-law decay "
        f"{cite_many('ye2022fsrs', 'wickelgren1974')}, and a permastore floor with at least 88% "
        f"stability, read out by a Birnbaum 3PL Item Response Theory head {cite('birnbaum1968irt')} "
        "that guarantees Easy > Medium > Hard pass probabilities in O(1) time per step.",
    )
    add_bullet(
        doc,
        "tabular.py, training.py, behavioral.py, and 3/model/run_v22.py: ",
        "tabular.py fits task_logistic (19 dims), history_logistic (53 dims = 19 task + 34 summary "
        "features), and history_boosting (HistGradientBoostingClassifier). training.py verifies "
        "native mps:0 GPU gradients (rejecting silent CPU fallback), sorts sequences by length "
        "into padded batches of 64, and trains for up to 35 epochs using AdamW (lr = 0.003, "
        "weight_decay = 0.001), OneCycleLR (pct_start = 0.15), gradient clipping at 5.0, and "
        "early-stopping patience 8 after 75% of epochs. behavioral.py runs the 6-check sanity "
        "suite (zero-score drop across all speeds including 6.426s, full-success gain, "
        "unaided > assisted > fail ordering, Easy > Medium > Hard ordering across all 5 formats, "
        "non-increasing 0-60d forgetting curves, and realistic replay of a real dashboard trace). "
        "3/model/run_v22.py selects the lowest validation selection log loss (0.5 * PracticeLL + "
        "0.5 * DelayedMeanLL) only among models passing 6/6 behavioral checks across all seeds.",
    )

    # ------------------------------------------------------------------
    # 4.3 Closed-Loop PPO Training and Policy Serving (3/ppo/)
    # ------------------------------------------------------------------
    add_h2(doc, "4.3 Closed-Loop PPO Training and Policy Serving (3/ppo/)")

    add_p(
        doc,
        f"Once the learner memory model can forecast how well a student will remember a skill "
        f"today and 30 days from now, the next module (3/ppo/) trains a reinforcement learning "
        f"teacher using Proximal Policy Optimization (PPO) {cite('schulman2017ppo')} to pick the "
        f"best teaching action within a fixed study budget.",
    )

    add_bullet(
        doc,
        "env.py (FrozenMemoryTracker & AdaptiveTutorEnv): ",
        "FrozenMemoryTracker loads the frozen monotonic_gru checkpoint "
        "(3/model/runs/v2.2-mps-main/monotonic_gru/seed_20261122/model.pt) with gradients "
        "disabled so online practice updates each student's 32-float state without altering "
        "memory weights. AdaptiveTutorEnv simulates a closed-loop tutor with TOTAL_BUDGET = 32.0 "
        "cost units (SESSION_BUDGET = 8.0 per session, with inter-session breaks of 1.5 days on "
        "main/power_law or 4.0 days on long_gap). At each step, the policy receives a 36-number "
        "state vector s_t combining immediate pass chances (p_easy, p_med, p_hard), 7- and "
        "30-day forecasts (p_7d, p_30d), fragility (p_med - p_30d), floor b_t, active strength "
        "k_t, recent score/hints/leakage, priming flags, prerequisite readiness, remaining budget "
        "ratio, per-skill 30-day retention across all 6 skills, and the active skill one-hot. "
        "For Ablation 3 (ppo_no_longterm_state), the 13 long-term channels are hidden (leaving "
        "23 dimensions) and prerequisite/topic routing uses short-term p_med only.",
    )
    add_bullet(
        doc,
        "Gated Option B Action Space (8 Actions & Masking): ",
        "Actions 0-2 ask an easier, same-difficulty, or harder question (cost 1.0); actions 3-4 "
        "explain a concept (0.6) or show a worked example (0.7); action 5 requests an active "
        "self-explanation (0.9); action 6 reviews a weak prerequisite or decayed earlier skill "
        "(1.0); and action 7 advances to the next topic (0.2). Gated Option B enforces common-sense "
        "rules through a boolean valid-action mask: after a passive lesson (actions 3 or 4), "
        "primed_mode is set and the next step is forced to be a question (no passive-lesson "
        "spamming); action 5 unlocks only after a question on the current skill; action 6 unlocks "
        "only when a prerequisite or earlier skill is weak or decayed; and action 7 unlocks only "
        "after at least one attempt on the current skill.",
    )
    add_bullet(
        doc,
        "agent.py (MaskedActorCritic & Baselines): ",
        f"Defines MaskedActorCritic—two parallel 2-hidden-layer neural networks (64 hidden units each, "
        "SiLU activations, orthogonal weight initialization) for the actor (action probabilities) "
        "and critic (state value estimate V(s_t)). Invalid actions have their logits set to -1e9 "
        "so they have zero probability. Advantage estimation uses Generalized Advantage Estimation "
        f"(GAE, gamma = 0.99, lambda = 0.95) {cite('schulman2016gae')}, updated with clipped PPO "
        "(clip epsilon = 0.20, value loss weight 0.5, 4 PPO epochs, minibatch size 128). "
        "agent.py also implements RandomValidPolicy, FixedCurriculumPolicy, and "
        "ThresholdHeuristicPolicy as comparators.",
    )
    add_bullet(
        doc,
        "train_ppo.py (Training Loop & 5-Check Behavioral Gate): ",
        "Trains each RL seed for 100 updates x 24 simulated episodes per update (2,400 episodes "
        "per seed) using AdamW (lr = 2.5e-3 decaying linearly to 15% of its initial value, "
        "weight_decay = 1e-4) while annealing the entropy bonus from 0.020 down to 0.003 "
        "(cur_ent = 0.020 - 0.017 * frac). The reward combines step-by-step improvements in "
        "mean 30-day retention, lowest-skill 30-day retention, and permastore floor b_t with a "
        "terminal +10.0 * delayed_mean bonus after simulated 1-, 7-, and 30-day unaided exams. "
        "A 5-check sanity gate verifies that question turns >= 40%, self-explanation <= 32%, "
        "topic advancement stays in [4.5%, 25%], floor b_t exceeds threshold_heuristic, and "
        "expected delayed retention beats all four baselines.",
    )

    # ------------------------------------------------------------------
    # 4.4 Recall Memory & PPO Dashboard (3/dashboard/, Port 8765)
    # ------------------------------------------------------------------
    add_h2(doc, "4.4 Recall Memory & PPO Dashboard (3/dashboard/, Port 8765)")

    add_p(
        doc,
        "Before adding live language-model generation, we built Recall (3/dashboard/), a clean "
        "local web application that lets a student or researcher answer real Python exercises from "
        "the 2,160-item v2.2 catalog and inspect live memory predictions and PPO recommendations "
        "without needing any external API key.",
    )

    add_bullet(
        doc,
        "server.py (Loopback HTTP Server, Port 8765): ",
        "Runs a Python ThreadingHTTPServer bound strictly to 127.0.0.1:8765. It checks Host and "
        "Origin headers to block remote access, enforces a per-session X-Dashboard-Token CSRF "
        "secret on all POST writes, attaches Content-Security-Policy headers, and caps POST "
        "payloads at 40,000 bytes.",
    )
    add_bullet(
        doc,
        "service.py & catalogue.py (Question-First Session Rules): ",
        "Dashboard in service.py enforces question-first interaction rules: learners receive up to "
        "3 progressive hints and 2 retries per practice item, while independent assessment checks "
        "grade only the first unaided response. catalogue.py loads the 2,160-item v2.2 bank "
        "read-only alongside authored hints and curated LESSONS for all 6 skills.",
    )
    add_bullet(
        doc,
        "grading.py (Deterministic Grading & Bounded AST Interpreter): ",
        "Grades single-choice, multiple-select, output-prediction, and fill-in-the-blank tasks "
        "deterministically against reference keys. For code_completion items, student code is "
        "never passed to Python's exec() or eval(). Instead, a custom bounded Abstract Syntax Tree "
        "(AST) Interpreter parses the submitted def solve(x): function and walks allowed syntax "
        "nodes under hard safety limits: at most 8,000 characters, 1,200 AST nodes, 10,000 "
        "operations per test case, 2,000 items per sequence, 256-bit integers, and only seven "
        "safe built-ins (range, list, len, sum, min, max, abs). Imports, attributes, recursion, "
        "while loops, and comprehensions are rejected safely as 'unresolved' rather than marked "
        "wrong.",
    )
    add_bullet(
        doc,
        "memory_service.py, ppo_service.py, & store.py (sessions.sqlite3): ",
        "store.py logs all profiles, interactions, append-only events, and idempotent request IDs "
        "in 3/dashboard/data/sessions.sqlite3 (SQLite WAL mode) so real sessions never touch the "
        "v2.2 dataset. MemoryService wraps the frozen v2.2 monotonic_gru checkpoint with cursor "
        "caching per (learner, skill, model_sha256). PPOTutorService loads the 3-seed "
        "delayed_retention_ppo ensemble (seeds 20261120-20261122) to serve GET /api/ppo and "
        "POST /api/ppo_step.",
    )
    add_bullet(
        doc,
        "0-60d Memory Explorer & Unassessed Guard (static/): ",
        "The browser interface lets users switch across all 5 question formats and view Easy, "
        "Medium, and Hard 0-to-60-day retention curves alongside the dashed permastore floor b_t. "
        "Skills with zero resolved attempts display an explicit 'Unassessed' guard instead of "
        "pretending an initial model prior is measured student mastery.",
    )

    # ------------------------------------------------------------------
    # 4.5 Adaptive Python Web Tutor and Glass-Box Inspector (3/adaptive_web/, Port 8780)
    # ------------------------------------------------------------------
    add_h2(doc, "4.5 Adaptive Python Web Tutor and Glass-Box Inspector (3/adaptive_web/, Port 8780)")

    add_p(
        doc,
        "Directory 3/adaptive_web/ combines the frozen monotonic_gru memory, the 3-seed "
        "delayed_retention_ppo ensemble, a Hybrid Retrieval-Augmented Generation (RAG) engine, "
        "difficulty-locked LLM question synthesis, anti-leakage guardrails (EduGuard), and a "
        "Glass-Box Research Inspector into a full adaptive Python learning website on port 8780.",
    )

    add_h3(doc, "Core Web Stack, Persistence, and Multi-Provider LLM Failover")
    add_bullet(
        doc,
        "server.py, app.py, & store.py (adaptive_web.sqlite3): ",
        "server.py serves static assets and JSON endpoints (/api/state, /api/profile, /api/select_skill, "
        "/api/ppo_step, /api/submit, /api/hint, /api/retry, /api/reveal, /api/finish, "
        "/api/chat, /api/curve, /api/export, /api/rag/ingest) on port 8780 with local-only "
        "Host/Origin checks, X-CSRF-Token validation, strict Content-Security-Policy, and a 1 MiB "
        "body limit. AdaptiveLearningApp (app.py) coordinates memory, PPO, RAG, and pedagogy "
        "without holding database locks during outbound LLM HTTP calls, and supports isolated "
        "in-memory/file snapshots via clone_snapshot() using sqlite3.Connection.backup(). "
        "store.py manages 3/adaptive_web/data/adaptive_web.sqlite3 in WAL mode across tables for "
        "profiles, interactions, events, requests, learning_paths, course_materials, tutor_turns, "
        "chat_messages, and learner-scoped generated_tasks (safely migrated via PRAGMA "
        "table_info and ALTER TABLE).",
    )
    add_bullet(
        doc,
        "catalogue_bridge.py, memory_adapter.py, & ppo_engine.py: ",
        "catalogue_bridge.py provides a read-only bridge to the 2,160-item v2.2 catalog, the "
        "canonical PPO prerequisite DAG, and validate_task_schema(). MemoryAdapter loads the "
        "frozen monotonic_gru checkpoint, enforces strict (learner, skill) event isolation, and "
        "exposes inspect_tokens() for the 47-dim token audit. PPOEngine evaluates the 3-seed "
        "delayed_retention_ppo ensemble over both v2.2 catalog items and newly generated SQLite "
        "tasks.",
    )
    add_bullet(
        doc,
        "config.py & llm_client.py (Multi-Provider Failover & Secret Redaction): ",
        "A pure-stdlib HTTP client reads .env and environment settings without leaking secrets. "
        "If a primary provider fails or times out, LLM_FALLBACK_PROVIDERS walks an ordered "
        "failover chain across groq, openai, anthropic, gemini, ollama, and optional "
        "deterministic_double, stripping every API key from error strings before returning status.",
    )

    add_h3(doc, "Hybrid RAG Engine (rag.py)")
    add_p(
        doc,
        f"To keep explanations and generated exercises grounded in verified course material "
        f"rather than unverified model guesswork {cite('lewis2020rag')}, HybridRAGEngine indexes "
        "three SQLite-backed corpora: (1) 24 curriculum chunks parsed from 6 Markdown guides in "
        "3/adaptive_web/knowledge/*.md (plus instructor-uploaded Markdown via /api/rag/ingest), "
        "(2) all 2,160 v2.2 exemplars, and (3) learner-scoped episodic memories (prior mistakes, "
        "solution reveals, and tutor doubts). Each query passes through five stages:",
    )
    add_bullet(
        doc,
        "1. Query Rewriting (rewrite_query): ",
        "Expands short student queries with canonical SKILL_META summaries, difficulty/format "
        "tags, active question tokens, and domain synonyms (_PYTHON_SYNONYMS, e.g., mapping "
        "'off by one' to 'range stop exclusive start inclusive boundary len').",
    )
    add_bullet(
        doc,
        "2. Sparse Lexical Search (SQLite FTS5 BM25): ",
        f"Queries an FTS5 virtual table (porter unicode61 tokenizer, BM25 field weights 2.0 "
        f"title, 1.5 section, 1.0 content) {cite('robertson2009bm25')} with strict SQL WHERE "
        "filters on source_type, skill, difficulty, task_type, and learner = ? for episodic rows.",
    )
    add_bullet(
        doc,
        "3. Lexical Vector Search (TF-IDF Cosine): ",
        "Computes cosine similarity using a scikit-learn TfidfVectorizer (1-2 word n-grams, "
        "8,192 max features, sublinear term frequency), adding a +0.015 offset for dynamic "
        "learner-scoped episodic records.",
    )
    add_bullet(
        doc,
        "4. Reciprocal Rank Fusion (RRF, k = 60): ",
        f"Merges the FTS5 rank and TF-IDF rank using 1 / (60 + rank_fts) + 1 / (60 + rank_vec) "
        f"{cite('cormack2009rrf')}, reducing sensitivity to raw score scale differences.",
    )
    add_bullet(
        doc,
        "5. Pedagogical Reranking & Context Budgets: ",
        "Applies small rerank_boost bonuses for exact skill/difficulty/format matches, exact "
        "query-token overlap, and mastery-tier relevance (boosting pitfalls and episodic history "
        "when mastery is lacking or unassessed). Doubt Tutor chat uses top k = 4 chunks, while "
        "question generation retrieves k = 3 difficulty-locked exemplars plus k = 2 curriculum/"
        "episodic chunks.",
    )

    add_h3(doc, "Difficulty-Locked Question Generation, 1-Shot Repair, and AST Sandbox")
    add_p(
        doc,
        "Inside 3/adaptive_web/llm_pedagogy.py, generate_question() retrieves 3 difficulty-locked "
        "v2.2 exemplars and 2 curriculum chunks, then injects explicit complexity bounds "
        "(_DIFFICULTY_GUARDRAILS) into the generation prompt for each level: easy items target "
        "1-5 lines and 15-35 s; medium items target 4-10 lines and 35-60 s; hard items target "
        "7-16 lines and 60-110 s. Post-generation, _validate_with_guardrails() enforces a hard "
        "line check on easy snippets at <= 7 lines (max_lines + 2), while _normalize_candidate_task() "
        "clamps reference_seconds to [10.0, 180.0].",
    )
    add_p(
        doc,
        "Every generated item must pass an anti-duplicate filter (max_text_similarity <= 0.85 "
        "against retrieved exemplars) and full structural/runnable checking via "
        "validate_task_schema(..., verify_code_tests=True). If a live LLM returns a schema or "
        "guardrail error, PedagogyRealizer makes a 1-shot question_repair call passing the exact "
        "validation_errors back to the model; if it still fails, the system safely falls back to "
        "an unseen v2.2 clone or deterministic template.",
    )
    add_p(
        doc,
        "Student answers are graded by 3/adaptive_web/grader.py, which wraps the bounded AST "
        "Interpreter from 3/dashboard/grading.py (max 8,000 characters, 1,200 AST nodes, 10,000 "
        "ops/case, 2,000 sequence items, 256-bit integers, safe built-ins only). For PPO "
        "Action 5 (request_self_explanation), grade_self_explanation() scores free-text student "
        "explanations strictly by rubric concept coverage (score >= 1.0 for full_success) while "
        "computing grader_confidence separately in [0.05, 0.99] from word count, causal "
        "connectors (e.g., 'because', 'therefore'), and hedging phrases—preserving the clean "
        f"separation between score and confidence {cite_many('typesafe1', 'typesafe2')}.",
    )

    add_h3(doc, "Mastery-Adaptive Doubt Tutor, EduGuard, and Glass-Box Inspector UI")
    add_p(
        doc,
        "When a student asks a question in the Learn tab or side-by-side AI Tutor drawer "
        "(POST /api/chat), _resolve_mastery_tier() reads the learner's monotonic_gru summary "
        "and classifies them into one of four tiers: unassessed, lacking (p_med < 70% or observed "
        "unaided pass rate < 70%), fragile (fragility > 0.15), or mastered. For unassessed or "
        "lacking learners, RAG automatically sets include_prerequisites=True and slows replies "
        "into step-by-step variable traces.",
    )
    add_p(
        doc,
        "To prevent students from using the chat drawer to cheat on an open quiz, EduGuard "
        "(_enforce_eduguard and _is_pasted_active_question) checks whether any unpassed question "
        "is open (even across skill tabs). When active, it injects anti-leakage rules into the "
        "system prompt, flags answer-fishing patterns (_ANSWER_FISHING_RE) or pasted prompts "
        "(similarity >= 0.35 or __blank_ markers), and scans the final reply and code snippet "
        "for leaked option IDs, answer keys, or reference solutions—replacing any leak with a "
        "grounded Socratic hint (guardrail_status = 'socratic_redirect').",
    )
    add_p(
        doc,
        "Finally, the toggleable Glass-Box Research Inspector in static/index.html and "
        "static/app.js exposes the internal decision pipeline live alongside the 3-tab learning "
        "workspace (Learn, Practice, Progress): it displays the 47-dim token audit (confirming "
        "zero text or code embeddings), the live 36-dim policy state s_t, the Gated Option B "
        "valid-action mask, the critic value V(s_t), the recommended action and policy_source, "
        "LLM failover telemetry, and the RAG & Guardrails Audit panel (corpus counts, FTS5/vector/"
        "RRF candidate counts, rewritten query, and search latency).",
    )

    add_code_block(
        doc,
        "End-to-End Runtime Turn Flow (3/adaptive_web/):\n"
        "  1. Student Action / Answer  ->  grader.py (deterministic / AST Interpreter / rubric)\n"
        "  2. Event Logged (SQLite WAL)->  memory_adapter.py (47-dim token -> 32-float monotonic_gru)\n"
        "  3. State Constructed        ->  ppo_engine.py (36-dim s_t + Gated Option B mask -> action)\n"
        "  4. Grounded Realization     ->  rag.py (FTS5 + TF-IDF + RRF k=60) + llm_pedagogy.py + EduGuard",
    )

    t41_headers = ["Module / Stage", "Primary Files", "Core Responsibility", "Key Safeguards"]
    t41_rows = [
        [
            "Stage 1 (Historical LoRA)",
            "1/generate_data.py, generate_data_2.py, test_comparison.py",
            "MLX LoRA style tuning on Llama-3.1-8B-4bit (Beginner vs. Advanced).",
            "Fixed seeds (42, 99); documented strictly as qualitative baseline.",
        ],
        [
            "Stage 2 (ASSISTments)",
            "2/compare_memory_models.py",
            "6.10M-row real-data comparison of LSTMDKT vs. SAKT on mps.",
            "Student-level SHA-1 80/10/10 split; excludes open_response & non-tutor.",
        ],
        [
            "Dataset v2.2",
            "3/data_Set/v2.2/ (catalog.py, simulator.py, build_dataset.py)",
            "2,160-item bank (6 skills, 5 formats); HLR + Permastore simulator.",
            "0.0 dose on unaided 0% fails; pilot gate; SHA-256 RELEASE_MANIFEST.json.",
        ],
        [
            "Memory Training",
            "3/model/memory/ (features.py, recurrent.py, training.py, behavioral.py)",
            "47-dim tokens; 9-model benchmark on mps:0; monotonic_gru O(1) state.",
            "Zero text/ID embeddings; _clean_token speed mask; 6-check behavioral gate.",
        ],
        [
            "PPO Policy",
            "3/ppo/ (env.py, agent.py, train_ppo.py)",
            "36-dim state s_t, 8-action Gated Option B MDP, MaskedActorCritic.",
            "No passive-lesson spam mask; frozen memory; 5-check sanity gate.",
        ],
        [
            "Recall Dashboard",
            "3/dashboard/ (server.py, service.py, grading.py, memory_service.py, ppo_service.py)",
            "Port 8765 offline practice app, 0-60d explorer, live PPO inspection.",
            "Loopback-only + CSRF; bounded AST Interpreter; Unassessed guard.",
        ],
        [
            "Adaptive Web + RAG + EduGuard",
            "3/adaptive_web/ (app.py, rag.py, llm_pedagogy.py, grader.py, llm_client.py)",
            "Port 8780 web tutor, Hybrid RAG (RRF k=60), Q-gen, Doubt Tutor, Glass-Box.",
            "1-shot repair; <=0.85 sim; score != confidence; EduGuard; key redaction.",
        ],
    ]
    add_table(
        doc,
        t41_headers,
        t41_rows,
        caption="Table 4.1 — Mapping of project modules to source files, responsibilities, and key safeguards.",
        col_widths=[1.25, 1.85, 1.90, 1.80],
    )
