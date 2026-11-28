"""Chapter 3 — Datasets and Methodology."""
from report_sections.references import cite, cite_many
from report_sections.style import (
    add_bullet,
    add_eq_block,
    add_h1,
    add_h2,
    add_h3,
    add_p,
    add_table,
)


def build(doc, ctx=None):
    """Append Chapter 3 — Datasets and Methodology to `doc`."""
    add_h1(doc, "Chapter 3 — Datasets and Methodology", page_break=True)

    add_p(
        doc,
        "This chapter tells the full story of how our data, learner-memory models, Reinforcement Learning (RL) "
        "teaching policy, and retrieval-grounded tutoring safeguards were designed. We present the work in "
        "chronological order: first our two early exploratory stages (style-only LoRA tuning and real-world "
        "public logs from ASSISTments), then our custom Python curriculum and simulator evolution (v1 through v2.2), "
        "followed by our nine-model memory benchmark, our eight-action closed-loop RL teacher, and finally our "
        "Hybrid Retrieval-Augmented Generation (RAG) and safety methodology. Every symbol, equation, and design "
        "choice is explained in simple, everyday words before any mathematical formula appears.",
    )

    # =========================================================================
    # 3.1 HISTORICAL FOUNDATIONS
    # =========================================================================
    add_h2(doc, "3.1 Historical Foundations: Stage 1 (LoRA Style Tuning) and Stage 2 (ASSISTments 2012–2013)")

    add_p(
        doc,
        "Before building the current Python tutoring system, we ran two smaller exploratory experiments in "
        "directories 1/ and 2/. Both stages are historical foundations rather than part of our final v2.2 "
        "evaluation, because each exposed a clear gap that shaped our final design.",
    )

    add_h3(doc, "3.1.1 Stage 1 (1/): Qualitative Style Tuning with LoRA")
    add_p(
        doc,
        f"In Stage 1 (directory 1/), we tested whether a large language model (LLM) could be taught to speak in "
        f"different tutoring personas using Low-Rank Adaptation (LoRA) {cite('hu2022lora')}. Instead of retraining "
        f"all 8 billion parameters of the model, LoRA freezes the original weights and adds small, trainable adapter "
        f"matrices inside a few transformer layers. We fine-tuned Meta-Llama-3.1-8B-Instruct-4bit locally using "
        f"Apple's MLX framework with identical adapter settings across both personas: rank = 8, scale = 20.0, "
        f"num_layers = 4, iters = 100, learning_rate = 1e-5, batch_size = 1, dropout = 0.0, max_seq_length = 2048, "
        f"and the Adam optimizer (seed = 0, mask_prompt = false; earlier 200- and 300-step checkpoints are also "
        f"archived in 1/adapters/ alongside the 100-step configuration).",
    )
    add_p(
        doc,
        "We generated two contrasting synthetic chat datasets (using an 85% / 15% train / validation split and "
        "the Llama-3.1 chat template) to represent two opposite student personalities:",
    )
    add_bullet(
        doc,
        "Adapter 1 — Beginner / Struggling Persona (1/generate_data.py): ",
        "75 total examples (63 train, 12 validation) across five emotional patterns: frustration (39), success (15), "
        "curiosity (8), tips (8), and boredom (5). Every reply followed one supportive rule: validate the student's "
        "feeling, pause the code, and explain with an everyday real-world analogy.",
    )
    add_bullet(
        doc,
        "Adapter 2 — Advanced / Fast Persona (1/generate_data_2.py): ",
        "43 total examples (36 train, 7 validation) across five fast-paced patterns: quick understanding (15), "
        "impatience (8), deep dive (8), challenging (4), and rapid fire (8). Replies were short, dense, peer-level, "
        "and stripped of analogies or hand-holding.",
    )
    add_p(
        doc,
        "Qualitative side-by-side checks (1/test_comparison.py at temperature = 0.7 and max_tokens = 300, plus "
        "1/make_ppt.py across loops, production functions, and Python memory management) showed that LoRA reliably "
        "changed tone—from textbook lectures in the vanilla model to reassuring analogies in Adapter 1 and terse "
        "four-word hints in Adapter 2. However, Stage 1 was superseded because it had no held-out accuracy metric, "
        "no blinded preference rating, no delayed memory tracking, and no teaching decision policy. It proved that "
        "changing speaking style alone cannot track what a student actually knows or chooses to forget weeks later, "
        "motivating our final architecture where one shared memory model and RL policy guide a retrieval-grounded "
        "LLM without per-student LoRA weights.",
    )

    add_h3(doc, "3.1.2 Stage 2 (2/): Real-Data Memory Comparison on ASSISTments 2012–2013")
    add_p(
        doc,
        f"Next, in Stage 2 (directory 2/), we tested whether standard deep sequence models could predict student "
        f"answers on a large real-world dataset: the ASSISTments 2012–2013 school math dataset "
        f"{cite('assistments2012')}. A full streaming audit scanned 6,123,270 log rows and removed 2,272 non-tutor "
        f"rows and 17,642 open-response rows (which ASSISTments marks correct automatically), retaining 6,103,356 "
        f"tutor rows across 46,674 unique students and 265 skill strings. Critically, 3,411,457 rows (55.713%) "
        f"were missing a skill_id label.",
    )
    add_p(
        doc,
        "To train models on this log, each interaction was represented by s:<skill_id> if a skill tag existed and "
        "p:<problem_id> otherwise, creating a large vocabulary of 118,232 item tokens, alongside 4 numeric history "
        "features (log-scaled hint count, attempt count, first-response time in milliseconds, and an original-problem "
        "flag). The target label was 1 only when the student answered completely right (correct = 1.0). We split "
        "students deterministically 80% / 10% / 10% (seed = 42, chunk length = 128) into 37,358 training students "
        "(62,469 chunks), 4,577 validation students (7,541 chunks; 582,244 targets), and 4,708 test students "
        "(7,795 chunks; 606,086 test targets).",
    )
    add_p(
        doc,
        f"We compared two classic architectures for 5 epochs (dim = 64, batch_size = 64, AdamW (lr = 1e-3, weight_decay = 1e-5)): a 2-layer "
        f"LSTM Deep Knowledge Tracing model (LSTM-DKT) {cite('piech2015dkt')} and a 4-head Self-Attentive "
        f"Knowledge Tracing Transformer (SAKT-Transformer) {cite('pandey2019sakt')}. On the 606,086 held-out test "
        f"targets, LSTM-DKT achieved 0.5028 test Log Loss (0.50280), 0.7852 ROC-AUC, and 75.58% accuracy "
        f"(where lower Log Loss means better-calibrated probability guesses and higher ROC-AUC means better "
        f"ranking of passing versus failing students), "
        f"outperforming SAKT-Transformer (0.5231 test Log Loss [0.52305], 0.7698 ROC-AUC, and 74.41% accuracy). "
        f"Although LSTM-DKT's win supported compact recurrent memory over vanilla self-attention, Stage 2 was "
        f"superseded by directory 3/ for three decisive reasons:",
    )
    add_bullet(
        doc,
        "1. Missing Skill Tags & Non-Transferable IDs: ",
        "With 55.713% of rows missing skill_id, the models had to memorize 118,232 problem/skill ID embeddings, "
        "which cannot transfer to new Python exercises or new courses.",
    )
    add_bullet(
        doc,
        "2. No Long-Term Forgetting Targets: ",
        "The log lacked explicit elapsed-time (Delta t) forgetting controls and held-out 1-, 7-, and 30-day "
        "unaided assessments to measure whether a student truly remembered concepts weeks later.",
    )
    add_bullet(
        doc,
        "3. Observational Confounding: ",
        "In observational classroom logs, item selection is mixed together with student skill, and there are no "
        "controlled counterfactual records for hints, solution reveals, or lessons needed to safely train and test "
        "a Reinforcement Learning teaching policy.",
    )

    # =========================================================================
    # 3.2 CUSTOM PYTHON CURRICULUM AND DATASET EVOLUTION
    # =========================================================================
    add_h2(doc, "3.2 Custom Python Curriculum and Dataset Evolution (v1 to v2.2)")

    add_p(
        doc,
        "To overcome those three gaps, we restarted dataset construction from scratch under 3/data_Set/ with "
        "Python programming as our first subject. Instead of memorizing problem IDs, our custom simulator generates "
        "clean, time-stamped learning histories where every event belongs to a known skill, every hint or retry is "
        "recorded separately, and independent delayed tests at 1 day, 7 days, and 30 days measure lasting memory "
        "without contaminating the practice timeline.",
    )

    add_h3(doc, "3.2.1 Six Canonical Python Skills and Prerequisite Graph")
    add_p(
        doc,
        "Our curriculum covers six foundational Python skills connected in a Directed Acyclic Graph (DAG)—a "
        "one-way map showing which earlier topics a student needs before tackling harder ones "
        "(3/data_Set/v2.2/catalog.py and 3/ppo/env.py):",
    )
    add_bullet(doc, "1. range_boundaries (Root Skill): ", "Understanding that range(start, stop, step) stops before the upper bound stop (no prerequisites).")
    add_bullet(doc, "2. loop_tracing: ", "Tracking how variables change step by step inside a for loop (prerequisite: range_boundaries).")
    add_bullet(doc, "3. accumulator_initialization: ", "Starting a running total or product at its right initial value before a loop (prerequisites: range_boundaries, loop_tracing).")
    add_bullet(doc, "4. conditional_evaluation: ", "Deciding whether an if or else branch executes for a given input (prerequisite: loop_tracing).")
    add_bullet(doc, "5. list_indexing: ", "Finding items in a Python list using zero-based positive indices and negative end-relative indices (prerequisite: range_boundaries).")
    add_bullet(doc, "6. function_returns: ", "Binding arguments to parameters and tracing what a function returns rather than prints (prerequisites: conditional_evaluation, accumulator_initialization).")

    add_h3(doc, "3.2.2 Question Bank (2,160 Items) and Five Task Formats")
    add_p(
        doc,
        "Across v2, v2.1, and v2.2, the item bank contains 2,160 authored Python tasks: 6 skills x 3 difficulties "
        "(easy, medium, hard) x 5 task formats x 24 variants. Within each skill, difficulty, and format, variants "
        "0 to 15 form the 1,440 practice items and variants 16 to 23 form the 720 held-out assessment items. "
        "Table 3.1 summarizes the five task formats, their target reference time, how strongly a full practice attempt "
        "builds skill (practice dose), and the built-in chance of guessing correctly by luck.",
    )
    add_table(
        doc,
        ["Task Format", "Base Time (s)", "Help Multiplier", "Practice Dose", "Guess Prob. & Grading Style"],
        [
            ["single_choice", "25", "0.75", "0.30", "0.25 (pick 1 of 4 options)"],
            ["multiple_select", "45", "0.90", "0.40", "0.50 per option (partial credit & penalty)"],
            ["output_prediction", "35", "1.00", "0.45", "0.02 (type exact printed output)"],
            ["fill_in_the_blank", "40", "1.10", "0.45", "0.04 (complete missing Python token)"],
            ["code_completion", "75", "1.25", "0.55", "0.01 (write working solve(x) body)"],
        ],
        caption="Table 3.1 — Five Python task formats in the 2,160-item bank (3/data_Set/v2.2/catalog.py).",
        col_widths=[1.40, 1.00, 1.15, 1.05, 2.20],
        right_cols={1, 2, 3},
    )

    add_h3(doc, "3.2.3 Evolution from v1 to v2.2 and Replay Flaws Fixed")
    add_p(
        doc,
        "Our dataset went through four disciplined versions, each fixing a concrete measurement weakness discovered "
        "during model training and interactive replay. Table 3.2 traces this progression from v1 to v2.2.",
    )
    add_table(
        doc,
        ["Version", "Task Bank & Formats", "Practice & Label Design", "Scale (2,600 Learners)", "Limitation Found / Upgrade Made"],
        [
            ["v1", "432 items (output_prediction only; max 1 hint, 1 retry)", "Blocked/mixed practice; practice targets only", "1,205,969 events; 347,861 responses; 146,035 main train targets", "Single format and fixed half-life decaying to 0; no multi-hint or delayed train labels."],
            ["v2", "2,160 items across 5 formats (up to 3 hints, 2 retries)", "Question-first practice + fresh follow-ups + 18 assessment forks", "1,700,389 events; 471,999 responses; 185,510 main train targets", "Max gap in main train was only 3.06 days, forcing 7d/30d tests to extrapolate."],
            ["v2.1", "Same 2,160-item 5-format bank", "Added spaced blocks (30% chance of 5–60d gaps) + train delayed labels", "1,952,928 events; 528,708 responses; 225,224 main train targets (198,651 + 26,573)", "Replay revealed 0% wrong answers raised predictions (42.96% -> 51.33%) and decay collapsed to 0%."],
            ["v2.2 (Current)", "Same 2,160-item 5-format bank", "Zero dose on unaided 0% fails, 35% rapid-guess RT, HLR + Permastore", "1,813,029 events; 498,383 responses; 384,215 tasks; 219,284 main train targets (192,779 + 26,505)", "Resolved all timing, zero-score gain, and zero-collapse issues cleanly."],
        ],
        caption="Table 3.2 — Evolution of the custom Python dataset from v1 to v2.2 (all cohorts have 2,600 simulated learners).",
        col_widths=[0.65, 1.35, 1.45, 1.35, 2.00],
    )

    add_p(
        doc,
        "During interactive testing on our local dashboard (3/dashboard/data/sessions.sqlite3), replaying a real "
        "Conditionals attempt uncovered why the earlier v2.1 model behaved unintuitively: submitting a completely "
        "wrong (0%) unaided answer in just 6.426 seconds (compared to an 84.96-second reference time) caused the "
        "predicted pass chance to jump upward from 42.96% to 51.33% (+8.37 percentage points). Controlled replay "
        "isolated two simulator causes in v2.1: first, +4.74 percentage points came from v2.1 granting a small "
        "unconditional learning gain even when score = 0; second, +3.63 percentage points came from v2.1's response "
        "timer always running faster when hidden skill k_t was higher, accidentally teaching models that fast "
        "clicks mean high mastery. Worse, the unconstrained v2.1 recurrent model decayed toward a shared average "
        "prior, causing forgetting curves after failures to drift upward during breaks, while naive exponential "
        "decay caused mastered skills to collapse toward 0% over 30–60 days.",
    )

    add_p(
        doc,
        "Version 2.2 (3/data_Set/v2.2/simulator.py) solved all three flaws with three principled upgrades:",
    )
    add_bullet(
        doc,
        "1. Zero Learning Dose on Unaided Zero-Score Failures: ",
        "In plain words, guessing wrong with no hints and zero partial credit produces zero learning. Let score "
        "be the rubric score in [0, 1], f_retained be the learner's retained-practice fraction when hints are used, "
        "d_format be the task format's practice dose from Table 3.1, and eta be the learner's learning rate. "
        "Without hints (hints = 0), the credit multiplier is factor = score if full_success else 0.5 * score "
        "(so score = 0 gives factor = 0). With hints (hints > 0), factor = f_retained * (0.20 + 0.80 * score). "
        "Dividing by attempt_number gives dose = (d_format * factor) / attempt_number, updating hidden skill "
        "from k_before to k_after:",
    )
    add_eq_block(
        doc,
        "k_after = k_before + (1 - k_before) * (1 - exp(-eta * dose))",
        label="3.1",
    )
    add_bullet(
        doc,
        f"2. Rapid-Guess Timing Mixture (35% on Non-Success) {cite('wise2005rte')}: ",
        "When an attempt is not a full success, there is now a 35% chance that the student made a rapid guess, "
        "drawing response time from Uniform(0.05, 0.25) * t_ref (clipped to [3, 900] seconds), and a 65% chance "
        "of an engaged log-normal response time. This prevents models from treating fast wrong clicks as mastery.",
    )
    add_bullet(
        doc,
        f"3. Duolingo Half-Life Growth and Consolidated Permastore Floor {cite_many('settles2016hlr', 'bahrick1984permastore')}: ",
        "Think of a student's skill k_t (from 0 to 1) as having two parts: a durable 'bedrock' floor b_t "
        "(initialized at b_0 = 0.65 * k_0) that represents deep understanding resistant to forgetting, and a "
        "temporary fluency layer (k_t - b_t) that fades during breaks. After practice, b_t consolidates upward "
        "by a fraction gamma (0.45 on unaided full success, 0.25 otherwise): b_t = min(k_after, min(k_before, "
        "b_{t-1}) + gamma * (k_after - k_before)). Meanwhile, the skill's memory half-life h_t (in days, clamped "
        "to [2.0, 240.0]) grows with cumulative unaided successes n_+ and shrinks with failures n_-, and strength "
        "after an elapsed gap of Delta t days decays smoothly toward b_t:",
    )
    add_eq_block(
        doc,
        "h_t = clamp(h_0 * 2^(0.35 * (sqrt(1 + n_+) - 1) - 0.15 * (sqrt(1 + n_-) - 1)), 2.0, 240.0)",
        label="3.2",
    )
    add_eq_block(
        doc,
        "k(t + Delta t) = b_t + (k_t - b_t) * R(Delta t, h_t)",
        label="3.3",
    )
    add_p(
        doc,
        "Here R(Delta t, h_t) = 2^(-Delta t / h_t) for exponential-forgetting cohorts (pilot, main, and long_gap) "
        "and R(Delta t, h_t) = (1 + Delta t / h_t)^(-1) for the power_law stress cohort. Across all four cohorts "
        "(2,600 learners total: 200 pilot [160/20/20], 2,000 main [1,600 train / 200 validation / 200 test], 200 "
        "power_law test-only, and 200 long_gap test-only with 14- and 60-day breaks), v2.2 records 1,813,029 events, "
        "498,383 responses, 384,215 tasks, and 219,284 main training targets (192,779 practice + 26,505 train "
        "delayed labels). Each learner completes 96 ordinary tasks (16 per skill), fresh unaided follow-up checks "
        "after assisted tasks, and 18 cloned independent delayed tests (6 skills x {1, 7, 30} days).",
    )

    # =========================================================================
    # 3.3 LEARNER-MEMORY ARCHITECTURE AND 9-MODEL BENCHMARK METHODOLOGY
    # =========================================================================
    add_h2(doc, "3.3 Learner-Memory Architecture and 9-Model Benchmark Methodology")

    add_h3(doc, "3.3.1 Subject-Agnostic Input Contract (Zero Text or ID Embeddings)")
    add_p(
        doc,
        "To keep our memory brain reusable across topics without memorizing question wording or student IDs, "
        "every sequence is isolated by (learner_id, skill_id) and converted into fixed numerical vectors "
        "(3/model/memory/features.py):",
    )
    add_bullet(
        doc,
        "19-Dimensional Task Vector (TASK_SIZE = 19): ",
        "Describes the target question purely by structure: 5 format one-hots + 3 difficulty one-hots + 2 optional "
        "reference-time features (log1p(t_ref / 60) and a presence flag) + 3 log-scaled counts (snippets, options, "
        "criteria) + 6 rubric-weight summaries (count, sum, and max for both credit and penalty weights). Text, "
        "code, and IDs are strictly excluded.",
    )
    add_bullet(
        doc,
        "47-Dimensional Interaction Token (TOKEN_SIZE = 47): ",
        "Encodes one past event: 4 event-type flags (attempt, hint, solution_reveal, instruction) + 19 task "
        "features + 3 hint types (conceptual, targeted, scaffold) + 3 assistance flags (independent, assisted, "
        "unknown) + 2 grading statuses (resolved, unresolved) + 16 optional value-and-presence pairs for attempt "
        "number, hint count, response seconds, relative time, partial score, full success, answer revealed, and "
        "new-session flag.",
    )
    add_bullet(
        doc,
        "34-Dimensional History Summary (SUMMARY_SIZE = 34): ",
        "Aggregate statistics (log counts of attempts, retries, hints, reveals, instructions, mean unaided/assisted "
        "scores, mean response speeds, recent unaided accuracy, and time gaps) concatenated with the 19 task "
        "features to form a 53-dimensional input for tabular baselines.",
    )

    add_h3(doc, "3.3.2 Nine Benchmark Models Plus Two Reference Anchors")
    add_p(
        doc,
        "We compare nine trained models—spanning tabular classifiers, unconstrained recurrent/attention networks, "
        "and neuro-symbolic grey-box models—alongside two diagnostic anchors (constant base rate and the hidden "
        "simulator oracle). Table 3.3 summarizes each model's family, trainable parameter count, and how much "
        "history it needs during live prediction: O(1) constant-size memory versus O(L) replay over all L past tokens.",
    )
    add_table(
        doc,
        ["Model Name", "Family", "Params", "State Size", "Core Mechanism"],
        [
            ["constant (Anchor)", "Base-rate scalar", "1", "O(1)", "Predicts fixed main-train pass rate (0.54304)."],
            ["task_logistic", "Tabular linear", "20", "O(1)", "L2 logistic regression on 19 task features only (no history)."],
            ["history_logistic", "Tabular linear", "54", "O(1)", "L2 logistic regression on 53 features (19 task + 34 summary)."],
            ["history_boosting", "Gradient-boosted trees", "Tree ensemble", "O(1)", "HistGradientBoosting (15/31 leaves, 120/160 trees) on 53 dims."],
            ["time_input_gru", "Unconstrained GRU", "9,633", "O(1) (32-float)", "32-dim GRUCell receiving 47-dim token + ln(1 + Delta t) and 2-layer SiLU readout."],
            ["time_decay_gru", "Decaying GRU", "10,689", "O(1) (32-float)", "32-dim GRUCell whose hidden state decays exponentially toward learned prior h_0."],
            ["time_aware_transformer", "Causal Self-Attention", "11,845", "O(L) history", "4-head attention (32-dim) with -softplus(tau_h)*ln(1 + Delta t) time bias."],
            ["kt_decay_gru", "Grey-box (no GRU ctx)", "447", "O(1) (6-float)", "2-channel HLR + Power-Law + Permastore floor + 3PL readout without context vector."],
            ["monotonic_gru (Ours)", "Neuro-Symbolic GRU", "10,223", "O(1) (32-float)", "6-channel HLR + Power-Law + Permastore + 18-dim GRU context + 3PL readout."],
            ["monotonic_transformer", "Neuro-Symbolic Attn.", "10,208", "O(L) history", "Same grey-box state & 3PL head, replacing the 18-dim GRUCell with 3-head attention."],
            ["oracle (Ceiling)", "Simulator ceiling", "N/A", "Hidden truth", "Gauss-Hermite integration over true hidden simulator strength and demand."],
        ],
        caption="Table 3.3 — Nine benchmark memory models and two anchors evaluated in Tier 1.",
        col_widths=[1.50, 1.25, 0.90, 1.00, 2.15],
        right_cols={2},
    )

    add_h3(doc, "3.3.3 How monotonic_gru Works in Plain English")
    add_p(
        doc,
        f"Unlike black-box neural networks, monotonic_gru (3/model/memory/recurrent.py) is a neuro-symbolic "
        f"'grey-box' model: it keeps a compact 32-float Markov state per learner-skill pair where 14 numbers have "
        f"explicit psychological meaning and 18 numbers form a small auxiliary GRU context vector {cite('cho2014gru')}:",
    )
    add_eq_block(
        doc,
        "State_t = [ k_t (6 channels),  b_t (6 channels),  n_{+,t} (1),  n_{-,t} (1),  c_t (18 GRU context) ]  (32 floats total)",
        label="3.4",
    )
    add_p(
        doc,
        "Each step follows five transparent safeguards:",
    )
    add_bullet(
        doc,
        "1. Timing De-Confounding (_clean_token): ",
        "Before reading a past event token, the four response-speed fields (indices 35 to 38) are multiplied by "
        "full_success (0 or 1). Thus, a fast wrong guess has its timing zeroed out so speed can never inflate "
        "mastery.",
    )
    add_bullet(
        doc,
        f"2. HLR Half-Life and FSRS Power-Law Decay {cite_many('settles2016hlr', 'wickelgren1974', 'rubin1996', 'pavlik2005actr', 'ye2022fsrs')}: ",
        "For each of the 6 skill channels m = 1..6, half-life h_{t,m} (clamped to [5.0, 365.0] days) rises with "
        "positive retrieval counter n_{+,t}, shrinks with negative counter n_{-,t}, and receives a bounded context "
        "adjustment 0.4 * tanh(w_h^T c_t). Retention over Delta t days uses a power-law curve R(Delta t, h_{t,m}) = "
        "(1 + kappa_m * (Delta t / h_{t,m}))^(-w_m) with shape w_m > 0.5 and kappa_m = 2^(1/w_m) - 1, ensuring "
        "R(0, h) = 1, R(h, h) = 0.5, and a realistic long-term tail.",
    )
    add_bullet(
        doc,
        f"3. Bahrick Consolidated Permastore Floor b_t {cite_many('bahrick1984permastore', 'mozer2016dash')}: ",
        "Active skill k_t is split into a durable floor b~_t = min(k_t, b_t) and transient fluency (k_t - b~_t). "
        "Because the permastore drift rho = 0.12 * sigmoid(rho~) <= 0.12 and decays slowly at 4 * h_t, the floor "
        "retains at least 88% stability while transient fluency decays at h_t. Crucially, because both parts only "
        "shrink as Delta t increases, predicted retention can never drift upward during inactivity.",
    )
    add_bullet(
        doc,
        f"4. Outcome-Signed Bounded Updates {cite('corbett1995bkt')}: ",
        "Only resolved attempts trigger skill updates. Wrong or partial answers apply a bounded negative gate "
        "g_- <= 0.45, while positive scores apply a bounded gain gate g_+ <= 0.45 (weighted higher for unaided "
        "full successes than assisted or partial attempts), with gamma in [0.50, 0.75] consolidating gains into b_t.",
    )
    add_bullet(
        doc,
        f"5. Birnbaum 3PL Psychometric Readout {cite('birnbaum1968irt')}: ",
        "To predict pass probability on a target task, TaskReadoutHead takes a strictly positive weighted average "
        "of logit(k_m), subtracts task demand (enforcing Easy < Medium < Hard by at least +0.05 per step), scales "
        "by a positive slope alpha >= 0.4, and adds a format guessing floor g_fmt in (0, 0.35): "
        "P(pass) = g_fmt + (1 - g_fmt) * sigmoid(alpha * (theta_skill - d_task)).",
    )

    add_p(
        doc,
        "All neural sequence models are trained for 35 epochs (batch size = 64, AdamW lr = 0.003, weight_decay = "
        "0.001, grad clip = 5.0, OneCycleLR, patience = 8 after 75% epochs) and selected by 0.5 * PracticeLL_val "
        "+ 0.5 * DelayedMeanLL_val only if every seed passes all 6 checks of our Behavioral Sanity Suite "
        "(3/model/memory/behavioral.py): (1) zero_score_decreases_all_speeds (0% fails drop predictions at 5s, "
        "6.426s, 60s, 180s, and missing time), (2) full_success_increases_all_speeds, "
        "(3) assistance_and_outcome_ordering (Unaided 100% > Assisted 3-hint 100% > Unaided 0%), "
        "(4) difficulty_monotonicity_all_formats (Easy > Medium > Hard across all 5 formats), "
        "(5) non_increasing_forgetting_after_pass_and_fail (0–60d curves never rise, and 2-pass 30d/60d retention "
        "stays above cold-start and >= 60% of Day 0), and (6) user_conditionals_trace_behaves_sensibly (replaying "
        "the exact 6.426s 0% -> 3 hints -> 2.27s 0% trace cleanly).",
    )

    # =========================================================================
    # 3.4 CLOSED-LOOP REINFORCEMENT LEARNING METHODOLOGY
    # =========================================================================
    add_h2(doc, "3.4 Closed-Loop Reinforcement Learning Methodology (Gated Option B PPO)")

    add_p(
        doc,
        f"A memory model alone only watches passively—it predicts what a student will remember, but does not decide "
        f"what to teach next. To turn memory forecasts into an active tutor under a fixed study budget B = 32.0 "
        f"(split into sessions of SESSION_BUDGET = 8.0, separated by 1.5-day breaks in main/power_law or 4.0-day "
        f"breaks in long_gap), we formulate teaching as a Markov Decision Process solved with Proximal Policy "
        f"Optimization (PPO) {cite('schulman2017ppo')} and Generalized Advantage Estimation (GAE) "
        f"{cite('schulman2016gae')} in 3/ppo/env.py.",
    )

    add_h3(doc, "3.4.1 Eight-Action Gated Option B Menu")
    add_p(
        doc,
        "In our 'Option B' design, teaching actions (concept explanations, worked examples, and self-explanation "
        "prompts) are first-class steps that return control to PPO rather than hard-coded rules. However, if "
        "passive lessons were always available, an unconstrained agent could spam cheap lessons without testing "
        "the student. Gated Option B solves this with strict action masking (invalid actions are masked to -1e9): "
        "after explain_concept or show_worked_example, the environment activates a primed_readiness state and "
        "masks out everything except a follow-up practice question (actions 0, 1, or 2), which also boosts "
        "learning consolidation (1.45x dose multiplier). Meanwhile, choosing give_same_difficulty_question when "
        "unverified_assisted = True triggers a no-hints verification check to confirm real mastery. Table 3.4 "
        "lists all eight actions, their budget costs, and their gating rules.",
    )
    add_table(
        doc,
        ["ID", "Action Name", "Cost", "Gating Condition & Pedagogical Effect"],
        [
            ["0", "give_easier_question", "1.0", "Unmasked when current difficulty > easy; steps difficulty down by 1 and runs a practice question."],
            ["1", "give_same_difficulty_question", "1.0", "Always unmasked; if unverified_assisted is True, runs an unaided verification check (allow_help=False)."],
            ["2", "give_harder_question", "1.0", "Unmasked when current difficulty < hard; steps difficulty up by 1 and runs a practice question."],
            ["3", "explain_concept", "0.6", "Unmasked only when primed_mode is None; delivers concept lesson, sets primed_readiness, forces question next."],
            ["4", "show_worked_example", "0.7", "Unmasked only when primed_mode is None; shows worked example, sets primed_readiness, forces question next."],
            ["5", "request_self_explanation", "0.9", "Unmasked after a question attempt on current topic; graded micro-assessment consolidating b_t and n_+."],
            ["6", "review_prerequisite", "1.0", "Unmasked only if a prerequisite (p_30d < 0.56 or b_t < 0.40) or visited skill (< 0.52 or < 0.36) is weak/decayed."],
            ["7", "advance_next_topic", "0.2", "Unmasked after >= 1 attempt on current skill; moves to next unvisited skill (or lowest-P_30d revisited skill)."],
        ],
        caption="Table 3.4 — Eight-action Gated Option B menu in AdaptiveTutorEnv (3/ppo/env.py).",
        col_widths=[0.55, 1.85, 0.70, 3.70],
        right_cols={0, 2},
    )

    add_h3(doc, "3.4.2 Policy State s_t (36 Dims) vs. Ablation 3 (23 Dims)")
    add_p(
        doc,
        "At each step t, the frozen v2.2 monotonic_gru tracker produces a 36-dimensional state vector s_t "
        "(3/ppo/env.py:429-457) summarizing both short-term performance and 30-day durability:",
    )
    add_bullet(
        doc,
        "Current-Skill Memory Forecasts (8 dims): ",
        "p_easy, p_med, p_hard, p_7d, p_30d, fragility = max(0, p_med - p_30d), permastore floor b_t, and active "
        "strength k_t. Notice how fragility exposes the 'illusion of competence': when short-term k_t is high "
        "but b_t is low, p_med - p_30d is large, signaling PPO to consolidate via self-explanation or unaided checks.",
    )
    add_bullet(
        doc,
        "Recent Interaction, Priming & Difficulty (11 dims): ",
        "unverified_assisted flag, last_score, last_hints/3, last_leakage (mapping continuous hint leakage l_t in "
        "[0, 1] into the observation), attempts_on_topic/10, awaiting_question, primed_concept, primed_example, "
        "and 3 difficulty one-hots.",
    )
    add_bullet(
        doc,
        "Prerequisite, Budget & Curriculum Map (17 dims): ",
        "minimum prerequisite p_med, minimum prerequisite floor b_t, remaining budget ratio, min(p_30d) and "
        "mean(p_30d) across all 6 skills, individual p_30d for all 6 skills, and 6 active-topic one-hots.",
    )
    add_p(
        doc,
        "To test whether PPO genuinely relies on monotonic_gru's long-term forecasts rather than just short-term "
        "scores, Ablation 3 (ppo_no_longterm_state, 23 dims) removes all 13 long-term dimensions (p_7d, p_30d, "
        "fragility, b_t, prereq_min_floor_b, min_p30_all, mean_p30_all, and p30_all[0..5]) and also replaces "
        "Action 6/7 masking and target routing with short-term p_med only (p_med < 0.60 for prerequisites, "
        "< 0.55 for visited skills) so zero long-term signal leaks through masks or routing.",
    )

    add_h3(doc, "3.4.3 Reward Function r_t, Training Setup, Baselines & Behavioral Checks")
    add_p(
        doc,
        "Let Delta mean_P_30d be the step-to-step change in average 30-day pass probability across all 6 skills, "
        "Delta min_s P_30d(s) be the change in the weakest skill's 30-day forecast (so hard topics like "
        "function_returns are not neglected), and Delta mean_b_t be the change in average permastore floor. "
        "Indicators I(cleared_unverified) and I(premature_advance) reward verifying assisted mastery (+0.08) and "
        "penalize abandoning fragile/unverified skills (-0.04), while I(done) adds a terminal anchor equal to 10 "
        "times the true simulated delayed_mean across Day 1, Day 7, and Day 30 (3/ppo/env.py:837-856):",
    )
    add_eq_block(
        doc,
        "r_t = 2.0*Delta(mean P_30d) + 1.5*Delta(min_s P_30d(s)) + 1.5*Delta(mean b_t) + 0.08*I(cleared_unverified) - 0.04*I(premature_advance) + 10.0*delayed_mean*I(done)",
        label="3.5",
    )
    add_p(
        doc,
        "Both actor and critic use a MaskedActorCritic 2-hidden-layer 64-unit SiLU MLP with orthogonal initialization. "
        "PPO trains for 100 updates x 24 episodes/update using gamma = 0.99, GAE lambda = 0.95, clip epsilon = "
        "0.20, value loss weight = 0.5, 4 PPO epochs, minibatch = 128, AdamW lr = 2.5e-3 (linearly decayed to "
        "0.15 * lr), and entropy coefficient decaying linearly from 0.020 to 0.003 (3/ppo/train_ppo.py:151). "
        "We compare delayed_retention_ppo against four matched-budget baselines: (1) threshold_heuristic (a strong "
        "rule-based tutor), (2) myopic_rl_bandit (same PPO network with gamma = 0.0, lambda = 0.0, optimizing only "
        "immediate step score), (3) fixed_curriculum (round-robin medium practice), and (4) random_valid. Every "
        "trained PPO seed is also audited by a 5-check PPO behavioral sanity suite (3/ppo/train_ppo.py:248-279): "
        "(1) question turns (including review_prerequisite) >= 40%, (2) no action spam <= 32% for self-explanation, "
        "prerequisite review, or passive instruction, (3) advance_next_topic in [4.5%, 25%], (4) floor b_t > "
        "threshold_heuristic, and (5) delayed expected mean beats all four baselines.",
    )

    # =========================================================================
    # 3.5 HYBRID RAG, QUESTION GENERATION, AND SAFETY GUARDRAIL METHODOLOGY
    # =========================================================================
    add_h2(doc, "3.5 Hybrid RAG, Question Generation, and Safety Guardrail Methodology")

    add_p(
        doc,
        f"In the live web application (3/adaptive_web/), the LLM acts as a bounded 'realizer'—turning PPO's chosen "
        f"skill and action into clear Python lessons, fresh practice items, and Socratic doubt replies—while "
        f"never modifying memory weights directly. Methodologically, this layer rests on three safeguards "
        f"(code modules and API endpoints are detailed in Chapter 4; retrieval and live-LLM evaluation design "
        f"appear in Chapter 5):",
    )
    add_bullet(
        doc,
        f"1. Three-Corpus Hybrid RAG with Reciprocal Rank Fusion (RRF k = 60) {cite_many('lewis2020rag', 'robertson2009bm25', 'cormack2009rrf')}: ",
        "Retrieval searches three SQLite-backed corpora: curriculum (24 chunks from 6 Python Markdown guides), "
        "v22_exemplar (all 2,160 authored v2.2 tasks, pre-filtered by skill, difficulty, and format), and episodic "
        "(strictly learner-scoped past mistakes, revealed solutions, and doubts so Student A's data never leaks to "
        "Student B). Queries pass through domain-synonym expansion, SQLite FTS5 BM25 keyword search, TF-IDF "
        "n-gram (1, 2) cosine similarity (max 8,192 features), Reciprocal Rank Fusion (score(d) = sum_r 1 / "
        "(60 + rank_r(d))), and metadata reranking. Context budgets are fixed at k = 4 chunks for Doubt Tutor "
        "replies and k = 3 exemplars + k = 2 curriculum/episodic chunks for Question Generation. Dense bi-encoder "
        "arms (Dense and BM25 + Dense) are supported via the DenseEncoder interface and remain pending pretrained "
        "encoder weights.",
    )
    add_bullet(
        doc,
        "2. Difficulty-Locked Question Generation, 1-Shot Repair & Bounded AST Sandbox: ",
        "Question prompts inject explicit complexity bounds from _DIFFICULTY_GUARDRAILS (easy: 1–5 lines, "
        "15–35s; medium: 4–10 lines, 35–60s; hard: 7–16 lines, 60–110s), with easy additionally enforced by a "
        "hard post-generation check at <= 7 lines (max_lines + 2) and reference_seconds clamped to [10.0, 180.0]. "
        "Generated items must also pass an anti-duplication filter (text similarity <= 0.85) and pass "
        "deterministic schema and code-test checks, with a 1-shot LLM schema-repair retry before falling back to "
        "a fresh v2.2 item. Submitted student code is graded in a bounded Python Abstract Syntax Tree (AST) "
        "interpreter permitting only safe def solve(x): bodies (blocking exec, eval, imports, attributes, "
        "recursion, while loops, and comprehensions; capped at 8,000 chars, 1,200 AST nodes, and 10,000 ops per "
        "case) and marking unsupported syntax as unresolved rather than wrong.",
    )
    add_bullet(
        doc,
        f"3. Mastery-Adaptive Doubt Tutor, EduGuard & Separated Grading {cite_many('typesafe1', 'typesafe2')}: ",
        "The Doubt Tutor reads monotonic_gru's state to classify each skill as unassessed, lacking (p_med < 70% "
        "or observed pass rate < 70%), fragile (p_med - p_30d > 0.15), or mastered, adjusting explanation depth "
        "accordingly. Before replying, EduGuard blocks answer-fishing phrases, detects pasted unpassed quiz items "
        "(even across skills), scans LLM drafts for leaked answers/blanks/options, and substitutes a Socratic hint "
        "when triggered. Finally, self-explanation grading computes score in [0, 1] strictly from rubric concept "
        "coverage separately from grader_confidence in [0.05, 0.99], so confident-sounding fluff without core "
        "concepts never earns false mastery.",
    )
