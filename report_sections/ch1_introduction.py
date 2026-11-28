"""Chapter 1 — Introduction builder for the SNU B.Tech CSE thesis report."""
from .references import cite, cite_many
from .style import add_bullet, add_h1, add_h2, add_mixed_p, add_p


def build(doc, ctx=None):
    """Append Chapter 1 — Introduction to `doc`."""
    ctx = ctx if ctx is not None else {}
    page_break = not bool(ctx.get("arabic_section_started", False))
    add_h1(doc, "Chapter 1 \u2014 Introduction", page_break=page_break)

    # -------------------------------------------------------------------------
    # 1.1 Motivation and Problem Statement
    # -------------------------------------------------------------------------
    add_h2(doc, "1.1 Motivation and Problem Statement")
    add_p(
        doc,
        "Learning Python programming requires more than solving a practice question right after reading "
        "a hint. When a student gets help on a loop or function-return exercise today, their short-term "
        "score often looks strong. Yet when that same student returns days or weeks later and faces a "
        "fresh problem without any hints, much of that skill can fade away. In educational psychology, "
        "this gap between immediate fluency and lasting mastery is known as the illusion of competence. "
        "Because real learners have limited study time\u2014modeled in our work as a fixed study budget "
        "B = 32.0 units across spaced sessions\u2014an automated tutor cannot afford to waste turns drilling "
        "topics the student already knows well, nor can it jump ahead before foundational skills stick.",
    )
    add_p(
        doc,
        f"Conventional tutoring tools fall short for two clear reasons. First, simple accuracy counters "
        f"ignore time gaps between sessions and treat aided wins the same as unaided mastery. Standard "
        f"knowledge tracing (KT)\u2014the task of estimating whether a student can solve a new problem "
        f"from their past learning history {cite_many('corbett1995bkt', 'piech2015dkt')}\u2014often focuses "
        f"on next-step prediction rather than long-term forgetting. Second, rule-based curricula and "
        f"unconstrained large language model (LLM) chatbots act myopically: a chatbot answers whatever "
        f"the student asks (sometimes accidentally giving away the answer), while a fixed rule-based "
        f"teacher repeats rigid thresholds without planning for how today\u2019s teaching action affects "
        f"retention 30 days later {cite_many('ruan2024', 'nam2024')}.",
    )
    add_p(
        doc,
        f"To solve both problems together, a tutor needs three simple, connected capabilities. First, it "
        f"needs a memory model that tracks two quantities for each skill: a short-term active strength k_t "
        f"and a durable permastore floor b_t (long-lasting knowledge that survives long breaks "
        f"{cite('bahrick1984permastore')}), along with a memory half-life h_t (the number of days before "
        f"recall drops halfway without practice {cite('settles2016hlr')}). Second, it needs a teaching "
        f"policy trained with reinforcement learning (RL)\u2014specifically Proximal Policy Optimization "
        f"(PPO) {cite('schulman2017ppo')}, an algorithm that improves step-by-step teaching decisions "
        f"toward long-term rewards without unstable jumps. Third, it needs grounded language generation "
        f"via Retrieval-Augmented Generation (RAG) {cite('lewis2020rag')}, which looks up verified Python "
        f"lessons and past examples before asking an LLM to explain a concept or write a new question.",
    )

    # -------------------------------------------------------------------------
    # 1.2 Evolution of Our Research Across Three Stages
    # -------------------------------------------------------------------------
    add_h2(doc, "1.2 Evolution of Our Research Across Three Stages")
    add_p(
        doc,
        "Rather than starting with a complex system all at once, our project progressed across three "
        "distinct empirical stages, where the limitations found in each stage directly shaped the design "
        "of the next:",
    )
    add_bullet(
        doc,
        "Stage 1 \u2014 Qualitative LLM Style Adaptation (Directory 1/): ",
        f"We began by fine-tuning Meta-Llama-3.1-8B-Instruct-4bit locally with Low-Rank Adaptation "
        f"(LoRA) {cite('hu2022lora')} on two small synthetic dialogue datasets: a beginner/struggling "
        f"persona (75 examples: 63 train, 12 validation) and an advanced/fast persona (43 examples: "
        f"36 train, 7 validation). Qualitative side-by-side comparisons showed that LoRA reliably changed "
        f"tone and brevity, but tone adaptation alone could not track whether a student actually knew "
        f"Python or decide which topic to review next. Maintaining per-student adapter weights also "
        f"scaled poorly, leading us to drop per-student LoRA weights in favor of a shared structured "
        f"memory model and an external RL policy.",
    )
    add_bullet(
        doc,
        "Stage 2 \u2014 Real-Data Memory Comparison on ASSISTments 2012\u20132013 (Directory 2/): ",
        f"Next, we tested whether recurrent or attention-based sequence models track student responses "
        f"better on 6,103,356 retained tutor interactions (from 6,123,270 scanned rows across 46,674 "
        f"students) in the public ASSISTments 2012\u20132013 dataset {cite('assistments2012')}. Across "
        f"606,086 held-out test interactions, a recurrent LSTM-DKT model {cite('piech2015dkt')} achieved "
        f"0.50280 log loss (0.7852 ROC-AUC, 75.58% accuracy), outperforming a self-attentive SAKT "
        f"Transformer {cite('pandey2019sakt')} (0.52305 log loss, 0.7698 ROC-AUC, 74.41% accuracy). "
        f"However, 55.713% of rows lacked skill_id tags, forcing reliance on 118,232 item/skill ID "
        f"embeddings, and the logs contained neither 1/7/30-day unaided delayed tests nor controlled "
        f"teaching actions for RL training.",
    )
    add_bullet(
        doc,
        "Stage 3 \u2014 Custom Python v1\u2013v2.2 Pipeline, 9-Model Benchmark, 8-Action PPO, and Web Tutor (Directory 3/): ",
        "To study long-term Python retention cleanly, we built a custom Python curriculum of 6 canonical "
        "skills and 2,160 tasks across 5 formats, evolving through releases v1, v2, v2.1, and v2.2. When "
        "interactive replay in our local Recall dashboard revealed that v2.1 models could raise predicted "
        "mastery after a fast 0% failure, we designed a 6-check behavioral sanity gate and upgraded the "
        "simulator to v2.2 (zero dose on unaided zero scores, a 35% rapid-guessing mixture, and Half-Life "
        "Regression with a Consolidated Permastore floor). We then benchmarked 9 learner-memory models, "
        "trained an 8-action Gated Option B PPO teacher using the winning grey-box memory state, and built "
        "the full interactive 3/adaptive_web Python tutor with Hybrid RAG and safety guardrails.",
    )

    # -------------------------------------------------------------------------
    # 1.3 System Overview and Key Contributions
    # -------------------------------------------------------------------------
    add_h2(doc, "1.3 System Overview and Key Contributions")
    add_p(
        doc,
        "Our final architecture separates what the system remembers (the Brain), what pedagogical move it "
        "chooses (the Teacher), and how it writes Python explanations or questions (the Voice). This "
        "decoupling keeps the memory tracker fast, interpretable, and independent of raw LLM text:",
    )
    add_bullet(
        doc,
        "1. The Brain \u2014 Grey-Box Monotonic Memory Tracker (monotonic_gru): ",
        f"Each (learner, skill) pair maintains its own compact 32-float Markov state (a fixed-size summary updated in constant O(1) time without replaying old logs) "
        f"using only structured evidence (47-dim interaction tokens and 19-dim task descriptors, with zero "
        f"text, code, or student-ID embeddings). Combining Duolingo Half-Life Regression {cite('settles2016hlr')}, "
        f"Wickelgren/FSRS power-law forgetting {cite_many('wickelgren1974', 'ye2022fsrs')}, Bahrick\u2019s "
        f"permastore floor b_t {cite('bahrick1984permastore')}, and a monotonic 3-parameter Item Response "
        f"Theory (3PL IRT) readout {cite('birnbaum1968irt')}, monotonic_gru passes all 6/6 behavioral "
        f"sanity checks across every seed and wins the 4-day long-gap delayed test (0.57190 log loss).",
    )
    add_bullet(
        doc,
        "2. The Teacher \u2014 8-Action Gated Option B PPO (delayed_retention_ppo): ",
        f"Before each turn, the RL policy reads a 36-dimensional state vector s_t containing immediate "
        f"predictions (easy, medium, hard), 7-day and 30-day forecasts, fragility (P_now \u2212 P_30d), "
        f"permastore floor b_t, prerequisite readiness, and remaining study budget. Valid-action masking "
        f"prevents spamming passive explanations without follow-up questions. Across 10 pre-registered "
        f"seeds and N = 400 fresh simulated learners at budget B = 32.0, delayed_retention_ppo achieves "
        f"62.77% delayed mean and 59.66% Day-30 unaided pass rate\u2014beating threshold_heuristic by "
        f"+2.50 percentage points (pp) delayed mean and +5.69 pp at Day 30. Ablation 3 proves that hiding the 13 long-term "
        f"memory features reduces Day-30 retention to 58.82% and roughly triples seed spread.",
    )
    add_bullet(
        doc,
        "3. The Voice & Guardrails \u2014 Hybrid RAG, AST Sandbox, and EduGuard (3/adaptive_web): ",
        f"A Hybrid RAG engine merges SQLite FTS5 BM25 {cite('robertson2009bm25')} and TF-IDF cosine "
        f"similarity via Reciprocal Rank Fusion (RRF, k = 60) {cite('cormack2009rrf')} over 24 curriculum "
        f"chunks, 2,160 v2.2 exemplars, and learner-scoped episodic records in under 5.2 ms 95th-percentile latency (p95) with "
        f"100% metadata match and 0 cross-learner leaks. Difficulty-locked question generation pairs "
        f"1-shot schema repair with a bounded Python AST sandbox grader, while EduGuard blocks direct "
        f"solution leaks during learner doubt chats.",
    )

    # -------------------------------------------------------------------------
    # 1.4 Scope, Honesty Boundaries, and Thesis Organization
    # -------------------------------------------------------------------------
    add_h2(doc, "1.4 Scope, Honesty Boundaries, and Thesis Organization")
    add_p(
        doc,
        "To keep every scientific claim transparent and verifiable, this thesis enforces four strict "
        "honesty boundaries throughout all chapters:",
    )
    add_bullet(
        doc,
        "Simulation vs. Human Classroom Scope: ",
        "All Tier 1 memory metrics and Tier 4 RL policy gains are established on simulated learners in our "
        "validated v2.2 simulator and monotonic_gru-backed PPO environment. Effectiveness with real human "
        "students belongs to an optional future IRB-approved Tier 5 pilot (not run here).",
    )
    add_bullet(
        doc,
        "Retrospective vs. Pre-Registered Runs: ",
        "Earlier 3-seed development runs (N = 200 test learners in Tier 4) are clearly labeled "
        "Retrospective and separated from our locked Pre-Registered runs (5 fresh seeds in Tier 1; 10 "
        "fresh seeds and N = 400 new learners in Tier 4).",
    )
    add_bullet(
        doc,
        "Pending Live-LLM and Human Adjudication: ",
        "Offline retrieval diagnostics (BM25 alone and BM25 + TF-IDF) are complete, whereas dense "
        "retrieval weights (Tier 2), live-LLM downstream comparisons (Ablation 1), Doubt Tutor and "
        "EduGuard trials (Tier 3a / Ablation 2), fresh question generation (Tier 3b), and self-explanation "
        "grading against human rubrics (Tier 3c) remain pending with blinded prompt packs exported and "
        "zero fabricated scores.",
    )
    add_bullet(
        doc,
        "Strict Separation of Engineering Checks: ",
        "Software engineering unit and end-to-end tests verify code correctness during development and are "
        "never reported as research evaluation metrics.",
    )

    add_mixed_p(
        doc,
        [
            ("Thesis Organization: ", True, False),
            (
                "Chapter 2 reviews prior work in knowledge tracing, human memory modeling, RL tutoring, and "
                "grounded RAG systems. Chapter 3 details our datasets (from Stages 1\u20132 through v2.2), "
                "the 9-model memory benchmark, and the 8-action Gated Option B PPO formulation. Chapter 4 "
                "describes the software implementation across 3/model, 3/ppo, 3/dashboard, and "
                "3/adaptive_web. Chapter 5 presents the four-tier evaluation protocol and plain-English "
                "metric definitions, and Chapter 6 reports all completed empirical results alongside the "
                "prepared live-LLM packets. Chapter 7 concludes with system limitations, threats to "
                "validity, and next steps, followed by IEEE References.",
                False,
                False,
            ),
        ],
        before=6,
        after=8,
    )
