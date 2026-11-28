"""Frozen Chapter 5 — Evaluation Protocol (verbatim from 3/build_experiments_docx.py lines 304-400)."""
from report_sections.style import (
    add_bullet,
    add_h1,
    add_h2,
    add_mixed_p,
    add_p,
    add_table,
)


def build(doc, ctx=None):
    # =========================================================================
    # CHAPTER 5 — EVALUATION PROTOCOL (PLAIN ENGLISH + SCIENTIFICALLY ACCURATE)
    # =========================================================================
    add_h1(doc, "Chapter 5 — Evaluation Protocol", page_break=True)

    add_h2(doc, "5.1 What We Are Testing and Our Main Goal")
    add_p(
        doc,
        "Imagine a student learning Python with an AI tutor. They only have a small amount of study time—a fixed "
        "study budget of B = 32.0 points (about four short lessons spaced days apart). Anyone can look smart right "
        "after reading an answer with hints. The real test of a good teacher is: when the student comes back weeks "
        "later with zero hints, do they still remember how to solve the problem?",
    )
    add_mixed_p(doc, [
        ("Main Outcome (Primary Metric): ", True, False),
        ("After studying ends, we give the simulated student fresh tests without any hints at ", False, False),
        ("1 day, 7 days, and 30 days later", True, False),
        (" (18 questions per day across 6 Python skills and 3 difficulty levels = 54 questions per student). "
         "Our main score is the ", False, False),
        ("mean delayed unaided pass rate", True, False),
        (" (the average of Day 1, Day 7, and Day 30), and we highlight ", False, False),
        ("Day-30 retention", True, False),
        (" separately because 30 days tests true lasting memory rather than short-term cramming. We compare our AI "
         "teacher against a standard rule-based teacher (Threshold Heuristic) using the exact same study budget and "
         "the exact same simulated students.", False, False),
    ])
    add_mixed_p(doc, [
        ("Important Scope Note: ", True, False),
        ("Tier 1 and Tier 4 test the memory model and teaching policy inside a computer simulator. They prove our "
         "system works in simulation, not yet in a live human classroom (which belongs to a future IRB-approved Tier 5 study).", False, True),
    ])

    add_h2(doc, "5.2 Simple Guide to Every Symbol, Variable, and Metric")
    add_p(
        doc,
        "Table 5.1 explains every symbol, variable, and test score in plain, everyday words so anyone reading this "
        "report for the first time can follow every number.",
    )
    add_table(
        doc,
        ["Symbol / Term", "Plain-English Meaning (What It Is)", "Good Value"],
        [
            ["N", "Number of students, questions, or test prompts in an experiment.", "Larger = more reliable"],
            ["B (Budget = 32.0)", "Total study coins a tutor may spend (1.0 per question, 0.6–0.9 per hint/lesson).", "Equal across policies"],
            ["k (Context Slots)", "Number of text chunks the search engine hands to the AI tutor (k=4 chat; k=3 examples + k=2 lessons for question making).", "Matched per arm"],
            ["b_t (Permastore Floor)", "Durable, long-lasting knowledge (from 0 to 1) that stays even after a 30-day break.", "Higher is better"],
            ["k_t (Active Strength)", "Short-term + long-term skill strength (from 0 to 1) right at the current time.", "Higher is better"],
            ["s_t (Policy State)", "The 36-number dashboard summary the RL teacher reads before picking the next action.", "36 dims (23 in Abl. 3)"],
            ["Log Loss (LL)", "Scores how accurate and honest a probability guess is. Punishes confident wrong guesses (0.693 = pure 50/50 coin flip).", "Lower is better"],
            ["ROC-AUC", "Pick one passing student and one failing student: how often did the model give the passing student a higher score? (0.5 = coin flip, 1.0 = perfect).", "Higher is better"],
            ["Brier Score", "Average squared gap between the model's 0–1 guess and the real 0/1 pass outcome.", "Lower is better"],
            ["ECE-10 (Calibration)", "Splits guesses into 10 buckets (0–10%, ..., 90–100%). Measures if a '70% guess' really passes ~70% of the time.", "Lower is better (near 0)"],
            ["Signal %", "How much of the gap between naive base-rate guessing (0%) and the hidden simulator Oracle ceiling (100%) a model captures.", "Higher is better"],
            ["6 Behavioral Checks", "Common-sense rules: getting 0% wrong must drop predictions; 0–60d forgetting curves must never drift upward without study.", "Must be 6 / 6"],
            ["Delayed Mean / Day-30", "Average unaided pass rate at 1, 7, and 30 days after study; Day-30 checks lasting memory one month later.", "Higher is better"],
            ["Lowest-Skill Day-30", "The Day-30 pass rate on the weakest of the 6 Python skills (makes sure hard topics like function_returns are not ignored).", "Higher is better"],
            ["Delta (Δ) & 95% CI", "Difference between two systems on the exact same students/queries, with a 95% Bootstrap interval from 2,000 resamples.", "CI excludes 0 = consistent"],
            ["nDCG@k / Recall@k", "Search quality: Recall@k checks if the right chunk is in the top k; nDCG@k rewards putting it at rank #1.", "Higher is better (max 1)"],
            ["p95 Latency (ms)", "Search speed: 95 out of 100 queries finish faster than this time in milliseconds.", "Lower is faster"],
            ["Meta Match / Violations", "Exemplar Meta Match checks exact skill/difficulty/format match (target 100%); Cross-Learner Violations checks student data leaks (must be 0).", "100% match / 0 leaks"],
            ["Leakage / Over-Refusal", "In EduGuard: Leakage = tutor gives away the answer on trick prompts; Over-Refusal = tutor wrongly blocks a genuine question.", "Both near 0%"],
            ["Usable Question Rate", "Share of AI-generated questions that are fresh, valid, solvable, and match the requested skill, format, and difficulty.", "Higher is better"],
            ["MAE (Self-Explanation)", "Mean Absolute Error between the automated grader's 0–1 score and human teacher scores.", "Lower is better (near 0)"],
        ],
        caption="Table 5.1 — Plain-English dictionary of every symbol, variable, and metric used in Chapters 5 and 6.",
        col_widths=[1.45, 3.65, 1.7],
    )

    add_h2(doc, "5.3 The Four-Tier Test Plan and Ablations")
    add_p(
        doc,
        "Because our tutor has four connected parts—a Memory Brain, a Search Engine, an LLM Teacher/Grader, and an "
        "RL Policy Planner—we test each part on its own and test them together in four tiers, plus three focused "
        "ablation (remove-one-piece) tests:",
    )
    add_table(
        doc,
        ["Tier / Test", "What Part It Tests", "How It Is Tested & Seeding", "Status in This Report"],
        [
            ["Tier 1\n(Memory)", "9 memory models + Constant & Oracle anchors", "Predicts first unaided pass/fail on 200 held-out students per cohort (main, power-law, long-gap). Retrospective: 3 seeds; Pre-Registered: 5 fresh seeds (20261201–20261205).", "Executed\n(Complete)"],
            ["Tier 2\n(Retrieval)", "BM25 alone vs. Deployed BM25+TF-IDF vs. Dense vs. BM25+Dense", "Tests speed (p95), metadata match, cross-learner isolation, and ranking on 66 dev + 132 held-out test queries at exact slots (k=4 chat; k=3 + k=2 question gen).", "Offline Diagnostics Done;\nDense & Human Labels Pending"],
            ["Tier 3a–3c\n(LLM & Grading)", "Doubt Tutor, EduGuard, Question Maker (180 items), Self-Explanation Grader (120 items)", "Blinded 2-reviewer human grading with live LLM failover turned off. Blind prompt/review packs exported.", "Blind Packs Ready;\nLive LLM & Human Review Pending"],
            ["Tier 4\n(RL Policy)", "Delayed PPO vs. Threshold Heuristic, Myopic Bandit, Fixed, Random", "Retrospective: 3 seeds (N=200). Pre-Registered: 10 fresh seeds (20261201–20261210) tested on 400 brand-new students (40000–40399) across 3 cohorts & 4 budgets.", "Executed\n(Complete)"],
            ["Ablation 1\n(Retrieval)", "Deployed BM25+TF-IDF vs. BM25+Dense (fixed LLM)", "Checks if better search improves live tutor Python correctness and question usability.", "Pack Ready (Pending Live LLM)"],
            ["Ablation 2\n(EduGuard)", "EduGuard ON vs. OFF on 120 paired prompts", "Measures answer leakage vs. over-refusal on identical questions and DB snapshots.", "Pack Ready (Pending Live LLM)"],
            ["Ablation 3\n(PPO State)", "Full 36-dim PPO vs. 23-dim No-Long-Term PPO", "Retrains 10 seeds from scratch after removing all 13 long-term (7d/30d/floor b_t) features and routing/mask leaks.", "Executed\n(Complete)"],
            ["Tier 5 (Opt.)\n(Human Pilot)", "Real classroom students (optional future step)", "Unaided pretest -> study session -> 7-day delayed posttest, kept strictly separate from simulation.", "Optional Outline Only"],
        ],
        caption="Table 5.2 — Summary of the four evaluation tiers, three ablations, and optional human pilot.",
        col_widths=[1.05, 1.85, 2.55, 1.35],
    )

    add_h2(doc, "5.4 Fair Testing Rules")
    add_bullet(doc, "Same Students & Same Prompts (Paired Testing): ", "Every competing model is tested on the exact same simulated students or exact same questions so differences come from the model, not lucky draws.")
    add_bullet(doc, "Honest Labels (Retrospective vs. Pre-Registered): ", "Our earlier 3-seed runs are clearly labeled Retrospective. Fresh 5-seed (Tier 1) and 10-seed (Tier 4, 400 new students) runs with locked rules are labeled Pre-Registered.")
    add_bullet(doc, "No Cherry-Picking & No Software Test Mixing: ", "We report all 3 test cohorts (main, power-law forgetting, and 4-day long-gap breaks) and all 6 Python skills. Software engineering unit tests are never mixed into research metrics.")
