"""Chapter 7 — Limitations and Conclusion."""
from report_sections.references import cite, cite_many
from report_sections.style import add_bullet, add_h1, add_h2, add_p


def build(doc, ctx=None):
    """Append Chapter 7 — Limitations and Conclusion to `doc`."""
    add_h1(doc, "Chapter 7 — Limitations and Conclusion", page_break=True)

    add_p(
        doc,
        "This final chapter pulls together what we learned across all stages of the project, "
        "states honestly what has been proven versus what remains pending or restricted to "
        "simulation, and outlines the next steps needed for classroom deployment.",
    )

    # ------------------------------------------------------------------
    # 7.1 Summary of Verified Findings
    # ------------------------------------------------------------------
    add_h2(doc, "7.1 Summary of Verified Findings")

    add_p(
        doc,
        f"Across directories 1/, 2/, and 3/, our results build step by step from early exploration "
        f"to a pre-registered closed-loop tutoring system. First, Stage 1 (1/) showed that tuning "
        f"a language model's conversational tone with LoRA {cite('hu2022lora')} is not enough "
        f"without a structured model of what the student knows. Second, Stage 2 (2/) compared two "
        f"sequence models on 6,103,356 retained rows from the ASSISTments 2012-2013 benchmark "
        f"{cite('assistments2012')}: the recurrent LSTMDKT model {cite('piech2015dkt')} beat "
        f"SAKT-Transformer {cite('pandey2019sakt')} (0.50280 vs. 0.52305 test log loss; 0.7852 vs. "
        f"0.7698 ROC-AUC), motivating a recurrent state design while exposing the need for clean "
        "skill tags, elapsed time gaps, and delayed retention exams.",
    )
    add_p(
        doc,
        "Third, under the v2.2 Python curriculum and simulator (3/data_Set/v2.2/ and 3/model/), "
        "combining predictive accuracy with common-sense behavioral rules proved decisive. In "
        "the 5-seed Pre-Registered Tier 1 evaluation, both monotonic_gru and time_input_gru passed "
        "all 6/6 behavioral sanity checks across every seed, whereas time_decay_gru failed at "
        "5.0/6 and was rejected because its forgetting curves drifted upward after failures. "
        "monotonic_gru achieved tight seed-to-seed stability (0.55183 +/- 0.00011 validation "
        "log loss, tied at five decimal places with monotonic_transformer and nearly 3x tighter "
        "than time_input_gru at +/- 0.00032; ranked #1 in Retrospective 3-seed selection at "
        "0.55175 +/- 0.00025), "
        "achieved the lowest delayed log loss on the 4-day long_gap_test stress cohort "
        "(0.57190 +/- 0.00050, capturing 80.4% of the oracle delayed signal), and uniquely "
        "exposes an O(1) 32-float grey-box state containing the student's permastore floor b_t "
        f"and half-life h_t {cite_many('settles2016hlr', 'bahrick1984permastore', 'ye2022fsrs')}.",
    )
    add_p(
        doc,
        f"Fourth, in the Pre-Registered Tier 4 evaluation across 10 seeds on N = 400 fresh "
        f"simulated learners at study budget B = 32.0, our 8-action Gated Option B policy "
        f"(delayed_retention_ppo) {cite('schulman2017ppo')} reached 62.77% +/- 1.12% mean "
        "delayed unaided pass rate and 59.66% +/- 1.92% Day-30 pass rate on main_test, "
        "beating the rule-based threshold_heuristic (60.27% delayed mean, 53.97% Day-30) by "
        "+2.50 pp (95% CI [+2.12, +2.86]) on delayed mean and +5.69 pp (95% CI [+5.05, +6.30]) "
        "on Day-30 retention. PPO also won on power_law_test (+2.30 pp [+1.96, +2.65]) and "
        "long_gap_test (+2.94 pp [+2.56, +3.33]), and outperformed threshold_heuristic across "
        "all four matched study budgets (B = 16, 24, 32, and 40). Crucially, Ablation 3 "
        "(ppo_no_longterm_state, 23 dimensions with short-term p_med-only masking and routing) "
        "proved that hiding long-term memory channels drops main_test delayed mean from 62.77% "
        "to 61.76% (paired loss +1.02 pp [+0.87, +1.16]) and Day-30 from 59.66% to 58.82% "
        "(+0.84 pp [+0.61, +1.05]), while roughly tripling seed standard deviation from +/- 1.12% "
        "to +/- 3.28%.",
    )
    add_p(
        doc,
        f"Finally, offline Tier 2 diagnostics on 3/adaptive_web/rag.py confirmed that the "
        f"deployed Hybrid RAG engine (SQLite FTS5 BM25 {cite('robertson2009bm25')} + TF-IDF "
        f"cosine + Reciprocal Rank Fusion k = 60 {cite('cormack2009rrf')}) runs in under "
        "5.2 ms at the 95th percentile (5.053 ms on 192 test retrieval tasks; 5.197 ms on 96 dev "
        "tasks) with 100.0% exemplar metadata match, 0 BM25 underfills, and 0 cross-learner "
        "episodic leaks across both the test split and the dedicated 24-query leakage probe.",
    )

    # ------------------------------------------------------------------
    # 7.2 Honest Scope Boundaries and Current Limitations
    # ------------------------------------------------------------------
    add_h2(doc, "7.2 Honest Scope Boundaries and Current Limitations")

    add_p(
        doc,
        "Scientific honesty requires separating what our verified experiments demonstrate from "
        "what has not yet been tested. Five explicit scope boundaries apply to this thesis:",
    )

    add_bullet(
        doc,
        "1. Simulation vs. Real Students: ",
        "All Tier 1 memory metrics and Tier 4 teaching-policy metrics measure simulated learners "
        "driven by 3/data_Set/v2.2/simulator.py and 3/ppo/env.py (backed by frozen "
        "monotonic_gru). Real-student classroom effectiveness belongs to the optional, "
        "ethics-board (IRB)-gated Tier 5 pilot, which is [NOT RUN] and is never merged with "
        "simulation numbers.",
    )
    add_bullet(
        doc,
        "2. Pending Live-LLM and Human-Adjudicated Items: ",
        "Several evaluations in Tiers 2 and 3 require live LLM API calls, pretrained dense "
        "embedding weights, or two-reviewer blinded human grading, and therefore remain "
        "explicitly pending with zero fabricated numbers: Dense and BM25 + Dense in Tier 2 "
        "(local sentence_transformers, transformers, and huggingface_hub weights are absent; "
        "1,090 query-document pairs are exported in blind_packs/), Ablation 1 downstream (T-A1, "
        "132 items), Tier 3a + Ablation 2 EduGuard ON vs. OFF (216 prompts), Tier 3b question "
        "generation (180 requests), Tier 3c self-explanation grading (120 prompts with "
        "rubric=None), and Tier 5 (human pilot).",
    )
    add_bullet(
        doc,
        "3. Out-of-Scope RAG Abstention (not_supported_deferred): ",
        "When a student asks an off-topic or out-of-scope question, 3/adaptive_web/rag.py "
        "currently returns the highest-scoring chunks within the matching Python skill rather "
        "than refusing or abstaining. Explicit out-of-scope abstention is marked "
        "not_supported_deferred.",
    )
    add_bullet(
        doc,
        "4. PPO Behavioral Action-Balance Edge Cases & Hardest Skill: ",
        "In the 10-seed Pre-Registered Tier 4 benchmark, 10/10 seeds beat every baseline on "
        "delayed retention and permastore floor b_t, while 8/10 seeds pass all 5/5 behavioral "
        "action-balance checks on main_test and power_law_test (7/10 on long_gap_test). "
        "Isolated action-share bound misses appear in seed 20261206 (question turns at 38.7%, "
        "38.8%, and 39.2%—just below the 40% threshold), seed 20261207 (combined passive "
        "instruction share explain_concept + show_worked_example at 37.4%–38.0%, above the "
        "32% limit, and topic advancement at 4.1%–4.2%, just below 4.5%), and long_gap_test "
        "seed 20261209 (combined passive instruction share at 32.6%, just above 32%). "
        "Furthermore, function_returns remains the hardest "
        "skill at Day 30 across every policy (36.72% +/- 2.50% for PPO on main_test) because "
        "weaknesses in earlier prerequisites (conditional_evaluation and "
        "accumulator_initialization) compound at the leaf of the 6-skill DAG.",
    )
    add_bullet(
        doc,
        "5. AST Sandbox & Narrow Topic Scope: ",
        "For security, the bounded AST Interpreter in 3/dashboard/grading.py intentionally "
        "rejects recursion, while loops, comprehensions, imports, and attribute calls—marking "
        "such submissions as 'unresolved' rather than wrong—and the current curriculum covers "
        "6 foundational introductory Python skills rather than the entire language.",
    )

    # ------------------------------------------------------------------
    # 7.3 Future Work
    # ------------------------------------------------------------------
    add_h2(doc, "7.3 Future Work")

    add_p(
        doc,
        "The codebase and blinded evaluation packets are structured so each remaining step can "
        "be executed cleanly without altering the frozen offline artifacts:",
    )
    add_bullet(
        doc,
        "Execute Prepared Live-LLM & Two-Reviewer Blind Evaluations: ",
        "Run the exported blind packs in 3/eval_runs/v2.2-preregistered/blind_packs/ with a "
        "single pinned LLM provider (failover disabled) and a local dense bi-encoder to populate "
        "Dense/BM25+Dense Tier 2 metrics, T-A1, T3a (EduGuard ON vs. OFF), T3b, and T3c.",
    )
    add_bullet(
        doc,
        "Optional IRB-Approved Classroom Pilot (Tier 5): ",
        "Conduct a controlled study with real introductory Python students comparing pretests "
        "against 7-to-30-day unaided delayed posttests under ethics-board approval.",
    )
    add_bullet(
        doc,
        "Out-of-Scope Abstention & Action-Share Regularization: ",
        "Add a similarity/confidence threshold in rag.py so off-topic queries trigger clean "
        "abstention, and refine PPO entropy/action-share regularization or checkpoint selection "
        "so 10/10 seeds stay inside every behavioral action-balance bound, alongside targeted "
        "scaffolding for function_returns.",
    )
    add_bullet(
        doc,
        "Curriculum Expansion Beyond Six Python Skills: ",
        "Because memory inputs (47-dim tokens and 19-dim task vectors) exclude text and ID "
        "embeddings, future work can add skills such as while loops, dictionaries, and recursion "
        "(plus cross-subject tests) while keeping separate per-skill learner records.",
    )

    # ------------------------------------------------------------------
    # 7.4 Conclusion
    # ------------------------------------------------------------------
    add_h2(doc, "7.4 Conclusion")

    add_p(
        doc,
        "This project set out to solve a practical limitation of modern AI tutors: fluent chat "
        "alone does not prevent students from forgetting what they practiced days or weeks "
        "earlier. By separating the system into three inspectable layers—a subject-independent, "
        "behaviorally gated cognitive memory model (monotonic_gru), a budget-aware reinforcement "
        "learning policy (delayed_retention_ppo) that plans for 30-day retention rather than "
        "immediate next-question accuracy, and a Hybrid RAG plus EduGuard web tutor "
        "(3/adaptive_web/)—the resulting architecture delivers both measurable long-term "
        "simulated retention gains (+5.69 pp at Day 30 on main_test) and transparent, safe "
        "pedagogical execution.",
    )
