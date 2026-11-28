"""Frozen Chapter 6 — Results and Performance Analysis (verbatim from 3/build_experiments_docx.py)."""
import csv
import gzip
import json
from pathlib import Path

from report_sections.style import (
    add_bullet,
    add_figure,
    add_h1,
    add_h2,
    add_h3,
    add_mixed_p,
    add_p,
    add_table,
    fmt_ll_ci,
    fmt_ms,
    fmt_pp_ci,
)

DEFAULT_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_runs" / "v2.2-preregistered"


def compute_t1b_bins(eval_dir=DEFAULT_EVAL_DIR):
    csv_gz = Path(eval_dir) / "tier1" / "main_test_predictions.csv.gz"
    with gzip.open(csv_gz, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seed_cols = [c for c in rows[0].keys() if c.startswith("monotonic_gru_seed_")]
    out = {}
    for sname, pred_fn in (
        ("ordinary", lambda r: r["purpose"] == "ordinary" and r["delay_days"] in ("", "0", "0.0")),
        ("assessment", lambda r: r["delay_days"] not in ("", "0", "0.0")),
    ):
        sub = [r for r in rows if pred_fn(r)]
        bins = []
        for b in range(10):
            lo, hi = b * 0.1, (b + 1) * 0.1
            preds, ys = [], []
            for r in sub:
                p = sum(float(r[c]) for c in seed_cols) / len(seed_cols)
                if (p >= lo and p < hi) or (b == 9 and p >= lo and p <= 1.0):
                    preds.append(p)
                    ys.append(int(r["label"]))
            bins.append((
                f"[{lo:.1f}, {hi:.1f}" + ("]" if b == 9 else ")"),
                len(preds),
                f"{sum(preds)/len(preds):.3f}" if preds else "—",
                f"{sum(ys)/len(ys):.3f}" if ys else "—",
            ))
        out[sname] = bins
    return out


def build(doc, ctx=None):
    ctx = ctx or {}
    eval_dir = Path(ctx.get("EVAL_DIR", DEFAULT_EVAL_DIR))
    fig_dir = Path(ctx.get("FIG_DIR", eval_dir / "figures"))
    T1 = ctx.get("T1") or json.loads((eval_dir / "tier1_run.json").read_text(encoding="utf-8"))
    T4 = ctx.get("T4") or json.loads((eval_dir / "tier4_run.json").read_text(encoding="utf-8"))
    T2 = ctx.get("T2") or json.loads((eval_dir / "tier2_offline_diagnostics.json").read_text(encoding="utf-8"))

    # =========================================================================
    # CHAPTER 6 — RESULTS AND PERFORMANCE ANALYSIS
    # =========================================================================
    add_h1(doc, "Chapter 6 — Results and Performance Analysis", page_break=True)

    # -------------------------------------------------------------------------
    # 6.1 TIER 1
    # -------------------------------------------------------------------------
    add_h2(doc, "6.1 Tier 1 — Does the Brain Guess Accurately and Obey Common Sense?")
    add_p(
        doc,
        "Before an AI teacher can plan good lessons, its memory brain must predict how likely a student is to pass a "
        "new question today (Ordinary Practice) and 1, 7, or 30 days later (Delayed Assessment). We trained all 9 "
        "models across 5 pre-registered seeds on 1,600 training students and tested them on 200 unseen students per cohort.",
    )

    pr1 = T1["pre_registered"]
    beh1 = pr1["behavioral_summary"]
    ev1_main = pr1["evaluation"]["main_test"]
    ev1_pl = pr1["evaluation"]["power_law_test"]
    ev1_lg = pr1["evaluation"]["long_gap_test"]

    t1_rows = []
    t1_models = [
        ("constant", "Constant (Base Rate)", "—"),
        ("task_logistic", "Task Logistic", "Tabular"),
        ("history_logistic", "History Logistic", "Tabular"),
        ("history_boosting", "History Boosting", "Tabular"),
        ("kt_decay_gru", "KT Decay GRU", None),
        ("time_aware_transformer", "Time-Aware Transformer", None),
        ("monotonic_transformer", "Monotonic Transformer", None),
        ("monotonic_gru", "Monotonic GRU (Ours)", None),
        ("time_input_gru", "Time-Input GRU", None),
        ("time_decay_gru", "Time-Decay GRU (Rejected)", None),
        ("oracle", "Oracle (Hidden Ceiling)", "—"),
    ]
    for mkey, mname, gate_fix in t1_models:
        sm = ev1_main["summary"][mkey]
        sig = ev1_main["constant_to_oracle_log_loss_fraction"]
        if gate_fix is not None:
            gate_s = gate_fix
        else:
            b = beh1[mkey]
            gate_s = f"{b['checks_passed_mean']:.0f}/6 ({'Pass' if b['all_seeds_passed'] else 'FAIL'})"
        ord_sig = "0.0%" if mkey == "constant" else ("100.0%" if mkey == "oracle" else f"{sig['ordinary'][mkey]*100:.1f}%")
        del_sig = "0.0%" if mkey == "constant" else ("100.0%" if mkey == "oracle" else f"{sig['delayed_equal_horizon_log_loss'][mkey]*100:.1f}%")
        t1_rows.append([
            mname,
            gate_s,
            fmt_ms(sm["ordinary"]["log_loss"], 4),
            fmt_ms(sm["ordinary"]["roc_auc"], 4),
            fmt_ms(sm["ordinary"]["ece_10"], 4),
            ord_sig,
            fmt_ms(sm["delayed_equal_horizon_log_loss"], 4),
            fmt_ms(sm["assessment"]["roc_auc"], 4),
            fmt_ms(sm["assessment"]["ece_10"], 4),
            del_sig,
        ])

    add_table(
        doc,
        ["Model", "6/6 Gate", "Ord. LL", "Ord. AUC", "Ord. ECE", "Ord. Sig%", "Del. LL", "Del. AUC", "Del. ECE", "Del. Sig%"],
        t1_rows,
        caption="Table 6.1 — Tier 1 Memory Benchmark on main_test (Pre-Registered, 5 Seeds, N=200 Held-Out Students).",
        col_widths=[1.45, 0.72, 0.65, 0.56, 0.54, 0.53, 0.65, 0.56, 0.54, 0.53],
        right_cols={2, 3, 4, 5, 6, 7, 8, 9},
    )

    add_figure(
        doc,
        fig_dir / "fig1_tier1_models.png",
        "Figure 6.1 — Tier 1 Log Loss (lower is better) and ROC-AUC (higher is better) on main_test (5 seeds).",
    )

    # Stress cohorts compact table
    stress_rows = []
    for cname, clab, cev in (("power_law_test", "Power-Law Stress", ev1_pl), ("long_gap_test", "Long-Gap (4d) Stress", ev1_lg)):
        for mkey, mname in (
            ("history_boosting", "History Boosting"),
            ("time_aware_transformer", "Time-Aware Transformer"),
            ("monotonic_transformer", "Monotonic Transformer"),
            ("monotonic_gru", "Monotonic GRU (Ours)"),
            ("time_input_gru", "Time-Input GRU"),
            ("oracle", "Oracle (Ceiling)"),
        ):
            sm = cev["summary"][mkey]
            sig = cev["constant_to_oracle_log_loss_fraction"]
            del_sig = "100.0%" if mkey == "oracle" else f"{sig['delayed_equal_horizon_log_loss'][mkey]*100:.1f}%"
            stress_rows.append([
                clab, mname,
                fmt_ms(sm["ordinary"]["log_loss"], 4),
                fmt_ms(sm["ordinary"]["roc_auc"], 4),
                fmt_ms(sm["day_1"]["log_loss"], 4),
                fmt_ms(sm["day_7"]["log_loss"], 4),
                fmt_ms(sm["day_30"]["log_loss"], 4),
                fmt_ms(sm["delayed_equal_horizon_log_loss"], 4),
                del_sig,
            ])

    add_table(
        doc,
        ["Cohort", "Model", "Ord. LL", "Ord. AUC", "1d LL", "7d LL", "30d LL", "Del. LL", "Del. Sig%"],
        stress_rows,
        caption="Table 6.2 — Tier 1 Stress-Cohort Generalization (Power-Law Forgetting and 4-Day Long-Gap Breaks, 5 Seeds).",
        col_widths=[1.15, 1.35, 0.65, 0.58, 0.58, 0.58, 0.58, 0.65, 0.58],
        right_cols={2, 3, 4, 5, 6, 7, 8},
    )

    # Plain-English findings for Tier 1
    pi1 = ev1_main["paired_intervals"]
    add_bullet(
        doc,
        "Why the 6/6 Sanity Gate Matters: ",
        "In 5-seed validation, time_decay_gru got a slightly lower raw Log Loss (0.55173), but it FAILED the 6/6 sanity gate (5/6) because its predictions drifted upward over 0–60 days after a mistake. Our gate automatically rejected it.",
    )
    add_bullet(
        doc,
        "Monotonic GRU vs. Transformers & Time-Input GRU: ",
        f"In the earlier Retrospective (3-seed) run, monotonic_gru ranked #1 on validation selection (0.55175 vs. 0.55177 for time_input_gru). In the fresh Pre-Registered (5-seed) run, both models are statistically tied within 0.00008 on validation (0.55175 vs. 0.55183), while monotonic_gru has the smallest seed-to-seed spread (±0.00011), beats time_aware_transformer on Ordinary Log Loss ({fmt_ll_ci(pi1['ordinary']['monotonic_gru minus time_aware_transformer'])}), beats history_boosting ({fmt_ll_ci(pi1['ordinary']['monotonic_gru minus history_boosting'])}), and uniquely exposes the explicit permastore floor b_t and half-life h_t in a tiny 32-float Markov state needed by PPO.",
    )

    # Calibration Table & Reliability Diagram
    bins_t1b = compute_t1b_bins(eval_dir=eval_dir)
    add_table(
        doc,
        ["Probability Bucket", "Ordinary Count", "Ord. Guess", "Ord. Real Pass", "Delayed Count", "Del. Guess", "Del. Real Pass"],
        [[b_o[0], b_o[1], b_o[2], b_o[3], b_d[1], b_d[2], b_d[3]] for b_o, b_d in zip(bins_t1b["ordinary"], bins_t1b["assessment"])],
        caption="Table 6.3 — Reliability (Calibration) Check for monotonic_gru on main_test (5-Seed Average). Guesses match reality closely.",
        col_widths=[1.1, 0.95, 0.9, 0.95, 0.95, 0.9, 0.95],
        right_cols={1, 2, 3, 4, 5, 6},
    )
    add_figure(
        doc,
        fig_dir / "fig2_tier1_reliability.png",
        "Figure 6.2 — Reliability diagram for monotonic_gru on main_test (Pre-Registered 5-seed average).",
        width_in=6.2,
    )

    # -------------------------------------------------------------------------
    # 6.2 TIER 2
    # -------------------------------------------------------------------------
    add_h2(doc, "6.2 Tier 2 — Can the Search Engine Find the Right Lessons Fast?")
    add_p(
        doc,
        "When a student asks a doubt (k=4 chunks) or the tutor creates a new question (k=3 examples + k=2 lesson chunks), "
        "the search engine must be fast, pick the right topic/difficulty, and never mix up different students' private mistakes. "
        "Offline checks ran on 66 Dev and 132 Held-Out Test queries (192 test retrieval tasks + 24 cross-learner leakage probes).",
    )
    bm25_t = T2["arms"]["bm25_alone"]["test"]
    hyb_t = T2["arms"]["hybrid_tfidf"]["test"]
    add_table(
        doc,
        ["Search Arm", "Chat (k=4) nDCG@4 / Recall@4", "Q-Gen (k=3 / k=2) Relevance", "Meta Match%", "Student Leaks", "Underfills", "p95 Speed"],
        [
            ["BM25 Alone (Keywords)", "[Pending Human Labels]", "[Pending Human Labels]", f"{bm25_t['metadata_match_rate']*100:.1f}%", f"{bm25_t['cross_learner_violation_total']} / 192", f"{bm25_t['bm25_underfilled_queries']} / 192", f"{bm25_t['p95_latency_ms']:.2f} ms"],
            ["Deployed BM25 + TF-IDF", "[Pending Human Labels]", "[Pending Human Labels]", f"{hyb_t['metadata_match_rate']*100:.1f}%", f"{hyb_t['cross_learner_violation_total']} / 192", f"{hyb_t['bm25_underfilled_queries']} / 192", f"{hyb_t['p95_latency_ms']:.2f} ms"],
            ["Dense Encoder (Dev-Sel.)", "[Pending Dense Weights & Human]", "[Pending Dense Weights & Human]", "[Pending]", "[Pending]", "—", "[Pending]"],
            ["BM25 + Dense (Abl. 1)", "[Pending Dense Weights & Human]", "[Pending Dense Weights & Human]", "[Pending]", "[Pending]", "[Pending]", "[Pending]"],
        ],
        caption="Table 6.4 — Tier 2 Held-Out Retrieval Diagnostics (N=132 Queries / 192 Tasks, plus 24-Query Cross-Learner Probe = 0 Leaks).",
        col_widths=[1.3, 1.2, 1.2, 0.75, 0.8, 0.75, 0.75],
        right_cols={3, 4, 5, 6},
    )
    add_p(
        doc,
        "Both offline search arms run in under 5.1 ms (p95), match the exact requested skill/difficulty/format 100.0% of the time, "
        "and show 0 cross-learner privacy leaks. Human relevance labels (1,090 query-document pairs exported) and dense weights are pending.",
        size=10.5, after=8,
    )

    # -------------------------------------------------------------------------
    # 6.3 TIER 3
    # -------------------------------------------------------------------------
    add_h2(doc, "6.3 Tier 3 — Tutor Replies, Safety Guard, Questions, and Grading (Pending Live LLM)")
    add_p(
        doc,
        "Zero live LLM API calls or fake human scores were run in this offline phase. Instead, all blinded prompt packets "
        "and isolated database test hooks are prepared and ready for the live-LLM + 2-reviewer human grading phase:",
    )
    add_table(
        doc,
        ["Tier / Test", "What It Measures", "Prepared Blind Pack Size", "Status"],
        [
            ["Tier 3a + Ablation 2\n(EduGuard ON/OFF)", "Python correctness, faithfulness, 3 rubric scores, answer leakage vs. over-refusal", "216 prompts (96 adaptation + 60 protected + 60 legit doubts)", "[Pending Live LLM & Human]"],
            ["Tier 3b\n(Question Generator)", "Usable fresh question rate, first-pass & post-repair validity, bank fallback rate", "180 requests (6 skills × 5 formats × 3 difficulties × 2)", "[Pending Live LLM & Human]"],
            ["Tier 3c\n(Explanation Grader)", "MAE vs. human 0–1 scores, full-success P/R/F1, confidence-vs-error check", "120 prompts across 18 production prompts (rubric=None)", "[Pending Human Corpus & LLM]"],
            ["Ablation 1 Downstream\n(Fixed-LLM Check)", "Does BM25+Dense improve real tutor replies & questions over BM25+TF-IDF?", "132 items (72 chat + 60 question requests)", "[Pending Live LLM & Human]"],
        ],
        caption="Table 6.5 — Tier 3 and Ablation 1–2 blinded evaluation packs (ready to execute with live LLM + 2 human reviewers).",
        col_widths=[1.4, 2.45, 1.7, 1.25],
    )

    # -------------------------------------------------------------------------
    # 6.4 TIER 4
    # -------------------------------------------------------------------------
    add_h2(doc, "6.4 Tier 4 — Does the AI Teacher Actually Help Students Remember 30 Days Later?")

    add_h3(doc, "6.4.1 Main Result (Primary Outcome & Day-30 Retention)")
    add_p(
        doc,
        "We trained our RL teacher (Delayed PPO) across 10 brand-new seeds and tested it on 400 brand-new simulated students "
        "(seeds 40000–40399) across all three environments at budget B = 32.0. We also keep our earlier 3-seed Retrospective run "
        "(200 students) cleanly separated in Table 6.6.",
    )

    pr4 = T4["pre_registered"]
    rt4 = T4["retrospective"]
    t4a_rows = []
    for run_lbl, blk in (("Pre-Reg (10 seeds, N=400)", pr4), ("Retro (3 seeds, N=200)", rt4)):
        for ckey, clab in (("main_test", "Main"), ("power_law_test", "Power-Law"), ("long_gap_test", "Long-Gap")):
            ev = blk["evaluation"][ckey]
            sm = ev["summary"]
            pi = ev["paired_intervals"]
            for pkey, plab in (
                ("delayed_retention_ppo", "Delayed PPO (Ours)"),
                ("threshold_heuristic", "Threshold Heuristic"),
                ("random_valid", "Random Valid"),
                ("fixed_curriculum", "Fixed Curriculum"),
                ("myopic_rl_bandit", "Myopic RL Bandit"),
            ):
                sp = sm[pkey]
                if pkey == "threshold_heuristic":
                    ci_s = "0.00 pp (Ref)"
                elif pkey in ("delayed_retention_ppo", "myopic_rl_bandit"):
                    ci_s = fmt_pp_ci(pi[f"{pkey} minus threshold_heuristic"]["observed_delayed"])
                else:
                    d_pp = (sp["delayed_mean"]["mean"] - sm["threshold_heuristic"]["delayed_mean"]["mean"]) * 100
                    ci_s = f"{d_pp:+.2f} pp"
                t4a_rows.append([
                    run_lbl if pkey == "delayed_retention_ppo" else "",
                    clab if pkey == "delayed_retention_ppo" else "",
                    plab,
                    fmt_ms(sp["day_1"], pct=True),
                    fmt_ms(sp["day_7"], pct=True),
                    fmt_ms(sp["day_30"], pct=True),
                    fmt_ms(sp["delayed_mean"], pct=True),
                    ci_s,
                    fmt_ms(sp["latent_mean_floor"], 3, False),
                ])

    add_table(
        doc,
        ["Run", "Cohort", "Policy", "1-Day", "7-Day", "30-Day", "Delayed Mean", "Diff vs. Heuristic (95% CI)", "Floor b_t"],
        t4a_rows,
        caption="Table 6.6 — Tier 4 Primary Outcome: Unaided Delayed Pass Rates at Study Budget B = 32.0 (Pre-Registered vs. Retrospective).",
        col_widths=[1.0, 0.62, 1.12, 0.58, 0.58, 0.62, 0.68, 1.15, 0.55],
        right_cols={3, 4, 5, 6, 8},
    )

    add_figure(
        doc,
        fig_dir / "fig3_tier4_policies.png",
        "Figure 6.3 — Pre-Registered (10 seeds, N=400) Mean Delayed Pass Rate and Day-30 Pass Rate across all 3 cohorts.",
    )

    pr4_main = pr4["evaluation"]["main_test"]
    d30_ci_main = pr4_main["paired_intervals"]["delayed_retention_ppo minus threshold_heuristic"]["observed_day_30"]
    d30_ci_lg = pr4["evaluation"]["long_gap_test"]["paired_intervals"]["delayed_retention_ppo minus threshold_heuristic"]["observed_day_30"]
    add_bullet(
        doc,
        "Clear Winner at 30 Days: ",
        f"On Main Test (N=400), Delayed PPO beats the Threshold Heuristic by +2.50 pp on overall delayed pass rate "
        f"({pr4_main['summary']['delayed_retention_ppo']['delayed_mean']['mean']*100:.2f}% vs. "
        f"{pr4_main['summary']['threshold_heuristic']['delayed_mean']['mean']*100:.2f}%) and by "
        f"{fmt_pp_ci(d30_ci_main)} on Day 30. Under harsh 4-day gaps (Long-Gap Test), the Day-30 lead stays strong at {fmt_pp_ci(d30_ci_lg)}.",
    )
    add_bullet(
        doc,
        "Why Short-Term Greedy AI Fails: ",
        "Myopic RL Bandit only cares about the very next turn (no future planning). It gets stuck drilling easy questions forever, scoring ~10 percentage points worse than Delayed PPO at Day 30.",
    )

    add_h3(doc, "6.4.2 Fairness Across All 6 Python Skills (Day-30 Per-Skill Pass Rates)")
    add_p(
        doc,
        "We checked all 6 individual Python skills at Day 30 on Main Test (N=400) to verify our AI teacher does not "
        "ignore hard topics just to boost its average score.",
    )
    skills = ["range_boundaries", "loop_tracing", "accumulator_initialization", "conditional_evaluation", "list_indexing", "function_returns"]
    skill_rows = []
    for pkey, plab in (
        ("delayed_retention_ppo", "Delayed PPO (Ours)"),
        ("threshold_heuristic", "Threshold Heuristic"),
        ("myopic_rl_bandit", "Myopic RL Bandit"),
        ("fixed_curriculum", "Fixed Curriculum"),
        ("random_valid", "Random Valid"),
        ("ppo_no_longterm_state", "Ablated 23-Dim PPO"),
    ):
        sp = pr4_main["summary"][pkey]
        ps = sp["per_skill_day_30"]
        skill_rows.append([
            plab,
            *[fmt_ms(ps[s], pct=True) for s in skills],
            f"{fmt_ms(sp['min_skill_day_30'], pct=True)}",
        ])
    add_table(
        doc,
        ["Policy", "Range", "Loops", "Accum.", "Cond.", "Lists", "Func. Ret.", "Lowest Skill"],
        skill_rows,
        caption="Table 6.7 — Main Test Day-30 Unaided Pass Rates Across All 6 Python Skills (Pre-Registered, 10 Seeds, N=400). Lowest skill is function_returns for all policies.",
        col_widths=[1.25, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.87],
        right_cols={1, 2, 3, 4, 5, 6, 7},
    )

    add_h3(doc, "6.4.3 Ablation 3 — Do Long-Term Memory Features Actually Cause the Gain?")
    add_p(
        doc,
        "In Ablation 3, we retrained a clean 23-dim PPO (ppo_no_longterm_state) across all 10 seeds with all 13 long-term "
        "features (7d/30d predictions and floor b_t) removed from its input, its action mask, and its topic routing—leaving "
        "only short-term 'today' predictions (p_med).",
    )
    abl_rows = []
    for ckey, clab in (("main_test", "Main Test"), ("power_law_test", "Power-Law Test"), ("long_gap_test", "Long-Gap Test")):
        ev = pr4["evaluation"][ckey]
        s_full = ev["summary"]["delayed_retention_ppo"]
        s_abl = ev["summary"]["ppo_no_longterm_state"]
        ci = ev["paired_intervals"]["delayed_retention_ppo minus ppo_no_longterm_state"]
        abl_rows.append([clab, "Full PPO (36 dims)", fmt_ms(s_full["delayed_mean"], pct=True), fmt_ms(s_full["day_30"], pct=True), "—", fmt_ms(s_full["latent_mean_floor"], 3, False), f"{s_full['action_shares']['request_self_explanation']*100:.1f}%"])
        abl_rows.append(["", "Ablated PPO (23 dims)", fmt_ms(s_abl["delayed_mean"], pct=True), fmt_ms(s_abl["day_30"], pct=True), f"Del: {fmt_pp_ci(ci['observed_delayed'])}\nD30: {fmt_pp_ci(ci['observed_day_30'])}", fmt_ms(s_abl["latent_mean_floor"], 3, False), f"{s_abl['action_shares']['request_self_explanation']*100:.1f}%"])
    add_table(
        doc,
        ["Cohort", "Model", "Delayed Mean", "Day-30 Pass", "Paired Loss from Ablation (95% CI)", "Floor b_t", "Self-Expl. %"],
        abl_rows,
        caption="Table 6.8 — Ablation 3 Results (10 Retrained Seeds, N=400). Hiding long-term memory cuts accuracy and triples seed instability.",
        col_widths=[0.9, 1.2, 0.85, 0.85, 1.65, 0.65, 0.7],
        right_cols={2, 3, 5, 6},
    )
    add_figure(
        doc,
        fig_dir / "fig4_ablation3.png",
        "Figure 6.4 — Ablation 3: Removing 30-day memory & floor b_t drops performance and makes training ~3x noisier (±1 SD bars).",
        width_in=6.0,
    )

    add_h3(doc, "6.4.4 Study Budget Curves (Budget B = 16, 24, 32, and 40)")
    bc = pr4["budget_curves"]
    bc_rows = []
    for b in (16.0, 24.0, 32.0, 40.0):
        bdata = bc[f"budget_{b:g}"]["main_test"]
        bsm = bdata["summary"]
        bpi = bdata["paired_intervals"]
        for pkey, plab in (("delayed_retention_ppo", "Delayed PPO (Ours)"), ("threshold_heuristic", "Threshold Heuristic"), ("myopic_rl_bandit", "Myopic RL Bandit")):
            sp = bsm[pkey]
            ci_s = "0.00 pp (Ref)" if pkey == "threshold_heuristic" else fmt_pp_ci(bpi[f"{pkey} minus threshold_heuristic"]["observed_delayed"])
            bc_rows.append([f"{int(b)}" if pkey == "delayed_retention_ppo" else "", plab, fmt_ms(sp["delayed_mean"], pct=True), fmt_ms(sp["day_30"], pct=True), fmt_ms(sp["latent_mean_floor"], 3, False), ci_s])
    add_table(
        doc,
        ["Budget (B)", "Policy", "Delayed Mean", "Day-30 Pass", "Floor b_t", "Diff vs. Heuristic (95% CI)"],
        bc_rows,
        caption="Table 6.9 — Main Test Budget Curves across B = 16 to 40 (Pre-Registered, 10 Seeds, N=400). PPO leads at every budget.",
        col_widths=[0.75, 1.35, 0.95, 0.95, 0.75, 1.45],
        right_cols={2, 3, 4},
    )
    add_figure(
        doc,
        fig_dir / "fig5_budget_curve.png",
        "Figure 6.5 — Day-30 pass rate as study budget grows from 16 to 40 across all 3 test cohorts.",
        width_in=6.3,
    )

    add_h3(doc, "6.4.5 What Teaching Actions Does the AI Teacher Pick?")
    act_rows = []
    for pkey, plab in (
        ("delayed_retention_ppo", "Delayed PPO (Ours)"),
        ("threshold_heuristic", "Threshold Heuristic"),
        ("myopic_rl_bandit", "Myopic RL Bandit"),
        ("fixed_curriculum", "Fixed Curriculum"),
        ("random_valid", "Random Valid"),
    ):
        sh = pr4_main["summary"][pkey]["action_shares"]
        act_rows.append([
            plab,
            f"{sh['give_easier_question']*100:.1f}%",
            f"{sh['give_same_difficulty_question']*100:.1f}%",
            f"{sh['give_harder_question']*100:.1f}%",
            f"{sh['explain_concept']*100:.1f}%",
            f"{sh['show_worked_example']*100:.1f}%",
            f"{sh['request_self_explanation']*100:.1f}%",
            f"{sh['review_prerequisite']*100:.1f}%",
            f"{sh['advance_next_topic']*100:.1f}%",
        ])
    add_table(
        doc,
        ["Policy", "Easier", "Same", "Harder", "Explain", "Example", "Self-Expl.", "Prereq", "Advance"],
        act_rows,
        caption="Table 6.10 — Share of each teaching action chosen on main_test (Pre-Registered, 10 Seeds, N=400).",
        col_widths=[1.3, 0.62, 0.62, 0.65, 0.65, 0.68, 0.72, 0.65, 0.68],
        right_cols={1, 2, 3, 4, 5, 6, 7, 8},
    )
    add_p(
        doc,
        "Notice how Threshold Heuristic never uses worked examples (0%), self-explanations (0%), or prerequisite reviews (0%). "
        "Delayed PPO mixes worked examples (16.4%), self-explanations (16.9%), prerequisite reviews (8.1%), and harder questions (15.1%) "
        "to lock knowledge into durable memory. Across the 10 pre-registered seeds, 10/10 seeds beat all baselines on delayed retention and "
        "floor b_t, while 8/10 seeds pass all 5 behavioral action-balance checks on Main and Power-Law tests (7/10 on Long-Gap).",
        size=10.5, after=8,
    )

    # -------------------------------------------------------------------------
    # 6.5 KEY TAKEAWAYS & NEXT STEP REMINDER
    # -------------------------------------------------------------------------
    add_h2(doc, "6.5 Key Takeaways and Pending Live-LLM Next Step")
    add_bullet(doc, "1. Delayed PPO Wins Where It Counts (30 Days Later): ", "At equal study budget (32.0), Delayed PPO improves 30-day unaided pass rates by +5.69 pp on Main Test, +4.69 pp on Power-Law stress, and +5.40 pp on 4-day Long-Gap breaks over the rule-based tutor.")
    add_bullet(doc, "2. Long-Term Memory Features Are Essential (Ablation 3): ", "Hiding 30-day predictions and permastore floor b_t cuts retention across every cohort and roughly triples training variance across seeds.")
    add_bullet(doc, "3. Common-Sense Memory Sanity Gate Protects Reliability: ", "monotonic_gru passes all 6/6 behavioral sanity checks across all seeds, matches reality closely on 10-bin calibration (ECE-10 = 0.0108 ordinary, 0.0235 delayed), and gives PPO the explicit b_t signal it needs.")
    add_bullet(doc, "4. Fast, Leak-Free Search Ready for Live Evaluation: ", "Hybrid retrieval runs in ~5 ms p95 with 100% metadata match and zero cross-learner privacy leaks.")
    add_mixed_p(doc, [
        ("Pending Next Step (Live-LLM Reminder): ", True, False),
        ("All offline experiments (Tiers 1, 2-offline, 4, and Ablation 3) are finished. The next step is running the live-LLM evaluation "
         "with failover disabled and 2 blinded human reviewers on the prepared packs (Tier 2 dense/relevance, Ablation 1 downstream, "
         "Tier 3a + Ablation 2 EduGuard ON/OFF, Tier 3b question generation, and Tier 3c self-explanation grading).", False, False),
    ], after=10)
