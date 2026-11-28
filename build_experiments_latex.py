"""Port Adaptive_Tutoring_Experiments_and_Results.docx into a modular, Overleaf-ready LaTeX thesis project."""
import csv
import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
EVAL_DIR = ROOT / "eval_runs" / "v2.2-preregistered"
FIG_SRC = EVAL_DIR / "figures"
OUT_DIR = REPO / "outputs" / "latex_report"

T1 = json.loads((EVAL_DIR / "tier1_run.json").read_text(encoding="utf-8"))
T4 = json.loads((EVAL_DIR / "tier4_run.json").read_text(encoding="utf-8"))
T2 = json.loads((EVAL_DIR / "tier2_offline_diagnostics.json").read_text(encoding="utf-8"))


def esc(text: str) -> str:
    """Escape plain text for pdflatex (never pass raw LaTeX commands through this)."""
    s = str(text)
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("$", r"\$")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    s = s.replace("±", r"$\pm$")
    s = s.replace("×", r"$\times$")
    s = s.replace("→", r"$\rightarrow$")
    s = s.replace("—", "---")
    s = s.replace("–", "--")
    return s


def fmt_ms_tex(entry, dec=4, pct=False):
    m = entry["mean"]
    sd = entry.get("sd")
    if pct:
        if sd is None:
            return f"{m * 100:.2f}\\%"
        return f"{m * 100:.2f}\\% $\\pm$ {sd * 100:.2f}\\%"
    if sd is None:
        return f"{m:.{dec}f}"
    return f"{m:.{dec}f} $\\pm$ {sd:.{dec}f}"


def fmt_pp_ci_tex(entry):
    return f"{entry['delta'] * 100:+.2f} pp [{entry['lower_95'] * 100:+.2f}, {entry['upper_95'] * 100:+.2f}]"


def fmt_ll_ci_tex(entry):
    return f"{entry['delta_log_loss']:+.5f} [{entry['lower_95']:+.5f}, {entry['upper_95']:+.5f}]"


def compute_t1b_bins():
    csv_gz = EVAL_DIR / "tier1" / "main_test_predictions.csv.gz"
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
                f"{{[}}{lo:.1f}, {hi:.1f}" + ("]" if b == 9 else ")"),
                str(len(preds)),
                f"{sum(preds)/len(preds):.3f}" if preds else "---",
                f"{sum(ys)/len(ys):.3f}" if ys else "---",
            ))
        out[sname] = bins
    return out


def make_table_env(caption, label, col_spec, headers, rows, resize=False, font_size=r"\small"):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        font_size,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        safe_row = [("{[}" + cell[1:] if cell.startswith("[") else cell) for cell in row]
        lines.append(" & ".join(safe_row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def build_latex():
    # Copy figures
    fig_dst = OUT_DIR / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    for png in sorted(FIG_SRC.glob("*.png")):
        shutil.copy2(png, fig_dst / png.name)

    # 1. main.tex
    main_tex = r"""\documentclass[12pt,a4paper,oneside]{report}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage[margin=0.9in]{geometry}
\usepackage{setspace}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{caption}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}

\definecolor{snublue}{RGB}{18,42,76}
\definecolor{snugrey}{RGB}{100,108,116}

\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries\color{snublue}}
  {\chaptertitlename\ \thechapter}{12pt}{\LARGE}
\titleformat{\section}
  {\normalfont\Large\bfseries\color{snublue}}{\thesection}{0.8em}{}
\titleformat{\subsection}
  {\normalfont\large\bfseries\color{snublue}}{\thesubsection}{0.8em}{}

\captionsetup{font=small,labelfont=bf}
\onehalfspacing

\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\small\color{snugrey}Page \thepage}
\renewcommand{\headrulewidth}{0pt}
\setlength{\headheight}{15pt}

\begin{document}

\pagenumbering{roman}
\input{frontmatter/titlepage}
\input{frontmatter/certificate}
\input{frontmatter/declaration}
\input{frontmatter/acknowledgement}
\input{frontmatter/abstract}

\tableofcontents
\clearpage
\listoftables
\clearpage
\listoffigures
\clearpage

\pagenumbering{arabic}
\input{chapters/ch1_introduction}
\input{chapters/ch2_literature_survey}
\input{chapters/ch3_datasets_methodology}
\input{chapters/ch4_implementation}
\input{chapters/ch5_evaluation_protocol}
\input{chapters/ch6_results}
\input{chapters/ch7_limitations_conclusion}

\chapter*{References}
\phantomsection
\addcontentsline{toc}{chapter}{References}
\textit{\color{snugrey}[Placeholder --- Bibliography and citations to be inserted.]}
% When ready to populate citations, replace the placeholder above with:
% \bibliographystyle{IEEEtran}
% \bibliography{references}

\end{document}
"""
    write_file(OUT_DIR / "main.tex", main_tex)

    # 2. Front matter placeholders
    write_file(
        OUT_DIR / "frontmatter" / "titlepage.tex",
        r"""\begin{titlepage}
\centering
\vspace*{1.2cm}
{\Huge\bfseries Adaptive Tutoring with Shared Temporal Memory and Reinforcement Learning\par}
\vspace{0.8cm}
{\itshape\normalsize Project Report Submitted in Partial Fulfilment of the Requirements for the Degree of\par}
\vspace{0.5cm}
{\LARGE\bfseries Bachelor of Technology\\in\\Computer Science and Engineering\par}
\vspace{1.2cm}
{\itshape\large Submitted by\\[0.3em] [Placeholder --- Student Name(s) and Roll Number(s)]\par}
\vspace{1.2cm}
{\itshape\large Under the Supervision of\\[0.3em] [Placeholder --- Supervisor Name and Designation]\par}
\vfill
{\normalsize Department of Computer Science and Engineering\\
Shiv Nadar Institution of Eminence, Delhi NCR\\
[Placeholder --- Date]\par}
\end{titlepage}
""",
    )
    for fname, title, note in (
        ("certificate.tex", "Certificate", "[Placeholder --- Supervisor certificate to be inserted.]"),
        ("declaration.tex", "Declaration", "[Placeholder --- Student declaration to be inserted.]"),
        ("acknowledgement.tex", "Acknowledgement", "[Placeholder --- Acknowledgements to be inserted.]"),
        ("abstract.tex", "Abstract", "[Placeholder --- Abstract to be written after final review.]"),
    ):
        write_file(
            OUT_DIR / "frontmatter" / fname,
            f"\\chapter*{{{title}}}\n\\phantomsection\n\\addcontentsline{{toc}}{{chapter}}{{{title}}}\n"
            f"\\textit{{\\color{{snugrey}}{note}}}\n\\clearpage\n",
        )

    # 3. Placeholder Chapters 1-4 and 7
    for fname, ch_title, note in (
        ("ch1_introduction.tex", "Introduction", "[Placeholder --- Problem statement, objectives, and contributions to be inserted.]"),
        ("ch2_literature_survey.tex", "Literature Survey", "[Placeholder --- Literature review to be inserted.]"),
        ("ch3_datasets_methodology.tex", "Datasets and Methodology", "[Placeholder --- Dataset v2.2 and model design to be inserted.]"),
        ("ch4_implementation.tex", "Implementation", "[Placeholder --- System architecture and implementation details to be inserted.]"),
        ("ch7_limitations_conclusion.tex", "Limitations and Conclusion", "[Placeholder --- Known limitations, threats to validity, and conclusion to be inserted.]"),
    ):
        write_file(
            OUT_DIR / "chapters" / fname,
            f"\\chapter{{{ch_title}}}\n\\textit{{\\color{{snugrey}}{note}}}\n",
        )

    # 4. references.bib placeholder
    write_file(
        OUT_DIR / "references.bib",
        "% Placeholder BibTeX file. Add references here when completing Chapters 1-4 and 7.\n",
    )

    # 5. Chapter 5 — Evaluation Protocol
    t5_1_rows = [
        [r"$N$", "Number of students, questions, or test prompts in an experiment.", "Larger = more reliable"],
        [r"$B$ (Budget $= 32.0$)", "Total study coins a tutor may spend (1.0 per question, 0.6--0.9 per hint/lesson).", "Equal across policies"],
        [r"$k$ (Context Slots)", r"Number of text chunks the search engine hands to the AI tutor ($k=4$ chat; $k=3$ examples + $k=2$ lessons for question making).", "Matched per arm"],
        [r"$b_t$ (Permastore Floor)", "Durable, long-lasting knowledge (from 0 to 1) that stays even after a 30-day break.", "Higher is better"],
        [r"$k_t$ (Active Strength)", "Short-term + long-term skill strength (from 0 to 1) right at the current time.", "Higher is better"],
        [r"$s_t$ (Policy State)", "The 36-number dashboard summary the RL teacher reads before picking the next action.", "36 dims (23 in Abl. 3)"],
        [r"Log Loss (LL)", "Scores how accurate and honest a probability guess is. Punishes confident wrong guesses (0.693 = pure 50/50 coin flip).", "Lower is better"],
        [r"ROC-AUC", "Pick one passing student and one failing student: how often did the model give the passing student a higher score? (0.5 = coin flip, 1.0 = perfect).", "Higher is better"],
        [r"Brier Score", "Average squared gap between the model's 0--1 guess and the real 0/1 pass outcome.", "Lower is better"],
        [r"ECE-10 (Calibration)", r"Splits guesses into 10 buckets (0--10\%, \ldots, 90--100\%). Measures if a \textquotedblleft 70\% guess\textquotedblright{} really passes $\sim$70\% of the time.", "Lower is better (near 0)"],
        [r"Signal \%", r"How much of the gap between naive base-rate guessing (0\%) and the hidden simulator Oracle ceiling (100\%) a model captures.", "Higher is better"],
        [r"6 Behavioral Checks", r"Common-sense rules: getting 0\% wrong must drop predictions; 0--60d forgetting curves must never drift upward without study.", "Must be 6 / 6"],
        [r"Delayed Mean / Day-30", "Average unaided pass rate at 1, 7, and 30 days after study; Day-30 checks lasting memory one month later.", "Higher is better"],
        [r"Lowest-Skill Day-30", r"The Day-30 pass rate on the weakest of the 6 Python skills (makes sure hard topics like \texttt{function\_returns} are not ignored).", "Higher is better"],
        [r"Delta ($\Delta$) \& 95\% CI", "Difference between two systems on the exact same students/queries, with a 95\\% Bootstrap interval from 2,000 resamples.", "CI excludes 0 = consistent"],
        [r"nDCG@k / Recall@k", "Search quality: Recall@k checks if the right chunk is in the top $k$; nDCG@k rewards putting it at rank \\#1.", "Higher is better (max 1)"],
        [r"p95 Latency (ms)", "Search speed: 95 out of 100 queries finish faster than this time in milliseconds.", "Lower is faster"],
        [r"Meta Match / Violations", "Exemplar Meta Match checks exact skill/difficulty/format match (target 100\\%); Cross-Learner Violations checks student data leaks (must be 0).", "100\\% match / 0 leaks"],
        [r"Leakage / Over-Refusal", "In EduGuard: Leakage = tutor gives away the answer on trick prompts; Over-Refusal = tutor wrongly blocks a genuine question.", "Both near 0\\%"],
        [r"Usable Question Rate", "Share of AI-generated questions that are fresh, valid, solvable, and match the requested skill, format, and difficulty.", "Higher is better"],
        [r"MAE (Self-Explanation)", "Mean Absolute Error between the automated grader's 0--1 score and human teacher scores.", "Lower is better (near 0)"],
    ]
    t5_1_tex = make_table_env(
        "Plain-English dictionary of every symbol, variable, and metric used in Chapters 5 and 6.",
        "tab:metrics_dict",
        r"p{3.4cm} p{8.4cm} p{3.3cm}",
        ["Symbol / Term", "Plain-English Meaning (What It Is)", "Good Value"],
        t5_1_rows,
        font_size=r"\footnotesize",
    )

    t5_2_rows = [
        ["Tier 1 (Memory)", "9 memory models + Constant \\& Oracle anchors", "Predicts first unaided pass/fail on 200 held-out students per cohort (main, power-law, long-gap). Retrospective: 3 seeds; Pre-Registered: 5 fresh seeds (20261201--20261205).", "Executed (Complete)"],
        ["Tier 2 (Retrieval)", "BM25 alone vs. Deployed BM25+TF-IDF vs. Dense vs. BM25+Dense", "Tests speed (p95), metadata match, cross-learner isolation, and ranking on 66 dev + 132 held-out test queries at exact slots ($k=4$ chat; $k=3$ + $k=2$ question gen).", "Offline Diagnostics Done; Dense \\& Human Labels Pending"],
        ["Tier 3a--3c (LLM \\& Grading)", "Doubt Tutor, EduGuard, Question Maker (180 items), Self-Explanation Grader (120 items)", "Blinded 2-reviewer human grading with live LLM failover turned off. Blind prompt/review packs exported.", "Blind Packs Ready; Live LLM \\& Human Review Pending"],
        ["Tier 4 (RL Policy)", "Delayed PPO vs. Threshold Heuristic, Myopic Bandit, Fixed, Random", "Retrospective: 3 seeds ($N=200$). Pre-Registered: 10 fresh seeds (20261201--20261210) tested on 400 brand-new students (40000--40399) across 3 cohorts \\& 4 budgets.", "Executed (Complete)"],
        ["Ablation 1 (Retrieval)", "Deployed BM25+TF-IDF vs. BM25+Dense (fixed LLM)", "Checks if better search improves live tutor Python correctness and question usability.", "Pack Ready (Pending Live LLM)"],
        ["Ablation 2 (EduGuard)", "EduGuard ON vs. OFF on 120 paired prompts", "Measures answer leakage vs. over-refusal on identical questions and DB snapshots.", "Pack Ready (Pending Live LLM)"],
        ["Ablation 3 (PPO State)", "Full 36-dim PPO vs. 23-dim No-Long-Term PPO", "Retrains 10 seeds from scratch after removing all 13 long-term (7d/30d/floor $b_t$) features and routing/mask leaks.", "Executed (Complete)"],
        ["Tier 5 (Opt. Human Pilot)", "Real classroom students (optional future step)", "Unaided pretest $\\rightarrow$ study session $\\rightarrow$ 7-day delayed posttest, kept strictly separate from simulation.", "Optional Outline Only"],
    ]
    t5_2_tex = make_table_env(
        "Summary of the four evaluation tiers, three ablations, and optional human pilot.",
        "tab:eval_plan",
        r"p{2.6cm} p{4.0cm} p{5.8cm} p{2.7cm}",
        ["Tier / Test", "What Part It Tests", "How It Is Tested \\& Seeding", "Status in This Report"],
        t5_2_rows,
        font_size=r"\footnotesize",
    )

    ch5_tex = f"""\\chapter{{Evaluation Protocol}}
\\label{{chap:eval_protocol}}

\\section{{What We Are Testing and Our Main Goal}}
Imagine a student learning Python with an AI tutor. They only have a small amount of study time---a fixed study budget of $B = 32.0$ points (about four short lessons spaced days apart). Anyone can look smart right after reading an answer with hints. The real test of a good teacher is: when the student comes back weeks later with zero hints, do they still remember how to solve the problem?

\\noindent\\textbf{{Main Outcome (Primary Metric):}} After studying ends, we give the simulated student fresh tests without any hints at \\textbf{{1 day, 7 days, and 30 days later}} (18 questions per day across 6 Python skills and 3 difficulty levels = 54 questions per student). Our main score is the \\textbf{{mean delayed unaided pass rate}} (the average of Day 1, Day 7, and Day 30), and we highlight \\textbf{{Day-30 retention}} separately because 30 days tests true lasting memory rather than short-term cramming. We compare our AI teacher against a standard rule-based teacher (Threshold Heuristic) using the exact same study budget and the exact same simulated students.

\\noindent\\textbf{{Important Scope Note:}} \\textit{{Tier 1 and Tier 4 test the memory model and teaching policy inside a computer simulator. They prove our system works in simulation, not yet in a live human classroom (which belongs to a future IRB-approved Tier 5 study).}}

\\section{{Simple Guide to Every Symbol, Variable, and Metric}}
Table~\\ref{{tab:metrics_dict}} explains every symbol, variable, and test score in plain, everyday words so anyone reading this report for the first time can follow every number.

{t5_1_tex}

\\section{{The Four-Tier Test Plan and Ablations}}
Because our tutor has four connected parts---a Memory Brain, a Search Engine, an LLM Teacher/Grader, and an RL Policy Planner---we test each part on its own and test them together in four tiers, plus three focused ablation (remove-one-piece) tests (Table~\\ref{{tab:eval_plan}}).

{t5_2_tex}

\\section{{Fair Testing Rules}}
\\begin{{itemize}}[leftmargin=1.5em]
  \\item \\textbf{{Same Students \\& Same Prompts (Paired Testing):}} Every competing model is tested on the exact same simulated students or exact same questions so differences come from the model, not lucky draws.
  \\item \\textbf{{Honest Labels (Retrospective vs. Pre-Registered):}} Our earlier 3-seed runs are clearly labeled \\textit{{Retrospective}}. Fresh 5-seed (Tier 1) and 10-seed (Tier 4, 400 new students) runs with locked rules are labeled \\textit{{Pre-Registered}}.
  \\item \\textbf{{No Cherry-Picking \\& No Software Test Mixing:}} We report all 3 test cohorts (main, power-law forgetting, and 4-day long-gap breaks) and all 6 Python skills. Software engineering unit tests are never mixed into research metrics.
\\end{{itemize}}
"""
    write_file(OUT_DIR / "chapters" / "ch5_evaluation_protocol.tex", ch5_tex)

    # 6. Chapter 6 — Results and Performance Analysis
    pr1 = T1["pre_registered"]
    beh1 = pr1["behavioral_summary"]
    ev1_main = pr1["evaluation"]["main_test"]
    ev1_pl = pr1["evaluation"]["power_law_test"]
    ev1_lg = pr1["evaluation"]["long_gap_test"]

    t1_rows = []
    t1_models = [
        ("constant", "Constant (Base Rate)", "---"),
        ("task_logistic", "Task Logistic", "Tabular"),
        ("history_logistic", "History Logistic", "Tabular"),
        ("history_boosting", "History Boosting", "Tabular"),
        ("kt_decay_gru", "KT Decay GRU", None),
        ("time_aware_transformer", "Time-Aware Transformer", None),
        ("monotonic_transformer", "Monotonic Transformer", None),
        ("monotonic_gru", "Monotonic GRU (Ours)", None),
        ("time_input_gru", "Time-Input GRU", None),
        ("time_decay_gru", "Time-Decay GRU (Rejected)", None),
        ("oracle", "Oracle (Hidden Ceiling)", "---"),
    ]
    for mkey, mname, gate_fix in t1_models:
        sm = ev1_main["summary"][mkey]
        sig = ev1_main["constant_to_oracle_log_loss_fraction"]
        if gate_fix is not None:
            gate_s = gate_fix
        else:
            b = beh1[mkey]
            gate_s = f"{b['checks_passed_mean']:.0f}/6 ({'Pass' if b['all_seeds_passed'] else 'FAIL'})"
        ord_sig = "0.0\\%" if mkey == "constant" else ("100.0\\%" if mkey == "oracle" else f"{sig['ordinary'][mkey]*100:.1f}\\%")
        del_sig = "0.0\\%" if mkey == "constant" else ("100.0\\%" if mkey == "oracle" else f"{sig['delayed_equal_horizon_log_loss'][mkey]*100:.1f}\\%")
        t1_rows.append([
            esc(mname),
            gate_s,
            fmt_ms_tex(sm["ordinary"]["log_loss"], 4),
            fmt_ms_tex(sm["ordinary"]["roc_auc"], 4),
            fmt_ms_tex(sm["ordinary"]["ece_10"], 4),
            ord_sig,
            fmt_ms_tex(sm["delayed_equal_horizon_log_loss"], 4),
            fmt_ms_tex(sm["assessment"]["roc_auc"], 4),
            fmt_ms_tex(sm["assessment"]["ece_10"], 4),
            del_sig,
        ])
    t6_1_tex = make_table_env(
        r"Tier 1 Memory Benchmark on \protect\texttt{main\_test} (Pre-Registered, 5 Seeds, $N=200$ Held-Out Students).",
        "tab:t1_main",
        "l l r r r r r r r r",
        ["Model", "6/6 Gate", "Ord. LL", "Ord. AUC", "Ord. ECE", "Ord. Sig\\%", "Del. LL", "Del. AUC", "Del. ECE", "Del. Sig\\%"],
        t1_rows,
        resize=True,
    )

    stress_rows = []
    for clab, cev in (("Power-Law Stress", ev1_pl), ("Long-Gap (4d) Stress", ev1_lg)):
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
            del_sig = "100.0\\%" if mkey == "oracle" else f"{sig['delayed_equal_horizon_log_loss'][mkey]*100:.1f}\\%"
            stress_rows.append([
                esc(clab), esc(mname),
                fmt_ms_tex(sm["ordinary"]["log_loss"], 4),
                fmt_ms_tex(sm["ordinary"]["roc_auc"], 4),
                fmt_ms_tex(sm["day_1"]["log_loss"], 4),
                fmt_ms_tex(sm["day_7"]["log_loss"], 4),
                fmt_ms_tex(sm["day_30"]["log_loss"], 4),
                fmt_ms_tex(sm["delayed_equal_horizon_log_loss"], 4),
                del_sig,
            ])
    t6_2_tex = make_table_env(
        "Tier 1 Stress-Cohort Generalization (Power-Law Forgetting and 4-Day Long-Gap Breaks, 5 Seeds).",
        "tab:t1_stress",
        "l l r r r r r r r",
        ["Cohort", "Model", "Ord. LL", "Ord. AUC", "1d LL", "7d LL", "30d LL", "Del. LL", "Del. Sig\\%"],
        stress_rows,
        resize=True,
    )

    bins_t1b = compute_t1b_bins()
    t6_3_tex = make_table_env(
        r"Reliability (Calibration) Check for \protect\texttt{monotonic\_gru} on \protect\texttt{main\_test} (5-Seed Average). Guesses match reality closely.",
        "tab:t1_calibration",
        "l r r r r r r",
        ["Probability Bucket", "Ordinary Count", "Ord. Guess", "Ord. Real Pass", "Delayed Count", "Del. Guess", "Del. Real Pass"],
        [[b_o[0], b_o[1], b_o[2], b_o[3], b_d[1], b_d[2], b_d[3]] for b_o, b_d in zip(bins_t1b["ordinary"], bins_t1b["assessment"])],
        font_size=r"\footnotesize",
    )

    bm25_t = T2["arms"]["bm25_alone"]["test"]
    hyb_t = T2["arms"]["hybrid_tfidf"]["test"]
    t6_4_tex = make_table_env(
        "Tier 2 Held-Out Retrieval Diagnostics ($N=132$ Queries / 192 Tasks, plus 24-Query Cross-Learner Probe = 0 Leaks).",
        "tab:t2_retrieval",
        "l c c r r r r",
        ["Search Arm", "Chat ($k=4$) Relevance", "Q-Gen ($k=3/2$) Relevance", "Meta Match\\%", "Student Leaks", "Underfills", "p95 Speed"],
        [
            ["BM25 Alone (Keywords)", "[Pending Human Labels]", "[Pending Human Labels]", f"{bm25_t['metadata_match_rate']*100:.1f}\\%", f"{bm25_t['cross_learner_violation_total']} / 192", f"{bm25_t['bm25_underfilled_queries']} / 192", f"{bm25_t['p95_latency_ms']:.2f} ms"],
            ["Deployed BM25 + TF-IDF", "[Pending Human Labels]", "[Pending Human Labels]", f"{hyb_t['metadata_match_rate']*100:.1f}\\%", f"{hyb_t['cross_learner_violation_total']} / 192", f"{hyb_t['bm25_underfilled_queries']} / 192", f"{hyb_t['p95_latency_ms']:.2f} ms"],
            ["Dense Encoder (Dev-Sel.)", "[Pending Dense \\& Human]", "[Pending Dense \\& Human]", "[Pending]", "[Pending]", "---", "[Pending]"],
            ["BM25 + Dense (Abl. 1)", "[Pending Dense \\& Human]", "[Pending Dense \\& Human]", "[Pending]", "[Pending]", "[Pending]", "[Pending]"],
        ],
        resize=True,
    )

    t6_5_tex = make_table_env(
        "Tier 3 and Ablation 1--2 blinded evaluation packs (ready to execute with live LLM + 2 human reviewers).",
        "tab:t3_packs",
        r"p{3.4cm} p{5.2cm} p{3.8cm} p{2.7cm}",
        ["Tier / Test", "What It Measures", "Prepared Blind Pack Size", "Status"],
        [
            ["Tier 3a + Ablation 2 (EduGuard ON/OFF)", "Python correctness, faithfulness, 3 rubric scores, answer leakage vs. over-refusal", "216 prompts (96 adaptation + 60 protected + 60 legit doubts)", "[Pending Live LLM \\& Human]"],
            ["Tier 3b (Question Generator)", "Usable fresh question rate, first-pass \\& post-repair validity, bank fallback rate", "180 requests (6 skills $\\times$ 5 formats $\\times$ 3 difficulties $\\times$ 2)", "[Pending Live LLM \\& Human]"],
            ["Tier 3c (Explanation Grader)", "MAE vs. human 0--1 scores, full-success P/R/F1, confidence-vs-error check", "120 prompts across 18 production prompts (\\texttt{rubric=None})", "[Pending Human Corpus \\& LLM]"],
            ["Ablation 1 Downstream (Fixed-LLM Check)", "Does BM25+Dense improve real tutor replies \\& questions over BM25+TF-IDF?", "132 items (72 chat + 60 question requests)", "[Pending Live LLM \\& Human]"],
        ],
        font_size=r"\footnotesize",
    )

    pr4 = T4["pre_registered"]
    rt4 = T4["retrospective"]
    t4a_rows = []
    for run_lbl, blk in (("Pre-Reg (10s, N=400)", pr4), ("Retro (3s, N=200)", rt4)):
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
                    ci_s = fmt_pp_ci_tex(pi[f"{pkey} minus threshold_heuristic"]["observed_delayed"])
                else:
                    d_pp = (sp["delayed_mean"]["mean"] - sm["threshold_heuristic"]["delayed_mean"]["mean"]) * 100
                    ci_s = f"{d_pp:+.2f} pp"
                t4a_rows.append([
                    run_lbl if pkey == "delayed_retention_ppo" else "",
                    clab if pkey == "delayed_retention_ppo" else "",
                    plab,
                    fmt_ms_tex(sp["day_1"], pct=True),
                    fmt_ms_tex(sp["day_7"], pct=True),
                    fmt_ms_tex(sp["day_30"], pct=True),
                    fmt_ms_tex(sp["delayed_mean"], pct=True),
                    ci_s,
                    fmt_ms_tex(sp["latent_mean_floor"], 3, False),
                ])
    t6_6_tex = make_table_env(
        "Tier 4 Primary Outcome: Unaided Delayed Pass Rates at Study Budget $B = 32.0$ (Pre-Registered vs. Retrospective).",
        "tab:t4_primary",
        "l l l r r r r l r",
        ["Run", "Cohort", "Policy", "1-Day", "7-Day", "30-Day", "Delayed Mean", "Diff vs. Heuristic (95\\% CI)", "Floor $b_t$"],
        t4a_rows,
        resize=True,
    )

    pr4_main = pr4["evaluation"]["main_test"]
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
            *[fmt_ms_tex(ps[s], pct=True) for s in skills],
            fmt_ms_tex(sp["min_skill_day_30"], pct=True),
        ])
    t6_7_tex = make_table_env(
        r"\protect\texttt{main\_test} Day-30 Unaided Pass Rates Across All 6 Python Skills (Pre-Registered, 10 Seeds, $N=400$). Lowest skill is \protect\texttt{function\_returns} for all policies.",
        "tab:t4_skills",
        "l r r r r r r r",
        ["Policy", "Range", "Loops", "Accum.", "Cond.", "Lists", "Func. Ret.", "Lowest Skill"],
        skill_rows,
        resize=True,
    )

    abl_rows = []
    for ckey, clab in (("main_test", "Main Test"), ("power_law_test", "Power-Law Test"), ("long_gap_test", "Long-Gap Test")):
        ev = pr4["evaluation"][ckey]
        s_full = ev["summary"]["delayed_retention_ppo"]
        s_abl = ev["summary"]["ppo_no_longterm_state"]
        ci = ev["paired_intervals"]["delayed_retention_ppo minus ppo_no_longterm_state"]
        abl_rows.append([clab, "Full PPO (36 dims)", fmt_ms_tex(s_full["delayed_mean"], pct=True), fmt_ms_tex(s_full["day_30"], pct=True), "---", fmt_ms_tex(s_full["latent_mean_floor"], 3, False), f"{s_full['action_shares']['request_self_explanation']*100:.1f}\\%"])
        abl_rows.append(["", "Ablated PPO (23 dims)", fmt_ms_tex(s_abl["delayed_mean"], pct=True), fmt_ms_tex(s_abl["day_30"], pct=True), f"Del: {fmt_pp_ci_tex(ci['observed_delayed'])}; D30: {fmt_pp_ci_tex(ci['observed_day_30'])}", fmt_ms_tex(s_abl["latent_mean_floor"], 3, False), f"{s_abl['action_shares']['request_self_explanation']*100:.1f}\\%"])
    t6_8_tex = make_table_env(
        "Ablation 3 Results (10 Retrained Seeds, $N=400$). Hiding long-term memory cuts accuracy and triples seed instability.",
        "tab:t4_ablation3",
        "l l r r l r r",
        ["Cohort", "Model", "Delayed Mean", "Day-30 Pass", "Paired Loss from Ablation (95\\% CI)", "Floor $b_t$", "Self-Expl. \\%"],
        abl_rows,
        resize=True,
    )

    bc = pr4["budget_curves"]
    bc_rows = []
    for b in (16.0, 24.0, 32.0, 40.0):
        bdata = bc[f"budget_{b:g}"]["main_test"]
        bsm = bdata["summary"]
        bpi = bdata["paired_intervals"]
        for pkey, plab in (("delayed_retention_ppo", "Delayed PPO (Ours)"), ("threshold_heuristic", "Threshold Heuristic"), ("myopic_rl_bandit", "Myopic RL Bandit")):
            sp = bsm[pkey]
            ci_s = "0.00 pp (Ref)" if pkey == "threshold_heuristic" else fmt_pp_ci_tex(bpi[f"{pkey} minus threshold_heuristic"]["observed_delayed"])
            bc_rows.append([f"{int(b)}" if pkey == "delayed_retention_ppo" else "", plab, fmt_ms_tex(sp["delayed_mean"], pct=True), fmt_ms_tex(sp["day_30"], pct=True), fmt_ms_tex(sp["latent_mean_floor"], 3, False), ci_s])
    t6_9_tex = make_table_env(
        r"\protect\texttt{main\_test} Budget Curves across $B = 16$ to $40$ (Pre-Registered, 10 Seeds, $N=400$). PPO leads at every budget.",
        "tab:t4_budget",
        "r l r r r l",
        ["Budget ($B$)", "Policy", "Delayed Mean", "Day-30 Pass", "Floor $b_t$", "Diff vs. Heuristic (95\\% CI)"],
        bc_rows,
        resize=True,
    )

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
            f"{sh['give_easier_question']*100:.1f}\\%",
            f"{sh['give_same_difficulty_question']*100:.1f}\\%",
            f"{sh['give_harder_question']*100:.1f}\\%",
            f"{sh['explain_concept']*100:.1f}\\%",
            f"{sh['show_worked_example']*100:.1f}\\%",
            f"{sh['request_self_explanation']*100:.1f}\\%",
            f"{sh['review_prerequisite']*100:.1f}\\%",
            f"{sh['advance_next_topic']*100:.1f}\\%",
        ])
    t6_10_tex = make_table_env(
        r"Share of each teaching action chosen on \protect\texttt{main\_test} (Pre-Registered, 10 Seeds, $N=400$).",
        "tab:t4_actions",
        "l r r r r r r r r",
        ["Policy", "Easier", "Same", "Harder", "Explain", "Example", "Self-Expl.", "Prereq", "Advance"],
        act_rows,
        resize=True,
    )

    pi1 = ev1_main["paired_intervals"]
    d30_ci_main = pr4_main["paired_intervals"]["delayed_retention_ppo minus threshold_heuristic"]["observed_day_30"]
    d30_ci_lg = pr4["evaluation"]["long_gap_test"]["paired_intervals"]["delayed_retention_ppo minus threshold_heuristic"]["observed_day_30"]

    ch6_tex = f"""\\chapter{{Results and Performance Analysis}}
\\label{{chap:results}}

\\section{{Tier 1 --- Does the Brain Guess Accurately and Obey Common Sense?}}
Before an AI teacher can plan good lessons, its memory brain must predict how likely a student is to pass a new question today (Ordinary Practice) and 1, 7, or 30 days later (Delayed Assessment). We trained all 9 models across 5 pre-registered seeds on 1,600 training students and tested them on 200 unseen students per cohort (Tables~\\ref{{tab:t1_main}} and~\\ref{{tab:t1_stress}}, Figure~\\ref{{fig:t1_models}}).

{t6_1_tex}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{figures/fig1_tier1_models.png}}
\\caption{{Tier 1 Log Loss (lower is better) and ROC-AUC (higher is better) on \\protect\\texttt{{main\\_test}} (5 seeds).}}
\\label{{fig:t1_models}}
\\end{{figure}}

{t6_2_tex}

\\begin{{itemize}}[leftmargin=1.5em]
  \\item \\textbf{{Why the 6/6 Sanity Gate Matters:}} In 5-seed validation, \\texttt{{time\\_decay\\_gru}} got a slightly lower raw Log Loss (0.55173), but it FAILED the 6/6 sanity gate (5/6) because its predictions drifted upward over 0--60 days after a mistake. Our gate automatically rejected it.
  \\item \\textbf{{Monotonic GRU vs. Transformers \\& Time-Input GRU:}} In the earlier Retrospective (3-seed) run, \\texttt{{monotonic\\_gru}} ranked \\#1 on validation selection (0.55175 vs.\\ 0.55177 for \\texttt{{time\\_input\\_gru}}). In the fresh Pre-Registered (5-seed) run, both models are statistically tied within 0.00008 on validation (0.55175 vs.\\ 0.55183), while \\texttt{{monotonic\\_gru}} has the smallest seed-to-seed spread ($\\pm$0.00011), beats \\texttt{{time\\_aware\\_transformer}} on Ordinary Log Loss ({fmt_ll_ci_tex(pi1['ordinary']['monotonic_gru minus time_aware_transformer'])}), beats \\texttt{{history\\_boosting}} ({fmt_ll_ci_tex(pi1['ordinary']['monotonic_gru minus history_boosting'])}), and uniquely exposes the explicit permastore floor $b_t$ and half-life $h_t$ in a tiny 32-float Markov state needed by PPO.
\\end{{itemize}}

Table~\\ref{{tab:t1_calibration}} and Figure~\\ref{{fig:t1_reliability}} verify that \\texttt{{monotonic\\_gru}} probabilities match real observed pass rates across all 10 probability buckets.

{t6_3_tex}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figures/fig2_tier1_reliability.png}}
\\caption{{Reliability diagram for \\protect\\texttt{{monotonic\\_gru}} on \\protect\\texttt{{main\\_test}} (Pre-Registered 5-seed average).}}
\\label{{fig:t1_reliability}}
\\end{{figure}}

\\section{{Tier 2 --- Can the Search Engine Find the Right Lessons Fast?}}
When a student asks a doubt ($k=4$ chunks) or the tutor creates a new question ($k=3$ examples + $k=2$ lesson chunks), the search engine must be fast, pick the right topic/difficulty, and never mix up different students' private mistakes. Offline checks ran on 66 Dev and 132 Held-Out Test queries (192 test retrieval tasks + 24 cross-learner leakage probes; Table~\\ref{{tab:t2_retrieval}}).

{t6_4_tex}

Both offline search arms run in under 5.1~ms (p95), match the exact requested skill/difficulty/format 100.0\\% of the time, and show 0 cross-learner privacy leaks. Human relevance labels (1,090 query-document pairs exported) and dense weights are pending.

\\section{{Tier 3 --- Tutor Replies, Safety Guard, Questions, and Grading (Pending Live LLM)}}
Zero live LLM API calls or fake human scores were run in this offline phase. Instead, all blinded prompt packets and isolated database test hooks are prepared and ready for the live-LLM + 2-reviewer human grading phase (Table~\\ref{{tab:t3_packs}}).

{t6_5_tex}

\\section{{Tier 4 --- Does the AI Teacher Actually Help Students Remember 30 Days Later?}}

\\subsection{{Main Result (Primary Outcome \\& Day-30 Retention)}}
We trained our RL teacher (Delayed PPO) across 10 brand-new seeds and tested it on 400 brand-new simulated students (seeds 40000--40399) across all three environments at budget $B = 32.0$. We also keep our earlier 3-seed Retrospective run (200 students) cleanly separated in Table~\\ref{{tab:t4_primary}} and Figure~\\ref{{fig:t4_policies}}.

{t6_6_tex}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{figures/fig3_tier4_policies.png}}
\\caption{{Pre-Registered (10 seeds, $N=400$) Mean Delayed Pass Rate and Day-30 Pass Rate across all 3 cohorts.}}
\\label{{fig:t4_policies}}
\\end{{figure}}

\\begin{{itemize}}[leftmargin=1.5em]
  \\item \\textbf{{Clear Winner at 30 Days:}} On Main Test ($N=400$), Delayed PPO beats the Threshold Heuristic by $+2.50$ pp on overall delayed pass rate ({pr4_main['summary']['delayed_retention_ppo']['delayed_mean']['mean']*100:.2f}\\% vs.\\ {pr4_main['summary']['threshold_heuristic']['delayed_mean']['mean']*100:.2f}\\%) and by {fmt_pp_ci_tex(d30_ci_main)} on Day 30. Under harsh 4-day gaps (Long-Gap Test), the Day-30 lead stays strong at {fmt_pp_ci_tex(d30_ci_lg)}.
  \\item \\textbf{{Why Short-Term Greedy AI Fails:}} Myopic RL Bandit only cares about the very next turn (no future planning). It gets stuck drilling easy questions forever, scoring $\\sim$10 percentage points worse than Delayed PPO at Day 30.
\\end{{itemize}}

\\subsection{{Fairness Across All 6 Python Skills (Day-30 Per-Skill Pass Rates)}}
We checked all 6 individual Python skills at Day 30 on Main Test ($N=400$) to verify our AI teacher does not ignore hard topics just to boost its average score (Table~\\ref{{tab:t4_skills}}).

{t6_7_tex}

\\subsection{{Ablation 3 --- Do Long-Term Memory Features Actually Cause the Gain?}}
In Ablation 3, we retrained a clean 23-dim PPO (\\texttt{{ppo\\_no\\_longterm\\_state}}) across all 10 seeds with all 13 long-term features (7d/30d predictions and floor $b_t$) removed from its input, its action mask, and its topic routing---leaving only short-term 'today' predictions ($p_{{\\text{{med}}}}$). Results appear in Table~\\ref{{tab:t4_ablation3}} and Figure~\\ref{{fig:t4_ablation3}}.

{t6_8_tex}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figures/fig4_ablation3.png}}
\\caption{{Ablation 3: Removing 30-day memory \\& floor $b_t$ drops performance and makes training $\\sim$3$\\times$ noisier ($\\pm 1$ SD bars).}}
\\label{{fig:t4_ablation3}}
\\end{{figure}}

\\subsection{{Study Budget Curves (Budget $B = 16, 24, 32,$ and $40$)}}
Table~\\ref{{tab:t4_budget}} and Figure~\\ref{{fig:t4_budget}} show that Delayed PPO outperforms baselines across every tested study budget from 16 to 40.

{t6_9_tex}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{figures/fig5_budget_curve.png}}
\\caption{{Day-30 pass rate as study budget grows from 16 to 40 across all 3 test cohorts.}}
\\label{{fig:t4_budget}}
\\end{{figure}}

\\subsection{{What Teaching Actions Does the AI Teacher Pick?}}
Table~\\ref{{tab:t4_actions}} breaks down how often each policy chooses each of the 8 teaching actions on \\texttt{{main\\_test}}.

{t6_10_tex}

Notice how Threshold Heuristic never uses worked examples (0\\%), self-explanations (0\\%), or prerequisite reviews (0\\%). Delayed PPO mixes worked examples (16.4\\%), self-explanations (16.9\\%), prerequisite reviews (8.1\\%), and harder questions (15.1\\%) to lock knowledge into durable memory. Across the 10 pre-registered seeds, 10/10 seeds beat all baselines on delayed retention and floor $b_t$, while 8/10 seeds pass all 5 behavioral action-balance checks on Main and Power-Law tests (7/10 on Long-Gap).

\\section{{Key Takeaways and Pending Live-LLM Next Step}}
\\begin{{enumerate}}[leftmargin=1.5em]
  \\item \\textbf{{Delayed PPO Wins Where It Counts (30 Days Later):}} At equal study budget (32.0), Delayed PPO improves 30-day unaided pass rates by $+5.69$ pp on Main Test, $+4.69$ pp on Power-Law stress, and $+5.40$ pp on 4-day Long-Gap breaks over the rule-based tutor.
  \\item \\textbf{{Long-Term Memory Features Are Essential (Ablation 3):}} Hiding 30-day predictions and permastore floor $b_t$ cuts retention across every cohort and roughly triples training variance across seeds.
  \\item \\textbf{{Common-Sense Memory Sanity Gate Protects Reliability:}} \\texttt{{monotonic\\_gru}} passes all 6/6 behavioral sanity checks across all seeds, matches reality closely on 10-bin calibration (ECE-10 = 0.0108 ordinary, 0.0235 delayed), and gives PPO the explicit $b_t$ signal it needs.
  \\item \\textbf{{Fast, Leak-Free Search Ready for Live Evaluation:}} Hybrid retrieval runs in $\\sim$5~ms p95 with 100\\% metadata match and zero cross-learner privacy leaks.
\\end{{enumerate}}

\\noindent\\textbf{{Pending Next Step (Live-LLM Reminder):}} All offline experiments (Tiers 1, 2-offline, 4, and Ablation 3) are finished. The next step is running the live-LLM evaluation with failover disabled and 2 blinded human reviewers on the prepared packs (Tier 2 dense/relevance, Ablation 1 downstream, Tier 3a + Ablation 2 EduGuard ON/OFF, Tier 3b question generation, and Tier 3c self-explanation grading).
"""
    write_file(OUT_DIR / "chapters" / "ch6_results.tex", ch6_tex)
    print(f"Generated modular LaTeX project under {OUT_DIR}")


if __name__ == "__main__":
    build_latex()
