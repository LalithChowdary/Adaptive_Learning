"""Front Matter builder for the SNU B.Tech CSE thesis report.

Builds:
  1. Title Page (verbatim hierarchy from Adaptive_Tutoring_Report_With_Advisor_Signature.docx,
     adapted cleanly to US Letter 0.85 in margins, dated November 2026)
  2. Certificate (formal SNU B.Tech CSE supervisor certificate)
  3. Declaration (verbatim academic honesty declaration + 3 student sign-off blocks)
  4. Acknowledgement
  5. Abstract (beginner-friendly summary + Keywords)
  6. Table of Contents (uses ctx.get("page_map", {}) without fake page numbers)
  7. List of Tables (uses ctx.get("page_map", {}) without fake page numbers)
  8. List of Figures (uses ctx.get("page_map", {}) without fake page numbers)
"""
from pathlib import Path

from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor

from .style import (
    add_centered_p,
    add_h1,
    add_mixed_p,
    add_p,
    configure_page_geometry,
    set_arabic_footer,
    set_roman_footer,
    set_run_font,
)

DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

STUDENTS = [
    ("Lalith Chowdary Vasireddy", "231010343"),
    ("Srinivas Tirumalaraju", "2310110305"),
    ("Tanuj Kinjarapu", "2310110327"),
]

DECLARATION_TEXT = (
    "We declare that this written submission represents our ideas in our own words and where others\u2019 "
    "ideas or words have been included, we have adequately cited and referenced the original sources. "
    "We also declare that we have adhered to all principles of academic honesty and integrity and have "
    "not misrepresented or fabricated or falsified any idea, data, fact, or source in our submission. "
    "We understand that any violation of the above will be cause for disciplinary action by the "
    "University and can also evoke penal action from the sources which have thus not been properly "
    "cited or from whom proper permission has not been taken when needed."
)

DEFAULT_TOC_ENTRIES = [
    (0, "Certificate", "Certificate"),
    (0, "Declaration", "Declaration"),
    (0, "Acknowledgement", "Acknowledgement"),
    (0, "Abstract", "Abstract"),
    (0, "Table of Contents", "Table of Contents"),
    (0, "List of Tables", "List of Tables"),
    (0, "List of Figures", "List of Figures"),
    (0, "Chapter 1 \u2014 Introduction", "Chapter 1 \u2014 Introduction"),
    (1, "1.1 Motivation and Problem Statement", "1.1 Motivation and Problem Statement"),
    (1, "1.2 Evolution of Our Research Across Three Stages", "1.2 Evolution of Our Research Across Three Stages"),
    (1, "1.3 System Overview and Key Contributions", "1.3 System Overview and Key Contributions"),
    (1, "1.4 Scope, Honesty Boundaries, and Thesis Organization", "1.4 Scope, Honesty Boundaries, and Thesis Organization"),
    (0, "Chapter 2 \u2014 Literature Survey", "Chapter 2 \u2014 Literature Survey"),
    (1, "2.1 Knowledge Tracing: From Rule-Based Mastery to Deep Sequences", "2.1 Knowledge Tracing: From Rule-Based Mastery to Deep Sequences"),
    (1, "2.2 Human Memory, Spaced Repetition, and Long-Term Retention", "2.2 Human Memory, Spaced Repetition, and Long-Term Retention"),
    (1, "2.3 Reinforcement Learning for Adaptive Tutoring", "2.3 Reinforcement Learning for Adaptive Tutoring"),
    (1, "2.4 Grounded LLM Tutoring, Hybrid RAG, and Anti-Leakage Guardrails", "2.4 Grounded LLM Tutoring, Hybrid RAG, and Anti-Leakage Guardrails"),
    (1, "2.5 Research Gaps Addressed by This Thesis", "2.5 Research Gaps Addressed by This Thesis"),
    (0, "Chapter 3 \u2014 Datasets and Methodology", "Chapter 3 \u2014 Datasets and Methodology"),
    (1, "3.1 Historical Foundations: Stage 1 (LoRA Style Tuning) and Stage 2 (ASSISTments 2012\u20132013)", "3.1 Historical Foundations: Stage 1 (LoRA Style Tuning) and Stage 2 (ASSISTments 2012\u20132013)"),
    (1, "3.2 Custom Python Curriculum and Dataset Evolution (v1 to v2.2)", "3.2 Custom Python Curriculum and Dataset Evolution (v1 to v2.2)"),
    (1, "3.3 Learner-Memory Architecture and 9-Model Benchmark Methodology", "3.3 Learner-Memory Architecture and 9-Model Benchmark Methodology"),
    (1, "3.4 Closed-Loop Reinforcement Learning Methodology (Gated Option B PPO)", "3.4 Closed-Loop Reinforcement Learning Methodology (Gated Option B PPO)"),
    (1, "3.5 Hybrid RAG, Question Generation, and Safety Guardrail Methodology", "3.5 Hybrid RAG, Question Generation, and Safety Guardrail Methodology"),
    (0, "Chapter 4 \u2014 Implementation", "Chapter 4 \u2014 Implementation"),
    (1, "4.1 Historical Prototypes (1/ LoRA Scripts and 2/ ASSISTments Pipeline)", "4.1 Historical Prototypes (1/ LoRA Scripts and 2/ ASSISTments Pipeline)"),
    (1, "4.2 Dataset Generation and Learner-Memory Training Pipeline (3/data_Set/ and 3/model/)", "4.2 Dataset Generation and Learner-Memory Training Pipeline (3/data_Set/ and 3/model/)"),
    (1, "4.3 Closed-Loop PPO Training and Policy Serving (3/ppo/)", "4.3 Closed-Loop PPO Training and Policy Serving (3/ppo/)"),
    (1, "4.4 Recall Memory & PPO Dashboard (3/dashboard/, Port 8765)", "4.4 Recall Memory & PPO Dashboard (3/dashboard/, Port 8765)"),
    (1, "4.5 Adaptive Python Web Tutor and Glass-Box Inspector (3/adaptive_web/, Port 8780)", "4.5 Adaptive Python Web Tutor and Glass-Box Inspector (3/adaptive_web/, Port 8780)"),
    (0, "Chapter 5 \u2014 Evaluation Protocol", "Chapter 5 \u2014 Evaluation Protocol"),
    (1, "5.1 What We Are Testing and Our Main Goal", "5.1 What We Are Testing and Our Main Goal"),
    (1, "5.2 Simple Guide to Every Symbol, Variable, and Metric", "5.2 Simple Guide to Every Symbol, Variable, and Metric"),
    (1, "5.3 The Four-Tier Test Plan and Ablations", "5.3 The Four-Tier Test Plan and Ablations"),
    (1, "5.4 Fair Testing Rules", "5.4 Fair Testing Rules"),
    (0, "Chapter 6 \u2014 Results and Performance Analysis", "Chapter 6 \u2014 Results and Performance Analysis"),
    (1, "6.1 Tier 1 \u2014 Does the Brain Guess Accurately and Obey Common Sense?", "6.1 Tier 1 \u2014 Does the Brain Guess Accurately and Obey Common Sense?"),
    (1, "6.2 Tier 2 \u2014 Can the Search Engine Find the Right Lessons Fast?", "6.2 Tier 2 \u2014 Can the Search Engine Find the Right Lessons Fast?"),
    (1, "6.3 Tier 3 \u2014 Tutor Replies, Safety Guard, Questions, and Grading (Pending Live LLM)", "6.3 Tier 3 \u2014 Tutor Replies, Safety Guard, Questions, and Grading (Pending Live LLM)"),
    (1, "6.4 Tier 4 \u2014 Does the AI Teacher Actually Help Students Remember 30 Days Later?", "6.4 Tier 4 \u2014 Does the AI Teacher Actually Help Students Remember 30 Days Later?"),
    (1, "6.5 Key Takeaways and Pending Live-LLM Next Step", "6.5 Key Takeaways and Pending Live-LLM Next Step"),
    (0, "Chapter 7 \u2014 Limitations and Conclusion", "Chapter 7 \u2014 Limitations and Conclusion"),
    (1, "7.1 Summary of Verified Findings", "7.1 Summary of Verified Findings"),
    (1, "7.2 Honest Scope Boundaries and Current Limitations", "7.2 Honest Scope Boundaries and Current Limitations"),
    (1, "7.3 Future Work", "7.3 Future Work"),
    (1, "7.4 Conclusion", "7.4 Conclusion"),
    (0, "References", "References"),
]

DEFAULT_TABLE_ENTRIES = [
    ("Table 2.1", "Table 2.1 \u2014 Comparison of Prior Tutoring Literature and This Thesis Across Four Research Gaps."),
    ("Table 3.1", "Table 3.1 \u2014 Five Python task formats in the 2,160-item bank (3/data_Set/v2.2/catalog.py)."),
    ("Table 3.2", "Table 3.2 \u2014 Evolution of the custom Python dataset from v1 to v2.2 (all cohorts have 2,600 simulated learners)."),
    ("Table 3.3", "Table 3.3 \u2014 Nine benchmark memory models and two anchors evaluated in Tier 1."),
    ("Table 3.4", "Table 3.4 \u2014 Eight-action Gated Option B menu in AdaptiveTutorEnv (3/ppo/env.py)."),
    ("Table 4.1", "Table 4.1 \u2014 Mapping of project modules to source files, responsibilities, and key safeguards."),
    ("Table 5.1", "Table 5.1 \u2014 Plain-English dictionary of every symbol, variable, and metric used in Chapters 5 and 6."),
    ("Table 5.2", "Table 5.2 \u2014 Summary of the four evaluation tiers, three ablations, and optional human pilot."),
    ("Table 6.1", "Table 6.1 \u2014 Tier 1 Memory Benchmark on main_test (Pre-Registered, 5 Seeds, N=200 Held-Out Students)."),
    ("Table 6.2", "Table 6.2 \u2014 Tier 1 Stress-Cohort Generalization (Power-Law Forgetting and 4-Day Long-Gap Breaks, 5 Seeds)."),
    ("Table 6.3", "Table 6.3 \u2014 Reliability (Calibration) Check for monotonic_gru on main_test (5-Seed Average). Guesses match reality closely."),
    ("Table 6.4", "Table 6.4 \u2014 Tier 2 Held-Out Retrieval Diagnostics (N=132 Queries / 192 Tasks, plus 24-Query Cross-Learner Probe = 0 Leaks)."),
    ("Table 6.5", "Table 6.5 \u2014 Tier 3 and Ablation 1\u20132 blinded evaluation packs (ready to execute with live LLM + 2 human reviewers)."),
    ("Table 6.6", "Table 6.6 \u2014 Tier 4 Primary Outcome: Unaided Delayed Pass Rates at Study Budget B = 32.0 (Pre-Registered vs. Retrospective)."),
    ("Table 6.7", "Table 6.7 \u2014 Main Test Day-30 Unaided Pass Rates Across All 6 Python Skills (Pre-Registered, 10 Seeds, N=400). Lowest skill is function_returns for all policies."),
    ("Table 6.8", "Table 6.8 \u2014 Ablation 3 Results (10 Retrained Seeds, N=400). Hiding long-term memory cuts accuracy and triples seed instability."),
    ("Table 6.9", "Table 6.9 \u2014 Main Test Budget Curves across B = 16 to 40 (Pre-Registered, 10 Seeds, N=400). PPO leads at every budget."),
    ("Table 6.10", "Table 6.10 \u2014 Share of each teaching action chosen on main_test (Pre-Registered, 10 Seeds, N=400)."),
]

DEFAULT_FIGURE_ENTRIES = [
    ("Figure 6.1", "Figure 6.1 \u2014 Tier 1 Log Loss (lower is better) and ROC-AUC (higher is better) on main_test (5 seeds)."),
    ("Figure 6.2", "Figure 6.2 \u2014 Reliability diagram for monotonic_gru on main_test (Pre-Registered 5-seed average)."),
    ("Figure 6.3", "Figure 6.3 \u2014 Pre-Registered (10 seeds, N=400) Mean Delayed Pass Rate and Day-30 Pass Rate across all 3 cohorts."),
    ("Figure 6.4", "Figure 6.4 \u2014 Ablation 3: Removing 30-day memory & floor b_t drops performance and makes training ~3x noisier (\u00b11 SD bars)."),
    ("Figure 6.5", "Figure 6.5 \u2014 Day-30 pass rate as study budget grows from 16 to 40 across all 3 test cohorts."),
]


def _lookup_page(page_map, key, title):
    if not page_map:
        return "-"
    for candidate in (key, title):
        if candidate and candidate in page_map:
            val = str(page_map[candidate]).strip()
            if val:
                return val
    prefix = key.split(" \u2014 ")[0].strip() if key else ""
    if prefix and prefix in page_map:
        val = str(page_map[prefix]).strip()
        if val:
            return val
    return "-"


def _add_leader_line(doc, title, page_str, *, indent_in=0.0, bold=False, size=10.5, after=3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.left_indent = Inches(indent_in)
    pf.space_before = Pt(1)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.2
    pf.tab_stops.add_tab_stop(Inches(6.80), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)

    r_title = p.add_run(title)
    set_run_font(r_title, size=size, bold=bold, color=RGBColor(0, 0, 0))
    r_tab = p.add_run("\t")
    set_run_font(r_tab, size=size, bold=False, color=RGBColor(80, 88, 98))
    r_page = p.add_run(page_str)
    set_run_font(r_page, size=size, bold=bold, color=RGBColor(40, 48, 56))
    return p


def _strip_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none"/><w:bottom w:val="none"/>'
        '<w:left w:val="none"/><w:right w:val="none"/>'
        '<w:insideH w:val="none"/><w:insideV w:val="none"/>'
        '</w:tblBorders>'
    )
    tbl_pr.append(borders)


def _build_title_page(doc, assets_dir: Path):
    add_centered_p(
        doc,
        "Adaptive Tutoring Using Reinforcement Learning",
        size=24,
        bold=False,
        before=24,
        after=10,
        line_spacing=1.15,
    )
    add_centered_p(doc, "Project Report", size=11, italic=True, before=4, after=8)
    add_centered_p(
        doc,
        "Project Report Submitted in Partial Fulfilment of the Requirements for the Degree of",
        size=10.5,
        before=2,
        after=8,
    )
    add_centered_p(doc, "Bachelor of Technology", size=17, bold=True, before=0, after=2, line_spacing=1.15)
    add_centered_p(doc, "in", size=11, italic=True, before=0, after=2, line_spacing=1.15)
    add_centered_p(doc, "Computer Science and Engineering", size=17, bold=True, before=0, after=22, line_spacing=1.15)

    add_centered_p(doc, "Submitted by", size=14, italic=True, before=4, after=8)
    for idx, (name, roll) in enumerate(STUDENTS):
        add_centered_p(
            doc,
            f"{name}  ({roll})",
            size=14,
            before=0,
            after=20 if idx == len(STUDENTS) - 1 else 4,
            line_spacing=1.2,
        )

    add_centered_p(doc, "Under the Supervision of", size=17, bold=True, italic=True, before=4, after=6)
    add_centered_p(doc, "Dr. Sonia Khetarpaul", size=14, bold=True, before=0, after=2, line_spacing=1.15)
    add_centered_p(doc, "Associate Professor", size=14, bold=True, before=0, after=16, line_spacing=1.15)

    logo_path = assets_dir / "snu_logo.jpeg"
    if logo_path.exists():
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(4)
        p_logo.paragraph_format.space_after = Pt(10)
        p_logo.add_run().add_picture(str(logo_path), width=Inches(3.7), height=Inches(1.526))

    add_centered_p(doc, "Shiv Nadar University, Delhi NCR", size=11, bold=True, before=4, after=2, line_spacing=1.2)
    add_centered_p(doc, "Department of Computer Science and Engineering", size=11, bold=True, before=0, after=4, line_spacing=1.2)
    add_centered_p(doc, "November 2026", size=11, bold=False, before=0, after=0, line_spacing=1.2)


def _build_certificate(doc, assets_dir: Path):
    add_h1(doc, "Certificate", page_break=True)
    add_p(
        doc,
        'This is to certify that the project report titled "Adaptive Tutoring Using Reinforcement Learning", '
        "submitted by Lalith Chowdary Vasireddy (231010343), Srinivas Tirumalaraju (2310110305), and "
        "Tanuj Kinjarapu (2310110327) to the Department of Computer Science and Engineering, Shiv Nadar "
        "University, Delhi NCR, in partial fulfilment of the requirements for the award of the degree of "
        "Bachelor of Technology in Computer Science and Engineering, is a bonafide record of the work "
        "carried out by them under my supervision and guidance during the academic year 2026.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=10,
        after=10,
        line_spacing=1.35,
    )
    add_p(
        doc,
        "To the best of my knowledge, the results and matter embodied in this report have not been "
        "submitted to any other University or Institute for the award of any other degree or diploma.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=0,
        after=36,
        line_spacing=1.35,
    )

    sig_table = doc.add_table(rows=1, cols=2)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _strip_table_borders(sig_table)
    left_cell, right_cell = sig_table.rows[0].cells
    left_cell.width = Inches(3.2)
    right_cell.width = Inches(3.6)

    p_left = left_cell.paragraphs[0]
    p_left.paragraph_format.space_before = Pt(28)
    p_left.paragraph_format.space_after = Pt(4)
    p_left.paragraph_format.line_spacing = 1.3
    r_l = p_left.add_run("Date: November 2026\nPlace: Greater Noida, Delhi NCR")
    set_run_font(r_l, size=11)

    p_right = right_cell.paragraphs[0]
    p_right.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_right.paragraph_format.space_before = Pt(0)
    p_right.paragraph_format.space_after = Pt(4)
    p_right.paragraph_format.line_spacing = 1.25

    sig_path = assets_dir / "advisor_signature.png"
    if sig_path.exists():
        r_img_lbl = p_right.add_run("Signature:  ")
        set_run_font(r_img_lbl, size=11)
        p_right.add_run().add_picture(str(sig_path), width=Inches(0.59), height=Inches(0.59))
        r_br = p_right.add_run("\n")
        set_run_font(r_br, size=11)
    else:
        r_line = p_right.add_run("Signature: __________________________\n")
        set_run_font(r_line, size=11)

    r_sup = p_right.add_run(
        "Dr. Sonia Khetarpaul\n"
        "Associate Professor\n"
        "Department of Computer Science and Engineering\n"
        "Shiv Nadar University, Delhi NCR"
    )
    set_run_font(r_sup, size=11, bold=False)


def _build_declaration(doc):
    add_h1(doc, "Declaration", page_break=True)
    add_p(
        doc,
        DECLARATION_TEXT,
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=8,
        after=24,
        line_spacing=1.35,
    )
    for name, roll in STUDENTS:
        add_p(doc, "Signature: __________________________", size=11, before=12, after=2, line_spacing=1.2)
        add_p(doc, f"{name}\nRoll number: {roll}", size=11, before=0, after=8, line_spacing=1.25)
    add_p(doc, "Date: ____________________", size=11, before=14, after=6, line_spacing=1.2)


def _build_acknowledgement(doc):
    add_h1(doc, "Acknowledgement", page_break=True)
    add_p(
        doc,
        "We express our sincere gratitude to our project supervisor, Dr. Sonia Khetarpaul, Associate Professor "
        "in the Department of Computer Science and Engineering at Shiv Nadar University, Delhi NCR, for her "
        "steady guidance, constructive criticism, and insightful feedback throughout every stage of this work.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=8,
        after=10,
        line_spacing=1.35,
    )
    add_p(
        doc,
        "We are thankful to the faculty and staff of the Department of Computer Science and Engineering, "
        "Shiv Nadar University, Delhi NCR, for providing a strong academic foundation and computing "
        "environment that enabled us to carry out our experiments systematically.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=0,
        after=10,
        line_spacing=1.35,
    )
    add_p(
        doc,
        "Finally, we thank our classmates, friends, and families for their patience, encouragement, and "
        "moral support during the design, implementation, and verification of this project.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=0,
        after=20,
        line_spacing=1.35,
    )
    add_p(
        doc,
        "Lalith Chowdary Vasireddy (231010343)\n"
        "Srinivas Tirumalaraju (2310110305)\n"
        "Tanuj Kinjarapu (2310110327)",
        size=11,
        bold=False,
        before=8,
        after=0,
        line_spacing=1.35,
    )


def _build_abstract(doc):
    add_h1(doc, "Abstract", page_break=True)
    add_p(
        doc,
        "Answering a Python question right after seeing a hint does not mean a student will remember it "
        "weeks later. Most automated tutors track only immediate accuracy and follow fixed rules, missing "
        "fragile knowledge that fades across multi-day breaks.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=6,
        after=8,
        line_spacing=1.3,
    )
    add_p(
        doc,
        "This thesis develops an adaptive Python tutor designed for long-term retention across three "
        "research stages. Stage 1 explored LoRA style fine-tuning on Llama-3.1-8B-Instruct-4bit, showing "
        "that tone changes alone cannot model forgetting or choose lessons. Stage 2 compared sequence "
        "models on 6,103,356 retained interactions from ASSISTments 2012\u20132013, where LSTM-DKT "
        "(0.50280 test log loss, 0.7852 ROC-AUC) beat SAKT-Transformer (0.52305, 0.7698), though "
        "55.71% missing skill tags and lack of delayed unaided tests motivated a controlled Stage 3 "
        "environment.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=0,
        after=8,
        line_spacing=1.3,
    )
    add_p(
        doc,
        "In Stage 3, we built dataset v2.2\u2014with 6 Python skills, 5 formats, 2,160 items, and a "
        "simulator combining Half-Life Regression with a Consolidated Permastore floor b_t (a durable "
        "long-term memory baseline)\u2014and benchmarked 9 memory models over 5 pre-registered seeds. "
        "Both time_input_gru (0.55175 validation log loss) and monotonic_gru (0.55183, tightest seed "
        "variance \u00b10.00011) passed all 6/6 behavioral sanity checks, whereas unconstrained "
        "time_decay_gru failed (5.0/6). Using monotonic_gru's compact 32-float state, we trained an "
        "8-action Gated Option B teacher (delayed_retention_ppo) with Proximal Policy Optimization "
        "(PPO). Across 10 pre-registered seeds and N = 400 fresh simulated learners at matched budget "
        "(B = 32.0), delayed_retention_ppo outperformed threshold_heuristic by +2.50 percentage points "
        "(pp) in delayed mean pass rate (62.77% vs. 60.27%, 95% CI [+2.12, +2.86]) and +5.69 pp at "
        "Day 30 (59.66% vs. 53.97%, 95% CI [+5.05, +6.30]), holding under power-law forgetting and "
        "4-day gaps. Ablation 3 confirmed that hiding long-term memory features lowers delayed mean "
        "retention to 61.76% (Day-30 to 58.82%) and roughly triples delayed-mean seed spread "
        "(\u00b11.12% to \u00b13.28%; Day-30 spread \u00b11.92% to \u00b14.24%). We also built the "
        "3/adaptive_web tutor with Hybrid RAG (FTS5 BM25 + TF-IDF + RRF), achieving 95th-percentile "
        "latency (p95) under 5.2 ms with 0 leaks, alongside difficulty-locked question generation, "
        "AST grading, and EduGuard.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=0,
        after=8,
        line_spacing=1.3,
    )
    add_p(
        doc,
        "Memory and policy gains are verified in simulation; dense retrieval weights, live-LLM human "
        "evaluations (T-A1, T3a\u2013T3c), and an optional IRB-gated human pilot (T5) remain pending "
        "with zero fabricated scores.",
        size=11,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=0,
        after=10,
        line_spacing=1.3,
    )
    add_mixed_p(
        doc,
        [
            ("Keywords: ", True, False),
            (
                "adaptive tutoring, reinforcement learning, proximal policy optimization, "
                "knowledge tracing, spaced repetition, retrieval-augmented generation.",
                False,
                True,
            ),
        ],
        size=10.5,
        before=2,
        after=6,
    )


def _build_toc_and_lists(doc, ctx: dict):
    page_map = ctx.get("page_map") or {}

    # Table of Contents
    add_h1(doc, "Table of Contents", page_break=True)
    toc_entries = ctx.get("toc_entries") or DEFAULT_TOC_ENTRIES
    for entry in toc_entries:
        if len(entry) == 3:
            level, key, title = entry
        else:
            level, title = entry
            key = title
        pg = _lookup_page(page_map, key, title)
        _add_leader_line(
            doc,
            title,
            pg,
            indent_in=0.25 if level >= 1 else 0.0,
            bold=(level == 0),
            size=10.5 if level == 0 else 10.0,
            after=4 if level == 0 else 2,
        )

    # List of Tables
    add_h1(doc, "List of Tables", page_break=True)
    table_entries = ctx.get("table_entries") or DEFAULT_TABLE_ENTRIES
    for entry in table_entries:
        if isinstance(entry, (tuple, list)):
            key, caption = entry[0], entry[1]
        else:
            caption = str(entry)
            key = caption.split(" \u2014 ")[0].strip()
        pg = _lookup_page(page_map, key, caption)
        _add_leader_line(doc, caption, pg, indent_in=0.0, bold=False, size=10.0, after=4)

    # List of Figures
    add_h1(doc, "List of Figures", page_break=True)
    figure_entries = ctx.get("figure_entries") or DEFAULT_FIGURE_ENTRIES
    for entry in figure_entries:
        if isinstance(entry, (tuple, list)):
            key, caption = entry[0], entry[1]
        else:
            caption = str(entry)
            key = caption.split(" \u2014 ")[0].strip()
        pg = _lookup_page(page_map, key, caption)
        _add_leader_line(doc, caption, pg, indent_in=0.0, bold=False, size=10.0, after=4)


def build(doc, ctx=None):
    """Append Front Matter (Title Page through List of Figures) to `doc`."""
    ctx = ctx if ctx is not None else {}
    assets_dir = Path(ctx.get("ASSETS_DIR", DEFAULT_ASSETS_DIR))

    # Configure Section 0 (Front Matter) with US Letter 0.85 in margins & Roman pagination
    if not ctx.get("manage_sections_externally", False):
        set_roman_footer(doc.sections[0], hide_first_page=True)
    else:
        configure_page_geometry(doc.sections[0])

    _build_title_page(doc, assets_dir)
    _build_certificate(doc, assets_dir)
    _build_declaration(doc)
    _build_acknowledgement(doc)
    _build_abstract(doc)
    _build_toc_and_lists(doc, ctx)

    # Start Arabic page 1 cleanly on Chapter 1 while preserving US Letter 0.85 in margins
    if not ctx.get("manage_sections_externally", False) and not ctx.get("single_section_only", False):
        body_sec = doc.add_section(WD_SECTION.NEW_PAGE)
        set_arabic_footer(body_sec, restart_at_1=True)
        ctx["arabic_section_started"] = True
