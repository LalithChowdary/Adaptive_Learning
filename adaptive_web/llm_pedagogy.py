"""Bounded PPO-to-LLM pedagogical realizer for adaptive Python learning."""
from __future__ import annotations

import copy
import json
import math
import random
import re
from typing import TYPE_CHECKING, Any, Iterable
import uuid

from .catalogue_bridge import (
    DEFAULT_LESSONS,
    DIFFICULTIES,
    FORMATS,
    PPO_SKILLS_ORDER,
    PREREQUISITES,
    SKILL_META,
    assert_valid_task,
    get_v22_task,
    sample_v22_task,
    validate_task_schema,
)
from .grader import grade_self_explanation
from .llm_client import LLMClient, LLMClientError
from .rag import max_text_similarity

if TYPE_CHECKING:
    from .rag import HybridRAGEngine

_DIFFICULTY_GUARDRAILS: dict[str, dict[str, Any]] = {
    "easy": {
        "max_lines": 5,
        "target_seconds_min": 15,
        "target_seconds_max": 35,
        "rules": (
            "Keep snippet concise (1-5 lines), test a single core rule with straightforward inputs, "
            "no nested loops, target reference_seconds in 15..35."
        ),
    },
    "medium": {
        "max_lines": 10,
        "target_seconds_min": 35,
        "target_seconds_max": 60,
        "rules": (
            "Use 4-10 lines, 1-2 coupled variables or strided/conditional logic, include one exact "
            "boundary check, target reference_seconds in 35..60."
        ),
    },
    "hard": {
        "max_lines": 16,
        "target_seconds_min": 60,
        "target_seconds_max": 110,
        "rules": (
            "Use 7-16 lines with compound branching, nested/dependent iteration, negative step/index, "
            "or subtle edge cases (empty range, early return), target reference_seconds in 60..110."
        ),
    },
}

_ANSWER_FISHING_RE = re.compile(
    r"(?:\b(give me the answer|give the answer|tell me the answer|what is the answer|just tell me|solve this|"
    r"solve it|solve for me|which option is correct|which option|which choice|what goes in the blank|"
    r"what should go in|exact output of|tell me the output|what is the output|what does it return|"
    r"what does this return|what does this print|full code solution|correct option|restore the missing|"
    r"fill in the blank)\b|__blank_\d*__)",
    re.IGNORECASE,
)


def _clean_excerpt(text: str, max_chars: int = 210) -> str:
    """Format a clean plain-text sentence excerpt without broken Markdown or partial backticks."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = []
    for ln in raw.splitlines():
        s = re.sub(r"^#{1,6}\s+", "", ln.strip())
        s = re.sub(r"^[-*•]\s+", "", s)
        s = re.sub(r"^\d+\.\s+", "", s)
        s = s.replace("**", "")
        if s:
            lines.append(s)
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(joined) > max_chars:
        clipped = joined[:max_chars]
        for end_ch in (". ", "! ", "? "):
            pos = clipped.rfind(end_ch)
            if pos >= max_chars // 2:
                joined = clipped[: pos + 1].strip()
                break
        else:
            sp = clipped.rfind(" ")
            joined = (clipped[:sp] if sp > max_chars // 3 else clipped).rstrip(" ,;:-") + "…"
    if joined.count("`") % 2 == 1:
        last_bt = joined.rfind("`")
        if last_bt != -1:
            joined = joined[:last_bt].rstrip(" ,;:-")
    return joined


def _resolve_mastery_tier(memory_summary: dict[str, Any] | None) -> dict[str, Any]:
    mem = memory_summary or {}
    assessed = bool(mem.get("assessed", False))
    p_med = float(mem.get("p_med", 0.5))
    p_30d = float(mem.get("p_30d", 0.5))
    fragility = float(mem.get("fragility", max(0.0, p_med - p_30d)))
    floor_b = float(mem.get("floor_b", 0.0))
    obs_rate = mem.get("observed_pass_rate")

    if not assessed:
        tier = "unassessed"
        instruction = (
            "Learner is UNASSESSED on this skill: start with an intuitive foundational mental model, "
            "walk through a tiny concrete step-by-step trace, and avoid assumed jargon."
        )
    elif p_med < 0.70 or (obs_rate is not None and float(obs_rate) < 0.70):
        tier = "lacking"
        instruction = (
            f"Learner is LACKING mastery (P(Med)={p_med:.0%} < 70%): enter Remediation Mode — slow down, "
            "write an explicit step-by-step variable trace, surface common off-by-one/boundary misconceptions, "
            "and reference any prior mistakes cleanly."
        )
    elif fragility > 0.15:
        tier = "fragile"
        instruction = (
            f"Learner has FRAGILE retention (30d drop={fragility:.0%}, floor b_t={floor_b:.2f}): "
            "skip basic syntax and reinforce durable causal rules and tricky boundary traps."
        )
    else:
        tier = "mastered"
        instruction = (
            f"Learner has MASTERED this skill (P(Med)={p_med:.0%}, floor b_t={floor_b:.2f}): keep basics "
            "concise and focus on subtle edge cases, corner inputs, and prerequisite/downstream transfer."
        )

    return {
        "mastery_tier": tier,
        "assessed": assessed,
        "p_med": round(p_med, 4),
        "p_30d": round(p_30d, 4),
        "fragility": round(fragility, 4),
        "floor_b": round(floor_b, 4),
        "observed_pass_rate": None if obs_rate is None else round(float(obs_rate), 4),
        "observed_unaided_passes": int(mem.get("observed_unaided_passes", 0)),
        "observed_unaided_attempts": int(mem.get("observed_unaided_attempts", 0)),
        "instruction": instruction,
    }


_SKILL_PATH_FOCUS: dict[str, dict[str, str]] = {
    "range_boundaries": {
        "learn": "Master half-open interval semantics [start, stop) and positive/negative step termination.",
        "code": "Trace 1-arg, 2-arg, and 3-arg range() calls across empty, single-element, and descending ranges.",
        "remember": "Explain why stop is strictly excluded and verify the last emitted integer.",
    },
    "loop_tracing": {
        "learn": "Build a state-table mental model tracking loop variable updates on every iteration.",
        "code": "Trace nested updates, early conditions, and final variable values after loop termination.",
        "remember": "Articulate the exact iteration order and post-loop variable state without running code.",
    },
    "accumulator_initialization": {
        "learn": "Separate initial identity/baseline values (0 for sums, 1 for products, [] for lists) from loop updates.",
        "code": "Spot bugs where accumulators are reset inside the loop or initialized to the wrong identity.",
        "remember": "Justify why the initial value before the loop determines the final accumulated result.",
    },
    "conditional_evaluation": {
        "learn": "Understand mutually exclusive if/elif/else short-circuiting vs. independent sequential if statements.",
        "code": "Evaluate boundary inequalities (<, <=, >, >=) and trace which single branch executes.",
        "remember": "Explain why branch ordering matters when conditions overlap.",
    },
    "list_indexing": {
        "learn": "Map zero-based positive indices (0..n-1) and negative indices (-1..-n) onto sequence positions.",
        "code": "Trace dynamic index expressions, off-by-one bounds, and IndexError boundaries.",
        "remember": "Explain how negative indices translate to len(seq) + idx.",
    },
    "function_returns": {
        "learn": "Trace parameter binding, local scope evaluation, operator precedence, and immediate return exit.",
        "code": "Distinguish return values from print() side effects and trace expressions in return statements.",
        "remember": "Explain why statements after an executed return never run and how the value reaches the caller.",
    },
}


_RICH_COURSE_CURRICULUM: dict[str, dict[str, Any]] = {
    "range_boundaries": {
        "title": "Complete Guide: Python range(start, stop, step) Boundaries",
        "concept_explanation": (
            "Python's built-in range() generates an immutable arithmetic progression of integers using half-open "
            "interval semantics: [start, stop). The start value is inclusive (always included if the sequence is non-empty), "
            "while the stop value is strictly exclusive (never included). This design aligns with zero-based indexing so "
            "range(n) produces exactly n integers from 0 to n - 1."
        ),
        "theory_sections": [
            {
                "heading": "1. The Three Signatures of range()",
                "body": "range(stop) defaults to start=0 and step=1. range(start, stop) uses step=1. range(start, stop, step) advances by step each iteration. A step of 0 raises ValueError.",
            },
            {
                "heading": "2. Termination Condition (Positive vs. Negative Step)",
                "body": "When step > 0, values x satisfy start <= x < stop. When step < 0, values x satisfy start >= x > stop. If start already violates the bound (e.g., range(5, 2) with step=1), range() produces an empty sequence [].",
            },
            {
                "heading": "3. Computing the Exact Last Element",
                "body": "For positive step, the sequence stops before reaching stop. For example, range(2, 9, 3) yields 2, 5, 8 (next would be 11 >= 9). For range(2, 8, 3), it yields 2, 5 because 8 is excluded.",
            },
        ],
        "pitfalls": [
            "Off-by-one stop inclusion: range(1, 5) yields [1, 2, 3, 4], NOT [1, 2, 3, 4, 5].",
            "Negative step without swapping start/stop: range(1, 5, -1) is empty []; you must write range(5, 1, -1).",
            "Forgetting that stop is excluded even when step lands exactly on it: range(0, 10, 2) ends at 8.",
        ],
        "quick_check": "What is the final element of list(range(10, 2, -3))?",
    },
    "loop_tracing": {
        "title": "Complete Guide: Step-by-Step Loop Tracing & State Tables",
        "concept_explanation": (
            "A for-loop binds its target variable to each element yielded by the iterable, one at a time, and executes "
            "the entire indented loop body before advancing to the next element. Tracing a loop accurately requires "
            "maintaining a state table of every variable's value at the start and end of each iteration."
        ),
        "theory_sections": [
            {
                "heading": "1. Anatomy of an Iteration",
                "body": "On iteration i: (1) Python requests the next item from the iterable; (2) binds the loop variable; (3) executes body statements top-to-bottom; (4) repeats until StopIteration.",
            },
            {
                "heading": "2. Variable Mutation & Order of Operations",
                "body": "Within the loop body, statements execute sequentially. If a variable x is updated on line 2 and read on line 3, line 3 sees the updated x—not the value from the start of the iteration.",
            },
            {
                "heading": "3. Post-Loop Variable State",
                "body": "In Python, the loop variable remains in scope after the loop finishes, holding the last value assigned during the final iteration (or remaining unbound if the loop ran 0 times).",
            },
        ],
        "pitfalls": [
            "Evaluating all lines of the loop body against the old value of a variable modified mid-body.",
            "Confusing how many times the loop runs with the final value of the loop variable.",
            "Skipping iterations mentally instead of writing out each pass in a state table.",
        ],
        "quick_check": "If total starts at 0 and for i in range(1, 4): total += i * 2, what are (i, total) after each pass?",
    },
    "accumulator_initialization": {
        "title": "Complete Guide: Accumulator Initialization & Loop Updates",
        "concept_explanation": (
            "The accumulator pattern maintains a running result across loop iterations. Correct accumulation requires two "
            "distinct steps: (1) initializing the accumulator once BEFORE the loop to the proper identity or baseline "
            "value, and (2) updating it cumulatively INSIDE the loop."
        ),
        "theory_sections": [
            {
                "heading": "1. Choosing the Right Identity / Baseline",
                "body": "Summation accumulators start at 0 (additive identity), product accumulators start at 1 (multiplicative identity), string builders start at '', and list builders start at []. Non-zero baselines (e.g., starting bonus points at 10) shift the entire final result.",
            },
            {
                "heading": "2. Initialization Scope (Before vs. Inside Loop)",
                "body": "Initializing an accumulator inside the loop resets it on every iteration, discarding all prior work so only the final iteration survives.",
            },
            {
                "heading": "3. Conditional Accumulation",
                "body": "When combining an accumulator with an if-statement, only items matching the condition modify the accumulator, while non-matching items leave it unchanged.",
            },
        ],
        "pitfalls": [
            "Initializing a product accumulator to 0 instead of 1 (multiplying by 0 always yields 0).",
            "Placing total = 0 inside the for-loop body instead of before it.",
            "Using = instead of += inside the loop, overwriting previous iterations.",
        ],
        "quick_check": "Why does initializing prod = 0 before a multiplication loop always produce 0?",
    },
    "conditional_evaluation": {
        "title": "Complete Guide: Conditional Evaluation & Branch Mutual Exclusion",
        "concept_explanation": (
            "Conditional statements (if / elif / else) control execution flow using boolean expressions. In an if/elif/else "
            "chain, Python tests conditions top-to-bottom and executes AT MOST ONE branch—the very first branch whose "
            "condition evaluates to True—before jumping past the entire block."
        ),
        "theory_sections": [
            {
                "heading": "1. if/elif/else vs. Sequential if Statements",
                "body": "In an if/elif/else chain, once a branch matches, all remaining elif and else branches are skipped even if their conditions would also be True. By contrast, separate sequential if statements are tested independently and multiple can execute.",
            },
            {
                "heading": "2. Strict vs. Inclusive Boundary Operators",
                "body": "Pay close attention to < vs <= and > vs >= at exact boundary values (e.g., x == 10 satisfies x >= 10 and x <= 10, but NOT x > 10 or x < 10).",
            },
            {
                "heading": "3. Precedence & Chained Comparisons",
                "body": "Python evaluates comparisons first, then not, then and, then or. Chained comparisons like 0 <= x < 10 are equivalent to (0 <= x) and (x < 10).",
            },
        ],
        "pitfalls": [
            "Putting a broad condition (e.g., score >= 50) before a stricter one (score >= 90) in an if/elif chain, making the 90+ branch unreachable.",
            "Confusing elif (mutually exclusive) with a second independent if statement.",
            "Off-by-one errors on exact equality boundaries (< vs <=).",
        ],
        "quick_check": "Given x = 10, if x > 5: a = 1; elif x >= 10: a = 2, what is a and why?",
    },
    "list_indexing": {
        "title": "Complete Guide: Zero-Based & Negative List Indexing",
        "concept_explanation": (
            "Python lists are ordered, mutable sequences indexed by integers. Positive indices count forward from 0 "
            "(first element) to len(lst) - 1 (last element). Negative indices count backward from -1 (last element) "
            "to -len(lst) (first element)."
        ),
        "theory_sections": [
            {
                "heading": "1. Dual Coordinate System",
                "body": "For a list of length n, valid positive indices are 0, 1, ..., n-1 and valid negative indices are -n, ..., -2, -1. Every negative index k maps to positive index n + k.",
            },
            {
                "heading": "2. Valid Bounds vs. IndexError",
                "body": "Accessing lst[n] or lst[-n - 1] raises IndexError because the valid positive range stops at n - 1.",
            },
            {
                "heading": "3. Dynamic Index Expressions",
                "body": "When an index is an expression like lst[i - 1] or lst[len(lst) // 2], evaluate the integer inside the brackets completely before looking up the element.",
            },
        ],
        "pitfalls": [
            "Using lst[len(lst)] to get the last item (raises IndexError; use lst[-1] or lst[len(lst) - 1]).",
            "Assuming 1-based indexing so lst[1] returns the first item (it actually returns the second).",
            "Forgetting that -1 is the last item and -len(lst) is the first item.",
        ],
        "quick_check": "For nums = [10, 20, 30, 40], what are nums[1] and nums[-3]?",
    },
    "function_returns": {
        "title": "Complete Guide: Function Parameters, Precedence & Return Flow",
        "concept_explanation": (
            "Calling a Python function creates a local frame, binds each passed argument to its corresponding parameter, "
            "executes the body sequentially, and immediately terminates the function when a return statement is reached, "
            "handing the evaluated expression back to the caller."
        ),
        "theory_sections": [
            {
                "heading": "1. Parameter Binding & Local Scope",
                "body": "Arguments are evaluated before entering the function and bound to local parameter names. Reassigning a parameter inside the function does not rebind variables in the caller.",
            },
            {
                "heading": "2. Operator Precedence in Return Expressions",
                "body": "Python evaluates return expressions using standard precedence: parentheses (), exponentiation **, multiplicative (*, /, //, %), then additive (+, -).",
            },
            {
                "heading": "3. Immediate Exit & return vs. print()",
                "body": "Once return executes, any lines below it in the same execution path are skipped. Unlike print() (which displays text and returns None), return sends a usable value back to the call site.",
            },
        ],
        "pitfalls": [
            "Using print(result) instead of return result, causing the function to return None to the caller.",
            "Expecting code after an executed return statement to still run.",
            "Ignoring operator precedence (e.g., evaluating 2 * x + 1 left-to-right when x is negative or grouped).",
        ],
        "quick_check": "If a function executes return a + b on line 3, when does line 4 run?",
    },
}


class PedagogyRealizer:
    """Realizes pedagogical interventions (curated path, course material, fresh questions, hints, self-explanation)."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        rag: "HybridRAGEngine | None" = None,
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.rag = rag

    def _is_true_live_call(self) -> bool:
        if not self.llm.is_live():
            return False
        if hasattr(self.llm, "last_call_info"):
            info = self.llm.last_call_info() or {}
            if info.get("provider") in ("mock", "deterministic_double"):
                return False
        return True

    # ---- 1. Curate Learning Path ----

    def curate_learning_path(
        self,
        skill: str,
        memory_summary: dict[str, Any] | None,
        all_skill_summaries: dict[str, dict[str, Any]] | None = None,
        prereqs: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Generate a personalized 'Learn it -> Code it -> Remember it' path based on cognitive memory state."""
        skill_info = SKILL_META.get(skill, {"title": skill.replace("_", " ").title(), "summary": ""})
        prereq_list = list(prereqs if prereqs is not None else PREREQUISITES.get(skill, ()))
        mem = memory_summary or {}
        p_med = float(mem.get("p_med", 0.43))
        p_30d = float(mem.get("p_30d", 0.28))
        floor_b = float(mem.get("floor_b", 0.20))
        assessed = bool(mem.get("assessed", False))
        fragility = max(0.0, p_med - p_30d)

        system_prompt = (
            "You are an expert pedagogical coordinator for an adaptive Python learning platform. "
            "Given a learner's cognitive state (P(Med), 30d retention, permastore floor b_t, fragility), "
            "curate a personalized 3-phase study path following the 'Learn it -> Code it -> Remember it' model. "
            "Output JSON with keys: title, summary, focus_skill, steps (list of 3 objects with phase, title, description, recommended_action)."
        )
        context_payload = {
            "intent": "curate_learning_path",
            "skill": skill,
            "skill_title": skill_info["title"],
            "prerequisites": prereq_list,
            "cognitive_state": {
                "assessed": assessed,
                "p_med": p_med,
                "p_30d": p_30d,
                "floor_b": floor_b,
                "fragility": fragility,
            },
        }

        try:
            resp = self.llm.complete_json(system_prompt, json.dumps(context_payload), schema_name="learning_path")
            steps = resp.get("steps")
            if isinstance(steps, list) and len(steps) >= 3 and all("phase" in s for s in steps):
                return {
                    "id": str(uuid.uuid4()),
                    "title": str(resp.get("title", f"Personalized Path: {skill_info['title']}")),
                    "summary": str(resp.get("summary", f"Personalized pathway for {skill_info['title']}.")),
                    "focus_skill": skill,
                    "steps": [
                        {
                            "phase": str(s.get("phase", "learn")),
                            "title": str(s.get("title", f"Step {i+1}")),
                            "description": str(s.get("description", "")),
                            "recommended_action": str(s.get("recommended_action", "give_same_difficulty_question")),
                        }
                        for i, s in enumerate(steps[:4])
                    ],
                    "prerequisites": list(prereq_list),
                    "cognitive_state": context_payload["cognitive_state"],
                    "source": "llm" if self.llm.is_live() else "deterministic_double",
                }
        except LLMClientError:
            pass

        # High-fidelity skill-specific fallback
        skill_focus = _SKILL_PATH_FOCUS.get(skill, {})
        prereq_note = (
            f" Prerequisites to keep warm: {', '.join(prereq_list)}."
            if prereq_list
            else " Root skill (no prerequisites)."
        )
        if not assessed:
            p1_desc = f"Unassessed diagnostic & core theory: {skill_focus.get('learn', skill_info['summary'])}"
            p1_act = "explain_concept"
            p2_desc = f"Foundational practice: {skill_focus.get('code', 'Solve Easy and Medium items with progressive hints.')}"
            p2_act = "give_same_difficulty_question"
            p3_desc = f"Consolidate permastore b_t: {skill_focus.get('remember', 'Complete an active self-explanation check.')}"
            p3_act = "request_self_explanation"
        elif fragility > 0.15:
            p1_desc = f"Combat Fragility ({fragility*100:.0f}% 30d drop): {skill_focus.get('learn', 'Review core rules and worked trace.')}"
            p1_act = "show_worked_example"
            p2_desc = f"Deliberate error-focused practice: {skill_focus.get('code', 'Target weak difficulty variants.')}"
            p2_act = "give_same_difficulty_question"
            p3_desc = f"Lock in floor b_t={floor_b:.2f}: {skill_focus.get('remember', 'Active unaided self-explanation.')}"
            p3_act = "request_self_explanation"
        else:
            p1_desc = f"Targeted refresher (P(Med)={p_med:.0%}): {skill_focus.get('learn', 'Review edge cases.')}"
            p1_act = "show_worked_example"
            p2_desc = f"Step-up challenge: {skill_focus.get('code', 'Solve harder problems to stretch skill channels.')}"
            p2_act = "give_harder_question"
            p3_desc = f"Spaced verification (b_t={floor_b:.2f}): {skill_focus.get('remember', 'Verify retention before advancing.')}"
            p3_act = "advance_next_topic"

        return {
            "id": str(uuid.uuid4()),
            "title": f"Study Path: {skill_info['title']}",
            "summary": f"Adaptive Learn → Code → Remember sequence for {skill_info['title']}.{prereq_note}",
            "focus_skill": skill,
            "steps": [
                {
                    "phase": "learn",
                    "title": f"1. Learn: {skill_info['title']} Theory",
                    "description": p1_desc,
                    "recommended_action": p1_act,
                },
                {
                    "phase": "code",
                    "title": f"2. Code: {skill_info['title']} Practice",
                    "description": p2_desc,
                    "recommended_action": p2_act,
                },
                {
                    "phase": "remember",
                    "title": "3. Remember: Self-Explanation & Lock-In",
                    "description": p3_desc,
                    "recommended_action": p3_act,
                },
            ],
            "prerequisites": list(prereq_list),
            "cognitive_state": context_payload["cognitive_state"],
            "source": "deterministic_double",
        }

    # ---- 2. Generate Course Material ----

    def generate_course_material(
        self,
        skill: str,
        memory_summary: dict[str, Any] | None = None,
        focus_gap: str | None = None,
        weak_topics: list[str] | None = None,
        *,
        learner: str | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive course explanation, mental model, pitfalls, and worked example."""
        default_data = DEFAULT_LESSONS.get(skill, {})
        skill_info = SKILL_META.get(skill, {"title": skill.replace("_", " ").title(), "summary": ""})
        mem = memory_summary or {}
        weak_list = [str(w).strip() for w in (weak_topics or []) if str(w).strip()]

        tier_info = _resolve_mastery_tier(mem)
        mastery_tier = tier_info["mastery_tier"]
        p_med = tier_info["p_med"]
        fragility = tier_info["fragility"]
        floor_b = tier_info["floor_b"]
        obs_rate = mem.get("observed_pass_rate")

        weak_note = f" Specific areas missed: {'; '.join(weak_list[:3])}." if weak_list else ""

        if mastery_tier == "unassessed":
            adaptation_note = (
                f"First time studying {skill_info['title']}: delivering the complete foundational theory, "
                "mental model, and step-by-step worked trace."
            )
        elif mastery_tier == "lacking":
            unaided_ok = tier_info["observed_unaided_passes"]
            unaided_tot = tier_info["observed_unaided_attempts"]
            obs_str = (
                f"unaided pass rate {unaided_ok}/{unaided_tot} ({float(obs_rate):.0%}), "
                if obs_rate is not None
                else ""
            )
            adaptation_note = (
                f"Remediation Mode ({obs_str}P(Med)={p_med:.0%} < 70%): slowing down to reinforce "
                f"step-by-step trace rules, boundary pitfalls, and scaffolded examples.{weak_note}"
            )
        elif mastery_tier == "fragile":
            adaptation_note = (
                f"Retention Refresher (30-day drop of {fragility * 100:.0f}%, floor b_t={floor_b:.2f}): "
                f"skipping basic syntax and focusing on durable mental models to prevent forgetting.{weak_note}"
            )
        else:
            adaptation_note = (
                f"Advanced Mastery Mode (P(Med)={p_med:.0%}, floor b_t={floor_b:.2f}): condensing basics "
                "and emphasizing tricky edge cases, boundary combinations, and downstream transfer."
            )

        citations: list[dict[str, Any]] = []
        rag_passages: list[dict[str, str]] = []
        if self.rag is not None:
            rag_query = f"{skill} {skill_info['title']} {' '.join(weak_list[:3])} {focus_gap or ''}"
            rag_res = self.rag.retrieve(
                rag_query,
                purpose="course_material",
                skill=skill,
                learner=learner,
                source_types=("curriculum", "episodic"),
                include_prerequisites=(mastery_tier in ("unassessed", "lacking")),
                mastery_tier=mastery_tier,
                top_k=3,
            )
            citations = rag_res["citations"]
            rag_passages = [
                {
                    "citation": (ch.get("metadata") or {}).get("citation_label") or f"{ch['title']} — {ch['section']}",
                    "source_type": ch["source_type"],
                    "excerpt": ch["content"][:500],
                }
                for ch in rag_res["chunks"]
            ]

        system_prompt = (
            "You are an instructional designer for computer science. "
            f"Generate a complete, textbook-quality mini-course tailored to mastery_tier='{mastery_tier}'. "
            "Ground your explanations in the provided curriculum and learner-history passages when present. "
            "If mastery_tier='lacking', emphasize step-by-step remediation on weak_topics and common mistakes. "
            "If mastery_tier='fragile', focus on retention rules and preventing decay. "
            "If mastery_tier='mastered', condense basics and focus on advanced edge cases and transfer. "
            "Output JSON with keys: title, concept_explanation (multi-sentence detailed theory), "
            "theory_sections (list of 3 objects with {heading, body}), mental_model, pitfalls (list of 3 specific mistakes), "
            "worked_example (object with title, code, subgoals (list of objects with label, detail)), quick_check."
        )
        context_payload = {
            "intent": "generate_course_material",
            "skill": skill,
            "skill_title": skill_info["title"],
            "mastery_tier": mastery_tier,
            "adaptation_note": adaptation_note,
            "weak_topics": weak_list,
            "focus_gap": focus_gap,
            "cognitive_state": mem,
            "rag_curriculum_context": rag_passages,
        }

        try:
            resp = self.llm.complete_json(
                system_prompt,
                json.dumps(context_payload),
                schema_name="course_material",
                max_tokens=1600,
            )
            if (
                isinstance(resp, dict)
                and "concept_explanation" in resp
                and "worked_example" in resp
                and isinstance(resp.get("worked_example"), dict)
            ):
                return {
                    "id": str(uuid.uuid4()),
                    "skill": skill,
                    "mastery_tier": mastery_tier,
                    "adaptation_note": adaptation_note,
                    "title": str(resp.get("title", f"Course Material: {skill_info['title']}")),
                    "concept_explanation": str(resp["concept_explanation"]),
                    "theory_sections": resp.get("theory_sections") or [],
                    "mental_model": str(resp.get("mental_model", "")),
                    "pitfalls": [str(p) for p in resp.get("pitfalls", [])],
                    "worked_example": resp["worked_example"],
                    "quick_check": str(resp.get("quick_check", "")),
                    "citations": citations,
                    "rag_used": bool(citations),
                    "source": "llm" if self.llm.is_live() else "deterministic_double",
                }
        except LLMClientError:
            pass

        # High-fidelity mastery-tailored fallback from _RICH_COURSE_CURRICULUM + DEFAULT_LESSONS + RAG passages
        rich = _RICH_COURSE_CURRICULUM.get(skill, {})
        concept = default_data.get("concept", {})
        we = default_data.get("worked_example", {})
        subgoals = []
        for step in we.get("steps", []):
            if ":" in step:
                lbl, dtl = step.split(":", 1)
                subgoals.append({"label": lbl.strip(), "detail": dtl.strip()})
            else:
                subgoals.append({"label": "Step", "detail": step.strip()})

        base_title = rich.get("title") or concept.get("title", f"Course Material: {skill_info['title']}")
        base_theory = list(rich.get("theory_sections", []))
        base_pitfalls = list(rich.get("pitfalls", []))
        base_expl = rich.get("concept_explanation") or concept.get("summary", skill_info["summary"])

        if mastery_tier == "lacking":
            title = f"Remediation & Core Trace: {skill_info['title']}"
            expl = f"Focused review: {base_expl} Slow down and verify each intermediate value explicitly before moving to the next line."
            weak_body = (
                f"Your recent missed items on this skill involved: {'; '.join(weak_list[:3])}. "
                "Write out each variable's value line-by-line and double-check boundary operators (< vs <=, stop exclusion)."
                if weak_list
                else "Avoid guessing from surface patterns. Write out each variable's value line-by-line and double-check boundary operators (< vs <=, stop exclusion)."
            )
            theory = [
                {
                    "heading": "Remediation Priority: Trace Your Missed Patterns",
                    "body": weak_body,
                },
                *base_theory[:2],
            ]
        elif mastery_tier == "fragile":
            title = f"Retention Refresher: {skill_info['title']}"
            expl = f"Memory consolidation review: {base_expl}"
            theory = [
                {
                    "heading": "Why This Skill Decays (and How to Lock It In)",
                    "body": "Short-term pattern recognition fades over days. Focus on the underlying causal rule so you can reconstruct answers unaided after a 30-day gap.",
                },
                *base_theory[1:],
            ]
        elif mastery_tier == "mastered":
            title = f"Advanced Edge Cases: {skill_info['title']}"
            expl = f"You have strong mastery on {skill_info['title']} (P(Med)={p_med:.0%}). Use this concise brief to guard against subtle edge cases and prepare for harder multi-concept problems."
            theory = [
                *base_theory[1:],
                {
                    "heading": "Advanced Transfer & Boundary Stress Cases",
                    "body": f"Test yourself on empty inputs, negative values, and combining {skill_info['title']} with downstream skills rather than repeating routine Easy exercises.",
                },
            ]
        else:
            title = base_title
            expl = base_expl
            theory = base_theory

        return {
            "id": str(uuid.uuid4()),
            "skill": skill,
            "mastery_tier": mastery_tier,
            "adaptation_note": adaptation_note,
            "title": title,
            "concept_explanation": expl,
            "theory_sections": theory,
            "mental_model": "\n".join(concept.get("rules", [])),
            "pitfalls": base_pitfalls,
            "worked_example": {
                "title": we.get("title", "Worked Trace Example"),
                "problem": we.get("problem", ""),
                "code": we.get("code", concept.get("code", "")),
                "subgoals": subgoals,
            },
            "quick_check": rich.get("quick_check") or default_data.get("self_explanation_prompt", "Trace the values before submitting."),
            "citations": citations,
            "rag_used": bool(citations),
            "source": "deterministic_double",
        }

    # ---- 3. Generate Fresh Question ----

    @staticmethod
    def _normalize_candidate_task(
        resp: dict[str, Any],
        *,
        skill: str,
        difficulty: str,
        task_format: str,
        pool: str,
        guardrail: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = f"llm.{skill}.{difficulty}.{task_format}.{uuid.uuid4().hex[:8]}"
        raw_ref_sec = resp.get("reference_seconds", (guardrail["target_seconds_min"] + guardrail["target_seconds_max"]) / 2.0)
        try:
            ref_sec = float(raw_ref_sec)
        except (TypeError, ValueError):
            ref_sec = float(guardrail["target_seconds_max"])
        ref_sec = max(10.0, min(180.0, ref_sec))

        raw_tests = resp.get("tests", [])
        norm_tests: list[Any] = []
        if isinstance(raw_tests, list):
            for tc in raw_tests:
                if isinstance(tc, dict) and "input" in tc:
                    fixed = dict(tc)
                    if "expected" not in fixed and "output" in fixed:
                        fixed["expected"] = fixed.pop("output")
                    norm_tests.append(fixed)
                else:
                    norm_tests.append(tc)

        cand_task = {
            "task_id": task_id,
            "skill_id": skill,
            "task_type": task_format,
            "difficulty": difficulty,
            "pool": pool,
            "prompt": str(resp.get("prompt", "")),
            "snippets": [str(s) for s in (resp.get("snippets") or [])],
            "options": resp.get("options") if isinstance(resp.get("options"), list) else [],
            "answer_key": resp.get("answer_key"),
            "reference_solution": resp.get("reference_solution"),
            "tests": norm_tests,
            "criteria": resp.get("criteria") if isinstance(resp.get("criteria"), list) and resp.get("criteria") else [{"kind": "exact_match", "weight": 1.0}],
            "reference_seconds": ref_sec,
            "hints": resp.get("hints", []),
            "version": "llm-generated-v2",
        }
        raw_hints = cand_task.get("hints")
        if not isinstance(raw_hints, list) or len(raw_hints) < 3:
            cand_task["hints"] = [
                {"level": 1, "kind": "conceptual", "text": "Carefully review the core Python rule."},
                {"level": 2, "kind": "targeted", "text": "Trace variables step by step."},
                {"level": 3, "kind": "scaffold", "text": "Check your boundary or operator precedence."},
            ]
        else:
            norm_hints = []
            for idx, h in enumerate(raw_hints[:3]):
                default_kind = ("conceptual", "targeted", "scaffold")[min(idx, 2)]
                if isinstance(h, str) and h.strip():
                    norm_hints.append({"level": idx + 1, "kind": default_kind, "text": h.strip()})
                elif isinstance(h, dict):
                    raw_lvl = h.get("level")
                    lvl = raw_lvl if isinstance(raw_lvl, int) and not isinstance(raw_lvl, bool) else (idx + 1)
                    norm_hints.append(
                        {
                            "level": lvl,
                            "kind": h.get("kind") if h.get("kind") in ("conceptual", "targeted", "scaffold") else default_kind,
                            "text": str(h.get("text", "")).strip() or "Review the step.",
                        }
                    )
            while len(norm_hints) < 3:
                nxt = len(norm_hints) + 1
                default_kind = ("conceptual", "targeted", "scaffold")[min(nxt - 1, 2)]
                norm_hints.append({"level": nxt, "kind": default_kind, "text": "Trace boundary conditions carefully."})
            cand_task["hints"] = norm_hints
        return cand_task

    def _validate_with_guardrails(
        self,
        cand_task: dict[str, Any],
        *,
        difficulty: str,
        exemplar_texts: list[str],
    ) -> list[str]:
        _, errors = validate_task_schema(cand_task, verify_code_tests=True)
        guardrail = _DIFFICULTY_GUARDRAILS.get(difficulty, _DIFFICULTY_GUARDRAILS["medium"])
        total_lines = sum(len(s.splitlines()) for s in cand_task.get("snippets") or [])
        if difficulty == "easy" and total_lines > guardrail["max_lines"] + 2:
            errors.append(
                f"Easy question snippet has {total_lines} lines; keep easy questions <= {guardrail['max_lines']} lines."
            )
        if cand_task.get("task_type") in ("output_prediction", "fill_in_the_blank") and not cand_task.get("reference_solution"):
            ak = cand_task.get("answer_key")
            if isinstance(ak, list) and ak:
                cand_task["reference_solution"] = str(ak[0])
        if exemplar_texts:
            cand_text = f"{cand_task.get('prompt', '')} {' '.join(cand_task.get('snippets') or [])}"
            sim = max_text_similarity(cand_text, exemplar_texts)
            if sim > 0.85:
                errors.append(
                    f"Candidate question is too similar to a retrieved v2.2 exemplar (similarity={sim:.2f} > 0.85); "
                    "use fresh variable names, constants, and a novel scenario."
                )
        return errors

    def generate_question(
        self,
        skill: str,
        difficulty: str = "medium",
        task_format: str = "output_prediction",
        pool: str = "practice",
        recent_mistakes: list[str] | None = None,
        *,
        learner: str | None = None,
        memory_summary: dict[str, Any] | None = None,
        exclude_task_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Generate a difficulty-calibrated Python question using RAG exemplars, guardrails, and 1-shot schema repair."""
        if difficulty not in DIFFICULTIES:
            difficulty = "medium"
        if task_format not in FORMATS:
            task_format = "output_prediction"

        guardrail = _DIFFICULTY_GUARDRAILS[difficulty]
        tier_info = _resolve_mastery_tier(memory_summary) if memory_summary else None
        exemplars: list[dict[str, Any]] = []
        curriculum_notes: list[str] = []
        citations: list[dict[str, Any]] = []
        rag_audit: dict[str, Any] | None = None
        if self.rag is not None:
            ex_res = self.rag.retrieve_exemplars(
                skill=skill,
                difficulty=difficulty,
                task_format=task_format,
                query=f"{skill} {difficulty} {task_format} {' '.join(recent_mistakes or [])}",
                exclude_task_ids=exclude_task_ids,
                top_k=3,
            )
            exemplars = ex_res["exemplars"]
            citations.extend(ex_res["citations"])
            rag_audit = ex_res["audit"]
            cur_res = self.rag.retrieve(
                f"{skill} {difficulty} pitfalls edge cases",
                purpose="question_curriculum",
                skill=skill,
                learner=learner,
                source_types=("curriculum", "episodic"),
                mastery_tier=tier_info["mastery_tier"] if tier_info else None,
                top_k=2,
                record_audit=False,
            )
            curriculum_notes = [f"{ch['section']}: {ch['content'][:260]}" for ch in cur_res["chunks"]]
            citations.extend(cur_res["citations"])

        sanitized_exemplars = [
            {
                "task_id": ex["task_id"],
                "difficulty": ex["difficulty"],
                "task_type": ex["task_type"],
                "reference_seconds": ex.get("reference_seconds"),
                "prompt": ex.get("prompt"),
                "snippets": ex.get("snippets", []),
                "options": ex.get("options", []),
                "criteria": ex.get("criteria", []),
                "tests": ex.get("tests", []),
            }
            for ex in exemplars
        ]
        exemplar_texts = [
            f"{ex.get('prompt', '')} {' '.join(ex.get('snippets') or [])}"
            for ex in exemplars
        ]

        format_rule = {
            "single_choice": "Include options (list of >=2 {id, text}), answer_key (single string option id), criteria (1 entry summing to 1.0), reference_solution.",
            "multiple_select": "Include options (list of >=2 {id, text}), answer_key (list of option ids), criteria (1 entry per option), reference_solution.",
            "output_prediction": "Include non-empty snippets (list of runnable Python snippets), answer_key (list of exact stdout strings), criteria (matching snippets length), reference_solution.",
            "fill_in_the_blank": "Include non-empty snippets with ___, answer_key (list of integer strings), criteria (matching blanks length), reference_solution.",
            "code_completion": "Include snippets (def solve(x): starter), reference_solution (runnable def solve(x): body), tests (list of {input, expected}), criteria (matching tests length).",
        }[task_format]
        system_prompt = (
            f"You are a computer science question author. Generate a NOVEL, runnable Python exercise on skill '{skill}', "
            f"difficulty '{difficulty}', format '{task_format}'. "
            f"DIFFICULTY GUARDRAIL ({difficulty}): {guardrail['rules']} "
            "Do NOT copy any exemplar verbatim; write fresh variable names, numbers, and code paths. "
            f"{format_rule} "
            "Output ONLY a JSON object with keys: prompt, snippets, options, answer_key, reference_solution, tests, criteria, "
            "reference_seconds (number), hints (list of 3 {level, kind, text})."
        )
        context_payload = {
            "intent": "generate_question",
            "skill": skill,
            "difficulty": difficulty,
            "difficulty_guardrail": guardrail,
            "task_format": task_format,
            "pool": pool,
            "recent_mistakes": recent_mistakes or [],
            "mastery_tier": tier_info["mastery_tier"] if tier_info else None,
            "curriculum_guidance": curriculum_notes,
            "v22_few_shot_exemplars": sanitized_exemplars,
        }

        repair_attempted = False
        try:
            resp = self.llm.complete_json(system_prompt, json.dumps(context_payload, ensure_ascii=False), schema_name="question")
            if isinstance(resp, dict) and "prompt" in resp:
                cand_task = self._normalize_candidate_task(
                    resp,
                    skill=skill,
                    difficulty=difficulty,
                    task_format=task_format,
                    pool=pool,
                    guardrail=guardrail,
                )
                errors = self._validate_with_guardrails(
                    cand_task,
                    difficulty=difficulty,
                    exemplar_texts=exemplar_texts,
                )
                if errors and self.llm.is_live():
                    repair_attempted = True
                    repair_payload = {
                        "intent": "repair_question_schema",
                        "required_skill": skill,
                        "required_difficulty": difficulty,
                        "required_format": task_format,
                        "validation_errors": errors,
                        "invalid_candidate": resp,
                        "difficulty_guardrail": guardrail,
                    }
                    repaired_resp = self.llm.complete_json(
                        system_prompt + " Fix every item in validation_errors and return valid JSON only.",
                        json.dumps(repair_payload, ensure_ascii=False),
                        schema_name="question_repair",
                    )
                    if isinstance(repaired_resp, dict) and "prompt" in repaired_resp:
                        cand_task = self._normalize_candidate_task(
                            repaired_resp,
                            skill=skill,
                            difficulty=difficulty,
                            task_format=task_format,
                            pool=pool,
                            guardrail=guardrail,
                        )
                        errors = self._validate_with_guardrails(
                            cand_task,
                            difficulty=difficulty,
                            exemplar_texts=exemplar_texts,
                        )

                if not errors:
                    cand_task["rag_citations"] = citations
                    cand_task["difficulty_guardrail"] = guardrail["rules"]
                    cand_task["repair_attempted"] = repair_attempted
                    cand_task["source"] = "llm" if self._is_true_live_call() else "deterministic_double"
                    if rag_audit is not None:
                        cand_task["rag_audit"] = rag_audit
                    return cand_task
        except LLMClientError:
            pass

        # Safe fallback: sample unseen task from v2.2 catalog or synthesize clean item
        v22_item = sample_v22_task(
            skill=skill,
            difficulty=difficulty,
            task_format=task_format,
            pool=pool,
            exclude_ids=exclude_task_ids,
        )
        if v22_item is not None:
            cloned = copy.deepcopy(v22_item)
            cloned["task_id"] = f"llm.clone.{cloned['task_id']}.{uuid.uuid4().hex[:6]}"
            cloned["difficulty_source"] = "llm_pedagogy_fallback"
            cloned["source"] = "v22_clone"
            if citations:
                cloned["rag_citations"] = citations
            cloned["difficulty_guardrail"] = guardrail["rules"]
            cloned["repair_attempted"] = repair_attempted
            return cloned

        # Deterministic synthetic template
        task_id = f"synth.{skill}.{difficulty}.{task_format}.{uuid.uuid4().hex[:6]}"
        return {
            "task_id": task_id,
            "skill_id": skill,
            "task_type": task_format,
            "difficulty": difficulty,
            "pool": pool,
            "prompt": f"Trace the Python code below and find the output for {skill.replace('_', ' ')}.",
            "snippets": ["print(list(range(2, 7, 2)))"],
            "options": [
                {"id": "o0", "text": "[2, 4, 6]"},
                {"id": "o1", "text": "[2, 4, 6, 8]"},
                {"id": "o2", "text": "[2, 5]"},
                {"id": "o3", "text": "[2, 7]"},
            ],
            "answer_key": "o0" if task_format == "single_choice" else ["[2, 4, 6]"],
            "reference_solution": "[2, 4, 6]",
            "tests": [],
            "criteria": [{"kind": "exact_match", "weight": 1.0}],
            "reference_seconds": 30.0,
            "hints": [
                {"level": 1, "kind": "conceptual", "text": "Start at 2 and step by 2."},
                {"level": 2, "kind": "targeted", "text": "Stop at 7 is strictly excluded."},
                {"level": 3, "kind": "scaffold", "text": "The sequence contains 2, 4, and 6."},
            ],
            "rag_citations": citations,
            "difficulty_guardrail": guardrail["rules"],
            "repair_attempted": repair_attempted,
            "source": "synth",
            "version": "synth-v1",
        }

    # ---- 4. Generate Progressive Hint with Leakage l_t ----

    def generate_hint(
        self,
        task: dict[str, Any],
        learner_answer: Any,
        hint_level: int,
        prior_hints: list[dict[str, Any]] | None = None,
        *,
        learner: str | None = None,
    ) -> dict[str, Any]:
        """Generate an adaptive 3-tier progressive hint with measured leakage l_t in [0, 1]."""
        level = max(1, min(3, int(hint_level)))
        kind = ("conceptual", "targeted", "scaffold")[level - 1]
        authored_hints = task.get("hints", [])
        authored_text = ""
        if isinstance(authored_hints, list):
            for h in authored_hints:
                if isinstance(h, dict) and h.get("level") == level:
                    authored_text = str(h.get("text", ""))
                    break

        citations: list[dict[str, Any]] = []
        rag_passages: list[dict[str, str]] = []
        skill = task.get("skill_id")
        if self.rag is not None and skill in PPO_SKILLS_ORDER:
            rag_res = self.rag.retrieve(
                f"{task.get('prompt', '')} {authored_text}",
                purpose="hint",
                skill=skill,
                learner=learner,
                source_types=("curriculum", "episodic"),
                top_k=2,
            )
            citations = rag_res["citations"]
            rag_passages = [
                {"section": ch["section"], "excerpt": ch["content"][:320]}
                for ch in rag_res["chunks"]
            ]

        system_prompt = (
            "You are a Socratic tutor giving a hint on a Python problem. "
            f"Provide a level {level} ({kind}) hint WITHOUT revealing the final answer or exact solution line. "
            "Output JSON with keys: hint_level (int), hint_kind (string), text (string), leakage_l_t (float between 0.0 and 1.0)."
        )
        context_payload = {
            "intent": "generate_hint",
            "task_id": task.get("task_id"),
            "prompt": task.get("prompt"),
            "hint_level": level,
            "hint_kind": kind,
            "learner_answer": learner_answer,
            "prior_hints": prior_hints or [],
            "rag_curriculum_context": rag_passages,
        }

        try:
            resp = self.llm.complete_json(system_prompt, json.dumps(context_payload), schema_name="hint")
            if isinstance(resp, dict) and "text" in resp:
                raw_leak = float(resp.get("leakage_l_t", (0.25, 0.55, 0.80)[level - 1]))
                leak = max(0.05, min(0.95, raw_leak))
                return {
                    "hint_level": level,
                    "hint_kind": kind,
                    "text": str(resp["text"]),
                    "leakage_l_t": leak,
                    "citations": citations,
                    "source": "llm" if self.llm.is_live() else "deterministic_double",
                }
        except LLMClientError:
            pass

        # Deterministic fallback
        text = authored_text or (
            f"Level {level} hint: Focus on the {kind} requirement and trace boundary inputs."
        )
        leakage = (0.25, 0.55, 0.80)[level - 1]
        return {
            "hint_level": level,
            "hint_kind": kind,
            "text": text,
            "leakage_l_t": leakage,
            "citations": citations,
            "source": "deterministic_double",
        }

    # ---- 5. Evaluate Self-Explanation ----

    def evaluate_self_explanation(
        self,
        skill: str,
        task: dict[str, Any] | None,
        prompt: str,
        learner_explanation: str,
    ) -> dict[str, Any]:
        """Evaluate free-text self-explanation response and produce rubric score + feedback."""
        # Always run deterministic grader for authoritative scoring
        det_grade = grade_self_explanation(prompt, learner_explanation)

        system_prompt = (
            "You are an expert computer science teacher evaluating a student's self-explanation. "
            "Assess conceptual understanding, identify misconceptions, and provide concise encouraging feedback. "
            "Output JSON with keys: feedback, key_concepts_hit (list of strings), missing_concepts (list of strings), "
            "grader_confidence (float 0..1), next_tip (string)."
        )
        context_payload = {
            "intent": "evaluate_self_explanation",
            "skill": skill,
            "prompt": prompt,
            "learner_explanation": learner_explanation,
            "deterministic_score": det_grade["score"],
            "deterministic_full_success": det_grade["full_success"],
        }

        try:
            resp = self.llm.complete_json(
                system_prompt, json.dumps(context_payload), schema_name="self_explanation_eval"
            )
            if isinstance(resp, dict) and "feedback" in resp:
                raw_conf = resp.get("grader_confidence")
                try:
                    grader_conf = det_grade["grader_confidence"] if raw_conf is None else float(raw_conf)
                except (TypeError, ValueError):
                    grader_conf = det_grade["grader_confidence"]
                if not math.isfinite(grader_conf):
                    grader_conf = det_grade["grader_confidence"]
                hits = resp.get("key_concepts_hit")
                missing = resp.get("missing_concepts")
                return {
                    "score": det_grade["score"],
                    "full_success": det_grade["full_success"],
                    "grading_status": det_grade["grading_status"],
                    "det_confidence": det_grade["grader_confidence"],
                    "grader_confidence": grader_conf,
                    "key_concepts_hit": hits if isinstance(hits, list) else det_grade["key_concepts_hit"],
                    "missing_concepts": missing if isinstance(missing, list) else det_grade["missing_concepts"],
                    "feedback": str(resp["feedback"]),
                    "next_tip": str(resp.get("next_tip") or "Continue to the next practice task."),
                    "source": "llm" if self._is_true_live_call() else "deterministic_double",
                }
        except LLMClientError:
            pass

        return {
            **det_grade,
            "det_confidence": det_grade["grader_confidence"],
            "next_tip": "Keep connecting the rule to concrete variable traces.",
            "source": "deterministic_double",
        }

    # ---- 6. Realize Bounded PPO Pedagogical Turn ----

    def realize_ppo_turn(
        self,
        action_name: str,
        target_skill: str,
        target_difficulty: str,
        memory_summary: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Realize the PPO-selected action strictly within Gated Option B constraints."""
        ctx = context or {}
        mem = memory_summary or {}

        if action_name == "explain_concept":
            mat = self.generate_course_material(target_skill, mem)
            return {
                "action_name": action_name,
                "kind": "lesson",
                "skill": target_skill,
                "difficulty": target_difficulty,
                "primed_mode": "concept",
                "title": mat["title"],
                "content": mat["concept_explanation"],
                "mental_model": mat["mental_model"],
                "rules": mat["pitfalls"],
                "worked_example": mat["worked_example"],
            }

        if action_name == "show_worked_example":
            mat = self.generate_course_material(target_skill, mem)
            return {
                "action_name": action_name,
                "kind": "lesson",
                "skill": target_skill,
                "difficulty": target_difficulty,
                "primed_mode": "example",
                "title": mat["worked_example"]["title"],
                "content": mat["worked_example"].get("problem", "Worked Trace Example"),
                "worked_example": mat["worked_example"],
            }

        if action_name == "request_self_explanation":
            default_prompt = DEFAULT_LESSONS.get(target_skill, {}).get(
                "self_explanation_prompt",
                "Explain why this boundary/rule holds and verify your trace.",
            )
            # Question mode for self-explanation
            task = self.generate_question(
                skill=target_skill,
                difficulty="medium",
                task_format=ctx.get("task_format", "output_prediction"),
                pool="practice",
            )
            return {
                "action_name": action_name,
                "kind": "question",
                "mode": "assessment",
                "skill": target_skill,
                "difficulty": "medium",
                "task": task,
                "self_explanation_prompt": default_prompt,
            }

        # Questions (give_easier_question, give_same_difficulty_question, give_harder_question, review_prerequisite, advance_next_topic)
        mode = "assessment" if ctx.get("unverified_assisted") and action_name == "give_same_difficulty_question" else "practice"
        task = self.generate_question(
            skill=target_skill,
            difficulty=target_difficulty,
            task_format=ctx.get("task_format", "output_prediction"),
            pool=mode,
        )
        return {
            "action_name": action_name,
            "kind": "question",
            "mode": mode,
            "skill": target_skill,
            "difficulty": target_difficulty,
            "task": task,
        }

    @staticmethod
    def _extract_active_answers(active_question: dict[str, Any] | None) -> list[str]:
        if not isinstance(active_question, dict):
            return []
        raw_task = active_question.get("raw_task") or active_question.get("question") or {}
        if not isinstance(raw_task, dict):
            return []
        secrets: list[str] = []
        ak = raw_task.get("answer_key")
        opt_map = {
            str(o.get("id")): str(o.get("text"))
            for o in (raw_task.get("options") or [])
            if isinstance(o, dict) and o.get("id") and o.get("text")
        }
        if isinstance(ak, str) and ak.strip():
            secrets.append(ak.strip())
            if ak.strip() in opt_map:
                secrets.append(opt_map[ak.strip()].strip())
        elif isinstance(ak, list):
            for item in ak:
                s = str(item).strip()
                if s:
                    secrets.append(s)
                    if s in opt_map:
                        secrets.append(opt_map[s].strip())
        ref_sol = raw_task.get("reference_solution")
        if isinstance(ref_sol, str) and ref_sol.strip():
            secrets.append(ref_sol.strip())
        # Deduplicate while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for s in secrets:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @classmethod
    def _is_pasted_active_question(cls, question: str, active_question: dict[str, Any] | None) -> bool:
        if not isinstance(active_question, dict):
            return False
        raw_task = active_question.get("raw_task") or active_question.get("question") or {}
        if not isinstance(raw_task, dict):
            return False
        q_low = (question or "").lower()
        if "__blank_" in q_low and raw_task.get("task_type") == "fill_in_the_blank":
            return True
        active_text = f"{raw_task.get('prompt', '')} {' '.join(str(s) for s in (raw_task.get('snippets') or []))}".strip()
        if not active_text:
            return False
        return max_text_similarity(question, [active_text]) >= 0.35

    @classmethod
    def _enforce_eduguard(
        cls,
        *,
        question: str,
        reply: str,
        code_example: str,
        active_question: dict[str, Any] | None,
        mastery_info: dict[str, Any],
        rag_chunks: list[dict[str, Any]],
        skill_title: str,
    ) -> tuple[str, str, str]:
        has_open_unpassed = bool(
            isinstance(active_question, dict)
            and active_question.get("status") == "open"
            and not any(r.get("full_success") for r in (active_question.get("results") or []))
        )
        if not has_open_unpassed:
            return reply, code_example, "grounded_open"

        secrets = cls._extract_active_answers(active_question)
        combined_resp = f"{reply}\n{code_example}".strip()
        low_resp = combined_resp.lower()
        leaked = bool(_ANSWER_FISHING_RE.search(question or "")) or cls._is_pasted_active_question(
            question, active_question
        )

        if not leaked:
            for ans in secrets:
                ans_low = ans.lower()
                if len(ans_low) <= 2:
                    if re.search(
                        rf"(?:\b(?:answer|output|prints|returns|result|equals|option|choice|blank|fill|put|missing|replace|use)\s*(?:is|with|:|=|be)?\s*{re.escape(ans_low)}\b|^\s*{re.escape(ans_low)}\s*$)",
                        low_resp,
                        flags=re.MULTILINE,
                    ):
                        leaked = True
                        break
                elif len(ans_low) >= 3:
                    if ans_low in low_resp:
                        leaked = True
                        break

        if not leaked:
            return reply, code_example, "grounded_protected"

        # Pull conceptual grounding strictly from curriculum chunks (never echo episodic "Learner asked: ...")
        curriculum_only = [ch for ch in rag_chunks if ch.get("source_type") == "curriculum"]
        top_excerpt = _clean_excerpt(curriculum_only[0]["content"], 200) if curriculum_only else ""
        tier = mastery_info["mastery_tier"]
        if tier in ("unassessed", "lacking"):
            lines = [
                f"**EduGuard Active — Let's Solve {skill_title} Step by Step**",
                "",
                "Since you have an unpassed exercise open on this topic, I'll guide your reasoning without giving away the answer:",
                "1. **Identify the rule:** Check each bound, operator, or parameter in order.",
                "2. **Trace concrete values:** Track every variable before the step runs, during each pass, and at exit/return.",
                "3. **Test your hypothesis:** Plug your candidate value back into the code and verify the condition.",
            ]
        else:
            lines = [
                f"**EduGuard Active — Socratic Check ({skill_title})**",
                "",
                "An active problem is still open, so I'm keeping the solution hidden:",
                "1. Check edge boundaries (`<` vs `<=`, exclusive `stop`, negative indices, or early `return`).",
                "2. Trace operator precedence and line-by-line execution with a small test input.",
            ]
        if top_excerpt:
            lines.extend(["", f"**Key Concept to Apply:** {top_excerpt}"])
        return "\n".join(lines).strip(), "", "socratic_redirect"

    def answer_learner_doubt(
        self,
        skill: str,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        course_material: dict[str, Any] | None = None,
        learner: str | None = None,
        memory_summary: dict[str, Any] | None = None,
        active_question: dict[str, Any] | None = None,
        eduguard_enabled: bool = True,
    ) -> dict[str, Any]:
        """Answer a learner's question adaptively by mastery level, grounded by Hybrid RAG + EduGuard."""
        if skill not in PPO_SKILLS_ORDER:
            raise ValueError(f"Unknown skill {skill!r}")
        cleaned_q = (question or "").strip()
        if not cleaned_q:
            raise ValueError("Question cannot be empty.")

        meta = SKILL_META.get(skill, {})
        skill_title = meta.get("title") or meta.get("name") or skill
        rich = _RICH_COURSE_CURRICULUM.get(skill, {})
        def_lesson = DEFAULT_LESSONS.get(skill, {})
        mat = course_material or {
            "concept_explanation": rich.get("concept_explanation") or def_lesson.get("concept_explanation", ""),
            "mental_model": def_lesson.get("mental_model", ""),
            "pitfalls": rich.get("pitfalls") or def_lesson.get("rules", []),
            "worked_example": def_lesson.get("worked_example", {}),
        }

        mastery_info = _resolve_mastery_tier(memory_summary)
        mastery_tier = mastery_info["mastery_tier"]
        has_open_unpassed = bool(
            isinstance(active_question, dict)
            and active_question.get("status") == "open"
            and not any(r.get("full_success") for r in (active_question.get("results") or []))
        )
        active_preview = ""
        if isinstance(active_question, dict):
            q_obj = active_question.get("question") or {}
            if isinstance(q_obj, dict):
                active_preview = f"{q_obj.get('prompt', '')} {' '.join(q_obj.get('snippets') or [])}"[:220]

        citations: list[dict[str, Any]] = []
        rag_chunks: list[dict[str, Any]] = []
        rag_audit: dict[str, Any] | None = None
        if self.rag is not None:
            rag_res = self.rag.retrieve(
                cleaned_q,
                purpose="tutor_chat",
                skill=skill,
                learner=learner,
                source_types=("curriculum", "episodic"),
                include_prerequisites=(mastery_tier in ("unassessed", "lacking")),
                mastery_tier=mastery_tier,
                extra_context=active_preview,
                top_k=4,
            )
            rag_chunks = rag_res["chunks"]
            citations = rag_res["citations"]
            rag_audit = rag_res["audit"]

        rag_context_for_llm = [
            {
                "citation": (ch.get("metadata") or {}).get("citation_label") or f"{ch['title']} — {ch['section']}",
                "source_type": ch["source_type"],
                "content": ch["content"][:550],
            }
            for ch in rag_chunks
        ]

        if self.llm.is_live():
            eduguard_rule = (
                "EDUGUARD ANTI-LEAKAGE POLICY (ACTIVE): The student currently has an unpassed practice/assessment question open. "
                "NEVER output the active question's exact answer, output line, option choice, missing blank, or solution code. "
                "Guide via Socratic questions and a DIFFERENT small toy example."
                if (has_open_unpassed and eduguard_enabled)
                else "No unpassed exercise is blocking explanation; answer clearly using concrete examples."
            )
            system_prompt = (
                "You are a supportive, clear Python tutor answering a student's doubt. "
                f"Adapt your depth, tone, and scaffolding to the student's cognitive mastery level: {mastery_info['instruction']} "
                "Format `reply` cleanly with Markdown headings/bold labels, short bullet or numbered steps (`1.`, `2.`), and inline code ticks. "
                "Cite relevant curriculum sections or prior mistakes naturally. "
                f"{eduguard_rule} "
                "Return ONLY a JSON object with keys:\n"
                '{"reply": "Well-formatted Markdown mastery-adapted explanation", "code_example": "Optional short toy Python snippet or empty string"}'
            )
            recent_history = []
            for turn in (history or [])[-6:]:
                if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("text"):
                    recent_history.append({"role": turn["role"], "text": str(turn["text"])[:400]})
            user_payload = {
                "skill": skill,
                "skill_title": skill_title,
                "learner_mastery": mastery_info,
                "lesson_summary": {
                    "concept_explanation": mat.get("concept_explanation", ""),
                    "mental_model": mat.get("mental_model", ""),
                    "pitfalls": mat.get("pitfalls", []),
                    "worked_example": mat.get("worked_example", {}),
                },
                "rag_context": rag_context_for_llm,
                "history": recent_history,
                "student_question": cleaned_q,
            }
            try:
                raw = self.llm.complete_json(
                    system_prompt,
                    json.dumps(user_payload, ensure_ascii=False),
                    schema_name="tutor_chat",
                    temperature=0.3,
                    max_tokens=750,
                )
                reply = str(raw.get("reply") or raw.get("answer") or "").strip()
                code_ex = str(raw.get("code_example") or raw.get("code") or "").strip()
                if reply:
                    if eduguard_enabled:
                        guarded_reply, guarded_code, guard_status = self._enforce_eduguard(
                            question=cleaned_q,
                            reply=reply,
                            code_example=code_ex,
                            active_question=active_question,
                            mastery_info=mastery_info,
                            rag_chunks=rag_chunks,
                            skill_title=skill_title,
                        )
                    else:
                        guarded_reply, guarded_code, guard_status = reply, code_ex, "disabled"
                    return {
                        "reply": guarded_reply,
                        "code_example": guarded_code,
                        "source": "llm" if self._is_true_live_call() else "deterministic_double",
                        "citations": citations,
                        "mastery_tier": mastery_tier,
                        "guardrail_status": guard_status,
                        "rag_audit": rag_audit,
                    }
            except LLMClientError:
                pass

        # Mastery-adaptive + RAG-grounded deterministic offline fallback (clean Markdown formatting)
        we = mat.get("worked_example") or {}
        pitfalls = mat.get("pitfalls") or rich.get("pitfalls") or []
        ql = cleaned_q.lower()
        episodic_chunks = [ch for ch in rag_chunks if ch.get("source_type") == "episodic"]
        curriculum_chunks = [ch for ch in rag_chunks if ch.get("source_type") != "episodic"]
        rag_note = ""
        if curriculum_chunks:
            cleaned_cur = _clean_excerpt(curriculum_chunks[0]["content"], 200)
            if cleaned_cur:
                rag_note = f"\n\n**From {curriculum_chunks[0]['section']}:** {cleaned_cur}"

        if mastery_tier == "lacking":
            header = f"**Remediation Guide ({mastery_info['p_med']:.0%} < 70% mastery on {skill_title})**"
            intro = "Let's slow down and trace every variable cleanly before moving on:"
        elif mastery_tier == "fragile":
            header = f"**Retention Refresher ({skill_title} · P(Med) {mastery_info['p_med']:.0%}, 30d drop {mastery_info['fragility']:.0%})**"
            intro = "Keep the core causal rule crisp so it holds after a gap:"
        elif mastery_tier == "mastered":
            header = f"**Precision Summary ({skill_title} · {mastery_info['p_med']:.0%} mastered)**"
            intro = "Watch for subtle boundary and precedence cases:"
        else:
            header = f"**Core Mental Model ({skill_title})**"
            intro = "Here is the step-by-step mental model to use:"

        episodic_note = ""
        if episodic_chunks and mastery_tier in ("unassessed", "lacking", "fragile"):
            cleaned_ep = _clean_excerpt(episodic_chunks[0]["content"], 160)
            if cleaned_ep and not cleaned_ep.lower().startswith("learner asked:"):
                episodic_note = f"\n\n**Focus Area from Recent Practice:** {cleaned_ep}"

        if any(k in ql for k in ("example", "code", "trace", "step", "subgoal", "work")):
            subgoals = we.get("subgoals") or []
            steps_md = "\n".join(
                f"{idx}. **{sg.get('label', f'Step {idx}')}:** {sg.get('detail', '')}"
                for idx, sg in enumerate(subgoals, start=1)
            )
            body = (
                f"**Worked Example — {we.get('title', 'Trace')}:**\n"
                f"{steps_md or mat.get('mental_model', '')}"
            )
            reply = f"{header}\n\n{intro}\n\n{body}{rag_note}{episodic_note}".strip()
            code_ex = "" if (has_open_unpassed and eduguard_enabled) else str(we.get("code") or "")
        elif any(k in ql for k in ("mistake", "pitfall", "wrong", "error", "bug", "watch", "careful")):
            pit_md = "\n".join(f"- {p}" for p in pitfalls[:3])
            reply = f"{header}\n\n**Common Pitfalls to Avoid:**\n{pit_md}{rag_note}{episodic_note}".strip()
            code_ex = ""
        else:
            core_expl = str(mat.get("concept_explanation", "")).strip()
            if mastery_tier == "mastered" and curriculum_chunks:
                core_expl = _clean_excerpt(curriculum_chunks[0]["content"], 240)
            mental = _clean_excerpt(str(mat.get("mental_model", "")), 220)
            lines = [header, "", core_expl]
            if mental:
                lines.extend(["", f"**Mental Model:** {mental}"])
            reply = ("".join([]) + "\n".join(lines) + f"{rag_note}{episodic_note}").strip()
            code_ex = "" if ((has_open_unpassed and eduguard_enabled) or mastery_tier == "mastered") else str(we.get("code") or "")

        if eduguard_enabled:
            guarded_reply, guarded_code, guard_status = self._enforce_eduguard(
                question=cleaned_q,
                reply=reply,
                code_example=code_ex,
                active_question=active_question,
                mastery_info=mastery_info,
                rag_chunks=rag_chunks,
                skill_title=skill_title,
            )
        else:
            guarded_reply, guarded_code, guard_status = reply, code_ex, "disabled"
        return {
            "reply": guarded_reply,
            "code_example": guarded_code,
            "source": "fallback",
            "citations": citations,
            "mastery_tier": mastery_tier,
            "guardrail_status": guard_status,
            "rag_audit": rag_audit,
        }

