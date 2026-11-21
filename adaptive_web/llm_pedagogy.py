"""Bounded PPO-to-LLM pedagogical realizer for adaptive Python learning."""
from __future__ import annotations

import copy
import json
import random
import uuid
from typing import Any

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

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

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
    ) -> dict[str, Any]:
        """Generate comprehensive course explanation, mental model, pitfalls, and worked example."""
        default_data = DEFAULT_LESSONS.get(skill, {})
        skill_info = SKILL_META.get(skill, {"title": skill.replace("_", " ").title(), "summary": ""})
        mem = memory_summary or {}
        weak_list = [str(w).strip() for w in (weak_topics or []) if str(w).strip()]

        assessed = bool(mem.get("assessed", False))
        p_med = float(mem.get("p_med", 0.5))
        p_30d = float(mem.get("p_30d", 0.5))
        fragility = float(mem.get("fragility", max(0.0, p_med - p_30d)))
        floor_b = float(mem.get("floor_b", 0.0))
        obs_rate = mem.get("observed_pass_rate")

        weak_note = f" Specific areas missed: {'; '.join(weak_list[:3])}." if weak_list else ""

        if not assessed:
            mastery_tier = "unassessed"
            adaptation_note = (
                f"First time studying {skill_info['title']}: delivering the complete foundational theory, "
                "mental model, and step-by-step worked trace."
            )
        elif p_med < 0.70 or (obs_rate is not None and float(obs_rate) < 0.70):
            mastery_tier = "lacking"
            unaided_ok = int(mem.get("observed_unaided_passes", 0))
            unaided_tot = int(mem.get("observed_unaided_attempts", 0))
            obs_str = (
                f"unaided pass rate {unaided_ok}/{unaided_tot} ({float(obs_rate):.0%}), "
                if obs_rate is not None
                else ""
            )
            adaptation_note = (
                f"Remediation Mode ({obs_str}P(Med)={p_med:.0%} < 70%): slowing down to reinforce "
                f"step-by-step trace rules, boundary pitfalls, and scaffolded examples.{weak_note}"
            )
        elif fragility > 0.15:
            mastery_tier = "fragile"
            adaptation_note = (
                f"Retention Refresher (30-day drop of {fragility * 100:.0f}%, floor b_t={floor_b:.2f}): "
                f"skipping basic syntax and focusing on durable mental models to prevent forgetting.{weak_note}"
            )
        else:
            mastery_tier = "mastered"
            adaptation_note = (
                f"Advanced Mastery Mode (P(Med)={p_med:.0%}, floor b_t={floor_b:.2f}): condensing basics "
                "and emphasizing tricky edge cases, boundary combinations, and downstream transfer."
            )

        system_prompt = (
            "You are an instructional designer for computer science. "
            f"Generate a complete, textbook-quality mini-course tailored to mastery_tier='{mastery_tier}'. "
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
                    "source": "llm" if self.llm.is_live() else "deterministic_double",
                }
        except LLMClientError:
            pass

        # High-fidelity mastery-tailored fallback from _RICH_COURSE_CURRICULUM + DEFAULT_LESSONS
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
            "source": "deterministic_double",
        }

    # ---- 3. Generate Fresh Question ----

    def generate_question(
        self,
        skill: str,
        difficulty: str = "medium",
        task_format: str = "output_prediction",
        pool: str = "practice",
        recent_mistakes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a fresh verifiable Python question matching the v2.2 schema and 8-key memory_task."""
        if difficulty not in DIFFICULTIES:
            difficulty = "medium"
        if task_format not in FORMATS:
            task_format = "output_prediction"

        system_prompt = (
            f"You are a computer science question author. Generate a fresh, valid Python exercise on skill '{skill}', "
            f"difficulty '{difficulty}', format '{task_format}'. "
            "Output JSON with keys: prompt, snippets (list of strings), options (list of {id, text} for choice formats), "
            "answer_key (string for single_choice, list for others), reference_solution, tests (list of {input, output} for code_completion), "
            "criteria (list of {kind, weight}), reference_seconds (number), hints (list of 3 {level, kind, text})."
        )
        context_payload = {
            "intent": "generate_question",
            "skill": skill,
            "difficulty": difficulty,
            "task_format": task_format,
            "pool": pool,
            "recent_mistakes": recent_mistakes or [],
        }

        try:
            resp = self.llm.complete_json(system_prompt, json.dumps(context_payload), schema_name="question")
            if isinstance(resp, dict) and "prompt" in resp:
                task_id = f"llm.{skill}.{difficulty}.{task_format}.{uuid.uuid4().hex[:8]}"
                cand_task = {
                    "task_id": task_id,
                    "skill_id": skill,
                    "task_type": task_format,
                    "difficulty": difficulty,
                    "pool": pool,
                    "prompt": str(resp["prompt"]),
                    "snippets": [str(s) for s in resp.get("snippets", [])],
                    "options": resp.get("options", []),
                    "answer_key": resp.get("answer_key"),
                    "reference_solution": resp.get("reference_solution"),
                    "tests": resp.get("tests", []),
                    "criteria": resp.get("criteria", [{"kind": "exact_match", "weight": 1.0}]),
                    "reference_seconds": float(resp.get("reference_seconds", 35.0)),
                    "hints": resp.get("hints", []),
                    "version": "llm-generated-v1",
                }
                # Fix up hints structure if necessary
                raw_hints = cand_task.get("hints")
                if not isinstance(raw_hints, list) or len(raw_hints) < 3:
                    cand_task["hints"] = [
                        {"level": 1, "kind": "conceptual", "text": "Carefully review the syntax rule."},
                        {"level": 2, "kind": "targeted", "text": "Trace variables step by step."},
                        {"level": 3, "kind": "scaffold", "text": "Check your boundary or operator precedence."},
                    ]
                else:
                    norm_hints = []
                    for idx, h in enumerate(raw_hints):
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
                    cand_task["hints"] = norm_hints
                # Validate against schema
                is_valid, _ = validate_task_schema(cand_task)
                if is_valid:
                    return cand_task
        except LLMClientError:
            pass

        # Safe fallback: sample unseen task from v2.2 catalog or synthesize clean item
        v22_item = sample_v22_task(skill=skill, difficulty=difficulty, task_format=task_format, pool=pool)
        if v22_item is not None:
            cloned = copy.deepcopy(v22_item)
            cloned["task_id"] = f"llm.clone.{cloned['task_id']}.{uuid.uuid4().hex[:6]}"
            cloned["difficulty_source"] = "llm_pedagogy_fallback"
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
            "version": "synth-v1",
        }

    # ---- 4. Generate Progressive Hint with Leakage l_t ----

    def generate_hint(
        self,
        task: dict[str, Any],
        learner_answer: Any,
        hint_level: int,
        prior_hints: list[dict[str, Any]] | None = None,
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

        system_prompt = (
            "You are a Socratic tutor giving a hint on a Python problem. "
            f"Provide a level {level} ({kind}) hint. "
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
                return {
                    "score": det_grade["score"],
                    "full_success": det_grade["full_success"],
                    "grading_status": "resolved",
                    "grader_confidence": float(resp.get("grader_confidence", det_grade["grader_confidence"])),
                    "key_concepts_hit": resp.get("key_concepts_hit", det_grade["key_concepts_hit"]),
                    "missing_concepts": resp.get("missing_concepts", det_grade["missing_concepts"]),
                    "feedback": str(resp["feedback"]),
                    "next_tip": str(resp.get("next_tip", "Continue to the next practice task.")),
                    "source": "llm" if self.llm.is_live() else "deterministic_double",
                }
        except LLMClientError:
            pass

        return {
            **det_grade,
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
