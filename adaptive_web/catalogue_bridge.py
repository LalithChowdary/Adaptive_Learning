"""Read-only bridge to 3/data_Set/v2.2/ item bank, canonical PPO DAG, and task schema validator."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from . import CODE_3_ROOT

DATA_V22_DIR = CODE_3_ROOT / "data_Set" / "v2.2"
V22_CATALOG_PY = DATA_V22_DIR / "catalog.py"
V22_TASK_CATALOG_JSON = DATA_V22_DIR / "dataset" / "task_catalog.json"

# Canonical topological skill order from 3/ppo/env.py:39-46
PPO_SKILLS_ORDER: tuple[str, ...] = (
    "range_boundaries",
    "loop_tracing",
    "accumulator_initialization",
    "conditional_evaluation",
    "list_indexing",
    "function_returns",
)
SKILLS_ORDER = PPO_SKILLS_ORDER

FORMATS: tuple[str, ...] = (
    "single_choice",
    "multiple_select",
    "output_prediction",
    "fill_in_the_blank",
    "code_completion",
)
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
HINT_KINDS: tuple[str, ...] = ("conceptual", "targeted", "scaffold")

# Exact prerequisite DAG from 3/ppo/env.py:48-55
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "range_boundaries": (),
    "loop_tracing": ("range_boundaries",),
    "accumulator_initialization": ("range_boundaries", "loop_tracing"),
    "conditional_evaluation": ("loop_tracing",),
    "list_indexing": ("range_boundaries",),
    "function_returns": ("conditional_evaluation", "accumulator_initialization"),
}

_SKILL_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "range_boundaries": (
        "Range boundaries",
        "Where Python range(start, stop, step) sequences begin, step, and stop exclusively",
    ),
    "loop_tracing": (
        "Loop tracing",
        "Follow state variables step-by-step across every loop iteration",
    ),
    "accumulator_initialization": (
        "Accumulators",
        "Combine explicit initial baselines with loop contributions",
    ),
    "conditional_evaluation": (
        "Conditionals",
        "Evaluate boundary inequalities and mutually exclusive if/elif/else branches",
    ),
    "list_indexing": (
        "List indexing",
        "Locate items with zero-based positive and negative sequence indices",
    ),
    "function_returns": (
        "Function returns",
        "Bind arguments, evaluate expressions with precedence, and trace return values",
    ),
}


def _build_skill_meta() -> dict[str, dict[str, Any]]:
    dependents: dict[str, list[str]] = {s: [] for s in PPO_SKILLS_ORDER}
    for skill, prereqs in PREREQUISITES.items():
        for pre in prereqs:
            dependents[pre].append(skill)
    meta: dict[str, dict[str, Any]] = {}
    for idx, skill in enumerate(PPO_SKILLS_ORDER):
        name, summary = _SKILL_DESCRIPTIONS[skill]
        meta[skill] = {
            "id": skill,
            "order": idx,
            "number": f"{idx + 1:02d}",
            "name": name,
            "title": name,
            "summary": summary,
            "prerequisites": list(PREREQUISITES[skill]),
            "dependents": list(dependents[skill]),
        }
    return meta


SKILL_META: dict[str, dict[str, Any]] = _build_skill_meta()

# Baseline fallback lessons owned by adaptive_web (never imports 3/dashboard/catalogue.py)
DEFAULT_LESSONS: dict[str, dict[str, Any]] = {
    "range_boundaries": {
        "concept": {
            "title": "How Python range(start, stop, step) boundaries work",
            "summary": "A Python range generates arithmetic values starting at start and stopping strictly before stop.",
            "rules": [
                "Inclusive start: the first generated number is always start (if it has not crossed stop).",
                "Exclusive stop: stop is never included in the sequence, even if start + k * step == stop.",
                "Step direction: for positive step keep values < stop; for negative step keep values > stop.",
            ],
            "code": "list(range(2, 9, 3))  # -> [2, 5, 8]",
        },
        "worked_example": {
            "title": "Worked Example: Tracing range(3, 11, 2)",
            "problem": "Find every value produced by list(range(3, 11, 2)) and its length.",
            "code": "values = list(range(3, 11, 2))\nprint(values[0], values[-1], len(values))",
            "steps": [
                "Identify start = 3, stop = 11 (excluded), step = +2.",
                "Generate terms while value < 11: 3, 5, 7, 9.",
                "Next term 9 + 2 = 11 equals stop, so generation halts immediately.",
                "Result: [3, 5, 7, 9] -> first=3, last=9, len=4.",
            ],
        },
        "self_explanation_prompt": (
            "Explain in your own words why the stop value of range(start, stop, step) is excluded "
            "and how you verify the final generated element."
        ),
    },
    "loop_tracing": {
        "concept": {
            "title": "Tracing state updates across loop iterations",
            "summary": "A for-loop executes its body once per sequence item in order, updating variables step by step.",
            "rules": [
                "Record the initial variable value before iteration 1 begins.",
                "On each iteration, bind the loop variable to the next element and evaluate the update using current state.",
                "After the final element completes, the variable retains its last assigned value.",
            ],
            "code": "x = 4\nfor step in range(1, 4):  # 1, 2, 3\n    x = x + 2 * step      # 4 -> 6 -> 10 -> 16",
        },
        "worked_example": {
            "title": "Worked Example: Iteration table for a loop update",
            "problem": "Trace the final value of value after the loop finishes.",
            "code": "value = 5\nfor k in range(2, 7, 2):\n    value = value + k - 1\nprint(value)",
            "steps": [
                "Sequence from range(2, 7, 2) is [2, 4, 6]. Initial state: value = 5.",
                "Iteration 1 (k = 2): value = 5 + 2 - 1 = 6.",
                "Iteration 2 (k = 4): value = 6 + 4 - 1 = 9.",
                "Iteration 3 (k = 6): value = 9 + 6 - 1 = 14.",
            ],
        },
        "self_explanation_prompt": (
            "Explain how you determine the exact number of loop iterations and how the updated "
            "variable carries from one iteration into the next."
        ),
    },
    "accumulator_initialization": {
        "concept": {
            "title": "Accumulator initialization and running totals",
            "summary": "An accumulator combines its starting baseline with every contribution added inside the loop.",
            "rules": [
                "Always inspect the initialization line before the loop — accumulators do not always start at 0.",
                "Final total = (Initial Value) + sum of each iteration's contribution.",
                "Conditional accumulators only update when their guard condition is True.",
            ],
            "code": "total = 10\nfor n in [2, 5, 3]:\n    total += n  # 10 + 2 + 5 + 3 = 20",
        },
        "worked_example": {
            "title": "Worked Example: Non-zero accumulator baseline",
            "problem": "Compute total after accumulating transformed values.",
            "code": "total = 3\nfor x in range(1, 4):\n    total = total + (2 * x)\nprint(total)",
            "steps": [
                "Baseline before loop: total = 3. Loop runs for x = 1, 2, 3.",
                "x = 1: total = 3 + 2 = 5.",
                "x = 2: total = 5 + 4 = 9.",
                "x = 3: total = 9 + 6 = 15.",
            ],
        },
        "self_explanation_prompt": (
            "Explain why an accumulator's initial value before the loop must be kept separate from "
            "the contributions added on each iteration."
        ),
    },
    "conditional_evaluation": {
        "concept": {
            "title": "Mutual exclusion in if / elif / else chains",
            "summary": "Python evaluates conditions top-to-bottom and runs only the first matching branch.",
            "rules": [
                "Check strict (> / <) vs inclusive (>= / <=) boundary operators at equality.",
                "In an if / elif / else chain, the first True branch executes and skips all later branches.",
                "Separate consecutive if statements are independent and each condition is tested.",
            ],
            "code": "x = 10\nif x > 10:\n    out = 'high'\nelif x >= 10:\n    out = 'edge'  # runs and skips else",
        },
        "worked_example": {
            "title": "Worked Example: Boundary test in an if/elif/else chain",
            "problem": "Determine which branch executes when score = 15.",
            "code": "score = 15\nif score > 15:\n    tag = 1\nelif score >= 12:\n    tag = 2\nelse:\n    tag = 3\nprint(tag)",
            "steps": [
                "Check first condition: 15 > 15 is False (strict inequality).",
                "Check elif condition: 15 >= 12 is True.",
                "Execute tag = 2 and skip the else branch completely.",
            ],
        },
        "self_explanation_prompt": (
            "Explain the difference between strict (>, <) and inclusive (>=, <=) comparisons at a boundary "
            "and why later elif/else branches are skipped once a branch runs."
        ),
    },
    "list_indexing": {
        "concept": {
            "title": "Zero-based positive and negative list indexing",
            "summary": "Python lists index elements from 0 on the left and from -1 on the right.",
            "rules": [
                "Positive indices start at 0 (first element) up to len(items) - 1 (last element).",
                "Negative indices count backwards: -1 is the last element, -len(items) is the first.",
                "For computed indices items[expr], evaluate expr completely before looking up the slot.",
            ],
            "code": "items = [14, 28, 42, 56]\nprint(items[1], items[-2])  # 28 42",
        },
        "worked_example": {
            "title": "Worked Example: Positive and negative index lookup",
            "problem": "Find items[2] and items[-3] for a 5-element list.",
            "code": "items = [10, 25, 40, 55, 70]\nprint(items[2], items[-3])",
            "steps": [
                "From left (0-based): 0->10, 1->25, 2->40, 3->55, 4->70, so items[2] = 40.",
                "From right (-1-based): -1->70, -2->55, -3->40, -4->25, -5->10, so items[-3] = 40.",
            ],
        },
        "self_explanation_prompt": (
            "Explain how zero-based positive indexing and -1-based negative indexing map onto "
            "the same list positions."
        ),
    },
    "function_returns": {
        "concept": {
            "title": "Function parameters, operator precedence, and return values",
            "summary": "Calling a function binds arguments to parameters and exits immediately with the return expression.",
            "rules": [
                "Substitute the passed argument value for every occurrence of the parameter.",
                "Apply Python precedence: parentheses, then *, //, /, %, then + and -.",
                "Executing return ends the function immediately and hands the value back to the caller.",
            ],
            "code": "def f(x):\n    return 3 * x + 4\nprint(f(5))  # 19",
        },
        "worked_example": {
            "title": "Worked Example: Tracing a two-step function return",
            "problem": "Evaluate compute(4) by tracing local variables to the return expression.",
            "code": "def compute(x):\n    base = 2 * x - 1\n    return base + 5\nprint(compute(4))",
            "steps": [
                "Bind parameter x = 4.",
                "Evaluate base = 2 * 4 - 1 = 7.",
                "Return base + 5 = 12.",
            ],
        },
        "self_explanation_prompt": (
            "Explain how argument substitution and operator precedence determine the value handed "
            "back by a return statement."
        ),
    },
}


def _load_v22_catalog_module():
    mod_name = "adaptive_web_v22_catalog"
    spec = importlib.util.spec_from_file_location(mod_name, V22_CATALOG_PY)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load v2.2 catalog module from {V22_CATALOG_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V22_CATALOG_MODULE = _load_v22_catalog_module()
if not V22_TASK_CATALOG_JSON.is_file():
    raise FileNotFoundError(f"Missing v2.2 task catalog JSON: {V22_TASK_CATALOG_JSON}")
_RAW_V22_TASKS: list[dict[str, Any]] = json.loads(V22_TASK_CATALOG_JSON.read_text(encoding="utf-8"))
_TASKS_BY_ID: dict[str, dict[str, Any]] = {t["task_id"]: t for t in _RAW_V22_TASKS}


def get_v22_catalog_module():
    """Return the loaded read-only 3/data_Set/v2.2/catalog.py module."""
    return _V22_CATALOG_MODULE


def v22_task_count() -> int:
    return len(_TASKS_BY_ID)


def get_v22_task(task_id: str) -> dict[str, Any] | None:
    task = _TASKS_BY_ID.get(task_id)
    return copy.deepcopy(task) if task is not None else None


def list_v22_tasks(
    skill: str | None = None,
    difficulty: str | None = None,
    task_format: str | None = None,
    pool: str | None = None,
) -> list[dict[str, Any]]:
    results = []
    for t in _RAW_V22_TASKS:
        if skill is not None and t["skill_id"] != skill:
            continue
        if difficulty not in (None, "", "mixed") and t["difficulty"] != difficulty:
            continue
        if task_format not in (None, "", "mixed") and t["task_type"] != task_format:
            continue
        if pool is not None and t["pool"] != pool:
            continue
        results.append(copy.deepcopy(t))
    return results


def sample_v22_task(
    skill: str,
    difficulty: str | None = None,
    task_format: str | None = None,
    pool: str | None = None,
    exclude_ids: Iterable[str] = (),
    rng: random.Random | None = None,
    *,
    allow_repeat: bool = False,
) -> dict[str, Any] | None:
    """Sample a v2.2 item matching filters, returning a deep copy or None if exhausted."""
    if skill not in PPO_SKILLS_ORDER:
        raise ValueError(f"Unknown canonical skill {skill!r}")
    excluded = set(exclude_ids)
    picker = rng or random

    def _filter(fmt_filter: str | None, use_exclude: bool) -> list[dict[str, Any]]:
        out = []
        for t in _RAW_V22_TASKS:
            if t["skill_id"] != skill:
                continue
            if difficulty not in (None, "", "mixed") and t["difficulty"] != difficulty:
                continue
            if fmt_filter not in (None, "", "mixed") and t["task_type"] != fmt_filter:
                continue
            if pool is not None and t["pool"] != pool:
                continue
            if use_exclude and t["task_id"] in excluded:
                continue
            out.append(t)
        return out

    candidates = _filter(task_format, True)
    if not candidates and task_format not in (None, "", "mixed"):
        # Clean fallback if requested format has no unseen items under (skill, difficulty, pool)
        candidates = _filter(None, True)
    if not candidates and allow_repeat:
        candidates = _filter(task_format, False) or _filter(None, False)
    if not candidates:
        return None
    if rng is not None:
        chosen = rng.choice(candidates)
    else:
        ordered = sorted(
            candidates,
            key=lambda t: (
                abs(len(t.get("answer_key") or []) - 2)
                if t.get("task_type") == "output_prediction" and isinstance(t.get("answer_key"), list)
                else 0,
                t["task_id"],
            ),
        )
        chosen = ordered[0]
    return copy.deepcopy(chosen)


def get_v22_reference_task(
    skill: str = "range_boundaries",
    task_format: str = "output_prediction",
    difficulty: str = "medium",
    pool: str = "assessment",
) -> dict[str, Any]:
    for t in _RAW_V22_TASKS:
        if (
            t["skill_id"] == skill
            and t["task_type"] == task_format
            and t["difficulty"] == difficulty
            and t["pool"] == pool
        ):
            return copy.deepcopy(t)
    raise KeyError(f"No v2.2 reference task for {(skill, task_format, difficulty, pool)}")


def get_reference_tasks_by_skill() -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    refs: dict[str, dict[tuple[str, str], dict[str, Any]]] = {s: {} for s in PPO_SKILLS_ORDER}
    for t in _RAW_V22_TASKS:
        if t.get("pool") == "assessment":
            key = (t["task_type"], t["difficulty"])
            if key not in refs[t["skill_id"]]:
                refs[t["skill_id"]][key] = copy.deepcopy(t)
    return refs


def get_ppo_reference_tasks() -> dict[str, dict[str, Any]]:
    """Exact 3 range_boundaries output_prediction assessment references used by 3/ppo/env.py."""
    return {
        diff: get_v22_reference_task("range_boundaries", "output_prediction", diff, "assessment")
        for diff in DIFFICULTIES
    }


def task_memory_features(task: dict[str, Any]) -> dict[str, Any]:
    """Extract the exact 8-key structured memory_task dictionary (zero text/code/IDs)."""
    if not isinstance(task, dict):
        raise ValueError("Task must be a dict")
    task_type = task.get("task_type")
    difficulty = task.get("difficulty")
    ref_sec = task.get("reference_seconds")
    if task_type not in FORMATS:
        raise ValueError(f"Invalid task_type {task_type!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Invalid difficulty {difficulty!r}")
    if not isinstance(ref_sec, (int, float)) or isinstance(ref_sec, bool) or not math.isfinite(ref_sec) or ref_sec <= 0:
        raise ValueError(f"Invalid reference_seconds {ref_sec!r}")

    snippets = task.get("snippets", [])
    options = task.get("options", [])
    criteria = task.get("criteria", [])
    if not isinstance(snippets, list) or not isinstance(options, list) or not isinstance(criteria, list):
        raise ValueError("Task snippets, options, and criteria must be lists")

    credit_weights = sorted(
        float(c["weight"]) for c in criteria if c.get("kind") != "incorrect_selection_penalty"
    )
    penalty_weights = sorted(
        float(c["weight"]) for c in criteria if c.get("kind") == "incorrect_selection_penalty"
    )

    return {
        "difficulty": difficulty,
        "task_type": task_type,
        "reference_seconds": float(ref_sec),
        "snippet_count": len(snippets),
        "option_count": len(options),
        "criterion_count": len(criteria),
        "credit_weights": credit_weights,
        "penalty_weights": penalty_weights,
    }


def validate_task_schema(
    task: dict[str, Any],
    *,
    verify_code_tests: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a v2.2 or LLM-generated task across all 5 formats and memory_task invariants."""
    errors: list[str] = []
    if not isinstance(task, dict):
        return False, ["Task must be a dictionary."]

    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        errors.append("task_id must be a non-empty string.")

    skill_id = task.get("skill_id")
    if skill_id not in PPO_SKILLS_ORDER:
        errors.append(f"skill_id {skill_id!r} must be in {PPO_SKILLS_ORDER}.")

    fmt = task.get("task_type")
    if fmt not in FORMATS:
        errors.append(f"task_type {fmt!r} must be in {FORMATS}.")

    diff = task.get("difficulty")
    if diff not in DIFFICULTIES:
        errors.append(f"difficulty {diff!r} must be in {DIFFICULTIES}.")

    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("prompt must be a non-empty string.")

    ref_sec = task.get("reference_seconds")
    if (
        not isinstance(ref_sec, (int, float))
        or isinstance(ref_sec, bool)
        or not math.isfinite(ref_sec)
        or ref_sec <= 0
    ):
        errors.append("reference_seconds must be a finite number > 0.")

    snippets = task.get("snippets")
    if not isinstance(snippets, list) or not all(isinstance(s, str) for s in snippets):
        errors.append("snippets must be a list of strings.")

    options = task.get("options")
    if not isinstance(options, list):
        errors.append("options must be a list.")

    criteria = task.get("criteria")
    if not isinstance(criteria, list) or len(criteria) == 0:
        errors.append("criteria must be a non-empty list.")

    tests = task.get("tests")
    if not isinstance(tests, list):
        errors.append("tests must be a list.")

    # Validate criteria weights
    if isinstance(criteria, list) and criteria:
        credit_sum = 0.0
        penalty_sum = 0.0
        has_penalty = False
        for idx, c in enumerate(criteria):
            if not isinstance(c, dict):
                errors.append(f"criteria[{idx}] must be a dict.")
                continue
            w = c.get("weight")
            if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w) or w < 0:
                errors.append(f"criteria[{idx}].weight must be a finite number >= 0.")
                continue
            if c.get("kind") == "incorrect_selection_penalty":
                has_penalty = True
                penalty_sum += float(w)
            else:
                credit_sum += float(w)
        if abs(credit_sum - 1.0) > 1e-4:
            errors.append(f"Positive credit weights must sum to 1.0 (got {credit_sum:.6f}).")
        if has_penalty and abs(penalty_sum - 1.0) > 1e-4:
            errors.append(f"Penalty weights must sum to 1.0 when present (got {penalty_sum:.6f}).")

    # Validate hints if provided
    hints = task.get("hints")
    if hints is not None:
        if not isinstance(hints, list):
            errors.append("hints must be a list when provided.")
        else:
            for idx, h in enumerate(hints):
                if not isinstance(h, dict):
                    errors.append(f"hints[{idx}] must be a dict.")
                    continue
                if not isinstance(h.get("level"), int) or isinstance(h.get("level"), bool):
                    errors.append(f"hints[{idx}].level must be an integer.")
                if h.get("kind") not in HINT_KINDS:
                    errors.append(f"hints[{idx}].kind must be in {HINT_KINDS}.")
                if not isinstance(h.get("text"), str) or not h["text"].strip():
                    errors.append(f"hints[{idx}].text must be a non-empty string.")

    # Format-specific checks
    if fmt in FORMATS and isinstance(options, list) and isinstance(criteria, list) and isinstance(snippets, list) and isinstance(tests, list):
        answer_key = task.get("answer_key")
        if fmt == "single_choice":
            if len(options) < 2:
                errors.append("single_choice requires at least 2 options.")
            opt_ids = []
            for idx, opt in enumerate(options):
                if not isinstance(opt, dict) or not isinstance(opt.get("id"), str) or not opt["id"] or not isinstance(opt.get("text"), str):
                    errors.append(f"options[{idx}] must have string 'id' and 'text'.")
                else:
                    opt_ids.append(opt["id"])
            if len(opt_ids) != len(set(opt_ids)):
                errors.append("single_choice option IDs must be unique.")
            if not isinstance(answer_key, str) or answer_key not in opt_ids:
                errors.append("single_choice answer_key must match one option id.")
            if len(criteria) != 1:
                errors.append("single_choice requires exactly 1 criterion.")

        elif fmt == "multiple_select":
            if len(options) < 2:
                errors.append("multiple_select requires at least 2 options.")
            opt_ids = []
            for idx, opt in enumerate(options):
                if not isinstance(opt, dict) or not isinstance(opt.get("id"), str) or not opt["id"] or not isinstance(opt.get("text"), str):
                    errors.append(f"options[{idx}] must have string 'id' and 'text'.")
                else:
                    opt_ids.append(opt["id"])
            if len(opt_ids) != len(set(opt_ids)):
                errors.append("multiple_select option IDs must be unique.")
            if not isinstance(answer_key, list) or not answer_key or not all(isinstance(a, str) for a in answer_key):
                errors.append("multiple_select answer_key must be a non-empty list of option IDs.")
            elif len(set(answer_key)) != len(answer_key) or not set(answer_key) <= set(opt_ids):
                errors.append("multiple_select answer_key must contain unique valid option IDs.")
            if len(criteria) != len(options):
                errors.append("multiple_select requires len(criteria) == len(options).")
            elif isinstance(answer_key, list):
                correct_set = set(answer_key)
                for opt, crit in zip(options, criteria):
                    if isinstance(opt, dict) and isinstance(crit, dict):
                        expected_kind = "correct_selection" if opt.get("id") in correct_set else "incorrect_selection_penalty"
                        if crit.get("kind") != expected_kind:
                            errors.append(f"multiple_select criterion kind mismatch for option {opt.get('id')!r}.")

        elif fmt == "output_prediction":
            if len(snippets) < 1:
                errors.append("output_prediction requires at least 1 snippet.")
            if not isinstance(answer_key, list) or len(answer_key) < 1 or not all(isinstance(a, str) and a.strip() != "" for a in answer_key):
                errors.append("output_prediction answer_key must be a non-empty list of non-empty strings.")
            elif len(criteria) != len(answer_key):
                errors.append("output_prediction requires len(criteria) == len(answer_key).")

        elif fmt == "fill_in_the_blank":
            if len(snippets) < 1:
                errors.append("fill_in_the_blank requires at least 1 snippet.")
            if not isinstance(answer_key, list) or len(answer_key) < 1 or not all(isinstance(a, str) for a in answer_key):
                errors.append("fill_in_the_blank answer_key must be a non-empty list of integer strings.")
            else:
                for idx, val in enumerate(answer_key):
                    try:
                        int(val.strip())
                    except (ValueError, AttributeError):
                        errors.append(f"fill_in_the_blank answer_key[{idx}] ({val!r}) must parse as int.")
                if len(criteria) != len(answer_key):
                    errors.append("fill_in_the_blank requires len(criteria) == len(answer_key).")

        elif fmt == "code_completion":
            if len(snippets) < 1:
                errors.append("code_completion requires at least 1 snippet.")
            if len(tests) < 1:
                errors.append("code_completion requires at least 1 test case.")
            else:
                for idx, tc in enumerate(tests):
                    if not isinstance(tc, dict) or "input" not in tc or "expected" not in tc:
                        errors.append(f"tests[{idx}] must have 'input' and 'expected' keys.")
                if len(criteria) != len(tests):
                    errors.append("code_completion requires len(criteria) == len(tests).")
            if verify_code_tests and not errors:
                ref_sol = task.get("reference_solution")
                if not isinstance(ref_sol, str) or not ref_sol.strip():
                    errors.append("verify_code_tests=True requires a non-empty reference_solution.")
                else:
                    from .grader import Interpreter, Unsupported

                    try:
                        runner = Interpreter(ref_sol)
                        for idx, tc in enumerate(tests):
                            actual = runner.run(tc["input"])
                            if actual != tc["expected"]:
                                errors.append(
                                    f"reference_solution failed tests[{idx}]: expected {tc['expected']!r}, got {actual!r}."
                                )
                    except (SyntaxError, Unsupported, RecursionError, OverflowError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
                        errors.append(f"reference_solution execution failed: {exc}")

    # Ensure memory_task dict projection succeeds if no prior errors
    if not errors:
        try:
            task_memory_features(task)
        except ValueError as exc:
            errors.append(str(exc))

    return (len(errors) == 0), errors


def assert_valid_task(task: dict[str, Any], *, verify_code_tests: bool = False) -> dict[str, Any]:
    ok, errors = validate_task_schema(task, verify_code_tests=verify_code_tests)
    if not ok:
        raise ValueError("Invalid task schema: " + "; ".join(errors))
    return task
