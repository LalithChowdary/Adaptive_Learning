"""Deterministic & AST submission grader plus rubric self-explanation evaluator."""
import importlib.util
import re
import sys
import types
from typing import Any

from . import CODE_3_ROOT, catalogue_bridge

DASHBOARD_GRADING_PY = CODE_3_ROOT / "dashboard" / "grading.py"
_SENTINEL = object()


def _load_dashboard_grading():
    mod_name = "adaptive_web._dashboard_grading"
    existing = sys.modules.get(mod_name)
    if existing is not None:
        return existing
    v22_catalog = catalogue_bridge.get_v22_catalog_module()
    spec = importlib.util.spec_from_file_location(mod_name, DASHBOARD_GRADING_PY)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load dashboard grader at {DASHBOARD_GRADING_PY}")
    module = importlib.util.module_from_spec(spec)
    prev_cat = sys.modules.get("catalogue", _SENTINEL)
    try:
        if prev_cat is _SENTINEL or not hasattr(prev_cat, "source"):
            sys.modules["catalogue"] = types.SimpleNamespace(source=v22_catalog)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    finally:
        if prev_cat is _SENTINEL:
            sys.modules.pop("catalogue", None)
        else:
            sys.modules["catalogue"] = prev_cat
    return module


_DASHBOARD_GRADING = _load_dashboard_grading()

Interpreter = _DASHBOARD_GRADING.Interpreter
Unsupported = _DASHBOARD_GRADING.Unsupported
Limit = _DASHBOARD_GRADING.Limit
normalize_output = _DASHBOARD_GRADING.normalize_output


def grade_submission(task: dict[str, Any], answer: Any) -> dict[str, Any]:
    """Grade a v2.2 or LLM-generated task deterministically and return full structured results."""
    raw = dict(_DASHBOARD_GRADING.grade(task, answer))
    status = raw.get("grading_status", "resolved")

    if status == "unresolved":
        raw.setdefault("criteria_results", [])
        msg = raw.get("message") or "Submission could not be verified automatically."
        raw.setdefault("feedback_summary", f"Unresolved: {msg}")
        return raw

    criteria = task.get("criteria", [])
    flags = list(raw.get("criteria", []))
    details = list(raw.get("details", []))
    detail_by_idx = {}
    for d in details:
        if isinstance(d, dict) and "criterion" in d:
            detail_by_idx[int(d["criterion"]) - 1] = d.get("message", "")

    criteria_results: list[dict[str, Any]] = []
    for idx, crit in enumerate(criteria):
        passed = bool(flags[idx]) if idx < len(flags) else False
        msg = detail_by_idx.get(idx)
        if not msg:
            if details and idx == 0 and isinstance(details[0], dict) and "message" in details[0]:
                msg = details[0]["message"]
            else:
                msg = "Passed" if passed else "Did not match expected criterion"
        criteria_results.append(
            {
                "id": crit.get("id", f"c{idx}"),
                "kind": crit.get("kind", "exact_match"),
                "weight": float(crit.get("weight", 0.0)),
                "passed": passed,
                "message": msg,
            }
        )

    score = float(raw.get("score") or 0.0)
    full = bool(raw.get("full_success"))
    total_crit = len(criteria_results)
    passed_count = sum(1 for cr in criteria_results if cr["passed"])
    if full:
        summary = f"Passed all {total_crit} criteria (100% credit)."
    elif details and isinstance(details[0], dict) and "message" in details[0] and "Syntax error" in details[0]["message"]:
        summary = details[0]["message"]
    elif score > 0.0:
        summary = f"Partial credit ({score * 100:.0f}%): {passed_count} of {total_crit} criteria matched."
    else:
        summary = f"Incorrect (0%): {passed_count} of {total_crit} criteria matched."

    raw["criteria_results"] = criteria_results
    raw["feedback_summary"] = summary
    return raw


grade = grade_submission


# --- Self-Explanation Rubric Evaluator (Strictly Separates score from grader_confidence) ---

_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "start": ("start", "starts", "starting", "begin", "begins", "initial", "inclusive"),
    "stop": ("stop", "stops", "upper bound", "end", "limit", "boundary"),
    "excluded": ("exclude", "excluded", "excludes", "excluding", "exclusive", "strictly below", "not included", "before"),
    "step": ("step", "steps", "increment", "stride", "jump"),
    "iteration": ("iteration", "iterations", "loop", "loops", "pass", "passes", "each time", "repeat"),
    "initial": ("initial", "initialize", "initialized", "initialization", "baseline", "starting value", "before the loop"),
    "accumulator": ("accumulator", "running total", "total", "sum", "add", "accumulate"),
    "condition": ("condition", "conditions", "if", "elif", "else", "branch", "branches", "true", "false"),
    "strict": ("strict", "equality", "equal", "greater", "less", ">", "<", ">=", "<="),
    "return": ("return", "returns", "returned", "exit", "exits", "caller"),
    "precedence": ("precedence", "order of operations", "multiply", "multiplication", "parentheses", "before addition"),
    "index": ("index", "indices", "position", "slot", "offset", "element"),
    "zero-based": ("zero-based", "0-based", "index 0", "zero", "at 0", "from 0"),
    "negative": ("negative", "-1", "from the end", "from the right", "backwards", "last element"),
}

_HEDGING_PATTERNS = (
    "not sure",
    "maybe",
    "i guess",
    "probably",
    "might be",
    "unsure",
    "idk",
    "don't know",
    "do not know",
)
_CAUSAL_CONNECTORS = (
    "because",
    "so ",
    "therefore",
    "means",
    "since ",
    "which ",
    "before ",
    "after ",
    "instead",
    "thus",
    "until",
)


def _infer_concepts_from_prompt(prompt: str) -> list[str]:
    lower = prompt.lower()
    inferred: list[str] = []
    for canonical, aliases in _CONCEPT_ALIASES.items():
        if canonical in lower or any(a in lower for a in aliases[:3]):
            inferred.append(canonical)
    if not inferred:
        # Generic fallback: meaningful content words >= 5 chars
        words = [w for w in re.findall(r"[a-z]{5,}", lower) if w not in {"explain", "words", "about", "python", "check", "before", "verify"}]
        inferred = list(dict.fromkeys(words[:4]))
    return inferred[:5]


def _concept_matched(concept: str, text_lower: str) -> bool:
    c_lower = concept.strip().lower()
    if not c_lower:
        return False
    # Direct substring / word match
    if re.search(r"[a-z0-9]", c_lower):
        if len(c_lower) <= 3:
            if c_lower in text_lower:
                return True
        else:
            # Match exact phrase or prefix stem (len >= 5)
            if c_lower in text_lower:
                return True
            if " " not in c_lower and len(c_lower) >= 5:
                stem = c_lower[:5]
                if re.search(rf"\b{re.escape(stem)}[a-z]*\b", text_lower):
                    return True
    elif c_lower in text_lower:
        return True

    # Alias lookup if concept matches a known canonical concept or shares an alias
    aliases = _CONCEPT_ALIASES.get(c_lower)
    if aliases is None:
        for group_aliases in _CONCEPT_ALIASES.values():
            if c_lower in group_aliases:
                aliases = group_aliases
                break
    if aliases:
        for alias in aliases:
            if alias in text_lower:
                return True
    return False


def grade_self_explanation(
    prompt: str,
    explanation_text: str,
    rubric: dict[str, Any] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a learner's self-explanation against rubric concepts.

    Preserves PROJECT_DECISIONS.md strict separation:
      - `score` and `full_success` depend strictly on rubric concept coverage.
      - `grader_confidence` measures clarity/ambiguity separately and NEVER multiplies `score`.
    """
    if not isinstance(explanation_text, str):
        raise ValueError("explanation_text must be a string.")
    prompt_str = prompt if isinstance(prompt, str) else ""

    min_words = 6
    if isinstance(rubric, dict):
        concepts_raw = rubric.get("key_concepts") or rubric.get("required_concepts") or []
        min_words = int(rubric.get("min_words", 6))
    elif isinstance(rubric, (list, tuple)):
        concepts_raw = list(rubric)
    else:
        concepts_raw = _infer_concepts_from_prompt(prompt_str)

    key_concepts = [str(c).strip() for c in concepts_raw if str(c).strip()]
    cleaned = explanation_text.strip()
    lower = cleaned.lower()
    words = re.findall(r"[A-Za-z0-9_'\-]+", cleaned)

    # Compute grader_confidence strictly independently from score
    if not cleaned:
        confidence = 0.95
    else:
        conf = 0.45
        if len(words) >= max(1, min_words):
            conf += 0.25
        elif len(words) >= 3:
            conf += 0.10
        else:
            conf -= 0.15
        if any(conn in lower for conn in _CAUSAL_CONNECTORS):
            conf += 0.20
        hedge_hits = sum(1 for h in _HEDGING_PATTERNS if h in lower)
        if hedge_hits:
            conf -= min(0.45, 0.22 * hedge_hits)
        # Code-only paste without prose explanation lowers evaluator confidence
        if cleaned.startswith("def ") or (cleaned.count("=") >= 2 and len(re.findall(r"\b[a-z]{4,}\b", lower)) < 4):
            conf -= 0.20
        confidence = round(max(0.05, min(0.99, conf)), 4)

    if not cleaned:
        return {
            "score": 0.0,
            "full_success": False,
            "grading_status": "resolved",
            "grader_confidence": confidence,
            "key_concepts_hit": [],
            "missing_concepts": list(key_concepts),
            "feedback": "No explanation was provided. State the key rule or trace step in your own words.",
            "grader": "deterministic-self-explanation-v1",
        }

    if not key_concepts:
        if confidence < 0.35:
            return {
                "score": None,
                "full_success": None,
                "grading_status": "unresolved",
                "grader_confidence": confidence,
                "key_concepts_hit": [],
                "missing_concepts": [],
                "feedback": "Explanation was too brief or ambiguous to grade automatically without rubric concepts.",
                "grader": "deterministic-self-explanation-v1",
            }
        return {
            "score": 1.0,
            "full_success": True,
            "grading_status": "resolved",
            "grader_confidence": confidence,
            "key_concepts_hit": [],
            "missing_concepts": [],
            "feedback": "Clear explanation recorded.",
            "grader": "deterministic-self-explanation-v1",
        }

    hit = [c for c in key_concepts if _concept_matched(c, lower)]
    missing = [c for c in key_concepts if c not in hit]
    # Score depends strictly on rubric coverage, never multiplied by grader_confidence
    score = round(len(hit) / len(key_concepts), 4)
    full_success = len(missing) == 0 and score >= 1.0

    if full_success:
        feedback = f"Solid self-explanation: covered all {len(key_concepts)} key concepts ({', '.join(hit)})."
    elif hit:
        feedback = (
            f"Partial self-explanation ({len(hit)}/{len(key_concepts)} concepts): "
            f"addressed {', '.join(hit)}; also explain {', '.join(missing)}."
        )
    else:
        feedback = (
            f"Explanation did not explicitly address the core concepts: {', '.join(missing)}."
        )

    return {
        "score": score,
        "full_success": full_success,
        "grading_status": "resolved",
        "grader_confidence": confidence,
        "key_concepts_hit": hit,
        "missing_concepts": missing,
        "feedback": feedback,
        "grader": "deterministic-self-explanation-v1",
    }
