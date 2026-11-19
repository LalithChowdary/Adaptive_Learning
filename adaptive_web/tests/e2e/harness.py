"""Opaque-box E2E test harness for 3/adaptive_web/.

Uses ONLY Python 3.12 stdlib imports at module load time so `unittest discover`
succeeds cleanly at any stage of concurrent M1/M2/M3 development, while lazily
importing and booting the real `adaptive_web` server inside `AdaptiveWebHarness.start()`.
"""

from __future__ import annotations

import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import unittest.mock
import urllib.parse
import uuid

REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_3_DIR = REPO_ROOT / "3"
ADAPTIVE_WEB_DIR = CODE_3_DIR / "adaptive_web"
PROPOSAL_PATH = ADAPTIVE_WEB_DIR / "PROPOSAL_PPO_LLM_ARCHITECTURE.md"
TASK_CATALOG_PATH = CODE_3_DIR / "data_Set" / "v2.2" / "dataset" / "task_catalog.json"

PPO_SKILLS_ORDER = (
    "range_boundaries",
    "loop_tracing",
    "accumulator_initialization",
    "conditional_evaluation",
    "list_indexing",
    "function_returns",
)

PREREQUISITES = {
    "range_boundaries": (),
    "loop_tracing": ("range_boundaries",),
    "accumulator_initialization": ("range_boundaries", "loop_tracing"),
    "conditional_evaluation": ("loop_tracing",),
    "list_indexing": ("range_boundaries",),
    "function_returns": ("conditional_evaluation", "accumulator_initialization"),
}

TASK_FORMATS = (
    "single_choice",
    "multiple_select",
    "output_prediction",
    "fill_in_the_blank",
    "code_completion",
)

PPO_ACTIONS = (
    "give_easier_question",
    "give_same_difficulty_question",
    "give_harder_question",
    "explain_concept",
    "show_worked_example",
    "request_self_explanation",
    "review_prerequisite",
    "advance_next_topic",
)

PALETTE_HEX = (
    "#f7f8f3",
    "#ffffff",
    "#e0e5d9",
    "#242922",
    "#687064",
    "#c6f36a",
    "#d2f58d",
    "#839f50",
    "#293321",
    "#f4f2fa",
    "#fff0df",
    "#7058ba",
)

FORBIDDEN_GRADIENT_PATTERNS = (
    "linear-gradient",
    "radial-gradient",
    "conic-gradient",
    "repeating-linear-gradient",
    "repeating-radial-gradient",
    "repeating-conic-gradient",
)

FORBIDDEN_PURPLE_STOPS = (
    "#667eea",
    "#764ba2",
    "#8b5cf6",
    "#a855f7",
    "#7c3aed",
    "#9333ea",
    "#6d28d9",
    "#5b21b6",
    "#a78bfa",
    "#c084fc",
    "#9b59b6",
    "#8e44ad",
)

FORBIDDEN_PURPLE_NAMED = (
    "rebeccapurple",
    "blueviolet",
    "darkviolet",
    "mediumpurple",
)

LIVE_SECRET_KEYS = (
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "ADAPTIVE_WEB_PORT",
    "ADAPTIVE_WEB_DB",
    "DEVICE",
    "LLM_TIMEOUT_SECONDS",
)

_V22_CACHE: dict[str, dict[str, Any]] | None = None


def ensure_adaptive_web_on_sys_path() -> None:
    """Add 3/adaptive_web and 3/ to sys.path for lazy runtime resolution."""
    for path in (str(ADAPTIVE_WEB_DIR), str(CODE_3_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _reject_non_finite_constant(val: str) -> Any:
    raise ValueError(f"Non-finite JSON constant not allowed: {val}")


def loads_strict_json(raw: bytes | str) -> Any:
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    return json.loads(text, parse_constant=_reject_non_finite_constant)


def load_v22_catalog() -> dict[str, dict[str, Any]]:
    """Read-only lookup of all 2,160 v2.2 tasks keyed by task_id."""
    global _V22_CACHE
    if _V22_CACHE is None:
        raw = json.loads(TASK_CATALOG_PATH.read_text(encoding="utf-8"))
        _V22_CACHE = {item["task_id"]: item for item in raw}
    return _V22_CACHE


def solve_v22_task(task: dict[str, Any] | str) -> tuple[Any, Any]:
    """Return (valid_answer, invalid_answer) for any v2.2 task or task_id."""
    catalog = load_v22_catalog()
    if isinstance(task, str):
        task_id = task
    else:
        task_id = task.get("task_id") or task.get("task") or task.get("id")
    if not task_id or task_id not in catalog:
        raise KeyError(f"Task ID {task_id!r} not found in v2.2 task_catalog.json")
    item = catalog[task_id]
    fmt = item["task_type"]
    if fmt == "single_choice":
        valid = item["answer_key"]
        wrong = next(o["id"] for o in item["options"] if o["id"] != valid)
        return valid, wrong
    if fmt == "multiple_select":
        valid = list(item["answer_key"])
        wrong_opt = next(o["id"] for o in item["options"] if o["id"] not in valid)
        return valid, [wrong_opt]
    if fmt == "output_prediction":
        valid = list(item["answer_key"])
        return valid, ["WRONG_E2E_OUTPUT"] * len(valid)
    if fmt == "fill_in_the_blank":
        valid = list(item["answer_key"])
        return valid, ["999999"] * len(valid)
    if fmt == "code_completion":
        return item["reference_solution"], "def solve(x):\n    return -999999"
    raise ValueError(f"Unsupported task_type: {fmt!r}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_reference_hashes() -> dict[str, str]:
    """Return SHA-256 digests of immutable reference files in 3/dashboard, 3/data_Set/v2.2, 3/model, 3/ppo."""
    targets: list[Path] = [
        CODE_3_DIR / "data_Set" / "v2.2" / "dataset" / "task_catalog.json",
        CODE_3_DIR / "data_Set" / "v2.2" / "catalog.py",
        CODE_3_DIR / "model" / "runs" / "v2.2-mps-main" / "monotonic_gru" / "seed_20261122" / "model.pt",
        CODE_3_DIR / "ppo" / "runs" / "v2.2-ppo-main" / "delayed_retention_ppo" / "seed_20261120.pt",
        CODE_3_DIR / "ppo" / "runs" / "v2.2-ppo-main" / "delayed_retention_ppo" / "seed_20261121.pt",
        CODE_3_DIR / "ppo" / "runs" / "v2.2-ppo-main" / "delayed_retention_ppo" / "seed_20261122.pt",
        CODE_3_DIR / "ppo" / "env.py",
        CODE_3_DIR / "ppo" / "agent.py",
        CODE_3_DIR / "dashboard" / "server.py",
        CODE_3_DIR / "dashboard" / "service.py",
        CODE_3_DIR / "dashboard" / "catalogue.py",
        CODE_3_DIR / "dashboard" / "grading.py",
        CODE_3_DIR / "dashboard" / "static" / "style.css",
    ]
    sessions_db = CODE_3_DIR / "dashboard" / "data" / "sessions.sqlite3"
    if sessions_db.exists():
        targets.append(sessions_db)

    digests: dict[str, str] = {}
    for p in targets:
        rel = str(p.relative_to(REPO_ROOT))
        if not p.exists():
            raise FileNotFoundError(f"Expected reference artifact missing: {p}")
        digests[rel] = _sha256_file(p)
    return digests


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def strip_css_comments(css_text: str) -> str:
    """Remove CSS /* ... */ block comments."""
    return re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)


def strip_html_comments(html_text: str) -> str:
    """Remove HTML <!-- ... --> comments."""
    return re.sub(r"<!--.*?-->", "", html_text, flags=re.DOTALL)


def strip_js_comments(js_text: str) -> str:
    """Remove JS /* ... */ block comments and standalone // line comments."""
    no_block = re.sub(r"/\*.*?\*/", "", js_text, flags=re.DOTALL)
    return re.sub(r"(^|[\s;{},()])//[^\n]*", r"\1", no_block, flags=re.MULTILINE)


def normalize_hex_tokens(css_text: str) -> set[str]:
    """Extract lowercase normalized hex colors from CSS property values (ignoring comments & #id selectors)."""
    stripped = strip_css_comments(css_text)
    out: set[str] = set()
    # Match real CSS declarations `property: value;` or `property: value}` (never `selector:pseudo {`)
    for decl in re.finditer(r"(?:^|[{;])\s*[a-zA-Z_-][a-zA-Z0-9_-]*\s*:\s*([^;{}]+?)\s*(?=[;}])", stripped):
        val = decl.group(1)
        for token in re.findall(r"(?<![0-9a-zA-Z_])#[0-9a-fA-F]+\b", val):
            t = token.lower()
            hex_len = len(t) - 1
            if hex_len == 3:
                out.add("#" + "".join(ch * 2 for ch in t[1:]))
            elif hex_len == 4:
                out.add("#" + "".join(ch * 2 for ch in t[1:]))
            elif hex_len in (6, 8):
                out.add(t)
    return out


def find_forbidden_gradients_and_purples(
    css_text: str = "",
    html_text: str = "",
    js_text: str = "",
) -> list[str]:
    """Detect CSS gradients, SVG gradients, and forbidden purple colors across CSS/HTML/JS (comments stripped)."""
    clean_css = strip_css_comments(css_text).lower()
    clean_html = strip_html_comments(html_text).lower()
    clean_js = strip_js_comments(js_text).lower()
    combined = "\n".join(part for part in (clean_css, clean_html, clean_js) if part)

    hits: list[str] = []
    for pat in FORBIDDEN_GRADIENT_PATTERNS:
        if pat in combined:
            hits.append(f"grad:{pat}")
    if re.search(r"<\s*(?:lineargradient|radialgradient)\b", combined):
        hits.append("svg:gradient")

    css_hexes = normalize_hex_tokens(css_text) if css_text else set()
    other_hexes: set[str] = set()
    for raw in re.findall(r"(?<![0-9a-zA-Z_])#([0-9a-f]{3,8})\b", clean_html + "\n" + clean_js):
        if len(raw) == 3:
            other_hexes.add("#" + "".join(ch * 2 for ch in raw))
        elif len(raw) in (6, 8):
            other_hexes.add("#" + raw)
    all_hexes = css_hexes | other_hexes
    for stop in FORBIDDEN_PURPLE_STOPS:
        if stop in all_hexes or stop in combined:
            hits.append(f"purple:{stop}")

    for name in FORBIDDEN_PURPLE_NAMED:
        if re.search(rf"\b{name}\b", combined):
            hits.append(f"purple:{name}")
    if re.search(r":\s*purple\b", clean_css) or re.search(r"['\"]purple['\"]\s*(?:,|;|\})", clean_js):
        hits.append("purple:named")

    for m in re.finditer(r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})", combined):
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if b >= 160 and r >= 90 and g <= min(r, b) - 30:
            hits.append(f"purple:rgb({r},{g},{b})")

    return sorted(set(hits))


class _StubThreadingHTTPServer(ThreadingHTTPServer):
    request_queue_size = 64
    daemon_threads = True
    allow_reuse_address = True


class LocalLLMStubServer:
    """Loopback HTTP LLM stub speaking OpenAI, Anthropic, and Gemini REST formats."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        delay_seconds: float = 0.0,
        malformed_json: bool = False,
        fenced_json: bool = False,
        raw_text_override: str | None = None,
        responder: Callable[[str, dict[str, str], dict[str, Any]], Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.delay_seconds = delay_seconds
        self.malformed_json = malformed_json
        self.fenced_json = fenced_json
        self.raw_text_override = raw_text_override
        self.responder = responder
        self.received_requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int | None = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise RuntimeError("LocalLLMStubServer is not started")
        return f"http://127.0.0.1:{self.port}"

    @staticmethod
    def _extract_user_text(prompt_text: str) -> str:
        try:
            data = json.loads(prompt_text)
        except Exception:
            return ""
        if not isinstance(data, dict):
            return ""
        parts: list[str] = []
        msgs = data.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for blk in content:
                            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                                parts.append(blk["text"])
        contents = data.get("contents")
        if isinstance(contents, list):
            for c in contents:
                if isinstance(c, dict) and c.get("role", "user") == "user":
                    for p in c.get("parts") or []:
                        if isinstance(p, dict) and isinstance(p.get("text"), str):
                            parts.append(p["text"])
        return "\n".join(parts)

    @staticmethod
    def _stub_hints(h1: str, h2: str, h3: str) -> list[dict[str, Any]]:
        return [
            {"level": 1, "kind": "conceptual", "text": h1},
            {"level": 2, "kind": "targeted", "text": h2},
            {"level": 3, "kind": "scaffold", "text": h3},
        ]

    def _default_payload(self, prompt_text: str) -> dict[str, Any]:
        full_low = prompt_text.lower()
        user_low = self._extract_user_text(prompt_text).lower()
        search_low = user_low or full_low

        if ("learning_path" in full_low or "study path" in full_low or "curate" in full_low) and "question" not in search_low:
            return {
                "title": "Stub Personalized Study Path",
                "summary": "Adaptive Learn-Code-Remember sequence via LocalLLMStubServer.",
                "focus_skill": "range_boundaries",
                "steps": [
                    {
                        "phase": "learn",
                        "title": "Core Concept & Boundary Mental Model",
                        "description": "Review start inclusive, stop exclusive, and prerequisite foundations.",
                        "recommended_action": "explain_concept",
                    },
                    {
                        "phase": "code",
                        "title": "Subgoal-Labeled Worked Trace & Practice",
                        "description": "Work through progressive tasks with targeted hints.",
                        "recommended_action": "give_same_difficulty_question",
                    },
                    {
                        "phase": "remember",
                        "title": "Self-Explanation & Spaced Consolidation",
                        "description": "Articulate why the rule holds to consolidate permastore b_t.",
                        "recommended_action": "request_self_explanation",
                    },
                ],
                "source": "llm",
            }
        if "self_explanation" in full_low or "evaluate_self" in full_low:
            return {
                "score": 1.0,
                "full_success": True,
                "grading_status": "resolved",
                "grader_confidence": 0.91,
                "key_concepts_hit": ["start", "stop", "excluded", "step", "condition"],
                "missing_concepts": [],
                "feedback": "Clear explanation of inclusive start, step progression, and exclusive stop boundaries.",
                "next_tip": "Try a medium code completion exercise next.",
            }
        if (
            "course_material" in full_low
            or "worked_example" in full_low
            or ("explanation" in full_low and "self_explanation" not in full_low and "question" not in full_low and "task_type" not in full_low)
        ):
            return {
                "title": "Stub Course Material",
                "concept_explanation": "Python sequences and control flow follow deterministic evaluation rules.",
                "mental_model": "Trace each step explicitly from initial state to final output.",
                "pitfalls": ["Off-by-one stop boundary", "Unintended initial accumulator"],
                "worked_example": {
                    "title": "Subgoal Worked Trace",
                    "code": "vals = list(range(2, 8, 2))\nprint(vals)",
                    "subgoals": [
                        {"label": "Subgoal 1: Identify Bounds", "detail": "start=2, stop=8 (excluded), step=2"},
                        {"label": "Subgoal 2: Enumerate Elements", "detail": "2, 4, 6 (< 8); stop before 8"},
                    ],
                },
                "quick_check": "What is the last item of list(range(1, 6, 2))?",
                "source": "llm",
            }
        if (
            "hint_level" in full_low
            or "hint_kind" in full_low
            or ("hint" in full_low and "hints" not in full_low and "question" not in full_low and "task_type" not in full_low)
        ):
            return {
                "hint_level": 1,
                "hint_kind": "conceptual",
                "text": "Stub hint: check the boundary condition step by step.",
                "leakage_l_t": 0.25,
            }

        requested_fmt = None
        m = re.search(
            r'(?:format|task_type)["\':=\s]+(single_choice|multiple_select|output_prediction|fill_in_the_blank|code_completion)',
            search_low,
        )
        if m:
            requested_fmt = m.group(1)
        else:
            hits = [fmt for fmt in TASK_FORMATS if fmt in search_low]
            if len(hits) == 1:
                requested_fmt = hits[0]
            elif not user_low:
                for fmt in TASK_FORMATS:
                    if fmt in full_low:
                        requested_fmt = fmt
                        break

        if requested_fmt == "code_completion":
            return {
                "task_id": f"llm.range_boundaries.easy.code_completion.{uuid.uuid4().hex[:6]}",
                "skill_id": "range_boundaries",
                "task_type": "code_completion",
                "difficulty": "easy",
                "prompt": "Complete def solve(x): to return list(range(1, x, 2)).",
                "snippets": ["def solve(x):\n    # complete\n    pass"],
                "options": [],
                "answer_key": None,
                "reference_solution": "def solve(x):\n    return list(range(1, x, 2))",
                "tests": [
                    {"input": 6, "expected": [1, 3, 5]},
                    {"input": 8, "expected": [1, 3, 5, 7]},
                ],
                "criteria": [
                    {"kind": "behavior_test", "weight": 0.5},
                    {"kind": "behavior_test", "weight": 0.5},
                ],
                "reference_seconds": 60.0,
                "hints": self._stub_hints(
                    "Start at 1 and step by 2.",
                    "Use range(1, x, 2).",
                    "Return list(range(1, x, 2)).",
                ),
            }
        if requested_fmt == "output_prediction":
            return {
                "task_id": f"llm.range_boundaries.easy.output_prediction.{uuid.uuid4().hex[:6]}",
                "skill_id": "range_boundaries",
                "task_type": "output_prediction",
                "difficulty": "easy",
                "prompt": "Predict the exact printed output.",
                "snippets": ["print(list(range(1, 6, 2)))"],
                "options": [],
                "answer_key": ["[1, 3, 5]"],
                "reference_solution": None,
                "tests": [],
                "criteria": [{"kind": "exact_match", "weight": 1.0}],
                "reference_seconds": 35.0,
                "hints": self._stub_hints(
                    "Start at 1 and add step=2.",
                    "Stop at 6 is excluded.",
                    "The sequence is 1, 3, 5.",
                ),
            }
        if requested_fmt == "fill_in_the_blank":
            return {
                "task_id": f"llm.range_boundaries.easy.fill_in_the_blank.{uuid.uuid4().hex[:6]}",
                "skill_id": "range_boundaries",
                "task_type": "fill_in_the_blank",
                "difficulty": "easy",
                "prompt": "Fill in the integer stop value so list(range(1, ___, 2)) == [1, 3, 5].",
                "snippets": ["list(range(1, ___, 2))"],
                "options": [],
                "answer_key": ["6"],
                "reference_solution": None,
                "tests": [],
                "criteria": [{"kind": "exact_match", "weight": 1.0}],
                "reference_seconds": 30.0,
                "hints": self._stub_hints(
                    "Stop is excluded from range.",
                    "The next value after 5 with step 2 is 7, so 6 already excludes 7.",
                    "Enter integer 6.",
                ),
            }
        if requested_fmt == "multiple_select":
            return {
                "task_id": f"llm.range_boundaries.easy.multiple_select.{uuid.uuid4().hex[:6]}",
                "skill_id": "range_boundaries",
                "task_type": "multiple_select",
                "difficulty": "easy",
                "prompt": "Select every value included in list(range(2, 7, 2)).",
                "snippets": ["list(range(2, 7, 2))"],
                "options": [
                    {"id": "o0", "text": "2"},
                    {"id": "o1", "text": "4"},
                    {"id": "o2", "text": "6"},
                    {"id": "o3", "text": "7"},
                ],
                "answer_key": ["o0", "o1", "o2"],
                "reference_solution": None,
                "tests": [],
                "criteria": [
                    {"kind": "correct_selection", "weight": 1.0 / 3.0},
                    {"kind": "correct_selection", "weight": 1.0 / 3.0},
                    {"kind": "correct_selection", "weight": 1.0 / 3.0},
                    {"kind": "incorrect_selection_penalty", "weight": 1.0},
                ],
                "reference_seconds": 40.0,
                "hints": self._stub_hints(
                    "Start at 2 and step by 2.",
                    "Stop=7 is strictly excluded.",
                    "Select 2, 4, and 6.",
                ),
            }
        return {
            "task_id": f"llm.range_boundaries.easy.single_choice.{uuid.uuid4().hex[:6]}",
            "skill_id": "range_boundaries",
            "task_type": "single_choice",
            "difficulty": "easy",
            "prompt": "What does list(range(1, 5)) evaluate to?",
            "snippets": ["list(range(1, 5))"],
            "options": [
                {"id": "o0", "text": "[1, 2, 3, 4]"},
                {"id": "o1", "text": "[1, 2, 3, 4, 5]"},
                {"id": "o2", "text": "[0, 1, 2, 3, 4]"},
                {"id": "o3", "text": "[2, 3, 4, 5]"},
            ],
            "answer_key": "o0",
            "reference_solution": None,
            "tests": [],
            "criteria": [{"kind": "exact_match", "weight": 1.0}],
            "reference_seconds": 35.0,
            "hints": self._stub_hints(
                "Remember stop is excluded.",
                "Start is 1, stop is 5.",
                "Values are 1, 2, 3, 4.",
            ),
        }

    def start(self) -> "LocalLLMStubServer":
        parent = self

        class StubHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b""
                try:
                    parsed_body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except Exception:
                    parsed_body = {"_raw": raw_body.decode("utf-8", errors="replace")}
                headers_lower = {k.lower(): v for k, v in self.headers.items()}
                with parent._lock:
                    parent.received_requests.append(
                        {
                            "path": self.path,
                            "headers": headers_lower,
                            "body": parsed_body,
                        }
                    )
                    delay = parent.delay_seconds
                    status = parent.status_code
                    malformed = parent.malformed_json
                    fenced = parent.fenced_json
                    raw_override = parent.raw_text_override
                    responder = parent.responder

                if delay > 0:
                    time.sleep(delay)

                if status != 200:
                    err_bytes = json.dumps({"error": {"message": f"Stub HTTP {status}"}}).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_bytes)))
                    self.end_headers()
                    self.wfile.write(err_bytes)
                    return

                if malformed:
                    bad_bytes = b'{"incomplete": [1, 2'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(bad_bytes)))
                    self.end_headers()
                    self.wfile.write(bad_bytes)
                    return

                if raw_override is not None:
                    content_text = raw_override
                elif responder is not None:
                    res = responder(self.path, headers_lower, parsed_body)
                    content_text = res if isinstance(res, str) else json.dumps(res)
                else:
                    prompt_blob = json.dumps(parsed_body)
                    content_text = json.dumps(parent._default_payload(prompt_blob))

                if fenced:
                    content_text = f"```json\n{content_text}\n```"

                path_low = self.path.lower()
                if "messages" in path_low:
                    envelope = {
                        "id": "msg_e2e_stub",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": content_text}],
                    }
                elif "generatecontent" in path_low:
                    envelope = {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": content_text}], "role": "model"},
                                "finishReason": "STOP",
                            }
                        ]
                    }
                else:
                    envelope = {
                        "id": "chatcmpl_e2e_stub",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content_text},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                out_bytes = json.dumps(envelope).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out_bytes)))
                self.end_headers()
                self.wfile.write(out_bytes)

        self._server = _StubThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        self.port = int(self._server.server_port)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"aw-llm-stub-{self.port}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            finally:
                self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise AssertionError("LocalLLMStubServer thread did not exit cleanly")
            self._thread = None

    def __enter__(self) -> "LocalLLMStubServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


class AdaptiveWebHarness:
    """Isolated ephemeral server + HTTP/CSRF client for opaque-box E2E tests."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        tmp_dir: tempfile.TemporaryDirectory[str] | None = None,
        keep_tmp: bool = False,
        env_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        strip_keys: bool = True,
    ) -> None:
        self.keep_tmp = keep_tmp
        self._owns_tmp = tmp_dir is None
        self.tmp_dir = tmp_dir or tempfile.TemporaryDirectory(prefix="aw_e2e_")
        self.db_path = Path(db_path) if db_path is not None else Path(self.tmp_dir.name) / "adaptive_web_e2e.sqlite3"
        self.env_path = Path(env_path) if env_path is not None else None
        self.env_overrides = dict(env_overrides or {})
        self.strip_keys = strip_keys

        self._env_patcher: Any = None
        self.settings: Any = None
        self.app: Any = None
        self.server: Any = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None
        self.base_url: str | None = None
        self.csrf_token: str | None = None
        self._stopped = False

    def start(self) -> "AdaptiveWebHarness":
        ensure_adaptive_web_on_sys_path()
        try:
            config_mod = importlib.import_module("adaptive_web.config")
            app_mod = importlib.import_module("adaptive_web.app")
            server_mod = importlib.import_module("adaptive_web.server")
        except (ImportError, ModuleNotFoundError) as exc:
            self._cleanup_on_abort()
            raise AssertionError(
                "Real 3/adaptive_web server/app not yet available; cannot fake E2E target"
            ) from exc

        Settings = getattr(config_mod, "Settings", None)
        AdaptiveLearningApp = getattr(app_mod, "AdaptiveLearningApp", None)
        serve_fn = getattr(server_mod, "serve", None) or getattr(server_mod, "create_server", None)
        if Settings is None or AdaptiveLearningApp is None or serve_fn is None:
            self._cleanup_on_abort()
            raise AssertionError(
                "Real 3/adaptive_web server/app not yet available; cannot fake E2E target"
            )

        env_updates = {k: v for k, v in os.environ.items()}
        if self.strip_keys:
            for k in LIVE_SECRET_KEYS:
                env_updates.pop(k, None)
        env_updates.setdefault("ADAPTIVE_WEB_PORT", "0")
        env_updates["ADAPTIVE_WEB_DB"] = str(self.db_path)
        env_updates.setdefault("DEVICE", "cpu")
        if "LLM_PROVIDER" not in self.env_overrides:
            env_updates.setdefault("LLM_PROVIDER", "mock")
        env_updates.setdefault("LLM_TIMEOUT_SECONDS", "5.0")
        env_updates.update(self.env_overrides)

        self._env_patcher = unittest.mock.patch.dict(os.environ, env_updates, clear=True)
        self._env_patcher.start()

        try:
            self.settings = Settings.from_env(env_path=self.env_path)
            self.app = AdaptiveLearningApp(self.settings)
            self.server = serve_fn(self.app, 0)
            self.port = int(self.server.server_port)
            self.base_url = f"http://127.0.0.1:{self.port}"
            self.thread = threading.Thread(
                target=self.server.serve_forever,
                name=f"aw-e2e-{self.port}",
                daemon=True,
            )
            self.thread.start()

            deadline = time.monotonic() + 5.0
            last_err: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    status, _, body = self.raw_request("GET", "/api/state")
                    if status == 200:
                        state = loads_strict_json(body)
                        self.csrf_token = state.get("csrf_token") or state.get("token")
                        return self
                except Exception as exc:
                    last_err = exc
                    time.sleep(0.05)
            raise AssertionError(f"Ephemeral adaptive_web server failed readiness check: {last_err}")
        except Exception:
            self.stop()
            raise

    def _cleanup_on_abort(self) -> None:
        if self._owns_tmp and not self.keep_tmp:
            self.tmp_dir.cleanup()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.server is not None:
            try:
                self.server.shutdown()
            finally:
                self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                raise AssertionError("adaptive_web server thread failed to join cleanly")
            self.thread = None
        if self.app is not None:
            for closer in (
                lambda: self.app.close() if hasattr(self.app, "close") else None,
                lambda: self.app.store.close() if hasattr(getattr(self.app, "store", None), "close") else None,
                lambda: self.app.store.db.close() if hasattr(getattr(getattr(self.app, "store", None), "db", None), "close") else None,
            ):
                try:
                    closer()
                except Exception:
                    pass
            self.app = None
        if self._env_patcher is not None:
            self._env_patcher.stop()
            self._env_patcher = None
        if self._owns_tmp and not self.keep_tmp:
            self.tmp_dir.cleanup()

    def cleanup_tmp(self) -> None:
        if self._owns_tmp:
            self.tmp_dir.cleanup()

    def __enter__(self) -> "AdaptiveWebHarness":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def raw_request(
        self,
        method: str,
        path: str,
        body: bytes | str = b"",
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
    ) -> tuple[int, dict[str, str], bytes]:
        if self.port is None:
            raise RuntimeError("AdaptiveWebHarness not started")
        payload_bytes = body.encode("utf-8") if isinstance(body, str) else body
        req_headers = {"Host": f"127.0.0.1:{self.port}", "Connection": "close"}
        if headers:
            req_headers.update(headers)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request(method, path, body=payload_bytes if method != "GET" else None, headers=req_headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, resp_headers, resp_body
        finally:
            conn.close()

    def get(self, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
        return self.raw_request("GET", path, headers=headers)

    def get_json(self, path: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        status, _, body = self.get(path, headers=headers)
        if status != 200:
            raise AssertionError(f"GET {path} expected 200, got {status}: {body.decode('utf-8', errors='replace')}")
        data = loads_strict_json(body)
        if not isinstance(data, dict):
            raise AssertionError(f"GET {path} did not return a JSON object")
        return data

    def post_raw(
        self,
        path: str,
        payload: dict[str, Any] | bytes | str,
        *,
        auto_request_id: bool = True,
        csrf: bool | str = True,
        origin: bool | str = True,
        host: str | None = None,
        content_type: str | None = "application/json",
        timeout: float = 20.0,
    ) -> tuple[int, dict[str, str], bytes]:
        if self.port is None:
            raise RuntimeError("AdaptiveWebHarness not started")
        if isinstance(payload, dict):
            data = dict(payload)
            if auto_request_id and "request_id" not in data:
                data["request_id"] = str(uuid.uuid4())
            body_bytes = json.dumps(data).encode("utf-8")
        elif isinstance(payload, str):
            body_bytes = payload.encode("utf-8")
        else:
            body_bytes = payload

        headers: dict[str, str] = {
            "Host": host if host is not None else f"127.0.0.1:{self.port}",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        if origin is True:
            headers["Origin"] = f"http://127.0.0.1:{self.port}"
        elif isinstance(origin, str):
            headers["Origin"] = origin
        if csrf is True:
            if not self.csrf_token:
                state = self.get_json("/api/state")
                self.csrf_token = state.get("csrf_token") or state.get("token")
            if self.csrf_token:
                headers["X-CSRF-Token"] = self.csrf_token
                headers["X-Dashboard-Token"] = self.csrf_token
        elif isinstance(csrf, str):
            headers["X-CSRF-Token"] = csrf
            headers["X-Dashboard-Token"] = csrf

        return self.raw_request("POST", path, body=body_bytes, headers=headers, timeout=timeout)

    def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        auto_request_id: bool = True,
        csrf: bool | str = True,
        origin: bool | str = True,
        host: str | None = None,
        content_type: str | None = "application/json",
        expect_status: int = 200,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        status, _, body = self.post_raw(
            path,
            payload,
            auto_request_id=auto_request_id,
            csrf=csrf,
            origin=origin,
            host=host,
            content_type=content_type,
            timeout=timeout,
        )
        if status != expect_status:
            raise AssertionError(
                f"POST {path} expected status {expect_status}, got {status}: {body.decode('utf-8', errors='replace')}"
            )
        parsed = loads_strict_json(body)
        if not isinstance(parsed, dict):
            raise AssertionError(f"POST {path} expected JSON object, got {type(parsed).__name__}")
        return parsed

    # ---- Domain HTTP helpers ----

    def create_profile(self, name: str = "E2E Learner") -> str:
        res = self.post("/api/profile", {"name": name})
        learner = res.get("learner") or res.get("learner_id") or (res.get("profile") or {}).get("id")
        if not learner:
            raise AssertionError(f"/api/profile missing learner ID: {res}")
        return str(learner)

    def get_state(self, learner: str | None = None, skill: str | None = None) -> dict[str, Any]:
        params: dict[str, str] = {}
        if learner is not None:
            params["learner"] = learner
        if skill is not None:
            params["skill"] = skill
        qs = f"?{urllib.parse.urlencode(params)}" if params else ""
        return self.get_json(f"/api/state{qs}")

    def get_curve(
        self,
        learner: str,
        skill: str,
        fmt: str = "output_prediction",
        task: str | None = None,
    ) -> dict[str, Any]:
        params = {"learner": learner, "skill": skill, "format": fmt}
        if task is not None:
            params["task"] = task
        return self.get_json(f"/api/curve?{urllib.parse.urlencode(params)}")

    def get_export(self, learner: str) -> dict[str, Any]:
        return self.get_json(f"/api/export?{urllib.parse.urlencode({'learner': learner})}")

    def select_skill(self, learner: str, skill: str) -> dict[str, Any]:
        return self.post("/api/select_skill", {"learner": learner, "skill": skill})

    def ppo_step(self, learner: str, skill: str, action_id: int | None = None, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"learner": learner, "skill": skill}
        if action_id is not None:
            payload["action_id"] = action_id
        payload.update(extra)
        return self.post("/api/ppo_step", payload)

    def hint(self, learner: str, interaction_id: str) -> dict[str, Any]:
        return self.post("/api/hint", {"learner": learner, "interaction": interaction_id})

    def submit(
        self,
        learner: str,
        interaction_id: str,
        answer: Any,
        *,
        active_seconds: float | None = 5.0,
        self_explanation: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "learner": learner,
            "interaction": interaction_id,
            "answer": answer,
            "active_seconds": active_seconds,
        }
        if self_explanation is not None:
            payload["self_explanation"] = self_explanation
        return self.post("/api/submit", payload)

    def retry(self, learner: str, interaction_id: str) -> dict[str, Any]:
        return self.post("/api/retry", {"learner": learner, "interaction": interaction_id})

    def reveal(self, learner: str, interaction_id: str) -> dict[str, Any]:
        return self.post("/api/reveal", {"learner": learner, "interaction": interaction_id})

    def finish(self, learner: str, interaction_id: str) -> dict[str, Any]:
        return self.post("/api/finish", {"learner": learner, "interaction": interaction_id})


# ---- Subprocess CLI helper ----

def spawn_cli_server(
    *,
    env_overrides: dict[str, str] | None = None,
    cli_args: tuple[str, ...] = (),
    cwd: str | Path = "/tmp",
    startup_timeout: float = 12.0,
) -> tuple[subprocess.Popen[str], int, str]:
    """Launch `python -m adaptive_web.server` in a clean subprocess and wait for HTTP readiness."""
    ensure_adaptive_web_on_sys_path()
    try:
        importlib.import_module("adaptive_web.server")
    except (ImportError, ModuleNotFoundError) as exc:
        raise AssertionError(
            "Real 3/adaptive_web server/app not yet available; cannot fake E2E target"
        ) from exc

    clean_env = {k: v for k, v in os.environ.items()}
    for k in LIVE_SECRET_KEYS:
        clean_env.pop(k, None)
    port = int((env_overrides or {}).get("ADAPTIVE_WEB_PORT", "0")) or find_free_port()
    clean_env["ADAPTIVE_WEB_PORT"] = str(port)
    clean_env.setdefault("DEVICE", "cpu")
    clean_env.setdefault("LLM_PROVIDER", "mock")
    clean_env["PYTHONPATH"] = f"{ADAPTIVE_WEB_DIR}:{CODE_3_DIR}"
    if env_overrides:
        clean_env.update(env_overrides)
        if clean_env.get("ADAPTIVE_WEB_PORT", "0") == "0":
            clean_env["ADAPTIVE_WEB_PORT"] = str(port)

    proc = subprocess.Popen(
        [sys.executable, "-m", "adaptive_web.server", *cli_args],
        env=clean_env,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.monotonic() + startup_timeout
    base_url = f"http://127.0.0.1:{port}"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=2.0)
            raise AssertionError(f"CLI server exited early ({proc.returncode}): stdout={out!r} stderr={err!r}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            conn.request("GET", "/api/state", headers={"Host": f"127.0.0.1:{port}"})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return proc, port, base_url
        except Exception:
            time.sleep(0.1)

    proc.terminate()
    out, err = proc.communicate(timeout=5.0)
    raise AssertionError(f"CLI server timed out waiting for HTTP 200 on {base_url}: stdout={out!r} stderr={err!r}")


# ---- Response normalization helpers ----

def extract_ppo(payload: dict[str, Any]) -> dict[str, Any]:
    if "ppo" in payload and isinstance(payload["ppo"], dict):
        return payload["ppo"]
    return payload


def extract_obs_vector(ppo: dict[str, Any]) -> list[float]:
    vec = ppo.get("obs_vector_36")
    if vec is None:
        vec = ppo.get("observation_vector")
    if vec is None:
        raise AssertionError(f"Missing obs_vector_36 / observation_vector in PPO payload: {list(ppo.keys())}")
    return [float(x) for x in vec]


def extract_valid_mask(ppo: dict[str, Any]) -> list[bool]:
    mask = ppo.get("valid_mask")
    if mask is not None:
        return [bool(x) for x in mask]
    actions = ppo.get("actions")
    if isinstance(actions, list) and len(actions) == 8:
        return [bool(a.get("valid")) for a in actions]
    raise AssertionError(f"Missing valid_mask / actions in PPO payload: {list(ppo.keys())}")


def extract_ensemble_probs(ppo: dict[str, Any]) -> list[float]:
    probs = ppo.get("ensemble_probs")
    if probs is not None:
        return [float(x) for x in probs]
    actions = ppo.get("actions")
    if isinstance(actions, list) and len(actions) == 8:
        return [float(a.get("probability", 0.0)) for a in actions]
    raise AssertionError(f"Missing ensemble_probs / actions in PPO payload: {list(ppo.keys())}")


def extract_skill_summary(state: dict[str, Any], skill: str) -> dict[str, Any]:
    summaries = state.get("skill_summaries") or state.get("skills") or {}
    if isinstance(summaries, dict) and skill in summaries:
        return summaries[skill]
    if isinstance(summaries, list):
        for item in summaries:
            if item.get("skill") == skill or item.get("skill_id") == skill:
                return item
    metrics = state.get("metrics_by_skill") or (state.get("ppo") or {}).get("metrics_by_skill") or {}
    if isinstance(metrics, dict) and skill in metrics:
        return metrics[skill]
    raise AssertionError(f"Skill {skill!r} summary not found in state keys={list(state.keys())}")


def extract_question(resp: dict[str, Any]) -> dict[str, Any]:
    q = resp.get("question")
    if isinstance(q, dict):
        return q
    if "active" in resp and isinstance(resp["active"], dict):
        act = resp["active"]
        if isinstance(act.get("question"), dict):
            return act["question"]
        if "task_id" in act or "task_type" in act:
            return act
    if "task_id" in resp and "task_type" in resp:
        return resp
    raise AssertionError(f"No question payload found in response: {list(resp.keys())}")


def extract_interaction_id(resp: dict[str, Any]) -> str:
    for key in ("interaction_id", "id"):
        if resp.get(key):
            return str(resp[key])
    for nested_key in ("interaction", "active"):
        nested = resp.get(nested_key)
        if isinstance(nested, dict):
            for key in ("interaction_id", "id"):
                if nested.get(key):
                    return str(nested[key])
    raise AssertionError(f"No interaction_id found in response: {list(resp.keys())}")


def extract_token_audit(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize MemoryAdapter.inspect_tokens() and compact token_audit schemas strictly (no vacuous None)."""
    audit = state.get("token_audit")
    if audit is None and isinstance(state.get("glass_box"), dict):
        audit = state["glass_box"].get("token_audit")
    if not isinstance(audit, dict):
        raise AssertionError(f"Missing token_audit dict in state keys={list(state.keys())}")

    token_dim = audit.get("token_dim")
    if token_dim is None:
        token_dim = audit.get("expected_token_dim")
    if token_dim is None:
        token_dim = audit.get("last_vector_dim")
    if token_dim is None:
        raise AssertionError(f"Missing token_dim / expected_token_dim in token_audit: {list(audit.keys())}")

    task_dim = audit.get("task_feature_dim")
    if task_dim is None:
        task_dim = audit.get("expected_task_dim")
    if task_dim is None:
        raise AssertionError(f"Missing task_feature_dim / expected_task_dim in token_audit: {list(audit.keys())}")

    field_audit = audit.get("field_audit")
    if "uses_text_or_code_embeddings" in audit and audit["uses_text_or_code_embeddings"] is not None:
        uses_emb = bool(audit["uses_text_or_code_embeddings"])
    elif isinstance(field_audit, dict) and "zero_text_code_id_embeddings" in field_audit:
        uses_emb = not bool(field_audit["zero_text_code_id_embeddings"])
    else:
        raise AssertionError(
            "Missing explicit uses_text_or_code_embeddings or field_audit.zero_text_code_id_embeddings "
            f"in token_audit: {list(audit.keys())}"
        )

    return {
        "token_dim": int(token_dim),
        "task_feature_dim": int(task_dim),
        "uses_text_or_code_embeddings": uses_emb,
        "projected_token_count": audit.get("projected_token_count"),
        "raw_event_count": audit.get("raw_event_count"),
        "field_audit": field_audit,
        "raw": audit,
    }


def extract_memory_task(task_record: dict[str, Any]) -> dict[str, Any]:
    """Return exact 8-key memory_task metadata from an exported generated task or project via catalogue_bridge."""
    if isinstance(task_record.get("memory_task"), dict):
        return dict(task_record["memory_task"])
    ensure_adaptive_web_on_sys_path()
    bridge = importlib.import_module("adaptive_web.catalogue_bridge")
    return bridge.task_memory_features(task_record)
