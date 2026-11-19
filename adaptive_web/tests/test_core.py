"""Unit and contract tests for M1_core_engine (3/adaptive_web/)."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import types
import unittest

CODE_3_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_3_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_3_ROOT))

from adaptive_web import Settings  # noqa: E402
from adaptive_web import catalogue_bridge, grader  # noqa: E402
from adaptive_web.memory_adapter import CURVE_DAYS, MemoryAdapter  # noqa: E402
from adaptive_web.ppo_engine import PPOEngine  # noqa: E402
from adaptive_web.store import AdaptiveStore  # noqa: E402


def _make_synthetic_llm_task(fmt: str, skill: str = "range_boundaries", difficulty: str = "medium", task_id: str | None = None) -> dict:
    tid = task_id or f"llm.{skill}.{difficulty}.{fmt}.test01"
    base = {
        "task_id": tid,
        "skill_id": skill,
        "task_type": fmt,
        "difficulty": difficulty,
        "pool": "practice",
        "prompt": f"Synthetic {fmt} question for {skill}.",
        "snippets": [],
        "options": [],
        "answer_key": None,
        "reference_solution": None,
        "tests": [],
        "reference_seconds": 40.0,
        "hints": [
            {"level": 1, "kind": "conceptual", "text": "Check start and stop."},
            {"level": 2, "kind": "targeted", "text": "Stop is excluded."},
            {"level": 3, "kind": "scaffold", "text": "Trace elements one by one."},
        ],
    }
    if fmt == "single_choice":
        base["snippets"] = ["print(list(range(1, 5)))"]
        base["options"] = [
            {"id": "o0", "text": "[1, 2, 3, 4]"},
            {"id": "o1", "text": "[1, 2, 3, 4, 5]"},
            {"id": "o2", "text": "[2, 3, 4]"},
            {"id": "o3", "text": "[0, 1, 2, 3]"},
        ]
        base["answer_key"] = "o0"
        base["criteria"] = [{"id": "c0", "weight": 1.0, "required": True, "kind": "exact_match"}]
    elif fmt == "multiple_select":
        base["options"] = [
            {"id": "o0", "text": "list(range(1, 4)) == [1, 2, 3]"},
            {"id": "o1", "text": "list(range(1, 4)) == [1, 2, 3, 4]"},
            {"id": "o2", "text": "list(range(0, 3)) == [0, 1, 2]"},
            {"id": "o3", "text": "list(range(2, 2)) == [2]"},
        ]
        base["answer_key"] = ["o0", "o2"]
        base["criteria"] = [
            {"id": "c0", "weight": 0.5, "required": True, "kind": "correct_selection"},
            {"id": "c1", "weight": 0.5, "required": True, "kind": "incorrect_selection_penalty"},
            {"id": "c2", "weight": 0.5, "required": True, "kind": "correct_selection"},
            {"id": "c3", "weight": 0.5, "required": True, "kind": "incorrect_selection_penalty"},
        ]
    elif fmt == "output_prediction":
        base["snippets"] = ["print(list(range(2, 7, 2)))", "print(len(range(2, 7, 2)))"]
        base["answer_key"] = ["[2, 4, 6]", "3"]
        base["criteria"] = [
            {"id": "c0", "weight": 0.5, "required": True, "kind": "exact_match"},
            {"id": "c1", "weight": 0.5, "required": True, "kind": "exact_match"},
        ]
    elif fmt == "fill_in_the_blank":
        base["snippets"] = ["print(list(range(__BLANK_1__, __BLANK_2__, 2)))"]
        base["answer_key"] = ["2", "8"]
        base["criteria"] = [
            {"id": "c0", "weight": 0.5, "required": True, "kind": "exact_match"},
            {"id": "c1", "weight": 0.5, "required": True, "kind": "exact_match"},
        ]
    elif fmt == "code_completion":
        base["snippets"] = ["def solve(x):\n    pass"]
        base["reference_solution"] = "def solve(x):\n    return list(range(1, x, 2))"
        base["tests"] = [
            {"input": 6, "expected": [1, 3, 5]},
            {"input": 1, "expected": []},
            {"input": 7, "expected": [1, 3, 5]},
        ]
        base["criteria"] = [
            {"id": "c0", "weight": 1.0 / 3.0, "required": True, "kind": "behavior_test"},
            {"id": "c1", "weight": 1.0 / 3.0, "required": True, "kind": "behavior_test"},
            {"id": "c2", "weight": 1.0 / 3.0, "required": True, "kind": "behavior_test"},
        ]
    return base


def _append_resolved_attempt(
    store: AdaptiveStore,
    *,
    learner: str,
    skill: str,
    task: dict,
    at: float,
    score: float = 1.0,
    full_success: bool = True,
    hints: list[dict] | None = None,
    revealed: bool = False,
    ppo_action: str = "give_same_difficulty_question",
    interaction_id: str | None = None,
    duration: float = 22.0,
) -> dict:
    iid = interaction_id or f"int-{learner}-{skill}-{int(at * 1000)}"
    prior = store.events(learner, skill)
    hints_list = list(hints or [])
    assistance = "assisted" if (hints_list or revealed) else "independent"
    task_feats = catalogue_bridge.task_memory_features(task)

    def _base_fields(ev_type: str, ts: float, gap_prev: float | None) -> dict:
        return {
            "event_type": ev_type,
            "gap_seconds": gap_prev,
            "new_session": None if gap_prev is None else False,
            "task": task_feats,
            "attempt_number": None,
            "active_response_seconds": None,
            "relative_response_time": None,
            "hint_count": None,
            "hint_type": None,
            "answer_revealed": None,
            "assistance": None,
            "score": None,
            "full_success": None,
            "grading_status": None,
        }

    t_cur = at
    for idx, h in enumerate(hints_list, start=1):
        gap = None if not prior else max(0.0, t_cur - prior[-1]["at"])
        fields = _base_fields("hint", t_cur, gap)
        fields.update(
            hint_count=idx,
            hint_type=h.get("kind", "conceptual"),
            assistance="assisted",
        )
        ev = store.append_event(learner, skill, iid, t_cur, "hint", {"features": fields})
        prior.append(ev)
        t_cur += 4.0

    gap_resp = None if not prior else max(0.0, t_cur - prior[-1]["at"])
    resp_fields = _base_fields("response", t_cur, gap_resp)
    resp_fields.update(
        attempt_number=1,
        hint_count=len(hints_list),
        answer_revealed=revealed,
        assistance=assistance,
        active_response_seconds=duration,
        relative_response_time=duration / task["reference_seconds"],
    )
    ev_r = store.append_event(learner, skill, iid, t_cur, "response", {"features": resp_fields})
    prior.append(ev_r)

    t_grade = t_cur + 1.0
    grade_fields = _base_fields("grade", t_grade, 1.0)
    grade_fields.update(
        attempt_number=1,
        hint_count=len(hints_list),
        answer_revealed=revealed,
        assistance=assistance,
        score=float(score),
        full_success=bool(full_success),
        grading_status="resolved",
    )
    store.append_event(learner, skill, iid, t_grade, "grade", {"features": grade_fields})

    item = {
        "id": iid,
        "learner": learner,
        "skill": skill,
        "task": task["task_id"],
        "difficulty": task["difficulty"],
        "ppo_action": ppo_action,
        "hints": hints_list,
        "revealed": revealed,
        "results": [
            {
                "score": float(score),
                "full_success": bool(full_success),
                "grading_status": "resolved",
            }
        ],
        "created": at,
    }
    store.save(item)
    return item


class ConfigTests(unittest.TestCase):
    def test_defaults_without_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_env = Path(tmp) / ".env"
            saved_env = {
                k: os.environ.pop(k)
                for k in (
                    "ADAPTIVE_WEB_PORT",
                    "ADAPTIVE_WEB_DB",
                    "LLM_PROVIDER",
                    "LLM_API_KEY",
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GEMINI_API_KEY",
                    "LLM_MODEL",
                    "LLM_BASE_URL",
                    "LLM_TIMEOUT_SECONDS",
                    "DEVICE",
                )
                if k in os.environ
            }
            try:
                cfg = Settings.from_env(missing_env)
                self.assertEqual(cfg.adaptive_web_port, 8780)
                self.assertEqual(cfg.llm_timeout_seconds, 45.0)
                self.assertEqual(cfg.device, "cpu")
                self.assertEqual(cfg.effective_provider, "mock")
                self.assertFalse(cfg.is_live())
                self.assertTrue(str(cfg.adaptive_web_db).endswith("3/adaptive_web/data/adaptive_web.sqlite3"))
            finally:
                os.environ.update(saved_env)

    def test_dotenv_parsing_and_os_environ_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "# comment line\n"
                "export ADAPTIVE_WEB_PORT=8899 # inline comment\n"
                'ADAPTIVE_WEB_DB="data/custom_test.sqlite3"\n'
                "LLM_PROVIDER='anthropic'\n"
                "ANTHROPIC_API_KEY=\"sk-ant-secret-dotenv-12345\"\n"
                "LLM_TIMEOUT_SECONDS=60.5\n",
                encoding="utf-8",
            )
            saved_port = os.environ.get("ADAPTIVE_WEB_PORT")
            saved_model = os.environ.get("LLM_MODEL")
            os.environ["ADAPTIVE_WEB_PORT"] = "9001"
            os.environ["LLM_MODEL"] = "claude-3-7-sonnet"
            try:
                cfg = Settings.from_env(env_file)
                self.assertEqual(cfg.adaptive_web_port, 9001)
                self.assertEqual(cfg.effective_provider, "anthropic")
                self.assertEqual(cfg.llm_model, "claude-3-7-sonnet")
                self.assertEqual(cfg.llm_timeout_seconds, 60.5)
                self.assertTrue(cfg.is_live())
                # Secret must never appear in repr, str, or to_public_dict
                secret = "sk-ant-secret-dotenv-12345"
                self.assertNotIn(secret, repr(cfg))
                self.assertNotIn(secret, str(cfg))
                pub = cfg.to_public_dict()
                self.assertNotIn(secret, json.dumps(pub))
                self.assertTrue(pub["api_key_configured"])
                self.assertTrue(pub["is_live"])
            finally:
                if saved_port is None:
                    os.environ.pop("ADAPTIVE_WEB_PORT", None)
                else:
                    os.environ["ADAPTIVE_WEB_PORT"] = saved_port
                if saved_model is None:
                    os.environ.pop("LLM_MODEL", None)
                else:
                    os.environ["LLM_MODEL"] = saved_model

    def test_provider_key_fallback_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("GEMINI_API_KEY=gem-secret-999\n", encoding="utf-8")
            saved = {k: os.environ.pop(k) for k in ("LLM_PROVIDER", "LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY") if k in os.environ}
            try:
                cfg = Settings.from_env(env_file)
                self.assertEqual(cfg.effective_provider, "gemini")
                self.assertTrue(cfg.is_live())
                self.assertNotIn("gem-secret-999", json.dumps(cfg.to_public_dict()))
                bad_env = Path(tmp) / "bad.env"
                bad_env.write_text("ADAPTIVE_WEB_PORT=99999\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    Settings.from_env(bad_env)
            finally:
                os.environ.update(saved)


class StoreTests(unittest.TestCase):
    def test_wal_mode_tables_and_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite3"
            store = AdaptiveStore(db_path)
            try:
                mode = store.db.execute("PRAGMA journal_mode;").fetchone()[0]
                self.assertEqual(mode.lower(), "wal")
                tables = {
                    r[0]
                    for r in store.db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                expected = {
                    "profiles",
                    "interactions",
                    "events",
                    "requests",
                    "generated_tasks",
                    "learning_paths",
                    "course_materials",
                    "tutor_turns",
                }
                self.assertTrue(expected <= tables)
                p1 = store.create_profile("  Alice   Learner ")
                self.assertEqual(p1["name"], "Alice Learner")
                self.assertEqual(store.require_profile(p1["id"])["name"], "Alice Learner")
                self.assertEqual(len(store.list_profiles()), 1)
                with self.assertRaises(ValueError):
                    store.require_profile("missing-id")
            finally:
                store.close()

    def test_get_task_v22_and_generated_tasks_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "store.sqlite3")
            try:
                v22 = store.get_task("range_boundaries.easy.single_choice.00")
                self.assertIsNotNone(v22)
                self.assertEqual(v22["skill_id"], "range_boundaries")

                llm_task = _make_synthetic_llm_task("code_completion", skill="loop_tracing", difficulty="medium")
                store.save_generated_task(llm_task, verify_code_tests=True)
                loaded = store.get_task(llm_task["task_id"])
                self.assertEqual(loaded["task_id"], llm_task["task_id"])
                self.assertEqual(loaded["tests"], llm_task["tests"])

                # Learner & skill isolation across paths, materials, turns, events
                store.create_profile("Learner A", profile_id="lA")
                store.create_profile("Learner B", profile_id="lB")
                store.save_learning_path("lA", "loop_tracing", {"steps": ["step1"]})
                store.save_course_material("lA", "loop_tracing", {"title": "Loop Guide"})
                store.save_tutor_turn("lA", "loop_tracing", "delayed_retention_ppo", {"action": "explain_concept"})

                self.assertIsNotNone(store.get_latest_learning_path("lA", "loop_tracing"))
                self.assertIsNone(store.get_latest_learning_path("lB", "loop_tracing"))
                self.assertIsNone(store.get_latest_learning_path("lA", "range_boundaries"))
                self.assertIsNotNone(store.get_latest_course_material("lA", "loop_tracing"))
                self.assertIsNone(store.get_latest_course_material("lB", "loop_tracing"))
                self.assertEqual(len(store.get_tutor_turns("lA", "loop_tracing")), 1)
                self.assertEqual(len(store.get_tutor_turns("lB", "loop_tracing")), 0)
            finally:
                store.close()

    def test_idempotency_nested_rollback_and_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "concurrent.sqlite3")
            try:
                calls = {"count": 0}

                def _op():
                    calls["count"] += 1
                    store.create_profile("Idem User", profile_id="idem1")
                    return {"ok": True, "calls": calls["count"]}

                r1 = store.run_idempotent("req-idempotent-01", _op)
                r2 = store.run_idempotent("req-idempotent-01", _op)
                self.assertEqual(r1, r2)
                self.assertEqual(calls["count"], 1)

                # Exception inside transaction rolls back cleanly
                with self.assertRaises(RuntimeError):
                    with store.transaction():
                        store.create_profile("Should Rollback", profile_id="rb1")
                        raise RuntimeError("abort")
                self.assertIsNone(store.get_profile("rb1"))

                # Multi-thread concurrent reads & writes
                errors: list[Exception] = []

                def _worker(tid: int):
                    try:
                        for i in range(10):
                            pid = f"t{tid}_{i}"
                            store.create_profile(f"Worker {tid} {i}", profile_id=pid)
                            store.save_tutor_turn(pid, "range_boundaries", "ppo", {"step": i})
                            _ = store.list_profiles()
                            _ = store.get_tutor_turns(pid)
                    except Exception as exc:
                        errors.append(exc)

                threads = [threading.Thread(target=_worker, args=(k,)) for k in range(6)]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                self.assertEqual(errors, [])
            finally:
                store.close()


class CatalogueBridgeTests(unittest.TestCase):
    def test_v22_2160_items_and_ppo_skill_order(self):
        self.assertEqual(catalogue_bridge.v22_task_count(), 2160)
        from ppo.env import PREREQUISITES as ENV_PREREQS, SKILLS_ORDER as ENV_SKILLS

        self.assertEqual(catalogue_bridge.PPO_SKILLS_ORDER, tuple(ENV_SKILLS))
        self.assertEqual(catalogue_bridge.PPO_SKILLS_ORDER[4], "list_indexing")
        self.assertEqual(catalogue_bridge.PPO_SKILLS_ORDER[5], "function_returns")
        self.assertEqual(catalogue_bridge.PREREQUISITES, ENV_PREREQS)

        # Deep copy protection
        t1 = catalogue_bridge.get_v22_task("range_boundaries.easy.single_choice.00")
        t1["prompt"] = "MUTATED"
        t2 = catalogue_bridge.get_v22_task("range_boundaries.easy.single_choice.00")
        self.assertNotEqual(t2["prompt"], "MUTATED")

    def test_8_key_memory_features_match_dashboard_across_all_2160(self):
        spec = importlib.util.spec_from_file_location(
            "test_ref_dashboard_catalogue",
            CODE_3_ROOT / "dashboard" / "catalogue.py",
        )
        dash_cat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash_cat)

        all_tasks = catalogue_bridge.list_v22_tasks()
        self.assertEqual(len(all_tasks), 2160)
        for task in all_tasks:
            ok, errs = catalogue_bridge.validate_task_schema(task)
            self.assertTrue(ok, f"{task['task_id']} failed schema validation: {errs}")
            f_new = catalogue_bridge.task_memory_features(task)
            f_dash = dash_cat.features(task)
            self.assertEqual(set(f_new.keys()), {
                "difficulty",
                "task_type",
                "reference_seconds",
                "snippet_count",
                "option_count",
                "criterion_count",
                "credit_weights",
                "penalty_weights",
            })
            self.assertEqual(f_new, f_dash)

    def test_validate_task_schema_valid_and_invalid_all_5_formats(self):
        for fmt in catalogue_bridge.FORMATS:
            t = _make_synthetic_llm_task(fmt)
            ok, errs = catalogue_bridge.validate_task_schema(t, verify_code_tests=(fmt == "code_completion"))
            self.assertTrue(ok, f"{fmt} failed: {errs}")

        # Malformed cases
        bad_weight = _make_synthetic_llm_task("single_choice")
        bad_weight["criteria"][0]["weight"] = 0.5
        ok, _ = catalogue_bridge.validate_task_schema(bad_weight)
        self.assertFalse(ok)

        bad_blank = _make_synthetic_llm_task("fill_in_the_blank")
        bad_blank["answer_key"] = ["not_an_int", "2"]
        ok, _ = catalogue_bridge.validate_task_schema(bad_blank)
        self.assertFalse(ok)

        bad_code = _make_synthetic_llm_task("code_completion")
        bad_code["reference_solution"] = "def solve(x):\n    return []"
        ok, _ = catalogue_bridge.validate_task_schema(bad_code, verify_code_tests=True)
        self.assertFalse(ok)


class MemoryAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = MemoryAdapter(device="cpu")

    def test_checkpoint_contract_and_no_dashboard_import(self):
        self.assertEqual(
            self.memory.version,
            "5ae56a794af5b531eb10e36974c5be3d5f20b1b4d1166ebd71dbe4825d362912",
        )
        self.assertFalse(self.memory.model.training)
        for param in self.memory.model.parameters():
            self.assertFalse(param.requires_grad)
        dashboard_dir = str(CODE_3_ROOT / "dashboard")
        self.assertNotIn(dashboard_dir, sys.path)

    def test_cold_start_unassessed_and_observed_vs_predicted_separation(self):
        summary = self.memory.skill_summary("learner_cold", "range_boundaries", [], now=1000.0)
        self.assertFalse(summary["assessed"])
        self.assertEqual(summary["observed_attempts"], 0)
        self.assertEqual(summary["observed_unaided_attempts"], 0)
        self.assertEqual(summary["observed_unaided_passes"], 0)
        self.assertIsNone(summary["observed_pass_rate"])
        self.assertGreater(summary["p_easy"], summary["p_med"])
        self.assertGreater(summary["p_med"], summary["p_hard"])
        self.assertGreater(summary["floor_b"], 0.0)
        self.assertGreater(summary["half_life_days"], 0.0)

        # Instruction-only event still leaves assessed=False
        instr_events = [
            {
                "id": 1,
                "at": 1000.0,
                "kind": "instruction",
                "skill": "range_boundaries",
                "learner": "learner_cold",
                "features": {
                    "event_type": "instruction",
                    "gap_seconds": None,
                    "new_session": None,
                    "task": None,
                },
            }
        ]
        summary_instr = self.memory.skill_summary("learner_cold", "range_boundaries", instr_events, now=1010.0)
        self.assertFalse(summary_instr["assessed"])
        self.assertIsNone(summary_instr["observed_pass_rate"])

    def test_learner_skill_isolation_and_rollback_cache_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "mem.sqlite3")
            try:
                store.create_profile("A", profile_id="lA")
                store.create_profile("B", profile_id="lB")
                t_rb = catalogue_bridge.get_v22_task("range_boundaries.medium.output_prediction.00")
                _append_resolved_attempt(
                    store, learner="lA", skill="range_boundaries", task=t_rb, at=1000.0, score=1.0, full_success=True
                )

                ev_a_rb = store.events("lA", "range_boundaries")
                ev_a_lt = store.events("lA", "loop_tracing")
                ev_b_rb = store.events("lB", "range_boundaries")

                s_a_rb = self.memory.skill_summary("lA", "range_boundaries", ev_a_rb, now=1100.0)
                s_a_lt = self.memory.skill_summary("lA", "loop_tracing", ev_a_lt, now=1100.0)
                s_b_rb = self.memory.skill_summary("lB", "range_boundaries", ev_b_rb, now=1100.0)

                self.assertTrue(s_a_rb["assessed"])
                self.assertEqual(s_a_rb["observed_pass_rate"], 1.0)
                self.assertFalse(s_a_lt["assessed"])
                self.assertFalse(s_b_rb["assessed"])
                self.assertGreater(s_a_rb["p_med"], s_b_rb["p_med"])

                # Cross-learner / cross-skill injection raises ValueError
                with self.assertRaises(ValueError):
                    self.memory.skill_summary("lB", "range_boundaries", ev_a_rb, now=1100.0)
                with self.assertRaises(ValueError):
                    self.memory.skill_summary("lA", "loop_tracing", ev_a_rb, now=1100.0)

                # Rollback / truncation cache safety: adding a second attempt then truncating back to first
                _append_resolved_attempt(
                    store, learner="lA", skill="range_boundaries", task=t_rb, at=1200.0, score=0.0, full_success=False
                )
                ev_two = store.events("lA", "range_boundaries")
                _ = self.memory.skill_summary("lA", "range_boundaries", ev_two, now=1300.0)
                s_truncated = self.memory.skill_summary("lA", "range_boundaries", ev_a_rb, now=1100.0)
                self.assertAlmostEqual(s_truncated["p_med"], s_a_rb["p_med"], places=6)
            finally:
                store.close()

    def test_assisted_only_has_none_unaided_pass_rate_and_curves_monotonic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "curves.sqlite3")
            try:
                store.create_profile("C", profile_id="lC")
                t_rb = catalogue_bridge.get_v22_task("range_boundaries.hard.output_prediction.00")
                _append_resolved_attempt(
                    store,
                    learner="lC",
                    skill="range_boundaries",
                    task=t_rb,
                    at=1000.0,
                    score=1.0,
                    full_success=True,
                    hints=[{"level": 1, "kind": "conceptual", "text": "h1"}],
                )
                ev = store.events("lC", "range_boundaries")
                s = self.memory.skill_summary("lC", "range_boundaries", ev, now=1100.0)
                self.assertTrue(s["assessed"])
                self.assertEqual(s["observed_attempts"], 1)
                self.assertEqual(s["observed_unaided_attempts"], 0)
                self.assertIsNone(s["observed_pass_rate"])

                for fmt in catalogue_bridge.FORMATS:
                    curve = self.memory.retention_curve("lC", "range_boundaries", ev, task_format=fmt, now=1100.0)
                    self.assertEqual(curve["days"], list(CURVE_DAYS))
                    self.assertTrue(curve["assessed"])
                    self.assertIn("floor_b", curve)
                    self.assertIn("cold_start_reference", curve)
                    for diff in ("easy", "medium", "hard"):
                        pts = [p["probability"] for p in curve["curves_by_difficulty"][diff]]
                        for i in range(len(pts) - 1):
                            self.assertLessEqual(pts[i + 1], pts[i] + 1e-7)
                    for i in range(len(CURVE_DAYS)):
                        pe = curve["curves_by_difficulty"]["easy"][i]["probability"]
                        pm = curve["curves_by_difficulty"]["medium"][i]["probability"]
                        ph = curve["curves_by_difficulty"]["hard"][i]["probability"]
                        self.assertGreater(pe, pm)
                        self.assertGreater(pm, ph)
            finally:
                store.close()

    def test_inspect_tokens_47_dim_zero_text_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "tokens.sqlite3")
            try:
                t1 = _make_synthetic_llm_task("output_prediction", task_id="llm.op.1")
                t2 = _make_synthetic_llm_task("output_prediction", task_id="llm.op.2")
                t2["prompt"] = "Completely different wording, emojis, and code comments!"
                t2["snippets"] = ["print(999999)", "print('abc')"]
                t2["answer_key"] = ["999999", "abc"]

                _append_resolved_attempt(store, learner="lT1", skill="range_boundaries", task=t1, at=1000.0)
                _append_resolved_attempt(store, learner="lT2", skill="range_boundaries", task=t2, at=1000.0)
                ev1 = store.events("lT1", "range_boundaries")
                ev2 = store.events("lT2", "range_boundaries")
                # Inject forbidden text/code/ID keys into raw events
                for e in ev2:
                    e["prompt"] = "secret prompt text"
                    e["features"]["code_embedding"] = [1.0, 2.0, 3.0]
                    e["features"]["task"]["prompt"] = "forbidden"

                audit1 = self.memory.inspect_tokens(ev1)
                audit2 = self.memory.inspect_tokens(ev2)
                self.assertEqual(audit1["expected_token_dim"], 47)
                self.assertEqual(audit1["last_vector_dim"], 47)
                self.assertEqual(audit1["projected_token_count"], 1)
                self.assertTrue(audit2["field_audit"]["zero_text_code_id_embeddings"])
                self.assertIn("event.prompt", audit2["field_audit"]["excluded_raw_keys_seen"])
                self.assertIn("features.code_embedding", audit2["field_audit"]["excluded_raw_keys_seen"])
                self.assertIn("task.prompt", audit2["field_audit"]["excluded_raw_keys_seen"])
                self.assertEqual(audit1["tokens"][0]["vector"], audit2["tokens"][0]["vector"])
            finally:
                store.close()

    def test_equivalence_with_dashboard_memory_service(self):
        # Load 3/dashboard/memory_service.py cleanly in isolation without polluting sys.path permanently
        dash_ms_path = CODE_3_ROOT / "dashboard" / "memory_service.py"
        spec = importlib.util.spec_from_file_location("test_ref_dashboard_memory_service", dash_ms_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        dash_ms = mod.MemoryService(device="cpu")

        test_gz = CODE_3_ROOT / "data_Set" / "v2.2" / "dataset" / "main" / "model_data" / "test.jsonl.gz"
        with gzip.open(test_gz, "rt", encoding="utf-8") as fh:
            record = json.loads(fh.readline())

        checked = 0
        for query in record["queries"]:
            if not query.get("target"):
                continue
            prefix = record["events"][: query["history_end"]]
            events = []
            at = 0.0
            for idx, feat in enumerate(prefix, start=1):
                at += feat["gap_seconds"] or 0.0
                events.append(
                    {
                        "id": idx,
                        "at": at,
                        "kind": feat["event_type"],
                        "skill": record["skill_id"],
                        "features": feat,
                    }
                )
            now = at + (query["input"]["since_last_event_seconds"] or 0.0)
            target_task = query["input"]["target_task"]
            p_dash = dash_ms.predict(events, target_task, now)
            p_adapt = self.memory.predict(events, target_task, now)
            self.assertAlmostEqual(p_adapt, p_dash, places=6)
            checked += 1
        self.assertGreater(checked, 10)


class PPOEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = MemoryAdapter(device="cpu")
        cls.ppo = PPOEngine(cls.memory)

    def test_obs_vector_36_bounds_groups_and_skill_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "ppo1.sqlite3")
            try:
                store.create_profile("P1", profile_id="p1")
                dec_li = self.ppo.decide(store, "p1", focus_skill="list_indexing", now=1000.0)
                dec_fr = self.ppo.decide(store, "p1", focus_skill="function_returns", now=1000.0)

                vec_li = dec_li["obs_vector_36"]
                vec_fr = dec_fr["obs_vector_36"]
                self.assertEqual(len(vec_li), 36)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in vec_li))
                self.assertEqual(vec_li[34], 1.0)
                self.assertEqual(vec_li[35], 0.0)
                self.assertEqual(vec_fr[34], 0.0)
                self.assertEqual(vec_fr[35], 1.0)

                groups = dec_li["obs_groups"]
                self.assertEqual(sum(len(g["features"]) for g in groups), 36)
                self.assertEqual(groups[0]["start"], 0)
                self.assertEqual(groups[-1]["end"], 36)
            finally:
                store.close()

    def test_gated_option_b_masks_and_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "ppo_mask.sqlite3")
            try:
                store.create_profile("MaskUser", profile_id="mu")
                # Cold start at range_boundaries / easy: actions 1,2,3,4 valid; 0,5,6,7 masked
                dec0 = self.ppo.decide(store, "mu", focus_skill="range_boundaries", now=1000.0)
                self.assertEqual(dec0["valid_mask"], [False, True, True, True, True, False, False, False])
                for masked_id in (0, 5, 6, 7):
                    self.assertIn(masked_id, dec0["mask_reasons"])
                    self.assertTrue(dec0["mask_reasons"][masked_id])

                # Emit instruction -> awaiting_question=True -> only question actions valid (1,2 at easy)
                store.append_event(
                    "mu",
                    "range_boundaries",
                    "lesson-1",
                    1010.0,
                    "instruction",
                    {
                        "features": {
                            "event_type": "instruction",
                            "gap_seconds": None,
                            "new_session": None,
                            "task": None,
                        },
                        "ppo_action": "explain_concept",
                        "primed_mode": "concept",
                    },
                )
                dec_wait = self.ppo.decide(store, "mu", focus_skill="range_boundaries", now=1020.0)
                self.assertTrue(dec_wait["awaiting_question"])
                self.assertEqual(dec_wait["primed_mode"], "concept")
                self.assertEqual(dec_wait["valid_mask"], [False, True, True, False, False, False, False, False])
                self.assertIn("Gated Option B", dec_wait["mask_reasons"][3])

                # Complete a question -> unlocks self_explanation (5) and advance (7)
                t_rb = catalogue_bridge.get_v22_task("range_boundaries.medium.output_prediction.00")
                _append_resolved_attempt(
                    store, learner="mu", skill="range_boundaries", task=t_rb, at=1030.0, score=1.0, full_success=True
                )
                dec_post = self.ppo.decide(store, "mu", focus_skill="range_boundaries", now=1060.0)
                self.assertFalse(dec_post["awaiting_question"])
                self.assertTrue(dec_post["valid_mask"][0])  # now at medium -> easier valid
                self.assertTrue(dec_post["valid_mask"][5])  # self-explanation unlocked
                self.assertTrue(dec_post["valid_mask"][7])  # advance unlocked

                # Complete request_self_explanation -> re-locks Action 5
                _append_resolved_attempt(
                    store,
                    learner="mu",
                    skill="range_boundaries",
                    task=t_rb,
                    at=1090.0,
                    score=1.0,
                    full_success=True,
                    ppo_action="request_self_explanation",
                )
                dec_after_se = self.ppo.decide(store, "mu", focus_skill="range_boundaries", now=1120.0)
                self.assertFalse(dec_after_se["valid_mask"][5])

                # Loop tracing has range_boundaries as prerequisite; if weak, Action 6 unlocks
                dec_lt = self.ppo.decide(store, "mu", focus_skill="loop_tracing", now=1130.0)
                # At cold start/low floor_b on range_boundaries or loop_tracing, check Action 6 validity matches metrics
                rb_m = dec_lt["metrics_by_skill"]["range_boundaries"]
                expected_prereq_weak = rb_m["p_30d"] < 0.56 or rb_m["floor_b"] < 0.40
                self.assertEqual(dec_lt["valid_mask"][6], expected_prereq_weak)
            finally:
                store.close()

    def test_generalized_reconstruct_context_v22_and_generated_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "ppo_gen.sqlite3")
            try:
                store.create_profile("GenUser", profile_id="gu")
                t_hard_v22 = catalogue_bridge.get_v22_task("range_boundaries.hard.output_prediction.00")
                _append_resolved_attempt(
                    store,
                    learner="gu",
                    skill="range_boundaries",
                    task=t_hard_v22,
                    at=1000.0,
                    score=1.0,
                    full_success=True,
                    hints=[{"level": 1, "kind": "targeted", "text": "h", "leakage_l_t": 0.42}],
                )
                dec1 = self.ppo.decide(store, "gu", focus_skill="range_boundaries", now=1050.0)
                self.assertEqual(dec1["focus_difficulty"], "hard")
                self.assertTrue(dec1["unverified_assisted"])
                self.assertAlmostEqual(dec1["obs_vector_36"][11], 0.42, places=5)

                # Now save an LLM-generated medium task on loop_tracing and complete it
                llm_task = _make_synthetic_llm_task("code_completion", skill="loop_tracing", difficulty="medium")
                store.save_generated_task(llm_task, verify_code_tests=True)
                _append_resolved_attempt(
                    store,
                    learner="gu",
                    skill="loop_tracing",
                    task=llm_task,
                    at=1100.0,
                    score=1.0,
                    full_success=True,
                    hints=[{"level": 1, "kind": "conceptual", "text": "h", "leakage_l_t": 0.31}],
                )
                dec2 = self.ppo.decide(store, "gu", focus_skill="loop_tracing", now=1150.0)
                self.assertEqual(dec2["focus_difficulty"], "medium")
                self.assertEqual(dec2["last_difficulty_by_skill"]["loop_tracing"], "medium")
                self.assertEqual(dec2["last_difficulty_by_skill"]["range_boundaries"], "hard")
                self.assertAlmostEqual(dec2["obs_vector_36"][11], 0.31, places=5)
                self.assertTrue(dec2["can_self_explain"])
                self.assertGreater(dec2["budget_used"], 1.5)
            finally:
                store.close()

    def test_3_seed_ensemble_probs_critic_and_rationales(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdaptiveStore(Path(tmp) / "ppo_ens.sqlite3")
            try:
                store.create_profile("EnsUser", profile_id="eu")
                dec = self.ppo.decide(store, "eu", focus_skill="range_boundaries", now=1000.0)
                self.assertEqual(len(dec["per_seed_probs"]), 3)
                self.assertEqual(len(dec["ensemble_probs"]), 8)
                self.assertAlmostEqual(sum(dec["ensemble_probs"]), 1.0, places=5)
                for aid, is_valid in enumerate(dec["valid_mask"]):
                    if not is_valid:
                        self.assertEqual(dec["ensemble_probs"][aid], 0.0)
                best_id = max(range(8), key=lambda i: dec["ensemble_probs"][i])
                self.assertEqual(dec["action_id"], best_id)
                self.assertIsInstance(dec["critic_value"], float)
                self.assertTrue(dec["rationale_plain"].strip())
                self.assertTrue(dec["rationale_technical"].strip())
                self.assertNotEqual(dec["rationale_plain"], dec["rationale_technical"])
            finally:
                store.close()


class GraderTests(unittest.TestCase):
    def test_all_5_formats_deterministic(self):
        # single_choice
        t_sc = _make_synthetic_llm_task("single_choice")
        res_sc = grader.grade_submission(t_sc, "o0")
        self.assertEqual(res_sc["grading_status"], "resolved")
        self.assertEqual(res_sc["score"], 1.0)
        self.assertTrue(res_sc["full_success"])
        self.assertEqual(len(res_sc["criteria_results"]), 1)
        self.assertIn("Passed", res_sc["feedback_summary"])

        # multiple_select partial credit and penalty
        t_ms = _make_synthetic_llm_task("multiple_select")
        res_ms_full = grader.grade_submission(t_ms, ["o0", "o2"])
        self.assertEqual(res_ms_full["score"], 1.0)
        self.assertTrue(res_ms_full["full_success"])
        res_ms_partial = grader.grade_submission(t_ms, ["o0"])
        self.assertEqual(res_ms_partial["score"], 0.5)
        self.assertFalse(res_ms_partial["full_success"])
        res_ms_penalized = grader.grade_submission(t_ms, ["o0", "o1"])
        self.assertEqual(res_ms_penalized["score"], 0.0)

        # output_prediction
        t_op = _make_synthetic_llm_task("output_prediction")
        res_op = grader.grade_submission(t_op, ["[2, 4, 6]", "3"])
        self.assertEqual(res_op["score"], 1.0)
        self.assertTrue(res_op["full_success"])

        # fill_in_the_blank
        t_fib = _make_synthetic_llm_task("fill_in_the_blank")
        res_fib = grader.grade_submission(t_fib, ["2", "8"])
        self.assertEqual(res_fib["score"], 1.0)
        self.assertTrue(res_fib["full_success"])

        # code_completion
        t_cc = _make_synthetic_llm_task("code_completion")
        res_cc = grader.grade_submission(t_cc, t_cc["reference_solution"])
        self.assertEqual(res_cc["score"], 1.0)
        self.assertTrue(res_cc["full_success"])
        self.assertEqual(len(res_cc["criteria_results"]), 3)

    def test_ast_interpreter_pass_fail_syntax_and_unresolved(self):
        t_cc = _make_synthetic_llm_task("code_completion")

        wrong = grader.grade_submission(t_cc, "def solve(x):\n    return [0]")
        self.assertEqual(wrong["grading_status"], "resolved")
        self.assertFalse(wrong["full_success"])

        syn = grader.grade_submission(t_cc, "def solve(x)\n    return []")
        self.assertEqual(syn["grading_status"], "resolved")
        self.assertEqual(syn["score"], 0.0)
        self.assertFalse(syn["full_success"])

        for bad_code in (
            "import os\ndef solve(x):\n    return []",
            "def solve(x):\n    while True:\n        pass",
            "def solve(x):\n    return open('/etc/passwd')",
            "def solve(x):\n    return list(range(5000))",
        ):
            unres = grader.grade_submission(t_cc, bad_code)
            self.assertEqual(unres["grading_status"], "unresolved")
            self.assertIsNone(unres["score"])
            self.assertIsNone(unres["full_success"])

    def test_grade_self_explanation_separates_confidence_from_score(self):
        rubric = {"key_concepts": ["start", "stop", "excluded"], "min_words": 6}
        confident = grader.grade_self_explanation(
            "Explain range boundaries.",
            "Because range begins at start and stops before stop, the stop value is excluded.",
            rubric=rubric,
        )
        hedged = grader.grade_self_explanation(
            "Explain range boundaries.",
            "Maybe start stop excluded i guess not sure",
            rubric=rubric,
        )
        self.assertEqual(confident["score"], 1.0)
        self.assertEqual(hedged["score"], 1.0)
        self.assertTrue(confident["full_success"])
        self.assertTrue(hedged["full_success"])
        self.assertGreater(confident["grader_confidence"], hedged["grader_confidence"])

        empty_res = grader.grade_self_explanation("Explain range boundaries.", "   ", rubric=rubric)
        self.assertEqual(empty_res["grading_status"], "resolved")
        self.assertEqual(empty_res["score"], 0.0)
        self.assertFalse(empty_res["full_success"])

        ambiguous_no_rubric = grader.grade_self_explanation("", "idk", rubric=[])
        self.assertEqual(ambiguous_no_rubric["grading_status"], "unresolved")
        self.assertIsNone(ambiguous_no_rubric["score"])
        self.assertIsNone(ambiguous_no_rubric["full_success"])


if __name__ == "__main__":
    unittest.main()
